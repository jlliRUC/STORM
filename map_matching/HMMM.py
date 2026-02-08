# The code is mostly from Libdataset: https://github.com/Libdataset/Bigsdataset-Libdataset/blob/master/libdataset/model/map_matching/HMMM.py
# Some parts are customized and annotated accordingly

import sys
sys.path.append("../")
from config import Config
import networkx as nx
import math
from logging import getLogger
from GPS_utils import radian2angle, R_EARTH, angle2radian, haversine, init_bearing
import time
from tqdm import tqdm
import pandas as pd
import json
import pickle
import numpy as np
import multiprocessing
import os


class HMMM:
    """
    Hidden Markov Map Matching Through Noise and Sparseness
    """

    def __init__(self, config, data_feature):
        # logger
        self._logger = getLogger()

        # model param
        self.k = config.get('k', 10)  # maximum number of candidates of any sampling points
        self.r = config.get('r', 200)  # radius of road segment candidate
        self.mu = config.get('mu', 0)  # always 0 in HMMM
        self.sigma_z = config.get('sigma_z', 10)  # jiali: Filter out small noise
        self.sigma_w = config.get('sigma_w', 1000)  # jiali: Filter out outlier noise
        self.beta = config.get('beta', 5)
        # mem_rate is not introduced in the paper, however, it can lead to better performance
        self.mem_rate = config.get('mem_rate', 0.5)

        # data param
        self.with_time = data_feature.get('with_time', False)
        self.with_rd_speed = data_feature.get('with_rd_speed', False)
        self.delta_time = data_feature.get('delta_time', False)

        # data cache
        self.rd_nwk = None  # road network
        self.usr_id = None  # current user id
        self.traj_id = None  # current trajectory id
        self.trajectory = None  # current trajectory
        self.candidates = list()  # candidates
        self.res_dct = dict()  # result

        # To facilitate the search of candidate points, the road network is indexed using a grid.
        self.lon_r = None
        self.lat_r = None

    def run(self, rd_nwk, df_trajectories):
        # "data" consists of road network & trajectory
        self.rd_nwk = rd_nwk  # data['rd_nwk'] consists of nodes and edges

        # set lon_radius and lat_radius based on the first Node of rd_nwk
        self._set_lon_lat_radius(
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lon'],
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lat']
        )

        # Traverse all trajectories
        for i in tqdm(range(df_trajectories.shape[0])):
            self._logger.info(f'begin map matching, traj_id:{i}')
            self.traj_id = i
            self.trajectory = df_trajectories.loc[i, "Locations"]  # np.array
            self._run_one_traj()
            self._logger.info(f'finish map matching, traj_id:{i}')

        return self.res_dct

    def _run_one_traj(self):
        """
        run HMMM for one trajectory
        self.trajectory self.rd_nwk
        Returns:
        """
        self._preprocess()
        self._logger.info('finish preprocessing')
        self._get_candidates()
        self._logger.info('finish getting candidates')
        self._observation_probability()
        self._logger.info('finish calculating observation probability')
        self._transmission_probability()
        self._logger.info('finish calculating transmission probability')
        self._find_matched_sequence()
        self._logger.info('finish finding matched sequence')
        self.candidates = list()

    def run_valids(self, rd_nwk, df_trajectories):
        # Some of trajectory points might be filtered out due to "noises" in preprocess. Get the indexes of valid points,
        # so as to get the corresponding timestamps.
        self.rd_nwk = rd_nwk

        # set lon_radius and lat_radius based on the first Node of rd_nwk
        self._set_lon_lat_radius(
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lon'],
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lat']
        )

        # Traverse all trajectories
        valids_list = []
        for i in tqdm(range(df_trajectories.shape[0])):
            self._logger.info(f'begin map matching, traj_id:{i}')
            self.traj_id = i
            self.trajectory = df_trajectories.loc[i, "Locations"]  # np.array
            idx = self._preprocess()
            self._logger.info('finish preprocessing')
            valids_list.append(idx)

        return valids_list

    def run_aug(self, rd_nwk, df_trajectories, k):
        # Find multiple map-matched paths
        self.rd_nwk = rd_nwk

        # set lon_radius and lat_radius based on the first Node of rd_nwk
        self._set_lon_lat_radius(
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lon'],
            self.rd_nwk.nodes[list(self.rd_nwk.nodes)[0]]['lat']
        )

        # Traverse all trajectories
        for i in tqdm(range(df_trajectories.shape[0])):
            self._logger.info(f'begin map matching, traj_id:{i}')
            self.traj_id = i
            self.trajectory = df_trajectories.loc[i, "Locations"]  # np.array
            self._run_aug_one_traj(k)
            self._logger.info(f'finish map matching, traj_id:{i}')

        return self.res_dct

    def _run_aug_one_traj(self, k):
        """
        run HMMM for one trajectory
        self.trajectory self.rd_nwk
        Returns:
        """
        self._preprocess()
        self._logger.info('finish preprocessing')
        self._get_candidates()
        self._logger.info('finish getting candidates')
        self._observation_probability()
        self._logger.info('finish calculating observation probability')
        self._transmission_probability()
        self._logger.info('finish calculating transmission probability')
        self._find_k_matched_sequence(k)
        self._logger.info('finish finding matched sequence')
        self.candidates = list()

    def _preprocess(self):
        """
        removing points that are within 2 sigma_z of the previous included point.
        The justification for this step is that until we see a point that is at least 2sigma_z away from
        its temporal predecessor, our confidence is low that the apparent movement is due to
        actual vehicle movement and not noise

        jiali:
        1. Note that sigma_z should be related to the type of moving object and the original sampling rate.
        If it is a slow moving object (e.g., pedestrian) and with a high sampling rate, sigma_z should be small.
        2. Cars can stop or drive slowly due to the traffic congestion, so sigma_z should be small or zero.
        3. Instead, we should set another sigma_w, which filter out the outliers, i.e., two consecutive points cannot be
        too far way. Taking Porto as an example, its sampling rate is 15 s, the maximum speed of car is no more than 210 km/h
        (60 m/s). So sigma_w should be around 1000.
        Returns:

        """

        new_traj = list()
        lat_r_pre = None  # latitude in radius
        lon_r_pre = None  # longitude in radius
        idx = []
        for i in range(len(self.trajectory)):
            lat_r, lon_r = self.trajectory[i][1], self.trajectory[i][0]
            # if lat_r_pre is None or dist(lat_r, lon_r, lat_r_pre, lon_r_pre) > 2 * self.sigma_z:
            # jiali: modify dist() to haversine()
            if lat_r_pre is None or haversine(lon_r, lat_r, lon_r_pre, lat_r_pre) > 2 * self.sigma_z:
                # if distance > 2*sigma_z, take this point into calculation, otherwise remove it
                new_traj.append(self.trajectory[i])
                lat_r_pre = lat_r
                lon_r_pre = lon_r
                idx.append(i)
        self.trajectory = np.vstack(new_traj)

        return idx

    def _set_lon_lat_radius(self, lon, lat):
        """
        get longitude range & latitude range (because radius is actually achieved by a grid search)
        Args:
            lon: longitude local
            lat: latitude local
            self.r
        Returns:
            self.lon_r
            self.lat_r
        """

        # lat_r
        self.lat_r = radian2angle(self.r / R_EARTH)

        # lon_r
        r_prime = R_EARTH * math.cos(angle2radian(lat))
        self.lon_r = radian2angle(self.r / r_prime)

    def _point_edge_dist(self, lon, lat, edge):
        """
        lat_origin = angle2radian(self.rd_nwk.nodes[edge[0]]['lat'])
        lon_origin = angle2radian(self.rd_nwk.nodes[edge[0]]['lon'])
        lat_dest = angle2radian(self.rd_nwk.nodes[edge[1]]['lat'])
        lon_dest = angle2radian(self.rd_nwk.nodes[edge[1]]['lon'])

        a = dist(angle2radian(lat), angle2radian(lon), lat_origin, lon_origin)
        b = dist(angle2radian(lat), angle2radian(lon), lat_dest, lon_dest)
        c = dist(lat_origin, lon_origin, lat_dest, lon_dest)
        """
        node_origin = self.rd_nwk.nodes[edge[0]]
        node_dest = self.rd_nwk.nodes[edge[1]]
        lat_origin = node_origin['lat']
        lon_origin = node_origin['lon']
        lat_dest = node_dest['lat']
        lon_dest = node_dest['lon']

        a = haversine(lon, lat, lon_origin, lat_origin)
        b = haversine(lon, lat, lon_dest, lat_dest)
        c = haversine(lon_origin, lat_origin, lon_dest, lat_dest)

        # if origin point is the closest
        if b ** 2 > a ** 2 + c ** 2:
            return a, edge[0]  # distance, point

        # if destination point is the closest
        elif a ** 2 > b ** 2 + c ** 2:
            return b, edge[1]

        if c == 0:
            return a, edge[0]
        # otherwise, calculate the Vertical length
        p = (a + b + c) / 2
        s = math.sqrt(p * math.fabs(p - a) * math.fabs(p - b) * math.fabs(p - c))
        return 2 * s / c, None

    def _get_candidates(self):
        """
        get candidates of each GPS sample with given road network
        Returns:
            self.candidates: a list of list.
                In each list are tuples (edge, distance, node)
        """

        # get trajectory without time
        traj_lon_lat = self.trajectory[:, 0:2]
        assert traj_lon_lat.shape[1] == 2

        # for every GPS sample
        for i in range(traj_lon_lat.shape[0]):

            candidate_i = set()
            lon, lat = traj_lon_lat[i, :]

            # for every edge
            for j in self.rd_nwk.edges:
                origin, dest = j[:2]
                lat_origin = self.rd_nwk.nodes[origin]['lat']
                lon_origin = self.rd_nwk.nodes[origin]['lon']
                lat_dest = self.rd_nwk.nodes[dest]['lat']
                lon_dest = self.rd_nwk.nodes[dest]['lon']
                if lat - self.lat_r <= lat_origin <= lat + self.lat_r \
                        and lon - self.lon_r <= lon_origin <= lon + self.lon_r \
                        or lat - self.lat_r <= lat_dest <= lat + self.lat_r \
                        and lon - self.lon_r <= lon_dest <= lon + self.lon_r:
                    candidate_i.add((origin, dest))
                elif lat - self.lat_r <= lat_origin / 2 + lat_dest / 2 <= lat + self.lat_r \
                        and lon - self.lon_r <= lon_origin / 2 + lon_dest / 2 <= lon + self.lon_r:
                    candidate_i.add((origin, dest))
            candidate_i_m = list()  # (edge, distance, point)
            for edge in candidate_i:
                distance, node = self._point_edge_dist(lon, lat, edge)
                candidate_i_m.append((edge, distance, node))
            candidate_i_m.sort(key=lambda a: a[1])  # asc
            candidate_i_m = candidate_i_m[:min(self.k, len(candidate_i_m))]
            candidate_i_k = dict()
            for edge, distance, node in candidate_i_m:
                candidate_i_k[edge] = {'distance': distance, 'node': node}
            self.candidates.append(candidate_i_k)

    def _observation_probability(self):
        """

        Returns:

        """

        # for candidates of every node
        for candidate_i in self.candidates:
            for edge, dct in candidate_i.items():
                candidate_i[edge]['N'] = (1 / math.sqrt(2 * math.pi) / self.sigma_z * math.exp(
                    - (dct['distance'] - self.mu) ** 2 / (2 * self.sigma_z ** 2)))

    def _transmission_probability(self):
        """

        Returns:

        """
        i = 1
        while i < len(self.candidates):
            # j and k
            j = i - 1
            if len(self.candidates[i]) == 0:
                k = i + 1
                while k < len(self.candidates) and len(self.candidates[k]) == 0:
                    k += 1
                if k == len(self.candidates):
                    break
            else:
                k = i
            """
            d = dist(
                angle2radian(self.trajectory[j][2]),
                angle2radian(self.trajectory[j][1]),
                angle2radian(self.trajectory[k][2]),
                angle2radian(self.trajectory[k][1])
            )  # great circle distance
            """
            d = haversine(
                self.trajectory[j][0],
                self.trajectory[j][1],
                self.trajectory[k][0],
                self.trajectory[k][1]
            )  # great circle distance
            for edge_j, dct_j in self.candidates[j].items():
                for edge_k, dct_k in self.candidates[k].items():
                    brng_jk = init_bearing(
                        angle2radian(self.trajectory[j][1]),
                        angle2radian(self.trajectory[j][0]),
                        angle2radian(self.trajectory[k][1]),
                        angle2radian(self.trajectory[k][0])
                    )
                    brng_edge_j = init_bearing(
                        angle2radian(self.rd_nwk.nodes[edge_j[0]]['lat']),
                        angle2radian(self.rd_nwk.nodes[edge_j[0]]['lon']),
                        angle2radian(self.rd_nwk.nodes[edge_j[1]]['lat']),
                        angle2radian(self.rd_nwk.nodes[edge_j[1]]['lon']),
                    )
                    try:
                        if dct_j['node'] is not None and dct_k['node'] is not None:
                            dt = abs(d - nx.astar_path_length(self.rd_nwk, dct_j['node'], dct_k['node'],
                                                              weight='distance'))
                        elif dct_j['node'] is not None:
                            nd2_origin = edge_k[0]
                            lon, lat = self.rd_nwk.nodes[nd2_origin]['lon'], self.rd_nwk.nodes[nd2_origin]['lat']
                            path_len = nx.astar_path_length(self.rd_nwk, dct_j['node'], nd2_origin, weight='distance')
                            path_len += math.sqrt(
                                math.fabs(
                                    haversine(
                                        self.trajectory[k][0],
                                        self.trajectory[k][1],
                                        lon,
                                        lat
                                    ) ** 2 - dct_k['distance'] ** 2
                                )
                            )
                            if edge_j[1] == dct_j['edge']:
                                path_len += self.rd_nwk[edge_j[0]][edge_j[1]]['distance'] * 2
                            dt = abs(d - path_len)

                        elif dct_k['node'] is not None:
                            nd1_destination = edge_j[1]
                            lon, lat = self.rd_nwk.nodes[nd1_destination]['lon'], self.rd_nwk.nodes[nd1_destination][
                                'lat']
                            path_len = nx.astar_path_length(self.rd_nwk, nd1_destination, dct_k['node'],
                                                            weight='distance')
                            path_len += math.sqrt(
                                math.fabs(
                                    haversine(
                                        self.trajectory[j][0],
                                        self.trajectory[j][1],
                                        lon,
                                        lat
                                    ) ** 2 - dct_j['distance'] ** 2
                                )
                            )
                            if edge_k[1] == dct_k['node']:
                                path_len += self.rd_nwk[edge_k[0]][edge_k[1]]['distance'] * 2
                            dt = abs(d - path_len)
                        else:
                            if edge_j == edge_k and math.fabs(brng_edge_j - brng_jk) < 90:
                                dt = 1
                            else:
                                nd1_destination = edge_j[1]
                                lon1, lat1 = self.rd_nwk.nodes[nd1_destination]['lon'], \
                                    self.rd_nwk.nodes[nd1_destination]['lat']
                                nd2_origin = edge_k[0]
                                lon2, lat2 = self.rd_nwk.nodes[nd2_origin]['lon'], self.rd_nwk.nodes[nd2_origin]['lat']
                                dt = abs(d - (
                                        nx.astar_path_length(self.rd_nwk, nd1_destination, nd2_origin,
                                                             weight='distance')
                                        + math.sqrt(
                                            math.fabs(
                                                haversine(
                                                    self.trajectory[j][0],
                                                    self.trajectory[j][1],
                                                    lon1,
                                                    lat1
                                                ) ** 2 - dct_j['distance'] ** 2
                                            )
                                        )
                                        + math.sqrt(
                                            math.fabs(
                                                haversine(
                                                    self.trajectory[k][0],
                                                    self.trajectory[k][1],
                                                    lon2,
                                                    lat2
                                                ) ** 2 - dct_k['distance'] ** 2
                                            )
                                        )
                                ))
                        result = 1 / self.beta * math.exp(-dt / self.beta)
                        if 'V' in dct_j.keys():
                            dct_j['V'][edge_k] = min(result, 1)
                        else:
                            dct_j['V'] = {edge_k: min(result, 1)}
                    except:
                        if 'V' in dct_j.keys():
                            dct_j['V'][edge_k] = 0
                        else:
                            dct_j['V'] = {edge_k: 0}
            i += 1

    def _find_matched_sequence(self):
        """
        Viterbi Algorithm
        Returns:

        """
        pre = list()
        # for every GPS sample
        for i in range(len(self.candidates)):
            # current candidates: self.candidates[i]
            # prev candidates: self.candidates[j]
            pre_i = dict()
            # no current candidates
            if len(self.candidates[i]) == 0:
                pre.append(None)
                continue
            # if there are current candidates, find prev index j
            j = i - 1
            while j >= 0 and len(self.candidates[j]) == 0:
                j -= 1
            # if j < 0, then i is the first valid index with candidates, score = N
            if j < 0:
                nodes = self.candidates[i]  # jiali: Here nodes is the node of HMM model, not the road network
                for edge, dct in nodes.items():
                    dct['score'] = dct['N']  # jiali: 'N' is the observation probability
                    pre_i[edge] = None
                pre.append(pre_i)
                continue
            # j >= 0, calculate score of candidates of GPS sample i
            for edge, dct in self.candidates[i].items():
                max_score = -float("inf")
                for edge_pre, dct_pre in self.candidates[j].items():
                    tmp = dct_pre['score'] + dct['N'] * dct_pre['V'][edge] * (
                        1 if 'T' not in dct_pre.keys() else dct_pre['T'][edge])  # jiali: No 'T' in this version. Does it mean temporal?
                    if tmp > max_score:
                        max_score = tmp
                        pre_i[edge] = edge_pre
                dct['score'] = max_score * self.mem_rate
            pre.append(pre_i)
        assert len(pre) == len(self.trajectory)

        res_lst = []
        e = None
        for i in range(len(pre) - 1, -1, -1):

            if e is None:
                if pre[i] is not None:
                    # if there's not e, and current i have candidates, init e.
                    e = max(self.candidates[i], key=lambda k: self.candidates[i][k]['score'])
                    res_lst.append(e)
                else:
                    # if there's not e, and current i have no candidates, result is None
                    res_lst.append(None)
            else:
                if pre[i + 1] is not None:
                    # if there's an e, and current i+1 have candidates, do the 'pre' thing
                    e = pre[i + 1][e]
                    res_lst.append(e)
                else:
                    # if there's an e, and current i+1 have no candidates, result is None
                    res_lst.append(None)

        res_lst.reverse()

        # to geo_id
        #res_lst_rel = np.array(list(map(lambda x: self.rd_nwk.edges[x]['geo_id'] if x is not None else None, res_lst)))
        res_lst_rel = np.array(list(map(lambda x: self.rd_nwk.edges[x]['edge_id'] if x is not None else None, res_lst)))
        #dyna_id_lst = self.trajectory[:, 0].astype(int)
        if self.with_time:
            time_lst = self.trajectory[:, 3]
            #res_all = np.stack([dyna_id_lst, res_lst_rel, time_lst], axis=1)
            res_all = np.stack([res_lst_rel, time_lst], axis=1)
        else:
            #res_all = np.stack([dyna_id_lst, res_lst_rel], axis=1)
            res_all = res_lst_rel

        # set self.res_dct
        if self.usr_id in self.res_dct.keys():
            self.res_dct[self.usr_id][self.traj_id] = res_all

        else:
            self.res_dct[self.traj_id] = res_all

    def _find_k_matched_sequence(self, k):
        """
        top-k Viterbi Algorithm
        Returns:

        """
        #assert k <= self.k  # k here must be less than the number of candidates set by self.k

        pre = list()
        for i in range(k):
            pre.append([])
        # for every GPS sample
        for i in range(len(self.candidates)):
            # current candidates: self.candidates[i]
            # prev candidates: self.candidates[j]
            pre_i = [dict() for sub_k in range(k)]
            # no current candidates
            if len(self.candidates[i]) == 0:

                for sub_k in range(k):
                    pre[sub_k].append(None)
                continue
            # if there are current candidates, find prev index j
            j = i - 1
            while j >= 0 and len(self.candidates[j]) == 0:
                j -= 1
            # if j < 0, then i is the first valid index with candidates, score = N
            if j < 0:
                nodes = self.candidates[i]  # jiali: Here nodes is the node of HMM model, not the road network
                for edge, dct in nodes.items():
                    dct['score'] = [dct['N'] for sub_k in range(k)]  # jiali: 'N' is the observation probability
                    for sub_k in range(k):
                        pre_i[sub_k][edge] = None
                for sub_k in range(k):
                    pre[sub_k].append(pre_i[sub_k])
                continue
            # j >= 0, calculate score of candidates of GPS sample i
            for edge, dct in self.candidates[i].items():
                max_score_list = [-float("inf") for sub_k in range(k)]
                for k1 in range(k):
                    # jiali: make sure the first path is the best path
                    for edge_pre, dct_pre in self.candidates[j].items():
                        tmp = dct_pre['score'][k1] + dct['N'] * dct_pre['V'][edge] * (
                            1 if 'T' not in dct_pre.keys() else dct_pre['T'][
                                edge])  # jiali: No 'T' in this version. Does it mean temporal?
                        # jiali: the current score must be larger than the best score.
                        # Besides, unless it is the first path, its score should be smaller than the previous one
                        # for both sorting and avoiding overlapping
                        if tmp > max_score_list[k1] and (k1 == 0 or tmp < max_score_list[k1 - 1]):
                            max_score_list[k1] = tmp
                            pre_i[k1][edge] = edge_pre
                dct['score'] = [item * self.mem_rate for item in max_score_list]
            for sub_k in range(k):
                pre[sub_k].append(pre_i[sub_k])
        assert len(pre[0]) == len(self.trajectory)

        res_lst = [[] for sub_k in range(k)]
        for sub_k in range(k):
            e = None
            for i in range(len(pre[sub_k]) - 1, -1, -1):
                if e is None:
                    if pre[sub_k][i] is not None:
                        # if there's not e, and current i have candidates, init e.
                        e = max(self.candidates[i], key=lambda x: self.candidates[i][x]['score'])
                        res_lst[sub_k].append(e)
                    else:
                        # if there's not e, and current i have no candidates, result is None
                        res_lst[sub_k].append(None)
                else:
                    # TODO: something wrong here about the e = pre[sub_k][i + 1][e]
                    #if pre[sub_k][i + 1] is not None:
                    if pre[sub_k][i + 1] is not None and e in pre[sub_k][i+1].keys():
                        # if there's an e, and current i+1 have candidates, do the 'pre' thing
                        e = pre[sub_k][i + 1][e]
                        res_lst[sub_k].append(e)
                    else:
                        # if there's an e, and current i+1 have no candidates, result is None
                        res_lst[sub_k].append(None)

        for sub_k in range(k):
            res_lst[sub_k].reverse()

        # to geo_id
        # res_lst_rel = np.array(list(map(lambda x: self.rd_nwk.edges[x]['geo_id'] if x is not None else None, res_lst)))
        res_lst_rel = []
        for sub_k in range(k):
            res_lst_rel.append(np.array(list(map(lambda x: self.rd_nwk.edges[x]['edge_id'] if x is not None else None, res_lst[sub_k]))))

        res_all = res_lst_rel
        # set self.res_dct
        if self.usr_id in self.res_dct.keys():
            self.res_dct[self.usr_id][self.traj_id] = res_all

        else:
            self.res_dct[self.traj_id] = res_all

        return self.res_dct


def get_road_network(dataset_name):
    road_network = nx.DiGraph()

    nodes = pd.read_csv(f"/home/jiali/project2/data/{dataset_name}/{dataset_name}_node.csv", usecols=['node_idx', 'lon', 'lat'])
    for i in range(nodes.shape[0]):
        road_network.add_node(nodes.loc[i, 'node_idx'], lon=nodes.loc[i, 'lon'], lat=nodes.loc[i, 'lat'])

    edges = pd.read_csv(f"/home/jiali/project2/data/{dataset_name}/{dataset_name}_segment.csv", usecols=['edge_id', 's_idx', 'e_idx', 's_lon', 's_lat', 'e_lon', 'e_lat', 'c_lon', 'c_lat', 'length'])
    for j in range(edges.shape[0]):
        road_network.add_edge(edges.loc[j, 's_idx'],
                              edges.loc[j, 'e_idx'],
                              edge_id=edges.loc[j, 'edge_id'],
                              distance=edges.loc[j, 'length'],
                              s_lon=edges.loc[j, 's_lon'],
                              s_lat=edges.loc[j, 's_lat'],
                              e_lon=edges.loc[j, 'e_lon'],
                              e_lat=edges.loc[j, 'e_lat'],
                              c_lon=edges.loc[j, 'c_lon'],
                              c_lat=edges.loc[j, 'c_lat'])

    return road_network


def run_single(dataset_name, df_trajectories, k, group, suffix):
    # suffix is ["train", "val", "finetune", "test"]
    data_feature = {"default": 0}
    with open("HMMM.json", "r") as f:
        mapmatching_configs = json.load(f)

    mapmatching_model = HMMM(mapmatching_configs, data_feature)
    road_network = get_road_network(dataset_name)

    print(f"Group {group} Suffix {suffix}")
    t1 = time.time()
    result = mapmatching_model.run_aug(road_network, df_trajectories, k)
    print(f"Group {group} suffix {suffix} costs {time.time() - t1} s")

    result_mm = []
    result_aug = []
    for i in range(len(result)):
        result_mm.append(result[i][0])
        result_aug.append(result[i][1])
    if not os.path.exists(f"../data/{dataset_name}/parallel_repo/mm"):
        os.makedirs(f"../data/{dataset_name}/parallel_repo/mm")
    if not os.path.exists(f"../data/{dataset_name}/parallel_repo/aug"):
        os.makedirs(f"../data/{dataset_name}/parallel_repo/aug")
    with open(f"../data/{dataset_name}/parallel_repo/mm/{suffix}_{group}.pkl", "wb") as f:
        pickle.dump(result_mm, f)
    with open(f"../data/{dataset_name}/parallel_repo/aug/{suffix}_{group}.pkl", "wb") as f:
        pickle.dump(result_aug, f)


def get_valids(dataset_name):
    data_feature = {"default": 0}
    with open("HMMM.json", "r") as f:
        mapmatching_configs = json.load(f)

    mapmatching_model = HMMM(mapmatching_configs, data_feature)
    road_network = get_road_network(dataset_name)

    with open(f"../data/{dataset_name}/{dataset_name}.pkl", "rb") as f:
        df_trajectories = pickle.load(f)

    valids_list = mapmatching_model.run_valids(road_network, df_trajectories)

    with open(f"../data/{dataset_name}/{dataset_name}_valids.pkl", "wb") as f:
        pickle.dump(valids_list, f)
    return valids_list


def map_matching(configs):
    dataset_name = configs.dataset_name
    with open(f"../data/{dataset_name}/{dataset_name}.pkl", "rb") as f:
        df_trajectories = pickle.load(f)

    params = []
    df_train = df_trajectories.iloc[0: configs.num_train]
    df_train.reset_index(drop=True, inplace=True)
    num_group = df_train.shape[0] // 1000
    for i in range(num_group):
        df_input = df_train.iloc[i*1000: i*1000+1000]
        df_input.reset_index(drop=True, inplace=True)
        params.append((dataset_name, df_input, 2, i, "train"))

    df_val = df_trajectories.iloc[configs.num_train: configs.num_train+configs.num_val]
    df_val.reset_index(drop=True, inplace=True)
    num_group = df_val.shape[0] // 1000
    for i in range(num_group):
        df_input = df_val.iloc[i*1000: i*1000+1000]
        df_input.reset_index(drop=True, inplace=True)
        params.append((dataset_name, df_input, 2, i, "val"))
    df_fine_tune = df_trajectories.iloc[configs.num_train+configs.num_val: configs.num_train+configs.num_val+configs.num_fine_tune]
    df_fine_tune.reset_index(drop=True, inplace=True)
    num_group = df_fine_tune.shape[0] // 1000
    #for i in range(num_group):
    for i in [0]:
        df_input = df_fine_tune.iloc[i*1000: i*1000+1000]
        df_input.reset_index(drop=True, inplace=True)
        params.append((dataset_name, df_input, 2, i, "finetune"))

    df_test = df_trajectories.iloc[configs.test_start: configs.test_start+configs.num_test]
    df_test.reset_index(drop=True, inplace=True)
    num_group = df_test.shape[0] // 1000
    for i in range(num_group):
        df_input = df_test.iloc[i*1000: i*1000+1000]
        df_input.reset_index(drop=True, inplace=True)
        params.append((dataset_name, df_input, 2, i, "test"))

    pool_single = multiprocessing.Pool(processes=20)
    pool_single.starmap_async(run_single, params)
    pool_single.close()
    pool_single.join()


def merge(configs):
    scale = 1000
    for suffix in ["train", "val", "finetune", "test"]:
    #for suffix in ["train"]:
        if suffix == "train":
            num_group = configs.num_train // scale
            offset = 0
        elif suffix == "val":
            num_group = configs.num_val // scale
            offset = configs.num_train
        elif suffix == "finetune":
            num_group = configs.num_fine_tune // scale
            offset = configs.num_train + configs.num_val
        elif suffix == "test":
            num_group = configs.num_test // scale
            offset = configs.test_start
        for folder_name in ["mm", "aug"]:
            folder = f"../data/{configs.dataset_name}/parallel_repo/{folder_name}"
            segments_list = []
            for i in range(num_group):
                with open(f"{folder}/{suffix}_{i}.pkl", "rb") as f:
                    result_sub = pickle.load(f)
                    assert len(result_sub) == 1000
                    segments_list += result_sub
            df = pd.DataFrame.from_dict({'road_segments': segments_list})
            df.insert(loc=0, column='traj_id', value=[offset+j for j in range(df.shape[0])])
            print(f"Processed {df.shape[0]} trajectories.")
            with open(f"../data/{configs.dataset_name}/{configs.dataset_name}_{suffix}_{folder_name}_map_matching.pkl", "wb") as f:
                pickle.dump(df, f)


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    map_matching(configs)
    merge(configs)
    get_valids(configs.dataset_name)



