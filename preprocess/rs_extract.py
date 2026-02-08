"""
Note that we model the road network graph by considering the road segments as nodes, while the road intersections as edges.
1. Feature Tokenization:
1.1. Location: Disretization of start nodes' location of each segment  (Need a cell size, default 100 m)
1.2. Radian_token: Discretization of radian of each segment (Need a radian_unit)
1.3. Length_token: Discretization of length of each segment (Need a length_unit)
1.4. cls_token:

2. Edge weights: Different road segments may have different effects on passing trajectories.
Thus, we modify the edge weight between two connected road segments based on the potential effect

"""
import sys
sys.path.append("../")
sys.path.append("../preprocess")
import pandas as pd
from GPS_utils import haversine, lonlat2meters, meters2lonlat
import pickle
import math
import os
from config import Config
import time
import multiprocessing


class Location2Token:
    def __init__(self, segment_file, dataset_name, min_lon, min_lat, max_lon, max_lat, x_step, y_step):
        """
        :param segment_file: dataset name
        :param dataset_name: dataset name
        :param min_lon: Boundary, GPS coordinate
        :param min_lat: Boundary, GPS coordinate
        :param max_lon: Boundary, GPS coordinate
        :param max_lat: Boundary, GPS coordinate
        :param x_step: Cell size x (meters)
        :param y_step: Cell size y (meters)
        """
        self.segment_file = segment_file
        self.dataset_name = dataset_name
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.min_x, self.min_y = lonlat2meters(min_lon, min_lat)
        self.max_x, self.max_y = lonlat2meters(max_lon, max_lat)
        self.num_x = math.ceil(round((self.max_x - self.min_x), 6) / x_step)
        self.num_y = math.ceil(round((self.max_x - self.min_x), 6) / y_step)
        self.x_step = x_step
        self.y_step = y_step

    def coord2cell(self, x, y):
        """
        Calculate the cell ID of point (x, y) based on the corresponding offsets.
        :param x: spherical meters
        :param y: spherical meters
        :return:
        """
        x_offset = math.floor(round((x - self.min_x), 6) / self.x_step)
        y_offset = math.floor(round((y - self.min_y), 6) / self.y_step)

        return y_offset * self.num_x + x_offset

    def cell2coord(self, cell):
        """
        From cell ID to the coordinates of point in spherical meters. Here the point is actually the cell center.
        :param cell:
        :return:
        """
        y_offset = cell // self.num_x
        x_offset = cell % self.num_x
        y = self.min_y + (y_offset + 0.5) * self.y_step
        x = self.min_x + (x_offset + 0.5) * self.x_step

        return x, y

    def gps2cell(self, lon, lat):
        """
        From point (lon, lat) to cell ID
        :param lon:
        :param lat:
        :return:
        """
        x, y = lonlat2meters(lon, lat)

        return self.coord2cell(x, y)

    def cell2gps(self, cell):
        """
        From cell ID to the coordinates of point in GPS coordinates. Here the point is actually the cell center.
        :param cell:
        :return:
        """
        x, y = self.cell2coord(cell)

        return meters2lonlat(x, y)

    def makeVocab(self):
        """
        Build the vocabulary from the raw trajectories stored in the hdf5 file.
        For one trajectory, each point lies in a row if reading with Python.
        :return:
        """

        vocab_file = f"/home/jiali/project2/data/{self.dataset_name}/{self.dataset_name}_cell-{self.x_step}_grid_vocab.pkl"

        if not os.path.exists(vocab_file):
            start = time.time()
            num_out_region = 0  # useful for evaluating the size of region bounding box
            df = pd.read_csv(self.segment_file)

            num = df.shape[0]

            # cell_generation
            cell_list = []
            print("Cell generation begins")
            for i in range(0, num):
                cell_list.append(self.gps2cell(df.loc[i, "s_lon"], df.loc[i, "s_lat"]))
                cell_list.append(self.gps2cell(df.loc[i, "e_lon"], df.loc[i, "e_lat"]))

            print(f"Cell generation ends, we got {len(cell_list)} cells from the trajectories")

            # Find out all hot cells
            self.hotcell = list(set(cell_list))
            print(f"We have {len(self.hotcell)} cells for road network of {self.dataset_name}")

            print("Build the map between cell and vocab id")
            self.hotcell2vocab = dict([(v, k) for k, v in enumerate(self.hotcell)])
            self.vocab2hotcell = dict([(v, k) for k, v in self.hotcell2vocab.items()])

            print("Calculate vocabulary size")
            self.vocab_size = len(self.hotcell)
            print(f"vocab_size is {self.vocab_size}")

            self.saveregion(vocab_file)
            print(f"Time cost: {time.time() - start} s")
        else:
            self.built = True
            print(f"Loading Vocabulary from {vocab_file}")
            self.loadregion(vocab_file)

        return len(self.hotcell)

    def cell2vocab(self, cell):
        """
        Return the vocab id for a cell in the region.
        If the cell is not hot cell, the function will first search its nearest hotcell and return the corresponding vocab id
        :param region:
        :param cell:
        :return:
        """
        assert self.built, "Build index for region first"
        if cell in self.hotcell2vocab.keys():
            return self.hotcell2vocab[cell]
        else:
            print("Unseen cell id!")

    def gps2vocab(self, lon, lat):
        """
        Mapping a gps point to the vocab id in the vocabulary consists of hot cells,
        each hot cell has an unique vocab id (hotcell2vocab)
        If the point falls out of the region, 'UNK' will be returned.
        If the point falls into the region, but out of the hot cells, its nearest hot cell will be used.
        :param lon:
        :param lat:
        :return:
        """
        return self.cell2vocab(self.gps2cell(max(min(lon, self.max_lon), self.min_lon), max(min(lat, self.max_lat), self.min_lat)))

    def saveregion(self, param_file):
        """
        :param param_file:
        :return:
        """
        with open(param_file, 'wb') as f:
            pickle.dump({"hotcell": self.hotcell,
                         "hotcell2vocab": self.hotcell2vocab,
                         "vocab2hotcell": self.vocab2hotcell,
                         "vocab_size": self.vocab_size}, f)
        f.close()

    def loadregion(self, param_file):
        with open(param_file, 'rb') as f:
            region_temp = pickle.load(f)
            self.hotcell = region_temp["hotcell"]
            self.hotcell2vocab = region_temp["hotcell2vocab"]
            self.vocab2hotcell = region_temp["vocab2hotcell"]
            self.vocab_size = region_temp["vocab_size"]
        f.close()


def cls2weights(cls):
    highway_cls_to_weight = {
        0: 6.0,
        1: 5.0,
        2: 4.0,
        3: 3.0,
        4: 2.0,
        5: 1.0,
    }

    return highway_cls_to_weight[cls]


def generate_edge_single(group, dataset_name, node_index):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    edge_index_list = []
    for i in node_index:
        for j in range(df_segments.shape[0]):
            if df_segments.loc[i, "s_idx"] == df_segments.loc[j, "e_idx"] and df_segments.loc[i, "e_idx"] != df_segments.loc[j, "s_idx"]:
                edge_index_list.append([i, j])
            elif df_segments.loc[i, "e_idx"] == df_segments.loc[j, "s_idx"] and df_segments.loc[i, "s_idx"] != df_segments.loc[j, "e_idx"]:
                edge_index_list.append([j, i])

    with open(f"../data/{dataset_name}/parallel_repo/edge_index/{dataset_name}_road_network_edge_index_{group}.pkl", "wb") as f:
        pickle.dump(edge_index_list, f)
    print(f"Group {group} finished. we have {len(edge_index_list)} edges")


def generate_edge(dataset_name):
    params = []
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    if not os.path.exists(f"../data/{dataset_name}/parallel_repo/edge_index"):
        os.makedirs(f"../data/{dataset_name}/parallel_repo/edge_index")
    num_group = math.ceil(df_segments.shape[0] / 3000)
    print(f"we have {num_group} groups")
    for i in range(num_group - 1):
        params.append((i, dataset_name, [i*3000+j for j in range(3000)]))
    params.append((num_group-1, dataset_name, [i for i in range((num_group-1)*3000, df_segments.shape[0])]))
    print(len(params))
    pool_single = multiprocessing.Pool(processes=20)
    pool_single.starmap_async(generate_edge_single, params)
    pool_single.close()
    pool_single.join()


def generate_edge_merge(dataset_name):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    num_group = math.ceil(df_segments.shape[0] / 3000)
    edge_index_list = []
    for i in range(num_group):
        with open(f"../data/{dataset_name}/parallel_repo/edge_index/{dataset_name}_road_network_edge_index_{i}.pkl", "rb") as f:
            edge_index_temp = pickle.load(f)
            print(f"For group {i}, we have {len(edge_index_temp)} edges")
            edge_index_list += edge_index_temp
    edge_index_list = [(item[0], item[1]) for item in edge_index_list]
    edge_index_list = list(set(edge_index_list))

    with open(f"../data/{dataset_name}/{dataset_name}_edge_index.pkl", "wb") as f:
        pickle.dump(edge_index_list, f)
    print(f"We have {len(edge_index_list)} edges in total")


def tokenization(configs):
    """
    Tokenize the road segment features including highway type, length, radian, and locations of start and end points,
    :param dataset_name:
    :return:
    """
    segment_file = f"../data/{configs.dataset_name}/{configs.dataset_name}_segment.csv"
    df_segments = pd.read_csv(segment_file)
    loc_token_generator = Location2Token(segment_file, configs.dataset_name, configs.min_lon, configs.min_lat, configs.max_lon, configs.max_lat, configs.cell_size, configs.cell_size)
    num_loc_token = loc_token_generator.makeVocab()
    df_segments["length_token"] = df_segments["length"].apply(lambda row: row / configs.seg_length_unit).astype("int64")
    num_length_token = df_segments["length_token"].max() + 1
    df_segments["radian_token"] = df_segments["radian"].apply(lambda row: row / configs.seg_radian_unit).astype("int64")
    num_radian_token = df_segments["radian_token"].max() + 1

    df_segments["s_token"] = df_segments.apply(lambda row: loc_token_generator.gps2vocab(row["s_lon"], row["s_lat"]), axis=1).astype("int64")
    df_segments["e_token"] = df_segments.apply(lambda row: loc_token_generator.gps2vocab(row["e_lon"], row["e_lat"]), axis=1).astype("int64")
    num_cls_token = df_segments["highway_cls"].nunique()
    with open(f"../data/{configs.dataset_name}/{configs.dataset_name}_road_network_features.pkl", "wb") as f:
        pickle.dump(df_segments, f)
    with open(f"../data/{configs.dataset_name}/{configs.dataset_name}_features_param.pkl", "wb") as f:
        pickle.dump({"num_cls_token": num_cls_token,
                     "num_loc_token": num_loc_token,
                     "num_length_token": num_length_token,
                     "num_radian_token": num_radian_token}, f)


def generate_edge_weight(dataset_name):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    with open(f"../data/{dataset_name}/{dataset_name}_edge_index.pkl", "rb") as f:
        edge_index_list = pickle.load(f)
    hdist_list = []
    adist_list = []
    cdist_list = []
    for node1_idx, node2_idx in edge_index_list:
        # It is hard to define the "network distance" for two road segments. Since here we only consider the connected segments, the haversine distance should be close to network distance
        hdist_list.append(abs(haversine(df_segments.loc[node1_idx, "c_lon"], df_segments.loc[node1_idx, "c_lat"],
                                        df_segments.loc[node2_idx, "c_lon"], df_segments.loc[node2_idx, "c_lat"])))
        adist_list.append(abs(df_segments.loc[node1_idx, "radian"] - df_segments.loc[node2_idx, "radian"]))
        #cdist_list.append(abs(cls2weights(df_segments.loc[node1_idx, "highway_cls"]) - cls2weights(df_segments.loc[node2_idx, "highway_cls"])) + 1)
    hdist_min = min(hdist_list)
    hdist_max = max(hdist_list)
    adist_min = min(adist_list)
    adist_max = max(adist_list)
    hdist_list_norm = [item / hdist_max for item in hdist_list]
    adist_list_norm = [item / adist_max for item in adist_list]

    weight = 0.8  # haversine_distance should be more important than radian_distance

    traj_edge_weights = [round((hdist_list_norm[i] * weight + adist_list_norm[i] * (1 - weight)), 8) for i in range(len(edge_index_list))]

    with open(f"../data/{dataset_name}/{dataset_name}_edge_weights.pkl", "wb") as f:
        pickle.dump(traj_edge_weights, f)


def generate_spatial_edge_single(group, dataset_name, node_index):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    edge_index_list = []
    for i in node_index:
        for j in range(df_segments.shape[0]):
            if abs(haversine(df_segments.loc[i, "c_lon"], df_segments.loc[i, "c_lat"], df_segments.loc[j, "c_lon"], df_segments.loc[j, "c_lat"])) < 200\
                    and abs(df_segments.loc[i, "radian"] - df_segments.loc[j, "radian"]) < math.pi / 8:
                edge_index_list.append([i, j])
    with open(f"../data/{dataset_name}/parallel_repo/spatial_edge_index/{dataset_name}_road_network_spatial_edge_index_{group}.pkl", "wb") as f:
        pickle.dump(edge_index_list, f)
    print(f"Group {group} finished. we have {len(edge_index_list)} spatial edges")


def generate_spatial_edge(dataset_name):
    params = []
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    if not os.path.exists(f"../data/{dataset_name}/parallel_repo/spatial_edge_index"):
        os.makedirs(f"../data/{dataset_name}/parallel_repo/spatial_edge_index")
    num_group = math.ceil(df_segments.shape[0] / 3000)
    print(f"we have {num_group} groups")
    for i in range(num_group - 1):
        params.append((i, dataset_name, [i*3000+j for j in range(3000)]))
    params.append((num_group-1, dataset_name, [i for i in range((num_group-1)*3000, df_segments.shape[0])]))
    print(len(params))
    pool_single = multiprocessing.Pool(processes=20)
    pool_single.starmap_async(generate_spatial_edge_single, params)
    pool_single.close()
    pool_single.join()


def generate_spatial_edge_merge(dataset_name):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    num_group = math.ceil(df_segments.shape[0] / 3000)
    edge_index_list = []
    for i in range(num_group):
        with open(f"../data/{dataset_name}/parallel_repo/spatial_edge_index/{dataset_name}_road_network_spatial_edge_index_{i}.pkl", "rb") as f:
            edge_index_temp = pickle.load(f)
            print(f"For group {i}, we have {len(edge_index_temp)} spatial edges")
            edge_index_list += edge_index_temp
    with open(f"../data/{dataset_name}/{dataset_name}_spatial_edge_index.pkl", "wb") as f:
        pickle.dump(edge_index_list, f)
    print(f"We have {len(edge_index_list)} edges in total")


def generate_spatial_edge_weight(dataset_name):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    with open(f"../data/{dataset_name}/{dataset_name}_spatial_edge_index.pkl", "rb") as f:
        edge_index_list = pickle.load(f)
    hdist_list = []
    adist_list = []
    cdist_list = []
    for node1_idx, node2_idx in edge_index_list:
        # It is hard to define the "network distance" for two road segments. Since here we only consider the connected segments, the haversine distance should be close to network distance
        hdist_list.append(abs(haversine(df_segments.loc[node1_idx, "c_lon"], df_segments.loc[node1_idx, "c_lat"],
                                        df_segments.loc[node2_idx, "c_lon"], df_segments.loc[node2_idx, "c_lat"])))
        adist_list.append(abs(df_segments.loc[node1_idx, "radian"] - df_segments.loc[node2_idx, "radian"]))
        #cdist_list.append(abs(cls2weights(df_segments.loc[node1_idx, "highway_cls"]) - cls2weights(df_segments.loc[node2_idx, "highway_cls"])) + 1)
    hdist_min = min(hdist_list)
    hdist_max = max(hdist_list)
    adist_min = min(adist_list)
    adist_max = max(adist_list)
    hdist_list_norm = [item / hdist_max for item in hdist_list]
    adist_list_norm = [item / adist_max for item in adist_list]

    weight = 0.8  # haversine_distance should be more important than radian_distance

    traj_edge_weights = [round((hdist_list_norm[i] * weight + adist_list_norm[i] * (1 - weight)), 8) for i in range(len(edge_index_list))]

    with open(f"../data/{dataset_name}/{dataset_name}_spatial_edge_weights.pkl", "wb") as f:
        pickle.dump(traj_edge_weights, f)


def generate_traj_edge(dataset_name, threshold):
    from collections import Counter
    df_traj_list = []
    for suffix in ["train", "val", "finetune", "test"]:
        df_traj_suffix = pickle.load(open(f"../data/{dataset_name}/{dataset_name}_{suffix}_mm_feature_meta.pkl", "rb"))
        df_traj_list.append(df_traj_suffix)
    df_traj = pd.concat(df_traj_list).reset_index()
    edge_index_list = []
    for i in range(df_traj.shape[0]):
        traj = df_traj.loc[i, "rs_ids"]
        for j in range(len(traj) - 1):
            edge_index_list.append((traj[j], traj[j + 1]))
    counter = Counter(edge_index_list)
    print(f"We have {len(counter.keys())} edges before filtering")

    edge_index_list_filter = []
    for item in counter.most_common():
        if item[1] > threshold:
            edge_index_list_filter.append(item[0])
    print(f"We have {len(edge_index_list_filter)} traj edges after filtering")

    network_edge_index_list = pickle.load(open(f"../data/{dataset_name}/{dataset_name}_edge_index.pkl", "rb"))
    print(f"We have {len(network_edge_index_list)} network edges")

    for edge in network_edge_index_list:
        if edge not in edge_index_list_filter:
            edge_index_list_filter.append(edge)
    print(f"After merging, we have {len(edge_index_list_filter)} edges")

    with open(f"../data/{dataset_name}/{dataset_name}_traj_edge_index_{threshold}.pkl", "wb") as f:
        pickle.dump(edge_index_list_filter, f)


def generate_traj_edge_weight(dataset_name, threshold):
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    with open(f"../data/{dataset_name}/{dataset_name}_traj_edge_index_{threshold}.pkl", "rb") as f:
        edge_index_list = pickle.load(f)
    hdist_list = []
    adist_list = []
    cdist_list = []
    for node1_idx, node2_idx in edge_index_list:
        # It is hard to define the "network distance" for two road segments. Since here we only consider the connected segments, the haversine distance should be close to network distance
        hdist_list.append(abs(haversine(df_segments.loc[node1_idx, "c_lon"], df_segments.loc[node1_idx, "c_lat"],
                                        df_segments.loc[node2_idx, "c_lon"], df_segments.loc[node2_idx, "c_lat"])))
        adist_list.append(abs(df_segments.loc[node1_idx, "radian"] - df_segments.loc[node2_idx, "radian"]))
        cdist_list.append(abs(cls2weights(df_segments.loc[node1_idx, "highway_cls"]) - cls2weights(df_segments.loc[node2_idx, "highway_cls"])) + 1)
    hdist_min = min(hdist_list)
    hdist_max = max(hdist_list)
    adist_min = min(adist_list)
    adist_max = max(adist_list)
    hdist_list_norm = [item / hdist_max for item in hdist_list]
    adist_list_norm = [item / adist_max for item in adist_list]

    weight = 0.8  # haversine_distance should be more important than radian_distance

    traj_edge_weights = [1 / round(math.log((hdist_list_norm[i] * weight + adist_list_norm[i] * (1 - weight)) * cdist_list[i]), 8) for i in range(len(edge_index_list))]

    with open(f"../data/{dataset_name}/{dataset_name}_traj_edge_weights_{threshold}.pkl", "wb") as f:
        pickle.dump(traj_edge_weights, f)


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})

    t1 = time.time()
    generate_edge(configs.dataset_name)
    t2 = time.time()
    print(f"generate_edge costs: {t2 - t1} s")

    generate_edge_merge(configs.dataset_name)
    t3 = time.time()
    print(f"generate_edge merge costs: {t3 - t2} s")
    generate_edge_weight(configs.dataset_name)
    t4 = time.time()
    print(f"generate_edge_weight costs: {t4 - t3} s")
    tokenization(configs)
    t5 = time.time()
    print(f"tokenization costs: {t5 - t4} s")

    for threshold in [5]:
        t1 = time.time()
        generate_traj_edge(configs.dataset_name, threshold)
        t2 = time.time()
        print(f"generate_edge {threshold} costs: {t2 - t1} s")
        generate_traj_edge_weight(configs.dataset_name, threshold)

