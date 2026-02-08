from multiprocessing import Pool
import pickle
import numpy as np
import networkx as nx
import time
import numba
import pandas as pd
import collections
import os
import warnings
warnings.filterwarnings("ignore")
from config import Config


def batch_Point_distance(dataset_name, roadnetwork, num_nodes):
    num_nodes_int = int(num_nodes // 1000 * 1000)
    num_nodes_remain = num_nodes - num_nodes_int
    pool = Pool(processes=20)
    for i in range(num_nodes_int + 1):
        if i != 0 and i % 1000 == 0:
            pool.apply_async(parallel_point_com, (i, list(range(i - 1000, i)), dataset_name, roadnetwork, num_nodes))
    if num_nodes_remain != 0:
        pool.apply_async(parallel_point_com, (num_nodes, list(range(num_nodes_int, num_nodes)), dataset_name, roadnetwork, num_nodes))
    pool.close()
    pool.join()


def merge_Point_distance(dataset_name, num_nodes):
    res = []
    for i in range(num_nodes + 1):
        if i != 0 and i % 1000 == 0:
            res.append(np.load('data/{}/ground_truth/Point_dis_matrix_{}.npy'.format(dataset_name, str(i))))
    num_nodes_int = int(num_nodes // 1000 * 1000)
    num_nodes_remain = num_nodes - num_nodes_int
    if num_nodes_remain != 0:
        res.append(np.load('data/{}/ground_truth/Point_dis_matrix_{}.npy'.format(dataset_name, str(num_nodes))))
    res = np.concatenate(res, axis=0)
    np.save('data/{}/ground_truth/Point_dis_matrix.npy'.format(dataset_name), res)


def parallel_point_com(i, id_list, dataset_name, roadnetwork, num_nodes):
    batch_list = []
    t1 = time.time()
    for k in id_list:
        one_list = []
        if k in roadnetwork.nodes():
            # jiali: Shortest path exists only if the two nodes are reachable in the road network.
            length_list = nx.shortest_path_length(roadnetwork, source=k, weight='distance')
            for j in range(num_nodes):
                if j in length_list.keys():
                    one_list.append(length_list[j])
                else:
                    one_list.append(-1)
            batch_list.append(np.array(one_list, dtype=np.float32))
        else:
            one_list = [-1 for j in range(num_nodes)]
            batch_list.append(np.array(one_list, dtype=np.float32))
    print(f"For {len(id_list)} nodes costs: {time.time() - t1} s")
    batch_list = np.array(batch_list, dtype=np.float32)
    print(batch_list.shape)
    p = 'data/{}/ground_truth/'.format(dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('data/{}/ground_truth//Point_dis_matrix_{}.npy'.format(dataset_name, str(i)), batch_list)


def generate_point_matrix(dataset_name):
    res = np.load('data/{}/ground_truth/Point_dis_matrix.npy'.format(dataset_name))
    return res


def generate_node_edge_interation(dataset_name):
    node_edge_dict = collections.defaultdict(set)
    edge = pd.read_csv('data/{}/{}_segment.csv'.format(dataset_name, dataset_name))
    node_s, node_e = edge.s_idx, edge.e_idx

    for idx, (n_s, n_e) in enumerate(zip(node_s, node_e)):
        node_edge_dict[int(n_s)].add(idx)
        node_edge_dict[int(n_e)].add(idx)

    return node_edge_dict


def batch_similarity_ground_truth(configs, distance_type, max_len, suffix):
    dataset_name = configs.dataset_name
    # jiali: shuffle_node_file is the shuffled trajectory file in road node list.
    with open(f"data/{dataset_name}/{dataset_name}_{suffix}_mm_feature_meta.pkl", "rb") as f:
        df_features = pickle.load(f)

    # To control the scale of finetune, vali and test (optional)
    if suffix == 'finetune' or suffix == 'val':
        node_list_int = df_features.loc[0: 9999, "seg_starts"]  # based dataset and "validation or test")
    elif suffix == 'test':
        node_list_int = df_features.loc[0: 9999, "seg_starts"]

    sample_list = node_list_int

    pool = Pool(processes=20)
    for i in range(len(sample_list)):
        if i % 50 == 0:
            pool.apply_async(Traj_distance, (i, dataset_name, distance_type, sample_list[i:i+50], node_list_int, suffix))
    pool.close()
    pool.join()

    return len(sample_list)


def merge_similarity_ground_truth(sample_len, suffix, dataset_name, distance_type):
    res = []
    for i in range(sample_len):
        if i % 50 == 0:
            res.append(np.load('data/{}/ground_truth/{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset_name, distance_type, suffix, distance_type, str(i))))
    res = np.concatenate(res, axis=0)
    np.save('data/{}/ground_truth/{}/{}_spatial_distance.npy'.format(dataset_name, distance_type, suffix), res)


def Traj_distance(k, dataset_name, distance_type, sample_list=[[]], test_list=[[]], suffix=None):
    all_dis_list = []
    t1 = time.time()
    for sample in sample_list:
        one_dis_list = []
        for traj in test_list:
            if distance_type == 'TP':
                one_dis_list.append(TP_dis(sample, traj))
            elif distance_type == 'LCRS':
                one_dis_list.append(LCRS_dis(sample, traj))
            elif distance_type == 'NetERP':
                one_dis_list.append(NetERP_dis(sample, traj))
            elif distance_type == 'NetDTW':
                one_dis_list.append(NetDTW_dis(sample, traj))
            elif distance_type == 'NetEDR':
                one_dis_list.append(NetEDR_dis(sample, traj))
        all_dis_list.append(np.array(one_dis_list))

    all_dis_list = np.array(all_dis_list)
    p = 'data/{}/ground_truth/{}/{}_batch/'.format(dataset_name, distance_type, suffix)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('data/{}/ground_truth//{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset_name, distance_type, suffix, distance_type, str(k)), all_dis_list)

    print(f"complete spatial distance {k} {distance_type} {suffix} costs {(time.time() - t1)} s")


@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a=[], list_b=[]):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            if distance_matrix[tr1[i]][tr2[j]] != -1:
                temp = distance_matrix[tr1[i]][tr2[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            if distance_matrix[tr2[i]][tr1[j]] != -1:
                temp = distance_matrix[tr2[i]][tr1[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1, max2))


def LCRS_dis(list_a=[], list_b=[]):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if len(node_edge_dict[list_a[i]] & node_edge_dict[list_b[j]]) >= 1:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return max_len * 2
    else:
        return (lena + lenb - c[-1][-1]) / float(c[-1][-1])


def hot_node():
    max_num = 0
    max_idx = 0
    for idx, nodes_interaction in enumerate(distance_matrix):
        nodes_interaction = np.array(nodes_interaction)
        x = len(nodes_interaction[nodes_interaction != -1])
        if x > max_num:
            max_num = x
            max_idx = idx
    return max_idx


@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a=[], list_b=[]):

    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(1, lena + 1):
        tp = distance_matrix[hot_node_id][list_a[i - 1]]
        if tp == -1:
            return -1
        edit[i][0] = edit[i - 1][0] + tp
    for i in range(1, lenb + 1):
        tp = distance_matrix[hot_node_id][list_b[i - 1]]
        if tp == -1:
            return -1
        edit[0][i] = edit[0][i - 1] + tp

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            tp1 = distance_matrix[hot_node_id][list_a[i - 1]]
            tp2 = distance_matrix[hot_node_id][list_b[j - 1]]
            tp3 = distance_matrix[list_a[i - 1]][list_b[j - 1]]
            if tp1 == -1 or tp2 == -1 or tp3 == -1:
                return -1
            edit[i][j] = min(edit[i - 1][j] + tp1, edit[i][j - 1] + tp2, edit[i - 1][j - 1] + tp3)

    return edit[-1][-1]


@numba.jit(nopython=True, fastmath=True)
def NetDTW_dis(list_a=[], list_b=[]):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M+1, N+1))
    tp = distance_matrix[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    cost[0, 0] = tp
    for i in range(1, M+1):
        tp = distance_matrix[tr1[i-1]][tr2[0]]
        if tp == -1:
            return -1
        cost[i, 0] = cost[i - 1, 0] + tp
    for i in range(1, N+1):
        tp = distance_matrix[tr1[0]][tr2[i-1]]
        if tp == -1:
            return -1
        cost[0, i] = cost[0, i - 1] + tp
    for i in range(1, M+1):
        for j in range(1, N+1):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            tp = distance_matrix[tr1[i-1]][tr2[j-1]]
            if tp == -1:
                return -1
            cost[i, j] = min(small) + tp
    return cost[-1][-1]


@numba.jit(nopython=True, fastmath=True)
def NetEDR_dis(list_a=[], list_b=[]):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M+1, N+1))

    # Initialize the first row and column
    for i in range(1, M+1):
        cost[i][0] = i
    for j in range(1, N+1):
        cost[0][j] = j

    # Populate the EDR matrix
    for i in range(1, M+1):
        for j in range(1, N+1):
            if distance_matrix[tr1[i-1]][tr2[j-1]] == -1:
                return -1
            else:
                subcost = 0 if distance_matrix[tr1[i-1]][tr2[j-1]] <= 1000 else 1
                cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + subcost)
    edr_dis = cost[M][N]
    if abs(edr_dis) < 1e-10:
        edr_dis = 0
    return edr_dis


#global distance_matrix
distance_matrix = generate_point_matrix("porto")
#global node_edge_dict
node_edge_dict = generate_node_edge_interation("porto")
#global hot_node_id
hot_node_id = hot_node()
max_len = 2000


if __name__ == '__main__':
    #traj_distance_demo("porto")

    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})

    """
    # The distance between road network' nodes
    roadnetwork = get_road_network(configs.dataset_name)
    num_nodes = roadnetwork.number_of_nodes()
    print(num_nodes)
    batch_Point_distance(configs.dataset_name, roadnetwork, num_nodes)
    merge_Point_distance(configs.dataset_name, num_nodes)
    """
    #"""

    for distance_type in ["NetLCSS"]:
        for suffix in ["val", "test"]:
            batch_similarity_ground_truth(configs, distance_type, 2000, suffix)
            merge_similarity_ground_truth(10000, suffix, configs.dataset_name, distance_type)
    #"""










