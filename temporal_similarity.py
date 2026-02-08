from multiprocessing import Pool
import pickle
import numpy as np
import time
import numba
import os
import json
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/")
from config import Config


def find_trajtimelist(dataset_name):
    longest_traj = 0
    smallest_time = np.inf
    with open(f"data/{dataset_name}/{dataset_name}.pkl", "rb") as f:
        df = pickle.load(f)
        time_list_int = df["Timestamps"]

    for time_list in time_list_int:
        time_list = json.loads(time_list)
        if len(time_list) > longest_traj:
            longest_traj = len(time_list)
        for t in time_list:
            if t < smallest_time:
                smallest_time = t
    return longest_traj, smallest_time


def batch_timelist_ground_truth(configs, distance_type, suffix):
    dataset_name = configs.dataset_name
    # jiali: shuffle_node_file is the shuffled trajectory file in road node list.
    with open(f"data/{dataset_name}/{dataset_name}_{suffix}_mm_feature_meta.pkl", "rb") as f:
        df_features = pickle.load(f)

    if suffix == 'finetune' or suffix == 'val':
        node_list_int = df_features.loc[0: 9999, "seg_tss"]  # based dataset and "validation or test"  (train:validation:test = 1w:4k:1.6w)
    elif suffix == 'test':
        node_list_int = df_features.loc[0: 9999, "seg_tss"]

    sample_list = node_list_int  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=20)
    for i in range(len(sample_list)):
        if i % 50 == 0:
            pool.apply_async(timelist_distance, (i, dataset_name, distance_type, sample_list[i:i+50], node_list_int, suffix))
    pool.close()
    pool.join()

    return len(sample_list)


def merge_timelist_ground_truth(sample_len, suffix, dataset_name, distance_type):
    res = []
    for i in range(sample_len):
        if i % 50 == 0:
            res.append(np.load('data/{}/ground_truth//{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset_name, distance_type, suffix, distance_type, str(i))))
    res = np.concatenate(res, axis=0)
    np.save('data/{}/ground_truth/{}/{}_temporal_distance.npy'.format(dataset_name, distance_type, suffix), res)


def timelist_distance(k, dataset_name, distance_type, sample_list=[[]], test_list=[[]], suffix=None):
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
    np.save('data/{}/ground_truth/{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset_name, distance_type, suffix, distance_type, str(k)), all_dis_list)

    print(f"complete temporal distance {k} {distance_type} {suffix} costs {(time.time() - t1)} s")


@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a=[], list_b=[]):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            temp = abs(tr1[i]-tr2[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            temp = abs(tr2[i]-tr1[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1,max2))


@numba.jit(nopython=True, fastmath=True)
def LCRS_dis(list_a=[], list_b=[]):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if abs(list_a[i] - list_b[j]) <= 3600:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return longest_trajtime_len*2
    else:
        return (lena + lenb - c[-1][-1]) / float(c[-1][-1])


@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a=[], list_b=[]):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(lena + 1):
        edit[i][0] = i * smallest_trajtime
    for i in range(lenb + 1):
        edit[0][i] = i * smallest_trajtime

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            edit[i][j] = min(edit[i - 1][j] + list_a[i-1] - smallest_trajtime, edit[i][j - 1] + list_b[j-1] - smallest_trajtime, edit[i - 1][j - 1] + abs(list_a[i-1] - list_b[j-1]))

    return edit[-1][-1]


@numba.jit(nopython=True, fastmath=True)
def NetDTW_dis(list_a=[], list_b=[]):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M+1, N+1))
    tp = abs(tr1[0] - tr2[0])
    cost[0, 0] = tp
    for i in range(1, M+1):
        tp = abs(tr1[i-1] - tr2[0])
        cost[i, 0] = cost[i - 1, 0] + tp
    for i in range(1, N+1):
        tp = abs(tr1[0] - tr2[i-1])
        cost[0, i] = cost[0, i - 1] + tp
    for i in range(1, M+1):
        for j in range(1, N+1):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            tp = abs(tr1[i-1] - tr2[j-1])
            cost[i, j] = min(small) + tp
    return cost[-1][-1]


@numba.jit(nopython=True, fastmath=True)
def NetEDR_dis(list_a=[], list_b=[]):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M+1, N+1))

    # Initialize the first row and column
    for i in range(1, M+1):
        cost[i, 0] = i
    for j in range(1, N+1):
        cost[0, j] = j

    # Populate the EDR matrix
    for i in range(1, M+1):
        for j in range(1, N+1):
            subcost = 0 if abs(tr1[i - 1] - tr2[j - 1]) <= 1000 else 1
            cost[i, j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + subcost)

    edr_dis = cost[M][N]

    return edr_dis


longest_trajtime_len, smallest_trajtime = find_trajtimelist("porto")


if __name__ == '__main__':
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    #global lonlongest_trajtime_len
    #global smallest_trajtime
    #longest_trajtime_len, smallest_trajtime = find_trajtimelist(configs.dataset_name)

    for distance_type in ["NetLCSS"]:
        for suffix in ["val", "test"]:
            batch_timelist_ground_truth(configs, distance_type, suffix)
            merge_timelist_ground_truth(10000, suffix, configs.dataset_name, distance_type)

