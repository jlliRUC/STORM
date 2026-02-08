import torch
import torch.nn as nn
from Model import Date2VecConvert
import datetime
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.neighbors import BallTree
import spatial_similarity as spatial_com
import temporal_similarity as temporal_com
import os
from multiprocessing import Pool
from config import Config

random.seed(1933)


def check_distance(traj1, traj2, distance_type):
    # Return false because two trajectories might not be reachable in the road network (due to the incompleteness)
    if distance_type == "LCRS":
        if spatial_com.LCRS_dis(traj1, traj2) == 2000 * 2:
            return False
        else:
            return True
    else:
        if distance_type == "TP":
            dis = spatial_com.TP_dis(traj1, traj2)
        elif distance_type == "LCRS":
            dis = spatial_com.LCRS_dis(traj1, traj2)
        elif distance_type == "NetERP":
            dis = spatial_com.NetERP_dis(traj1, traj2)
        elif distance_type == "NetDTW":
            dis = spatial_com.NetDTW_dis(traj1, traj2)
        elif distance_type == "NetLCSS":
            dis = spatial_com.NetLCSS_dis(traj1, traj2)
        elif distance_type == "NetEDR":
            dis = spatial_com.NetEDR_dis(traj1, traj2)
    if dis == -1:
        return False
    else:
        return True


class Date2vec(nn.Module):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

    def forward(self, time_seq):
        all_list = []
        for one_seq in time_seq:
            one_list = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = torch.Tensor(t).float()
                embed = self.d2v(x)
                one_list.append(embed)

            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, 64)

            all_list.append(one_list.numpy().tolist())

        all_list = np.array(all_list)

        return all_list


def prepare_dataset(configs):
    """
    We sample a training set from the existing finetune dataset.
    In addition to all features, we'll also generate simplified trajectories' coors to construct the BallTree
    """
    kseg = configs.kseg
    df_nodes = pd.read_csv(f"data/{configs.dataset_name}/{configs.dataset_name}_node.csv")
    df_traj = pickle.load(open(f"data/{configs.dataset_name}/{configs.dataset_name}_finetune_mm_feature.pkl", "rb"))
    num_traj = df_traj.shape[0]

    node_list_int = []
    rs_list_int = []
    time_list_int = []
    coor_trajs = []
    kseg_coor_trajs = []
    st_path = f"data/{configs.dataset_name}/{configs.dataset_name}_finetune_mm_segmentwise_weights.pkl"
    with open(st_path, "rb") as f:
        st_list = np.array(pickle.load(f))

    for i in range(num_traj):
        node_list_int.append(df_traj.loc[i, "seg_starts"])
        rs_list_int.append(df_traj.loc[i, "rs_ids"].tolist())
        time_list_int.append(df_traj.loc[i, "seg_tss"])
        coor_trajs.append(
            np.array([[df_nodes.loc[node, "lon"], df_nodes.loc[node, "lat"]] for node in node_list_int[-1]]))
        t = coor_trajs[-1]
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        if seg == 0:  # jiali: num_segments less than kseg
            for i in range(kseg):
                kseg_coor.append(np.mean(t, axis=0))
        else:
            for i in range(kseg):
                if i == kseg - 1:
                    kseg_coor.append(np.mean(t[i * seg:], axis=0))
                else:
                    kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)

    node_list_int = np.array(node_list_int)
    rs_list_int = np.array(rs_list_int)
    time_list_int = np.array(time_list_int)
    coor_trajs = np.array(coor_trajs)
    kseg_coor_trajs = np.array(kseg_coor_trajs)
    print("complete: ksegment")

    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)
    # jiali: Following st2vec, we only select 50,000 trajectories and then find out the trajectory pairs.
    shuffle_index = shuffle_index[:50000]  # 5w size of dataset

    node_list_int = node_list_int[shuffle_index]
    rs_list_int = rs_list_int[shuffle_index]
    time_list_int = time_list_int[shuffle_index]
    st_list = st_list[shuffle_index]
    coor_trajs = coor_trajs[shuffle_index]
    kseg_coor_trajs = kseg_coor_trajs[shuffle_index]

    np.save(configs.shuffle_node_file, node_list_int)
    np.save(configs.shuffle_rs_file, rs_list_int)
    np.save(configs.shuffle_time_file, time_list_int)
    np.save(configs.shuffle_st_file, st_list)
    np.save(configs.shuffle_coor_file, coor_trajs)
    np.save(configs.shuffle_kseg_file, kseg_coor_trajs)

    d2vec = Date2vec()
    timelist = np.load(configs.shuffle_time_file, allow_pickle=True)
    d2v = d2vec(timelist)
    np.save(configs.shuffle_d2vec_file, d2v)


def prepare_porto(configs):
    """
    We sample a training set from the existing finetune dataset.
    In addition to all features, we'll also generate simplified trajectories' coors to construct the BallTree
    """
    kseg = configs.kseg
    df_nodes = pd.read_csv(f"data/{configs.dataset_name}/{configs.dataset_name}_node.csv")
    df_list = []
    st_list_list = []
    for suffix in ["train", "finetune", "val", "test"]:
        df_traj = pickle.load(open(f"data/{configs.dataset_name}/{configs.dataset_name}_{suffix}_mm_feature.pkl", "rb"))
        df_list.append(df_traj)
        st_path = f"data/{configs.dataset_name}/{configs.dataset_name}_{suffix}_mm_segmentwise_weights.pkl"
        st_list_list.append(np.array(pickle.load(open(st_path, "rb"))))
    df_traj = pd.concat(df_list)
    df_traj = df_traj.reset_index(drop=True)
    st_list = np.concatenate(st_list_list, axis=0)
    num_traj = df_traj.shape[0]
    print(f"we have {num_traj} / {len(st_list)} trajectories.")

    node_list_int = []
    rs_list_int = []
    time_list_int = []
    coor_trajs = []
    kseg_coor_trajs = []

    for i in range(num_traj):
        node_list_int.append(df_traj.loc[i, "seg_starts"])
        rs_list_int.append(df_traj.loc[i, "rs_ids"].tolist())
        time_list_int.append(df_traj.loc[i, "seg_tss"])
        coor_trajs.append(
            np.array([[df_nodes.loc[node, "lon"], df_nodes.loc[node, "lat"]] for node in node_list_int[-1]]))
        t = coor_trajs[-1]
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        if seg == 0:  # jiali: num_segments less than kseg
            for i in range(kseg):
                kseg_coor.append(np.mean(t, axis=0))
        else:
            for i in range(kseg):
                if i == kseg - 1:
                    kseg_coor.append(np.mean(t[i * seg:], axis=0))
                else:
                    kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)

    node_list_int = np.array(node_list_int)
    rs_list_int = np.array(rs_list_int)
    time_list_int = np.array(time_list_int)
    coor_trajs = np.array(coor_trajs)
    kseg_coor_trajs = np.array(kseg_coor_trajs)
    print("complete: ksegment")

    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)
    shuffle_index = shuffle_index[:50000]  # 5w size of dataset

    node_list_int = node_list_int[shuffle_index]
    rs_list_int = rs_list_int[shuffle_index]
    time_list_int = time_list_int[shuffle_index]
    st_list = st_list[shuffle_index]
    coor_trajs = coor_trajs[shuffle_index]
    kseg_coor_trajs = kseg_coor_trajs[shuffle_index]

    np.save(configs.shuffle_node_file, node_list_int)
    np.save(configs.shuffle_rs_file, rs_list_int)
    np.save(configs.shuffle_time_file, time_list_int)
    np.save(configs.shuffle_st_file, st_list)
    np.save(configs.shuffle_coor_file, coor_trajs)
    np.save(configs.shuffle_kseg_file, kseg_coor_trajs)

    d2vec = Date2vec()
    timelist = np.load(configs.shuffle_time_file, allow_pickle=True)
    d2v = d2vec(timelist)
    np.save(configs.shuffle_d2vec_file, d2v)


class DataLoader():
    """
    load("train") is for generating training set (triplets/sets), load("vali") and load("test") will load the data part directly
    """
    def __init__(self, configs):
        self.configs = configs
        self.train_set = 10000
        self.vali_set = 14000
        self.test_set = 24000

    def load(self, load_part):
        self.kseg = self.configs.kseg
        # split train, vali, test set
        node_list_int = np.load(self.configs.shuffle_node_file, allow_pickle=True)
        rs_list_int = np.load(self.configs.shuffle_rs_file, allow_pickle=True)
        st_list = np.load(self.configs.shuffle_st_file, allow_pickle=True)
        time_list_int = np.load(self.configs.shuffle_time_file, allow_pickle=True)
        d2vec_list_int = np.load(self.configs.shuffle_d2vec_file, allow_pickle=True)

        train_set = self.train_set
        vali_set = self.vali_set
        test_set = self.test_set

        if load_part == 'train':
            return node_list_int[:train_set], \
                   rs_list_int[:train_set], \
                   time_list_int[:train_set],\
                   st_list[:train_set],\
                   d2vec_list_int[:train_set]
        if load_part == 'vali':
            return node_list_int[train_set:vali_set], \
                   rs_list_int[train_set:vali_set], \
                   time_list_int[train_set:vali_set], \
                   st_list[train_set:vali_set], \
                   d2vec_list_int[train_set:vali_set]
        if load_part == 'test':
            return node_list_int[20000:test_set], \
                   rs_list_int[20000:test_set], \
                   time_list_int[20000:test_set], \
                   st_list[20000:test_set], \
                   d2vec_list_int[20000:test_set]

    def ksegment_ST(self):
        # Simplify the trajectory
        kseg_coor_trajs = np.load(self.configs.shuffle_kseg_file, allow_pickle=True)[:self.train_set]
        time_trajs = np.load(self.configs.shuffle_time_file, allow_pickle=True)[:self.train_set]

        kseg_time_trajs = []
        for t in time_trajs:
            kseg_time = []
            seg = len(t) // self.kseg
            t = np.array(t)
            if seg == 0:  # jiali: num_segments less than kseg
                for i in range(self.kseg):
                    kseg_time.append(np.mean(t))
            else:
                for i in range(self.kseg):
                    if i == self.kseg - 1:
                        kseg_time.append(np.mean(t[i * seg:]))
                    else:
                        kseg_time.append(np.mean(t[i * seg:i * seg + seg]))
            kseg_time_trajs.append(kseg_time)
        kseg_time_trajs = np.array(kseg_time_trajs)
        print(kseg_time_trajs.shape)

        max_lat = -np.inf  # jiali: note that lon could be negative, thus initialized with -inf, instead of 0.
        max_lon = -np.inf
        for traj in kseg_coor_trajs:
            for t in traj:
                if max_lat < t[1]:  # jiali: note that the order of [lon, lat] might be different
                    max_lat = t[1]
                if max_lon < t[0]:
                    max_lon = t[0]
        print(f"max_lat: {max_lat}, max_lon: {max_lon}")
        kseg_coor_trajs = kseg_coor_trajs / [max_lon, max_lat]
        kseg_coor_trajs = kseg_coor_trajs.reshape(-1, self.kseg * 2)
        kseg_time_trajs = kseg_time_trajs / np.max(kseg_time_trajs)

        kseg_ST = np.concatenate((kseg_coor_trajs, kseg_time_trajs), axis=1)
        print("kseg_ST len: ", len(kseg_ST))
        print("kseg_ST shape: ", kseg_ST.shape)

        return kseg_ST

    def get_triplets_st2vec(self):
        """
        We find num_companion pos and num_companion_neg for each anchor. Then we will save them in a triplet format,
        i.e., for each anchor we will have num_companion triplets
        """
        from collections import Counter
        num_companion = self.configs.num_companion

        train_node_list, train_rs_list, train_time_list, train_st_list, train_d2vec_list = self.load(load_part='train')

        # obtain simplified trajectories
        simplified_trajs = self.ksegment_ST()

        ball_tree = BallTree(simplified_trajs)

        # randomly select anchor
        anchor_index_list = list(range(len(train_node_list)))
        random.shuffle(anchor_index_list)

        apn_node_triplets = []
        apn_rs_triplets = []
        apn_st_triplets = []
        apn_time_triplets = []
        apn_d2vec_triplets = []
        apn_index_triplets = []

        for j in range(1, 1001):
            for i in anchor_index_list:
                dist, index = ball_tree.query([simplified_trajs[i]], j + 1)  # k nearest neighbors
                p_index = list(index[0])
                p_index = p_index[-1]
                p_sample = train_node_list[p_index]  # positive sample
                n_index = random.randint(0, len(train_node_list) - 1)
                n_sample = train_node_list[n_index]  # negative sample
                a_sample = train_node_list[i]  # anchor sample

                if check_distance(a_sample, p_sample, self.configs.distance_type) and check_distance(a_sample, n_sample, self.configs.distance_type):
                    apn_index_triplets.append([i, p_index, n_index])
                    apn_node_triplets.append([a_sample, p_sample, n_sample])
                    apn_rs_triplets.append([train_rs_list[i], train_rs_list[p_index], train_rs_list[n_index]])
                    apn_st_triplets.append([train_st_list[i], train_st_list[p_index], train_st_list[n_index]])
                    apn_time_triplets.append([train_time_list[i], train_time_list[p_index], train_time_list[n_index]])
                    apn_d2vec_triplets.append(
                        [train_d2vec_list[i], train_d2vec_list[p_index], train_d2vec_list[n_index]])

                if len(apn_index_triplets) >= num_companion * len(anchor_index_list):  # We've already found top-k pos
                    break
            if len(apn_index_triplets) >= num_companion * len(anchor_index_list):  # We've already found top-k pos
                print(f"current j is {j}")
                break

        print("St2vec Triplet done")
        print(f"We found {len(apn_time_triplets)} triplets")
        a_list = [item[0] for item in apn_index_triplets]
        counter = Counter(a_list)
        print(f"We have {len(counter.keys())} / {len(train_node_list)} valid anchors")

        p = f"data/{self.configs.dataset_name}/triplet/{self.configs.distance_type}/"
        if not os.path.exists(p):
            os.makedirs(p)
        pickle.dump(apn_node_triplets, open(self.configs.path_node_triplets_st2vec, 'wb'))
        pickle.dump(apn_rs_triplets, open(self.configs.path_rs_triplets_st2vec, 'wb'))
        pickle.dump(apn_time_triplets, open(self.configs.path_time_triplets_st2vec, 'wb'))
        pickle.dump(apn_st_triplets, open(self.configs.path_st_triplets_st2vec, 'wb'))
        pickle.dump(apn_d2vec_triplets, open(self.configs.path_d2vec_triplets_st2vec, 'wb'))
        pickle.dump(apn_index_triplets, open(self.configs.path_index_triplets_st2vec, 'wb'))

    def get_triplets(self):
        """
        We find num_companion pos and num_companion_neg for each anchor. Then we will save them in a triplet format,
        i.e., for each anchor we will have num_companion triplets
        """
        from collections import Counter
        num_companion = self.configs.num_companion

        train_node_list, train_rs_list, train_time_list, train_st_list, train_d2vec_list = self.load(load_part='train')
        # Due to the incompleteness of road network, some anchor may not have valid companions
        # (i.e, all pair spatial distance would be -1)
        failed_n = 0
        failed_p = 0

        # obtain simplified trajectories
        simplified_trajs = self.ksegment_ST()

        ball_tree = BallTree(simplified_trajs)

        # randomly select anchor
        anchor_index_list = list(range(len(train_node_list)))
        random.shuffle(anchor_index_list)

        apn_node_triplets = []
        apn_rs_triplets = []
        apn_st_triplets = []
        apn_time_triplets = []
        apn_d2vec_triplets = []
        apn_index_triplets = []
        for i in anchor_index_list:
            p_index_list = []
            n_index_list = []
            a_sample = train_node_list[i]  # anchor sample
            #  If the anchor is invalid, then its distance to top similar instances would be -1,
            #  thus we try at most num_companion * 3 times.
            dist, index = ball_tree.query([simplified_trajs[i]], num_companion * 3 + 1)
            p_index_cand = list(index[0])[1:]  # exclude itself
            for p_index in p_index_cand:
                if check_distance(a_sample, train_node_list[p_index], self.configs.distance_type):
                        p_index_list.append(p_index)
                else:
                    pass
                if len(p_index_list) >= num_companion:  # We've already found top-k pos
                    break
            if len(p_index_list) < num_companion:  # We can't find enough pos for this anchor from the candidates
                failed_p += 1
                continue

            # Next we randomly select num_companion neg from the rest instances.
            rest_index_list = [item for item in anchor_index_list if item not in [i] + p_index_list]
            # Here, we try at most 100 times since some "neg" might be invalid
            trial = 0
            while len(n_index_list) < num_companion:
                n_index = random.choice(rest_index_list)
                if check_distance(a_sample, train_node_list[n_index], self.configs.distance_type):
                    n_index_list.append(n_index)
                else:
                    trial += 1
                if trial > 100:
                    break
            if len(n_index_list) < num_companion:  # We can't find enough neg for this anchor after 100 trials
                failed_n += 1
                continue

            # Save all selected instances in a triplet format
            for m in range(num_companion):
                p_index = p_index_list[m]
                n_index = n_index_list[m]
                a_sample = train_node_list[i]
                p_sample = train_node_list[p_index]
                n_sample = train_node_list[n_index]

                apn_index_triplets.append([i, p_index, n_index])
                apn_node_triplets.append([a_sample, p_sample, n_sample])
                apn_rs_triplets.append([train_rs_list[i], train_rs_list[p_index], train_rs_list[n_index]])
                apn_st_triplets.append([train_st_list[i], train_st_list[p_index], train_st_list[n_index]])
                apn_time_triplets.append([train_time_list[i], train_time_list[p_index], train_time_list[n_index]])
                apn_d2vec_triplets.append([train_d2vec_list[i], train_d2vec_list[p_index], train_d2vec_list[n_index]])

        print("Triplet done")
        print(f"We found {len(apn_time_triplets)} triplets")
        a_list = [item[0] for item in apn_index_triplets]
        counter = Counter(a_list)
        print(f"We have {len(counter.keys())} / {len(train_node_list)} valid anchors")
        print(f"failed_n: {failed_n}, failed_p: {failed_p}")

        p = f"data/{self.configs.dataset_name}/triplet/{self.configs.distance_type}/"
        if not os.path.exists(p):
            os.makedirs(p)
        pickle.dump(apn_node_triplets, open(self.configs.path_node_triplets, 'wb'))
        pickle.dump(apn_rs_triplets, open(self.configs.path_rs_triplets, 'wb'))
        pickle.dump(apn_time_triplets, open(self.configs.path_time_triplets, 'wb'))
        pickle.dump(apn_st_triplets, open(self.configs.path_st_triplets, 'wb'))
        pickle.dump(apn_d2vec_triplets, open(self.configs.path_d2vec_triplets, 'wb'))
        pickle.dump(apn_index_triplets, open(self.configs.path_index_triplets, 'wb'))

    def get_sets(self):
        """
        We find num_companion pos and num_companion_neg for each anchor. Then we will save them in a set format,
        i.e., for each anchor we will have one list with num_companion * 2 + 1 instances
        """
        from collections import Counter
        num_companion = self.configs.num_companion

        train_node_list, train_rs_list, train_time_list, train_st_list, train_d2vec_list = self.load(load_part='train')
        # Due to the incompleteness of road network, some anchor may not have valid companions
        # (i.e, all pair spatial distance would be -1)
        failed_n = 0
        failed_p = 0

        # obtain simplified trajectories
        simplified_trajs = self.ksegment_ST()

        ball_tree = BallTree(simplified_trajs)

        # randomly select anchor
        anchor_index_list = list(range(len(train_node_list)))
        random.shuffle(anchor_index_list)

        apn_node_sets = []
        apn_rs_sets = []
        apn_st_sets = []
        apn_time_sets = []
        apn_d2vec_sets = []
        apn_index_sets = []
        for i in anchor_index_list:
            p_index_list = []
            n_index_list = []
            a_sample = train_node_list[i]  # anchor sample
            #  If the anchor is invalid, then its distance to top similar instances would be -1,
            #  thus we try at most num_companion * 3 times.
            dist, index = ball_tree.query([simplified_trajs[i]], num_companion * 3 + 1)
            p_index_cand = list(index[0])[1:]  # exclude itself
            for p_index in p_index_cand:
                if check_distance(a_sample, train_node_list[p_index], self.configs.distance_type):
                    p_index_list.append(p_index)
                else:
                    pass
                if len(p_index_list) >= num_companion:  # We've already found top-k pos
                    break
            if len(p_index_list) < num_companion:  # We can't find enough pos for this anchor from the candidates
                failed_p += 1
                continue

            # Next we randomly select num_companion neg from the rest instances.
            rest_index_list = [item for item in anchor_index_list if item not in [i] + p_index_list]
            # Here, we try at most 100 times since some "neg" might be invalid
            trial = 0
            while len(n_index_list) < num_companion:
                n_index = random.choice(rest_index_list)
                if check_distance(a_sample, train_node_list[n_index], self.configs.distance_type):
                    n_index_list.append(n_index)
                else:
                    trial += 1
                if trial > 100:
                    break
            if len(n_index_list) < num_companion:  # We can't find enough neg for this anchor after 100 trials
                failed_n += 1
                continue

            apn_index_sets.append([i] + p_index_list + n_index_list)
            apn_node_sets.append([train_node_list[idx] for idx in apn_index_sets[-1]])
            apn_rs_sets.append([train_rs_list[idx] for idx in apn_index_sets[-1]])
            apn_st_sets.append([train_st_list[idx] for idx in apn_index_sets[-1]])
            apn_time_sets.append([train_time_list[idx] for idx in apn_index_sets[-1]])
            apn_d2vec_sets.append([train_d2vec_list[idx] for idx in apn_index_sets[-1]])

        print("Sets done")
        print(f"We found {len(apn_time_sets)} sets")
        a_list = [item[0] for item in apn_index_sets]
        counter = Counter(a_list)
        print(f"We have {len(counter.keys())} / {len(train_node_list)} valid instances")
        print(f"failed_n: {failed_n}, failed_p: {failed_p}")

        p = f"data/{self.configs.dataset_name}/triplet/{self.configs.distance_type}/"
        if not os.path.exists(p):
            os.makedirs(p)
        pickle.dump(apn_node_sets, open(self.configs.path_node_sets, 'wb'))
        pickle.dump(apn_rs_sets, open(self.configs.path_rs_sets, 'wb'))
        pickle.dump(apn_time_sets, open(self.configs.path_time_sets, 'wb'))
        pickle.dump(apn_st_sets, open(self.configs.path_st_sets, 'wb'))
        pickle.dump(apn_d2vec_sets, open(self.configs.path_d2vec_sets, 'wb'))
        pickle.dump(apn_index_sets, open(self.configs.path_index_sets, 'wb'))


def set_ground_truth(configs):
    if configs.distance_type == "TP":
        spatial_func = spatial_com.TP_dis
        temporal_func = temporal_com.TP_dis
    elif configs.distance_type == "LCRS":
        spatial_func = spatial_com.LCRS_dis
        temporal_func = temporal_com.LCRS_dis
    elif configs.distance_type == "NetERP":
        spatial_func = spatial_com.NetERP_dis
        temporal_func = temporal_com.NetERP_dis
    elif configs.distance_type == "NetDTW":
        spatial_func = spatial_com.NetDTW_dis
        temporal_func = temporal_com.NetDTW_dis
    elif configs.distance_type == "NetLCSS":
        spatial_func = spatial_com.NetLCSS_dis
        temporal_func = temporal_com.NetLCSS_dis
    elif configs.distance_type == "NetEDR":
        spatial_func = spatial_com.NetEDR_dis
        temporal_func = temporal_com.NetEDR_dis

    apn_node_triplets = pickle.load(open(configs.path_node_sets, 'rb'))
    apn_time_triplets = pickle.load(open(configs.path_time_sets, 'rb'))
    com_max_s = []
    com_max_t = []
    for i in range(len(apn_time_triplets)):
        temp_s_list = []
        temp_t_list = []
        for j in range(1, len(apn_time_triplets[i])):
            temp_s_list.append(spatial_func(apn_node_triplets[i][0], apn_node_triplets[i][j]))
            temp_t_list.append(temporal_func(apn_time_triplets[i][0], apn_time_triplets[i][j]))
        com_max_s.append(temp_s_list)
        com_max_t.append(temp_t_list)

    com_max_s = np.array(com_max_s)
    com_max_t = np.array(com_max_t)

    if configs.dataset_name == "porto" or configs.dataset_name == "tdrive":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 4
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    if configs.dataset_name == "rome":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 2
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    # Fix effects of extreme values
    com_max_s = com_max_s / np.max(com_max_s) * coe
    com_max_t = com_max_t / np.max(com_max_t) * coe

    weight = 0.5
    train_sets_dis = com_max_s * weight + com_max_t * (1 - weight)

    np.save(configs.path_sets_truth, train_sets_dis)
    print("complete: set ground truth")
    print(f"Avg pos distance: {train_sets_dis[:, :configs.num_companion].mean()}, avg neg distance: {train_sets_dis[:, configs.num_companion:].mean()}")


def triplet_ground_truth(configs):
    apn_node_triplets = pickle.load(open(configs.path_node_triplets, 'rb'))
    apn_time_triplets = pickle.load(open(configs.path_time_triplets, 'rb'))
    com_max_s = []
    com_max_t = []
    if configs.distance_type == "TP":
        spatial_func = spatial_com.TP_dis
        temporal_func = temporal_com.TP_dis
    elif configs.distance_type == "LCRS":
        spatial_func = spatial_com.LCRS_dis
        temporal_func = temporal_com.LCRS_dis
    elif configs.distance_type == "NetERP":
        spatial_func = spatial_com.NetERP_dis
        temporal_func = temporal_com.NetERP_dis
    elif configs.distance_type == "NetDTW":
        spatial_func = spatial_com.NetDTW_dis
        temporal_func = temporal_com.NetDTW_dis
    elif configs.distance_type == "NetLCSS":
        spatial_func = spatial_com.NetLCSS_dis
        temporal_func = temporal_com.NetLCSS_dis
    elif configs.distance_type == "NetEDR":
        spatial_func = spatial_com.NetEDR_dis
        temporal_func = temporal_com.NetEDR_dis

    for i in range(len(apn_time_triplets)):
        ap_s = spatial_func(apn_node_triplets[i][0], apn_node_triplets[i][1])
        an_s = spatial_func(apn_node_triplets[i][0], apn_node_triplets[i][2])
        com_max_s.append([ap_s, an_s])
        ap_t = temporal_func(apn_time_triplets[i][0], apn_time_triplets[i][1])
        an_t = temporal_func(apn_time_triplets[i][0], apn_time_triplets[i][2])
        com_max_t.append([ap_t, an_t])

    com_max_s = np.array(com_max_s)
    com_max_t = np.array(com_max_t)

    if configs.dataset_name == "porto" or configs.dataset_name == "tdrive":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 4
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    if configs.dataset_name == "rome":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 2
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    # Fix effects of extreme values
    com_max_s = com_max_s / np.max(com_max_s) * coe
    com_max_t = com_max_t / np.max(com_max_t) * coe

    train_triplets_dis = (com_max_s + com_max_t) / 2

    np.save(configs.path_triplets_truth, train_triplets_dis)
    print("complete: triplet ground truth")
    print(f"Avg pos distance: {train_triplets_dis[:, 0].mean()}, avg neg distance: {train_triplets_dis[:, 1].mean()}")


def triplet_st2vec_ground_truth(configs):
    apn_node_triplets = pickle.load(open(configs.path_node_triplets_st2vec, 'rb'))
    apn_time_triplets = pickle.load(open(configs.path_time_triplets_st2vec, 'rb'))
    com_max_s = []
    com_max_t = []
    if configs.distance_type == "TP":
        spatial_func = spatial_com.TP_dis
        temporal_func = temporal_com.TP_dis
    elif configs.distance_type == "LCRS":
        spatial_func = spatial_com.LCRS_dis
        temporal_func = temporal_com.LCRS_dis
    elif configs.distance_type == "NetERP":
        spatial_func = spatial_com.NetERP_dis
        temporal_func = temporal_com.NetERP_dis
    elif configs.distance_type == "NetDTW":
        spatial_func = spatial_com.NetDTW_dis
        temporal_func = temporal_com.NetDTW_dis
    elif configs.distance_type == "NetLCSS":
        spatial_func = spatial_com.NetLCSS_dis
        temporal_func = temporal_com.NetLCSS_dis
    elif configs.distance_type == "NetEDR":
        spatial_func = spatial_com.NetEDR_dis
        temporal_func = temporal_com.NetEDR_dis

    for i in range(len(apn_time_triplets)):
        ap_s = spatial_func(apn_node_triplets[i][0], apn_node_triplets[i][1])
        an_s = spatial_func(apn_node_triplets[i][0], apn_node_triplets[i][2])
        com_max_s.append([ap_s, an_s])
        ap_t = temporal_func(apn_time_triplets[i][0], apn_time_triplets[i][1])
        an_t = temporal_func(apn_time_triplets[i][0], apn_time_triplets[i][2])
        com_max_t.append([ap_t, an_t])

    com_max_s = np.array(com_max_s)
    com_max_t = np.array(com_max_t)

    if configs.dataset_name == "porto" or configs.dataset_name == "tdrive":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 4
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    if configs.dataset_name == "rome":
        if configs.distance_type == "TP":
            coe = 8
        elif configs.distance_type == "LCRS":
            coe = 2
        elif configs.distance_type == "NetERP":
            coe = 8
        elif str(configs.distance_type) == "NetDTW":
            coe = 8
        elif str(configs.distance_type) == "NetLCSS":
            coe = 8
        elif str(configs.distance_type) == "NetEDR":
            coe = 8

    # Fix effects of extreme values
    com_max_s = com_max_s / np.max(com_max_s) * coe
    com_max_t = com_max_t / np.max(com_max_t) * coe

    train_triplets_dis = (com_max_s + com_max_t) / 2

    np.save(configs.path_triplets_st2vec_truth, train_triplets_dis)
    print("complete: triplet ground truth")
    print(f"Avg pos distance: {train_triplets_dis[:, 0].mean()}, avg neg distance: {train_triplets_dis[:, 1].mean()}")


def s_batch_similarity_ground_truth(configs, valiortest=None):
    from spatial_similarity import Traj_distance
    node_list_int = np.load(configs.shuffle_node_file, allow_pickle=True)
    print(f"Total sample: {len(node_list_int)}")
    if valiortest == 'vali':
        node_list_int = node_list_int[10000:14000]
    elif valiortest == 'test':
        node_list_int = node_list_int[20000:24000]
    sample_list = node_list_int[:10000]
    print(f"Selected sample: {len(sample_list)}")

    pool = Pool(processes=20)
    for i in range(len(sample_list)):
        if i % 50 == 0:
            pool.apply_async(Traj_distance,
                             (i, configs.dataset_name, configs.distance_type, sample_list[i:i + 50], node_list_int, valiortest))
    pool.close()
    pool.join()

    return len(sample_list)


def s_merge_similarity_ground_truth(configs, sample_len, valiortest):
    res = []
    for i in range(sample_len):
        if i % 50 == 0:
            res.append(np.load(
                f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_batch/{configs.distance_type}_spatial_distance_{i}.npy"))
    res = np.concatenate(res, axis=0)
    np.save(f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_spatial_distance.npy", res)


def t_batch_similarity_ground_truth(configs, valiortest=None):
    from temporal_similarity import timelist_distance
    time_list_int = np.load(configs.shuffle_time_file, allow_pickle=True)
    if valiortest == 'vali':
        time_list_int = time_list_int[10000:14000]
    elif valiortest == 'test':
        time_list_int = time_list_int[20000:24000]

    sample_list = time_list_int[:10000]  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=20)
    for i in range(len(sample_list)):
        if i % 50 == 0:
            pool.apply_async(timelist_distance,
                             (i, configs.dataset_name, configs.distance_type, sample_list[i:i + 50], time_list_int, valiortest))
    pool.close()
    pool.join()

    return len(sample_list)


def t_merge_similarity_ground_truth(configs, sample_len, valiortest):
    res = []
    for i in range(sample_len):
        if i % 50 == 0:
            res.append(np.load(
                f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_batch/{configs.distance_type}_temporal_distance_{i}.npy"))
    res = np.concatenate(res, axis=0)
    np.save(f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_temporal_distance.npy", res)


def test_merge_st_dis(configs, weight, valiortest=None):
    s = np.load(f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_spatial_distance.npy")
    t = np.load(f"data/{configs.dataset_name}/ground_truth/{configs.distance_type}/{valiortest}_temporal_distance.npy")
    print(s.shape)

    unreach = {}
    for i, dis in enumerate(s):
        tmp = []
        for j, und in enumerate(dis):
            if und == -1:
                tmp.append(j)
        if len(tmp) > 0:
            unreach[i] = tmp

    s = s / np.max(s)
    t = t / np.max(t)

    st = s * weight + t * (1 - weight)

    for i in unreach.keys():
        st[i][unreach[i]] = -1

    path = configs.path_vali_truth.split("vali_st_distance")[0] + f"{valiortest}_st_distance_{weight}.npy"
    np.save(path, st)
    if weight == 0.5:
        if valiortest == 'vali':
            np.save(configs.path_vali_truth, st)
        else:
            np.save(configs.path_test_truth, st)
    print("complete: merge_st_distance")


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    #prepare_dataset() is called before generating sets
    # Remember to compute the new ground_truth for vali and test after calling "prepare_dataset"
    if configs.dataset_name == "porto":
        prepare_porto(configs)  # Porto is special due to its scale
    else:
        prepare_dataset(configs)
    data = DataLoader(configs)
    print(f"DataLoader done")
    for distance_type in ["TP"]:
        for num_companion in [1]:
            print(num_companion)
            configs.dataset_update({"dataset_name": "porto", "num_companion": num_companion, "distance_type": distance_type})
            data.get_sets()
            print(f"data.get_sets done")
            for valiortest in ["vali", "test"]:
                sample_len = s_batch_similarity_ground_truth(configs, valiortest=valiortest)
                s_merge_similarity_ground_truth(configs, sample_len=sample_len, valiortest=valiortest)
                sample_len = t_batch_similarity_ground_truth(configs, valiortest=valiortest)
                t_merge_similarity_ground_truth(configs, sample_len=sample_len, valiortest=valiortest)
                #for weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for weight in [0.5]:
                   test_merge_st_dis(configs, weight, valiortest=valiortest)
            print(f"test_merge_st_dis done")
            data = DataLoader(configs)
            print(f"DataLoader done")
            set_ground_truth(configs)
            #"""
            print(f"set_ground_truth done")


