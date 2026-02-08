import numpy as np
import torch
import pickle
import random
from config import Config
import pandas as pd
import networkx as nx


def get_road_network(dataset_name):
    road_network = nx.DiGraph()

    nodes = pd.read_csv(f"data/{dataset_name}/{dataset_name}_node.csv", usecols=['node_idx', 'lon', 'lat'])
    for i in range(nodes.shape[0]):
        road_network.add_node(nodes.loc[i, 'node_idx'], lon=nodes.loc[i, 'lon'], lat=nodes.loc[i, 'lat'])

    edges = pd.read_csv(f"data/{dataset_name}/{dataset_name}_segment.csv", usecols=['edge_id', 's_idx', 'e_idx', 's_lon', 's_lat', 'e_lon', 'e_lat', 'c_lon', 'c_lat', 'length'])
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


def get_segment_embedding(configs):
    if configs.load_segment_embedding:
        embeddings = pickle.load(open(
            f"data/{configs.dataset_name}/{configs.dataset_name}_{configs.segment_embedding_type}_rs_embeddings.pkl",
            "rb"))  # [num_seg, num_seg_feature]
        if configs.segment_finetune:
            print(f"We load segment embedding from {configs.segment_embedding_type}, and finetune")
        else:
            print(f"We load segment embedding from {configs.segment_embedding_type}, and not finetune")
    else:
        with open(f"data/{configs.dataset_name}/{configs.dataset_name}_road_network_features.pkl", "rb") as f:
            df_segments = pickle.load(f)
            embeddings = torch.tensor(df_segments.loc[:, ["highway_cls", "s_token", "e_token", "length_token",
                                                          "radian_token"]].to_numpy())  # [num_seg, num_seg_dim]
        print("We train segment embeddings from scratch")

    # traj edge
    edge_index = pickle.load(open(f"data/{configs.dataset_name}/{configs.dataset_name}_traj_edge_index_5.pkl", "rb"))
    if configs.edge_dim is not None:
        edge_weights = pickle.load(
            open(f"data/{configs.dataset_name}/{configs.dataset_name}_traj_edge_weights_5.pkl", "rb"))
        return (torch.tensor(embeddings).to(configs.device),
                torch.tensor(edge_index).T.to(configs.device),
                torch.tensor(edge_weights, dtype=torch.float32).to(configs.device))
    else:
        edge_weights = None
        return (torch.tensor(embeddings).to(configs.device),
                torch.tensor(edge_index).T.to(configs.device),
                None)


class EvalDataLoader:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.st = []
        self.s = []
        self.t = []
        self.start = 0

    def load(self):
        # ST
        st_path = f"data/{self.dataset_name}/{self.dataset_name}_test_mm_segmentwise_weights.pkl"
        with open(st_path, "rb") as f:
            weights_list = pickle.load(f)

        self.st = np.array(weights_list)

        # S
        feature_path = f"data/{self.dataset_name}/{self.dataset_name}_test_mm_feature.pkl"
        with open(feature_path, "rb") as f:
            df = pickle.load(f)

        for i in range(df.shape[0]):
            self.s.append(df.loc[i, "rs_ids"].tolist())
        self.s = np.array(self.s)

        # T
        t_path = f"data/{self.dataset_name}/{self.dataset_name}_test_mm_feature_t.pkl"
        with open(t_path, "rb") as f:
            self.t = pickle.load(f)

        self.num_traj = len(self.s)
        print(f"Loaded {self.num_traj} trajectories.")
        self.start = 0

    def get_one_batch(self):
        if self.start >= self.num_traj:
            return None, None, None, None

        # sub_s_list = [s[idx] for idx in range(self.start, self.start + self.batch_size) for s in self.s_list]
        sub_s = self.s[self.start:self.start + self.batch_size]
        # sub_t_list = [t[idx] for idx in range(self.start, self.start + self.batch_size) for t in self.t_list]
        sub_t = self.t[self.start:self.start + self.batch_size]
        sub_st = self.st[self.start:self.start + self.batch_size]
        lengths = torch.LongTensor(list(map(len, self.s[self.start:self.start + self.batch_size])))

        self.start += self.batch_size

        return sub_s, sub_t, sub_st, lengths


class CLDataLoader:
    def __init__(self, dataset_name, num_train, num_val, augmentation_list, istrain, batch_size, shuffle=True):

        self.dataset_name = dataset_name
        self.num_train = num_train
        self.num_val = num_val
        self.augmentation_list = augmentation_list
        self.istrain = istrain
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self):
        # train_val index
        # Limited by the time-consuming map-matching operation, we split the dataset into train and val.
        # Thus the idx will only be the relative idx in each file, not the original idx in the full dataset.
        # index_list = [i for i in range(0, self.num_train)] if self.istrain else [i for i in range(0, self.num_val)]
        self.s_list = []
        self.t_list = []
        self.st_list = []

        group_name = "train" if self.istrain else "val"

        for aug_name in self.augmentation_list:
            print(aug_name)
            # ST
            st_path = f"data/{self.dataset_name}/{self.dataset_name}_{group_name}_{aug_name}_segmentwise_weights.pkl"
            with open(st_path, "rb") as f:
                weights_list = pickle.load(f)
            self.st_list.append(np.array(weights_list))

            # S
            feature_path = f"data/{self.dataset_name}/{self.dataset_name}_{group_name}_{aug_name}_feature.pkl"
            with open(feature_path, "rb") as f:
                df = pickle.load(f)
            s = []
            for i in range(df.shape[0]):
                temp = df.loc[i, "rs_ids"]
                if not isinstance(temp, list):
                    temp = temp.tolist()
                s.append(temp)
            self.s_list.append(np.array(s))

            # T
            t_path = f"data/{self.dataset_name}/{self.dataset_name}_{group_name}_{aug_name}_feature_t.pkl"
            with open(t_path, "rb") as f:
                self.t_list.append(pickle.load(f))

        self.num_traj = len(self.s_list[0])
        self.start = 0

    def get_one_batch(self):
        if self.start + self.batch_size >= self.num_traj:
            return None, None, None, None
        if self.shuffle:
            index = list(range(self.num_traj))
            random.shuffle(index)
            self.s_list = [s[index] for s in self.s_list]
            # self.s_list = [s[i] for i in index for s in self.s_list]
            self.t_list = [t[index] for t in self.t_list]
            # self.t_list = [t[i] for i in index for t in self.t_list]
            self.st_list = [st[index] for st in self.st_list]
            self.shuffle = False  # Just shuffle once

        # sub_s_list = [s[idx] for idx in range(self.start, self.start + self.batch_size) for s in self.s_list]
        sub_s_list = [s[self.start:self.start + self.batch_size] for s in self.s_list]
        # sub_t_list = [t[idx] for idx in range(self.start, self.start + self.batch_size) for t in self.t_list]
        sub_t_list = [t[self.start:self.start + self.batch_size] for t in self.t_list]
        sub_st_list = [st[self.start:self.start + self.batch_size] for st in self.st_list]
        lengths_list = [torch.LongTensor(list(map(len, s[self.start:self.start + self.batch_size]))) for s in
                        self.s_list]

        self.start += self.batch_size

        return sub_s_list, sub_t_list, sub_st_list, lengths_list


class FinetuneDataLoader:
    def __init__(self, dataset_name, batch_size, num_traj, distance_type, shuffle):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.st = []
        self.s = []
        self.t = []

        self.start = 0
        self.num_traj = num_traj
        self.distance_type = distance_type
        self.shuffle = shuffle

    def load(self, configs):
        if configs.istriplet:
            print(f"We load dataset from {configs.path_st_triplets_st2vec}")
            self.st = np.array(pickle.load(open(configs.path_st_triplets_st2vec, 'rb')))
            self.s = np.array(pickle.load(open(configs.path_rs_triplets_st2vec, 'rb')))
            self.t = np.array(pickle.load(open(configs.path_d2vec_triplets_st2vec, 'rb')))
            ground_truth = np.load(configs.path_triplets_st2vec_truth)
        else:
            print(f"We load dataset from {configs.path_st_triplets}")
            self.st = np.array(pickle.load(open(configs.path_st_sets, 'rb')))
            self.s = np.array(pickle.load(open(configs.path_rs_sets, 'rb')))
            self.t = np.array(pickle.load(open(configs.path_d2vec_sets, 'rb')))
            ground_truth = np.load(configs.path_sets_truth)
        lengths_list = []
        for i in range(len(self.s)):
            lengths_list.append([])
            for j in range(len(self.s[0])):
                lengths_list[-1].append(len(self.s[i][j]))
        self.lengths = np.array(lengths_list)

        self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        self.num_traj = len(self.st)
        print(f"Loaded {self.num_traj} training instances.")
        self.start = 0
        self.num_companion = len(self.st[0]) - 1
        print(f"For each sample, we have {self.num_companion} companion samples")

    def get_one_batch(self):
        if self.start >= self.num_traj:
            return [None] * 5
        if self.shuffle:
            index = list(range(self.num_traj))
            random.shuffle(index)
            self.s = self.s[index]
            self.t = self.t[index]
            # self.t_list = [t[i] for i in index for t in self.t_list]
            self.st = self.st[index]
            self.lengths = self.lengths[index]
            self.ground_truth = self.ground_truth[index]
            self.shuffle = False  # Just shuffle once

        batch_index = [idx for idx in range(self.start, self.start + self.batch_size) if idx < self.num_traj]

        s_list = [self.s[batch_index, i] for i in range(self.num_companion + 1)]
        t_list = [self.t[batch_index, i] for i in range(self.num_companion + 1)]
        st_list = [self.st[batch_index, i] for i in range(self.num_companion + 1)]
        length_list = [torch.LongTensor(self.lengths[batch_index, i]) for i in range(self.num_companion + 1)]

        self.start += self.batch_size

        return s_list, t_list, st_list, length_list, self.ground_truth[batch_index]

