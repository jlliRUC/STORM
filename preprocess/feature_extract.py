"""
Feature extract & tokenization for trajectories.
1. Due to the timeconsuming map-matching, we split the dataset into train, val, finetune and test
2. For train & val of pretraining, we need mm and augmented ["aug", "trim_rate_random", "temporal_distortion_rate_random"]
3. For finetune and test, we only need mm
4. Feature extract including 3 parts, S, T, ST
"""
import sys
sys.path.append("../")
sys.path.append("../model/")
import pandas as pd
import datetime
import pickle
import json
from utils import rle_index
import numpy as np
from config import Config
from Model import Date2VecConvert
import torch
import time
from augmentation import run_single
import warnings
warnings.filterwarnings("ignore")


def generate_point_matrix(dataset_name):
    res = np.load('../data/{}/ground_truth/Point_dis_matrix.npy'.format(dataset_name))
    return res


def get_meta_feature(configs):
    """
    Meta features include
    1. Road segment ids: already done in HMMM.map_matching
    2. Valid idxs of trajectory points: #trajectories is less than #points due to (1) one segment can contain multiple
    points, (2) those points that are too close to each other, are considered as noise points and thus filtered out
    during _preprocess in HMMM. (3) Once the moving object has at least one point record located in one segment, it is
    supposed to go across the whole road segment. Thus we consider the first point record of each segment as the corresponding
    point. (4) Our road network is bi-directional, and thus we don't have to identify the direction of each trajectory.
    3. Timestamps: we take the timestamp of each road segment's corresponding valid idx as its timestamp. Note that
    we cannot always generate a time interval for each segment since there maybe only one point record of that road
    segment. In this case, we use the time interval (traj_end - traj_start) instead.
    :param configs:
    :return:
    """
    dataset_name = configs.dataset_name
    with open(f"../data/{dataset_name}/{dataset_name}.pkl", "rb") as f:
        df_traj = pickle.load(f)
    with open(f"../data/{dataset_name}/{dataset_name}_valids.pkl", "rb") as f:
        valids_list = pickle.load(f)  # valid: some trajectory points might be filtered out as noises
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")

    for group_name in ["train", "val", "finetune", "test"]:
    #for group_name in ["finetune"]:
        if group_name == "train":
            offset = 0
        elif group_name == "val":
            offset = configs.num_train
        elif group_name == "finetune":
            offset = configs.num_train + configs.num_val
        elif group_name == "test":
            offset = configs.test_start

        invalid_idx_list = []
        with open(f"../data/{dataset_name}/{dataset_name}_{group_name}_mm_map_matching.pkl", "rb") as f:
            df_mm = pickle.load(f)
        with open(f"../data/{dataset_name}/{dataset_name}_{group_name}_aug_map_matching.pkl", "rb") as f:
            df = pickle.load(f)
            for i in range(df.shape[0]):
                # Filter out invalid probabilistic map-matching
                if None in df.loc[i, "road_segments"] or (
                        df_mm.loc[i, "road_segments"] == df.loc[i, "road_segments"]).all():
                    invalid_idx_list.append(i)

        for aug_name in ["mm", "aug"]:
            traj_id_list = []
            rs_ids_full_list = []
            rs_ids_list = []
            seg_starts_list = []
            seg_ends_list = []
            seg_tss_list = []  # timestamps of the first timestamp for each matched segment
            seg_tss_full_list = []
            seg_radians_list = []
            with open(f"../data/{dataset_name}/{dataset_name}_{group_name}_{aug_name}_map_matching.pkl", "rb") as f:
                df_traj_segment = pickle.load(f)
                for i in range(df_traj_segment.shape[0]):
                    road_segments = df_traj_segment.loc[i, "road_segments"]
                    rle_idx = rle_index(road_segments)
                    if i not in invalid_idx_list:
                        traj_id = offset + i
                        traj_id_list.append(traj_id)
                        rs_ids_full_list.append(road_segments)

                        # rle_idx: we merge the consecutively overlapping segments
                        rle_road_segment = road_segments[rle_idx]
                        rs_ids_list.append(rle_road_segment)

                        # Here we assume that the moving object always move from start to end of the map-matched road segments
                        # If map-matching is not reliable, we may need to find the "start" and "end" based on network distance
                        seg_starts_list.append([df_segments.loc[seg_id, "s_idx"] for seg_id in rle_road_segment])
                        seg_ends_list.append([df_segments.loc[seg_id, "e_idx"] for seg_id in rle_road_segment])
                        traj_timestamps = json.loads(df_traj.loc[traj_id, "Timestamps"])
                        full_timestamps = [traj_timestamps[idx] for idx in valids_list[traj_id]]
                        seg_tss_full_list.append(full_timestamps)
                        rle_timestamps = [traj_timestamps[valids_list[traj_id][idx]] for idx in rle_idx]
                        seg_tss_list.append(rle_timestamps)
                        seg_radians_list.append([df_segments.loc[seg_id, "radian"] for seg_id in rle_road_segment])

            df_results = pd.DataFrame.from_dict({"traj_id": traj_id_list,
                                                 "rs_ids_full": rs_ids_full_list,
                                                 "rs_ids": rs_ids_list,
                                                 "seg_starts": seg_starts_list,
                                                 "seg_ends": seg_ends_list,
                                                 "seg_tss": seg_tss_list,
                                                 "seg_tss_full": seg_tss_full_list,
                                                 "seg_radians": seg_radians_list})
            df_results.insert(loc=0, column='idx', value=[j for j in range(df_results.shape[0])])
            with open(f"../data/{dataset_name}/{dataset_name}_{group_name}_{aug_name}_feature_meta.pkl", "wb") as f:
                pickle.dump(df_results, f)
            print(group_name, aug_name, df_traj_segment.shape[0], len(invalid_idx_list), df_results.shape)


def get_feature_t(configs, file_meta):
    with open(f"../data/{configs.dataset_name}/{file_meta}", "rb") as f:
        df = pickle.load(f)

    d2v = Date2VecConvert(model_path="../d2v_model/d2v_98291_17.169918439404636.pth")
    all_list = []
    for i in range(df.shape[0]):
        one_seq = df.loc[i, "seg_tss"]
        one_list = []
        for timestamp in one_seq:
            t = datetime.datetime.fromtimestamp(timestamp)
            t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
            x = torch.Tensor(t).float()
            embed = d2v(x)
            one_list.append(embed)
        one_list = torch.cat(one_list, dim=0)
        one_list = one_list.view(-1, configs.date2vec_size)

        all_list.append(one_list.to(torch.float32).numpy())

    all_list = np.array(all_list)

    with open(f"../data/{configs.dataset_name}/{file_meta.split('_meta')[0]}_t.pkl", "wb") as f:
        pickle.dump(all_list, f)


def get_feature(configs, file_meta):
    """
    Generate features from meta features (original / augmented)
    :param configs:
    :return:
    """
    dataset_name = configs.dataset_name
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")
    dist_matrix = generate_point_matrix(dataset_name)
    with open(f"../data/{dataset_name}/{file_meta}", "rb") as f:
        df = pickle.load(f)

        seg_diss_list = []  # network distance between start points of two segments
        seg_tis_list = []  # time interval between start points of two segments
        seg_speed_list = []
        seg_turns_list = []
        seg_aspeed_list = []

        for i in range(df.shape[0]):
            rle_road_segment = df.loc[i, "rs_ids"]
            seg_starts = df.loc[i, "seg_starts"]
            seg_ends = df.loc[i, "seg_ends"]
            traj_timestamps = df.loc[i, "seg_tss"]
            full_tss = df.loc[i, "seg_tss_full"]
            seg_radians = df.loc[i, "seg_radians"]

            if len(rle_road_segment) == 1:
                #  For short (few number of points, or short distance due to congestion) trajectories, all points might be mapped to one segment
                seg_diss_list.append([abs(dist_matrix[seg_starts[0]][seg_ends[0]])])
                seg_tis_list.append([full_tss[-1] - full_tss[0]])
                seg_speed_list.append([seg_diss_list[-1][0] / seg_tis_list[-1][0]])
                seg_turns_list.append([0])
                seg_aspeed_list.append([0])
            else:
                # The last interval should be about the last segment
                diss = ([abs(dist_matrix[seg_starts[j]][seg_starts[j - 1]]) for j in range(1, len(seg_starts))]
                        + [abs(dist_matrix[seg_starts[-1]][seg_ends[-1]])])
                seg_diss_list.append(diss)

                tis = [traj_timestamps[j] - traj_timestamps[j - 1] for j in range(1, len(traj_timestamps))]
                if full_tss[-1] == traj_timestamps[-1]:  # The last segment only contains one point
                    tis += [traj_timestamps[-1] - full_tss[-2]]
                else:
                    tis += [full_tss[-1] - traj_timestamps[-1]]
                seg_tis_list.append(tis)

                turns = [abs(seg_radians[j] - seg_radians[j - 1]) for j in range(1, len(seg_radians))]
                turns += [np.array(turns).mean()]  # We assume the last turn is the avg turn...
                seg_turns_list.append(turns)

                seg_speed_list.append([abs(seg_diss_list[-1][j] / seg_tis_list[-1][j]) for j in range(0, len(seg_tis_list[-1]))])
                seg_aspeed_list.append([abs(seg_turns_list[-1][j] / seg_tis_list[-1][j]) for j in range(0, len(seg_tis_list[-1]))])

        df_results = pd.DataFrame.from_dict({"seg_diss": seg_diss_list,
                                             "seg_tis": seg_tis_list,
                                             "seg_speed": seg_speed_list,
                                             "seg_turns": seg_turns_list,
                                             "seg_aspeed": seg_aspeed_list})
        df_results = df.join(df_results)
        with open(f"../data/{dataset_name}/{file_meta.split('_meta')[0]}.pkl", "wb") as f:
            pickle.dump(df_results, f)
        print(file_meta, df_results.shape)


def get_segmentwise_weights(configs, file_feature):
    """
    For each trajectory, we give its segment-wise weight matrix based on the network distance, time inverval, radian change,
    weighted by the difference of their (speed, aspeed)
    :param configs:
    :return:
    """
    dataset_name = configs.dataset_name
    dist_matrix = generate_point_matrix(dataset_name)

    with open(f"../data/{dataset_name}/{file_feature}", "rb") as f:
        df = pickle.load(f)
    df_segments = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_segment.csv")

    weights_list = []

    for i in range(df.shape[0]):
        traj_length = len(df.loc[i, "rs_ids"])

        dis_array = np.zeros((traj_length, traj_length))
        ti_array = np.zeros((traj_length, traj_length))
        speed_array = np.zeros((traj_length, traj_length))
        starts = df.loc[i, "seg_starts"]
        tss = df.loc[i, "seg_tss"]
        speed = df.loc[i, "seg_speed"]
        for p in range(traj_length):
            for q in range(p + 1, traj_length):
                dis_array[p][q] = abs(dist_matrix[starts[q]][starts[p]])
                ti_array[p][q] = tss[q] - tss[p]
                speed_array[p][q] = abs(speed[q] - speed[p])

        dis_array = dis_array + dis_array.T  # complete the weight matrix
        ti_array = ti_array + ti_array.T
        speed_array = speed_array + speed_array.T

        dis_array = dis_array / np.max(dis_array)
        ti_array = ti_array / np.max(ti_array)
        speed_array = speed_array / np.max(speed_array) + 1  # The weight of two identical speeds should be 1, not 0

        weights_list.append(np.nan_to_num(np.multiply((dis_array * 0.5 + ti_array * 0.5), speed_array)))

    with open(f"../data/{dataset_name}/{file_feature.split('_feature')[0]}_segmentwise_weights.pkl", "wb") as f:
        pickle.dump(weights_list, f)


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    t1 = time.time()
    # 1. Meta features. For ["train", "val", "test", "finetune"] of ["mm", "aug"]
    get_meta_feature(configs)
    t2 = time.time()
    print(f"Get meta_feature costs {t2 - t1} s")

    # 2. Augmentation. For ["train", "val"] do augmenttion and obtain the meta_feature
    for name in ["trim", "temporal_distortion"]:
        temp_t1 = time.time()
        run_single(configs.dataset_name, {"name": name, "rate": "random"})
        temp_t2 = time.time()
        print(f"Augmentation-{name} costs {temp_t2 - temp_t1} s")
    t3 = time.time()
    print(f"Augmentation costs {t3 - t2} s")

    # 3. Generate feature and weights. For ["train", "val", "test", "finetune"] of "mm"
    # and ["train", "val"] of ["aug", "trim", "temporal_distortion"]
    num_diss_token_max, num_tis_token_max, num_speed_token_max, num_turns_token_max, num_aspeed_token_max = 0, 0, 0, 0, 0
    params = []
    for group_name in ["train", "val", "test", "finetune"]:
        for aug_name in ["mm"]:
            params.append((group_name, aug_name))
    for group_name in ["train", "val"]:
        for aug_name in ["trim_rate_random", "temporal_distortion_rate_random", "aug"]:
            params.append((group_name, aug_name))
    for group_name, aug_name in params:
        print(f"{group_name}-{aug_name}:")
        temp_t1 = time.time()
        get_feature(configs, f"{configs.dataset_name}_{group_name}_{aug_name}_feature_meta.pkl")
        temp_t2 = time.time()
        print(f"generate_feature costs {temp_t2 - temp_t1} s")
        get_feature_t(configs, f"{configs.dataset_name}_{group_name}_{aug_name}_feature_meta.pkl")
        temp_t3 = time.time()
        print(f"generate_t costs {temp_t3 - temp_t2} s")
        get_segmentwise_weights(configs, f"{configs.dataset_name}_{group_name}_{aug_name}_feature.pkl")
        temp_t4 = time.time()
        print(f"get_segmentwise_weights costs {temp_t4 - temp_t3} s")
    t4 = time.time()
    print(f"Total costs {t4 - t1} s")
