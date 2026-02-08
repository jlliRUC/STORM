import sys
sys.path.append("../")
import math
import random
import copy
from config import Config
import os
import pickle
import time
import functools


def data_augmentation(name):
    if name == "trim":
        return trim
    elif name == "temporal_distortion":
        return temporal_distortion
    elif name == "original":
        return original


def original(trajs):
    return trajs, [i for i in range(len(trajs))]


def trim(ori_length, rate, decay_rate=0.5):
    """
    We trim the trajectory by randomly removing some of its elements with a rate and the following elements with a decay_rate.
    :param ori_length:
    :param rate:
    :param decay_rate:
    :return:
    """
    if rate == 'random':
        trim_rate = random.randint(1, 4) / 10
    else:
        trim_rate = rate

    current_rate = copy.deepcopy(trim_rate)
    valid_idx = []
    full_idxs = [i for i in range(ori_length)]
    for i in range(len(full_idxs)):
        if random.random() <= current_rate:
            current_rate *= decay_rate  # i is removed, current_rate should be decayed
        else:
            valid_idx.append(i)
            current_rate = copy.deepcopy(trim_rate)  # i is kept, current_rate should be recovered to rate

    if 0 not in valid_idx:
        valid_idx.insert(0, 0)

    return valid_idx


def temporal_distortion(timestamps, rate):
    """
    We distort the time interval, instead of the timestamps directly, to maintain the monotonicity
    :param timestamps:
    :param rate:
    :return: Array of distorted traj timestamps data
    """
    distorted_timestamps = copy.copy(timestamps)

    for i in range(1, len(distorted_timestamps)):
        if rate == 'random':
            distort_rate = random.randint(1, 4) / 10
        else:
            distort_rate = rate

        if random.random() <= distort_rate:
            # Since the time intervals are not always even in original point-trajectory, let alone segment-trajectory,
            # we calculate time interval manually.
            radius_ti = timestamps[i] - timestamps[i - 1]
            # To strictly maintain the monotonicity, t should be the distorted timestamps, not the original one.
            t = distorted_timestamps[i - 1]
            if radius_ti == 0:  # Records error
                tnoise = 0
            else:
                tnoise = random.randint(1, radius_ti)  # Timestamps are always integers
            distorted_timestamps[i] = t + tnoise

    return distorted_timestamps


def run_single(dataset_name, method):
    start = time.time()

    # We only need to augment the train set & val set
    for group_suffix in ["train", "val"]:
        with open(f"../data/{dataset_name}/{dataset_name}_{group_suffix}_mm_feature_meta.pkl", "rb") as f:
            df = pickle.load(f)

        # Function for data augmentation
        transformer = functools.partial(data_augmentation(method["name"]), rate=method["rate"])

        # Information for file name
        suffix = f"{method['name']}_rate_{method['rate']}"
        print(f"{suffix} for {dataset_name}-{group_suffix}")

        aug_folder = f"../data/{dataset_name}"
        if not os.path.exists(aug_folder):
            os.mkdir(aug_folder)

        if method["name"] == "temporal_distortion":
            df["seg_tss"] = df["seg_tss"].map(lambda ts: transformer(ts))
        elif method["name"] == "trim":
            for i in range(df.shape[0]):
                valid_idx = transformer(len(df.loc[i, "rs_ids"]))
                for column_name in ["rs_ids", "seg_starts", "seg_ends", "seg_tss"]:
                    df.at[i, column_name] = [df.loc[i, column_name][item] for item in valid_idx]
        elif method["name"] == "original":
            df = df
        else:
            print("Unknown augmentation method!")

        with open(os.path.join(aug_folder, f"{dataset_name}_{group_suffix}_{suffix}_feature_meta.pkl"), "wb") as f:
            pickle.dump(df, f)

        print(f"{suffix} cost: {time.time() - start} s")


if __name__ == "__main__":
    # For testing
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    for name in ["trim", "temporal_distortion"]:
        run_single(configs.dataset_name, {"name": name, "rate": "random"})





