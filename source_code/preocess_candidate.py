import recommend
import utils
import os
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool


num_pool = 32
batch_size = 2000

mat_max = {}
mat_min = {}


def cal_min_max(dataset_name):
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy")).astype(np.float16)
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy")).astype(np.float16)

    ma, mi = float('-inf'), float('inf')
    print(f"start calculating max and min of {dataset_name} user: {emb_user.shape[0]}\titem: {emb_item.shape[0]}...")
    batch = 0
    while batch * batch_size < emb_user.shape[0]:
        start = batch * batch_size
        batch_usr = emb_user[start: start + batch_size]
        mat_dis_batch = np.dot(batch_usr, emb_item.T)

        ma, mi = max(ma, mat_dis_batch.max()), min(mi, mat_dis_batch.min())
        batch += 1
        print(f"batch {batch}: max={ma}\tmin={mi}")
    print(f"dataset: {dataset_name} max={ma}\tmin={mi}")
    return ma, mi


def create_candidates_stratification_batch(dataset_name, seed, K_c=1000, num_fold=10, epsilon=0.1):
    path_candidate = os.path.join("data", dataset_name, "rec", str(seed), "candidate.npy")
    # emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy")).astype(np.float16)
    # emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy")).astype(np.float16)
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy")).astype(np.float16)
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy")).astype(np.float16)
    if os.path.exists(path_candidate):
        mat_candidate = np.load(path_candidate, allow_pickle=True).item()
        if emb_user.shape[0] == len(mat_candidate):
            return np.load(path_candidate, allow_pickle=True).item()

    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))

    group_train = df_train.groupby("userInd")
    num_item = emb_item.shape[0]
    all_items = set([i for i in range(num_item)])
    # mat_dis = np.dot(emb_user, emb_item.T).astype(np.float16)
    max_dis, min_dis = mat_max[dataset_name] + epsilon, mat_min[dataset_name]
    inter = (max_dis - min_dis) / num_fold

    # mat_label = np.floor((mat_dis - min_dis) / inter).astype(np.int8)

    batch = 0
    print("Start processing...")
    random.seed(seed)
    list_res = []
    while batch * batch_size < emb_user.shape[0]:
        start = batch * batch_size
        batch_usr = emb_user[start: start+batch_size]
        mat_dis_batch = np.dot(batch_usr, emb_item.T)
        mat_label_batch = np.floor((mat_dis_batch- min_dis) / inter).astype(np.int8)

        i = start
        pool = Pool(num_pool)
        while i < start + batch_size and i < emb_user.shape[0]:
            list_label = mat_label_batch[i-start]
            group = group_train.get_group(i)
            list_no_train = list(all_items - set(group["itemInd"].values.tolist()))
            list_label = list_label[list_no_train].tolist()
            df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
            list_res.append(
                pool.apply_async(
                    recommend.create_candidates_stratification_sub,
                    (
                        df_item,
                        K_c,
                    ),
                )
            )
            i += 1

        batch += 1
        pool.close()
        pool.join()
        print(f"batch {batch} finished...")

    # multi-process
    # print("Start processing...")
    # random.seed(seed)
    # pool = Pool(num_pool)
    # list_res = []
    # for list_label, (uind, group) in zip(mat_label, group_train):
    #     list_no_train = list(all_items - set(group["itemInd"].values.tolist()))
    #     list_label = list_label[list_no_train].tolist()
    #     df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
    #     list_res.append(
    #         pool.apply_async(
    #             recommend.create_candidates_stratification_sub,
    #             (
    #                 df_item,
    #                 K_c,
    #             ),
    #         )
    #     )
    # pool.close()
    # pool.join()

    list_test_iind = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))["itemInd"].values.tolist()
    mat_candidate = dict()
    for uind, res in enumerate(list_res):
        mat_candidate[uind] = res.get()
        mat_candidate[uind].append(list_test_iind[uind])

    np.save(path_candidate, mat_candidate)
    return mat_candidate

if __name__ == "__main__":
    list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]
    list_method = ["rand", "nov", "pop", "qua", "ela", "acc", "dif", "div"]
    list_dataset_name = list_dataset_name[3:4]
    list_seed = [777, 7777, 77777, 73, 79, 83, 89, 97, 101, 103]#[0:1]
    # recommend.recommend(list_dataset_name, list_seed)
    # utils.evaluate(list_dataset_name, list_seed, list_method)

    for dataset_name in list_dataset_name:
        if mat_max.get(dataset_name, None) is None:
            ma, mi = cal_min_max(dataset_name)
            mat_min[dataset_name] = mi
            mat_max[dataset_name] = ma
        for seed in list_seed:
            path = os.path.join("data", dataset_name, "rec")
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, str(seed))
            if not os.path.exists(path):
                os.mkdir(path)
            create_candidates_stratification_batch(dataset_name, seed)
