"""
Overview of eight Recommendation Strategies
    - function random_: randomly recommend items in candidates
    - function novelty: recommend items based on their release time
    - function unpopularity: recommend items based on the number of occurrences
    - high_quality: recommend items based on their average ratings
    - function elasticity_item: recommend items whose accuracy level matches users' acceptance ability
        (measure user elasticity based on related items)
    - function accuracy_cf: recommend items based on collaborative filtering
    - function diversity: recommend items based on the item-level diversity of recommendations
    - function difference: recommend items based on the item-level diversity among recommendations and user histories
"""
import os
from multiprocessing import Pool
from collections import Counter
import math
import random

import pandas as pd
import numpy as np
import utils

num_pool = 8


def random_sub(list_candidate, K):
    return np.array(random.sample(list_candidate, K)).reshape(1, K)


def random_(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_rand.npy")
    if os.path.exists(save_path):
        return
    pool = Pool(num_pool)
    list_res = [
        pool.apply_async(
            random_sub,
            (
                mat_candidate[uind],
                K,
            ),
        )
        for uind in range(len(mat_candidate))
    ]
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def sub_argpartition(list_score, list_candidate, K):
    # descend
    return list_candidate[np.argpartition(list_score, -K)[-K:]].reshape(1, K)


def novelty(mat_candidate, dataset_name, seed, K=20):
    """
    - Generating recommendations based on item novelty.
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    """
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_nov.npy")
    if os.path.exists(save_path):
        return
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))

    list_date = df_item["date"].values
    pool = Pool(num_pool)
    list_res = [
        pool.apply_async(
            sub_argpartition,
            (
                list_date[mat_candidate[i]],
                np.array(mat_candidate[i]),
                K,
            ),
        )
        for i in range(len(mat_candidate))
    ]
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def unpopularity(mat_candidate, dataset_name, seed, K=20):
    """
    - Generating recommendations based on item unpopularity.
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    """
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_pop.npy")
    if os.path.exists(save_path):
        return
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))
    list_count = df_item["count"].values

    pool = Pool(num_pool)
    list_res = [
        pool.apply_async(
            sub_argpartition,
            (
                -list_count[mat_candidate[i]],
                np.array(mat_candidate[i]),
                K,
            ),
        )
        for i in range(len(mat_candidate))
    ]
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def high_quality(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_qua.npy")
    if os.path.exists(save_path):
        return
    df_rating = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"), usecols=["itemInd", "rating"])
    list_mean_rating = df_rating.groupby("itemInd")["rating"].mean().values

    pool = Pool(num_pool)
    list_res = [
        pool.apply_async(
            sub_argpartition,
            (
                list_mean_rating[mat_candidate[i]],
                np.array(mat_candidate[i]),
                K,
            ),
        )
        for i in range(len(mat_candidate))
    ]
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def elasticity_item_sub(list_factor, list_candidate, mean, K, alpha):
    return list_candidate[np.argpartition(np.abs(list_factor - alpha * mean), K)[:K]].reshape(1, K)


def elasticity_item(mat_candidate, dataset_name, seed, K=20, alpha=1, **kwargs):
    """
    - Generating recommendations based on user elasticity
    - Quantifying user elasticity based on item category
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    Returns:
        mat_rec: recommendation matrix
    """
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_ela.npy")
    if os.path.exists(save_path):
        return
    df_user = pd.read_csv(os.path.join("data", dataset_name, "user.csv"))
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))

    mat_dis = np.dot(emb_user, emb_item.T)
    max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)

    mat_similarity = []
    for uind in range(len(mat_candidate)):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten().tolist()
        mat_similarity.append(similarity_t)

    list_item_count = df_user["num_item"].values.tolist()
    # list_item_count = df_user["dot"].values.tolist()
    count_min, count_max = min(list_item_count), max(list_item_count)
    list_ela = ((np.array(list_item_count) - count_min) / (count_max - count_min)).reshape(emb_user.shape[0], 1)

    mat_factor = [
        ((np.array(line) - min_dis) / (max_dis - min_dis) + ela).tolist() for line, ela in zip(mat_similarity, list_ela)
    ]
    list_factor = []
    for line in mat_factor:
        list_factor += line
    mean_factor = np.mean(list_factor)

    pool = Pool(num_pool)
    list_res = []
    for uind in range(emb_user.shape[0]):
        list_res.append(
            pool.apply_async(
                elasticity_item_sub,
                (
                    np.array(mat_factor[uind]),
                    np.array(mat_candidate[uind]),
                    mean_factor,
                    K,
                    alpha,
                ),
            )
        )
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def accuracy_cf(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_acc.npy")
    if os.path.exists(save_path):
        return
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))

    pool = Pool(num_pool)
    list_res = []
    for uind in range(len(mat_candidate)):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten().tolist()
        list_res.append(pool.apply_async(sub_argpartition, (similarity_t, np.array(mat_candidate[uind]), K)))
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def dpp(kernel_matrix, max_length, epsilon=1e-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)

    selected_items.append(selected_item)
    # cnt = 0
    while len(selected_items) < max_length:
        # cnt += 1
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            print("break")
            break
        # if cnt > 2*max_length:
        #
        #     break
        # print("looping", cnt)
        selected_items.append(selected_item)

    if len(selected_items) < max_length:
        selected_items += random.sample(
            list(set(range(item_size)) - set(selected_items)),
            max_length - len(selected_items),
        )
    return selected_items


def diversity(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_div.npy")
    if os.path.exists(save_path):
        return
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))

    list_res = []
    pool = Pool(num_pool)
    for uind in range(len(mat_candidate)):
        mat_similarity = np.dot(emb_item[mat_candidate[uind]], emb_item[mat_candidate[uind]].T)
        list_res.append(
            pool.apply_async(
                dpp,
                (
                    mat_similarity,
                    K,
                ),
            )
        )
    pool.close()
    pool.join()

    mat_rec = np.concatenate(
        [np.array(mat_candidate[uind])[np.array(list_res[uind].get())].reshape(1, K) for uind in range(len(mat_candidate))]
    )

    np.save(save_path, mat_rec)


def _diversity(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_div.npy")
    if os.path.exists(save_path):
        return
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))

    list_res = []
    for uind in range(len(mat_candidate)):
        mat_similarity = np.dot(emb_item[mat_candidate[uind]], emb_item[mat_candidate[uind]].T)
        list_res.append(
                dpp(mat_similarity, K,),
        )

    mat_rec = np.concatenate(
        [np.array(mat_candidate[uind])[np.array(list_res[uind])].reshape(1, K) for uind in range(len(mat_candidate))]
    )

    np.save(save_path, mat_rec)


def difference(mat_candidate, dataset_name, seed, K=20):
    save_path = os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_dif.npy")
    if os.path.exists(save_path):
        return
    # load data
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))

    pool = Pool(num_pool)
    list_res = []
    for uind in range(len(mat_candidate)):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten()
        list_res.append(pool.apply_async(sub_argpartition, (-similarity_t, np.array(mat_candidate[uind]), K)))
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(save_path, mat_rec)


def create_candidates_stratification_sub(df_item, K_c):
    if len(df_item) < K_c:
        K_c = len(df_item)
    df_item = (
        df_item.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(int(np.rint(K_c * len(x) / len(df_item)))))
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return df_item["itemInd"].values.tolist()


def create_candidates_stratification(dataset_name, seed, K_c=1000, num_fold=10, epsilon=0.1):
    path_candidate = os.path.join("data", dataset_name, "rec", str(seed), "candidate.npy")
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
    mat_dis = np.dot(emb_user, emb_item.T).astype(np.float16)
    max_dis, min_dis = np.max(mat_dis) + epsilon, np.min(mat_dis)
    inter = (max_dis - min_dis) / num_fold

    mat_label = np.floor((mat_dis - min_dis) / inter).astype(np.int8)

    # multi-process
    print("Start processing...")
    random.seed(seed)
    pool = Pool(num_pool)
    list_res = []
    for list_label, (uind, group) in zip(mat_label, group_train):
        list_no_train = list(all_items - set(group["itemInd"].values.tolist()))
        list_label = list_label[list_no_train].tolist()
        df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
        list_res.append(
            pool.apply_async(
                create_candidates_stratification_sub,
                (
                    df_item,
                    K_c,
                ),
            )
        )
    pool.close()
    pool.join()

    list_test_iind = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))["itemInd"].values.tolist()
    mat_candidate = dict()
    for uind, res in enumerate(list_res):
        mat_candidate[uind] = res.get()
        mat_candidate[uind].append(list_test_iind[uind % len(list_test_iind)])

    np.save(path_candidate, mat_candidate)
    return mat_candidate


def create_user(dataset_name):
    if os.path.exists(os.path.join("data", dataset_name, "user.csv")):
        return
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    group_train = df_train.groupby("userInd")
    list_num_item, list_num_category = [], []
    for uind, group in group_train:
        list_num_item.append(len(group))

    df_user = pd.DataFrame({"num_item": list_num_item})
    df_user.to_csv(os.path.join("data", dataset_name, "user.csv"), index=False)


def recommend(list_dataset_name, list_seed, K):
    print("Recommending start...")
    for dataset_name in list_dataset_name:
        create_user(dataset_name)

        for seed in list_seed:
            print(f"{dataset_name} - {seed} - {K} start...")
            path_seed = os.path.join("data", dataset_name, f"rec{K}", str(seed))
            if not os.path.exists(path_seed):
                os.makedirs(path_seed)
            mat_candidate = create_candidates_stratification(dataset_name, seed)

            random_(mat_candidate, dataset_name, seed, K=K)
            novelty(mat_candidate, dataset_name, seed, K=K)
            unpopularity(mat_candidate, dataset_name, seed, K=K)
            high_quality(mat_candidate, dataset_name, seed, K=K)

            elasticity_item(mat_candidate, dataset_name, seed, K=K)
            _diversity(mat_candidate, dataset_name, seed, K=K)
            difference(mat_candidate, dataset_name, seed, K=K)
            accuracy_cf(mat_candidate, dataset_name, seed, K=K)
            print(f"{dataset_name} - {seed} - {K} finished...")
    print(f"Recomending finished...")
