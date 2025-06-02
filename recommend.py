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
import pickle
from multiprocessing import Pool
from collections import Counter
import math
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import utils
from recommend_combination import rec_factor_merge

num_pool = 4


def random_sub(list_candidate, K):
    return np.array(random.sample(list_candidate, K)).reshape(1, K)


def random_(mat_candidate, dataset_name, seed, K=20):
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_rand.npy"), mat_rec)


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
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))

    list_date = df_item["date"].values
    pool = Pool(num_pool)
    # for i in range(len(mat_candidate)):
    #     print("=====list_date[mat_candidate[i]]====")
    #     print(len(mat_candidate[i]))
    #     print(mat_candidate[i])
    #     print(list_date[mat_candidate[i]])
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_nov.npy"), mat_rec)


def unpopularity(mat_candidate, dataset_name, seed, K=20):
    """
    - Generating recommendations based on item unpopularity.
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    """
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_pop.npy"), mat_rec)


def high_quality(mat_candidate, dataset_name, seed, K=20):
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_qua.npy"), mat_rec)


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
    df_user = pd.read_csv(os.path.join("data", dataset_name, "user.csv"))
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))

    mat_dis = np.dot(emb_user, emb_item.T)
    max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)
    mat_dis=None
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_ela.npy"), mat_rec)


def accuracy_cf(mat_candidate, dataset_name, seed, K=20):
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
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_acc.npy"), mat_rec)


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
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    if len(selected_items) < max_length:
        selected_items += random.sample(
            list(set(range(item_size)) - set(selected_items)),
            max_length - len(selected_items),
        )
    return selected_items


def diversity(mat_candidate, dataset_name, seed, K=20):
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_item /= np.linalg.norm(emb_item, axis=1, keepdims=True)

    list_res = []
    for uind in tqdm(range(len(mat_candidate))):
        mat_similarity = np.dot(emb_item[mat_candidate[uind]], emb_item[mat_candidate[uind]].T)
        mat_similarity=(1+mat_similarity)/2

        L_mat=np.diag(np.ones_like(mat_candidate[uind]))*mat_similarity*np.diag(np.ones_like(mat_candidate[uind]))
        list_res.append(dpp(L_mat,K))
    '''
    for uind in range(len(mat_candidate)):
        
        arr1=np.array(mat_candidate[uind])
        arr2=arr1[np.array(list_res[uind].get())].reshape(1, K) 
        list1.append(arr1)
    [np.array(mat_candidate[uind])[np.array(list_res[uind].get())].reshape(1, K) ]
    '''
    mat_rec = np.concatenate(
        [np.array(mat_candidate[uind])[np.array(list_res[uind])].reshape(1, K) for uind in range(len(mat_candidate))]
    )

    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_div.npy"), mat_rec)


def difference(mat_candidate, dataset_name, seed, K=20):
    # load data
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))
    mat_dis = np.dot(emb_item, emb_item.T)
    max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)
    mat_dis=None
    pool = Pool(num_pool)
    list_res = []
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    groups_train = df_train.groupby("userInd")
    for (_, group_train),uind in zip(groups_train,range(len(mat_candidate))):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        list_rec=mat_candidate[uind]
        emb_rec = emb_item[list_rec]#[1000,dim]
        list_train=group_train["itemInd"].values.tolist()#[num]
        emb_train = emb_item[list_train]#[num,dim]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten()
        diffscore = 1 - (np.max(np.dot(emb_rec, emb_train.T), axis=-1) - min_dis) / (max_dis - min_dis)#[1000,]
        list_res.append(pool.apply_async(sub_argpartition, (diffscore, np.array(mat_candidate[uind]), K)))
    pool.close()
    pool.join()

    mat_rec = np.concatenate([res.get() for res in list_res])
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_dif.npy"), mat_rec)

def sample_list(lst, K_c):
    if len(lst) < K_c:
        # 当列表长度小于K_c时，进行随机重采样补齐长度
        sampled_lst = random.sample(lst, k=K_c-len(lst))
        lst.extend(sampled_lst)
    elif len(lst) == K_c:
        # 当列表长度等于K_c时，直接返回列表
        pass
    else:
        # 当列表长度大于K_c时，不重复采样至K_c长度
        lst = random.sample(lst, K_c)
    return lst

def create_candidates_stratification_sub(df_item, K_c):
    '''
    如果 df_item 的长度小于 K_c（候选项数量），则将 K_c 更新为 df_item 的长度，以确保不会超出可用的候选项数量。
    df_item:
    itemInd  label
    '''
    K_c_org=K_c
    if len(df_item) < K_c:
        K_c = len(df_item)
    '''
    df_item.groupby("label", group_keys=False)：按照 "label" 列进行分组，group_keys=False 表示不在结果中包含分组键。
    .apply(lambda x: x.sample(int(np.rint(K_c * len(x) / len(df_item)))))：
    对每个分组应用函数。该函数使用 x.sample 随机选择每个分组的一部分项，选择数量为  K_c * len(x) / len(df_item)（根据分组的大小和 K_c 的比例）。
    K_c * len(x) / len(df_item) 意为每个分组应该贡献K_c中的多少比例的item，即 K_c* [len(x) / len(df_item)]
    .sample(frac=1)：对整个 DataFrame 进行随机采样，frac=1 表示保留所有行，并打乱它们的顺序。
    .reset_index(drop=True)：重置索引，并丢弃之前的索引。
    return df_item["itemInd"].values.tolist()：返回经过处理的 DataFrame 中 "itemInd" 列的值作为候选项列表。
    '''
    # print("df_item before op: ",df_item)
    df_item = (
        df_item.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(int(np.rint(K_c * len(x) / len(df_item)))))
        .sample(frac=1)
        .reset_index(drop=True)
    )
    # print("df_item after op: ",df_item)
    cand=df_item["itemInd"].values.tolist()
    cand=sample_list(cand,K_c_org)

    return cand


def create_candidates_stratification(dataset_name, seed, K_c=1000, num_fold=10, epsilon=0.1):
    path_candidate = os.path.join("data", dataset_name, "rec", str(seed), "candidate.npy")
    path_list_res = os.path.join("data", dataset_name, "rec", str(seed), "list_res.pickle")
    print("load emb_item/user...")
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))
    if os.path.exists(path_candidate) :
        mat_candidate = np.load(path_candidate, allow_pickle=True).item()
        if emb_user.shape[0] == len(mat_candidate):
            return np.load(path_candidate, allow_pickle=True).item()
    print("read df_train...")
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))

    group_train = df_train.groupby("userInd")
    num_item = emb_item.shape[0]
    print("np.dot(emb_user, emb_item.T) begin...")
    mat_dis = np.dot(emb_user, emb_item.T).astype(np.float16)
    print("np.dot(emb_user, emb_item.T) fin...")
    max_dis, min_dis = np.max(mat_dis) + epsilon, np.min(mat_dis)
    print("max_dis, min_dis =maxmin  fin...")
    inter = (max_dis - min_dis) / num_fold# min_dis| | | | | | max_dis
    print("np.floor((mat_dis - min_dis) begin...")
    mat_label = np.floor((mat_dis - min_dis) / inter).astype(np.int8)# min_dis| | |mat_dis| ... | | max_dis
    print("np.floor((mat_dis - min_dis) fin...")
    mat_dis=None

    # if not os.path.exists(path_list_res):
    #     # multi-process
    #     random.seed(seed)
    #     pool = Pool(num_pool)
    #     list_res = []
    #     for list_label, (uind, group) in zip(mat_label, group_train):
    #         # print("uind: ",uind)0 1 2 ... len(user)-1
    #         list_no_train = list(set([i for i in range(num_item)]) - set(group["itemInd"].values.tolist()))# 从item全集中剔除训练集中用户的历史物品
    #         list_label = list_label[list_no_train].tolist()# 从item全集中剔除训练集中用户的历史物品
    #         df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
    #         list_res.append(
    #             pool.apply_async(
    #                 create_candidates_stratification_sub,
    #                 (
    #                     df_item,
    #                     K_c,
    #                 ),
    #             )#每个用户的K_c个候选物品
    #         )
    #     list_res=[res.get() for res in list_res]
    #     pool.close()
    #     pool.join()
    #     # 使用pickle序列化和反序列化
    #     # 存储到文件
    #     with open(path_list_res, 'wb') as file:
    #         pickle.dump(list_res, file)
    # list_test_iind = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))["itemInd"].values.tolist()
    list_test_uid2iinds = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv")) \
        .groupby('userInd')['itemInd'].apply(list).to_dict()
    import concurrent.futures
    print("list_res file exist? ={}".format(os.path.exists(path_list_res)))
    if not os.path.exists(path_list_res) :
        # 多进程
        random.seed(seed)
        with concurrent.futures.ProcessPoolExecutor(num_pool) as executor:
            list_res = []
            for list_label, (uind, group) in zip(mat_label, group_train):
                list_no_train = list(set(range(num_item)) - set(group["itemInd"].values.tolist()))
                list_label = list_label[list_no_train].tolist()
                df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
                # future = executor.submit(create_candidates_stratification_sub, df_item, K_c)
                future = executor.submit(create_candidates_stratification_sub, df_item, K_c-len(list_test_uid2iinds[uind]))
                list_res.append(future)
            # 等待所有任务完成
            concurrent.futures.wait(list_res)
            # 获取任务的结果
            list_res = [future.result() for future in list_res]
            # 使用pickle序列化和反序列化
            # 存储到文件
            with open(path_list_res, 'wb') as file:
                pickle.dump(list_res, file)
    else:
        # 从文件加载
        print("load list_res from pickle file...")
        with open(path_list_res, 'rb') as file:
            list_res = pickle.load(file)

    # print(list_test_uid2iinds)
    # print(list_test_iind)
    mat_candidate = dict()
    for uind, res in enumerate(list_res):
        mat_candidate[uind] = res
        # print(mat_candidate[uind])
        # mat_candidate[uind].append(list_test_iind[uind])#在candi候选集尾部添加测试物品
        # 2023 10 24 lisong update
        mat_candidate[uind].extend(list_test_uid2iinds[uind])#在candi候选集尾部添加测试物品
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


def recommend(list_dataset_name, list_seed,merge_factor_rec_strategy_names,merge_factor_wei_list):
    for dataset_name in list_dataset_name:
        create_user(dataset_name)
        for seed in tqdm(list_seed):
            path_seed = os.path.join("data", dataset_name, "rec", str(seed))
            if not os.path.exists(path_seed):
                os.makedirs(path_seed)
            print("begin gen mat candi")
            mat_candidate = create_candidates_stratification(dataset_name, seed,K_c=1000)
            print("create_candidates num: ",len(mat_candidate[0]))
            print("user num: ",len(mat_candidate))
            print("div rec")
            diversity(mat_candidate, dataset_name, seed)
            print("random nov unpop highQua rec")
            random_(mat_candidate, dataset_name, seed)
            novelty(mat_candidate, dataset_name, seed)
            unpopularity(mat_candidate, dataset_name, seed)
            high_quality(mat_candidate, dataset_name, seed)
            print("ela rec")
            elasticity_item(mat_candidate, dataset_name, seed)
            print("difference rec")
            difference(mat_candidate, dataset_name, seed)
            print("accuracy_cf rec")
            accuracy_cf(mat_candidate, dataset_name, seed)
            print("rand_nov rec")
            if  not(merge_factor_wei_list is None):
                for wei_list in merge_factor_wei_list:
                    rec_factor_merge(mat_candidate, dataset_name, seed,wei_list,merge_factor_rec_strategy_names)
