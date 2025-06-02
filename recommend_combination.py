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

num_pool = 4


def random_sub(list_candidate, K):
    return np.array(random.sample(list_candidate, K)).reshape(1, K)

def random_(mat_candidate, dataset_name, seed_parm,regenerate=False):
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed_parm), "rec_random_score_list.npy")) \
            and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed_parm), "rec_random_score_list.npy"))
        return list_res
    np.random.seed(seed_parm)
    list_res=[]
    for uind in range(len(mat_candidate)):
        cand=mat_candidate[uind]
        random_score_list = np.random.random(len(cand))
        list_res.append(random_score_list)

    cand_num=len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res=list_res.reshape(len(mat_candidate),cand_num)#[userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed_parm), "rec_random_score_list.npy"), list_res)
    return list_res

def sub_argpartition(list_score, list_candidate, K):
    # descend
    return list_candidate[np.argpartition(list_score, -K)[-K:]].reshape(1, K)


def novelty(mat_candidate, dataset_name, seed, K=20,regenerate=False):
    """
    - Generating recommendations based on item novelty.
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    """
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_nov_list_res.npy")) and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_nov_list_res.npy"))
        return list_res
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))
    list_date = df_item["date"].values
    list_res = []
    for i in range(len(mat_candidate)):
        novelty_scores=list_date[mat_candidate[i]]
        min_value = np.min(novelty_scores)
        max_value = np.max(novelty_scores)
        # 进行最小-最大缩放
        normalized_novelty_scores = (novelty_scores - min_value) / (max_value - min_value)
        list_res.append(normalized_novelty_scores)
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_nov_list_res.npy"), list_res)
    return list_res

def unpopularity(mat_candidate, dataset_name, seed, K=20,regenerate=False):
    """
    - Generating recommendations based on item unpopularity.
    Paras:
        dataset_name: dataset name
        K: recommend top K items
    """
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_pop_list_res.npy")) and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_pop_list_res.npy"))
        return list_res
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))
    list_count = df_item["count"].values
    list_res=[]
    for i in range(len(mat_candidate)):
        unpop_scores=-list_count[mat_candidate[i]]
        min_value = np.min(unpop_scores)
        max_value = np.max(unpop_scores)
        # 进行最小-最大缩放
        normalized_unpop_scores = (unpop_scores - min_value) / (max_value - min_value)
        list_res.append(normalized_unpop_scores)
    #[userNum,candNum]
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_pop_list_res.npy"), list_res)
    return list_res


def high_quality(mat_candidate, dataset_name, seed, K=20,regenerate=False):
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_qua_list_res.npy")) and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_qua_list_res.npy"))
        return list_res

    df_rating = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"), usecols=["itemInd", "rating"])
    list_mean_rating = df_rating.groupby("itemInd")["rating"].mean().values
    list_res = []
    for i in range(len(mat_candidate)):
        qua_scores = list_mean_rating[mat_candidate[i]]
        min_value = np.min(qua_scores)
        max_value = np.max(qua_scores)
        # 进行最小-最大缩放
        normalized_qua_scores = (qua_scores - min_value) / (max_value - min_value)
        list_res.append(normalized_qua_scores)
    # [userNum,candNum]
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_qua_list_res.npy"), list_res)
    return list_res


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

    mat_similarity = []
    for uind in range(len(mat_candidate)):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten().tolist()
        mat_similarity.append(similarity_t)#sim(,Cand)

    list_item_count = df_user["num_item"].values.tolist()#ela(u)
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

    list_res = []
    for uind in range(emb_user.shape[0]):
        list_factor=np.array(mat_factor[uind])
        ela_scores=-np.abs(list_factor - alpha * mean_factor)#argmin=>argmax
        min_value = np.min(ela_scores)
        max_value = np.max(ela_scores)
        normalized_ela_scores=(ela_scores - min_value) / (max_value - min_value)
        list_res.append(normalized_ela_scores)
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_ela_list_res.npy"), list_res)
    return list_res


def accuracy_cf(mat_candidate, dataset_name, seed, K=20,regenerate=False):
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_acc_list_res.npy")) and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_acc_list_res.npy"))
        return list_res
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))

    list_res = []
    for uind in range(len(mat_candidate)):
        emb_user_t = emb_user[uind].reshape(1, emb_user.shape[1])
        emb_item_t = emb_item[mat_candidate[uind]]
        similarity_t = np.dot(emb_user_t, emb_item_t.T).flatten().tolist()
        # list_res.append(pool.apply_async(sub_argpartition, (similarity_t, np.array(mat_candidate[uind]), K)))
        min_value = np.min(similarity_t)
        max_value = np.max(similarity_t)
        # 进行最小-最大缩放
        normalized_similarity_t = (similarity_t - min_value) / (max_value - min_value)
        list_res.append(normalized_similarity_t)
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_acc_list_res.npy"), list_res)
    return list_res

def dpp(mat_similarity, max_length, epsilon=1e-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    相似性矩阵：给定一组项目，我们首先计算它们之间的相似性或相关性，并将这些相似性值组合成一个矩阵，称为相似性矩阵。相似性矩阵的大小为 N x N，其中 N 是项目的数量。相似性矩阵中的每个元素表示对应项目之间的相似性度量，可以是基于内容、协同过滤或其他方法得到的。
边缘概率：对于 DPP，我们定义了一个边缘概率，表示选择某个项目的概率。边缘概率与项目的自相似性或核函数值有关，在代码中用 di2s 表示。
条件概率：DPP 还定义了条件概率，表示在已选择的项目集合下，选择下一个项目的概率。条件概率与已选择的项目之间的相关性有关，在代码中用 cis 表示。
贪心选择：DPP 的关键思想是通过贪心选择的方式构建推荐列表。从初始的项目集合开始，依次选择下一个项目，每次选择时考虑到已选择的项目和候选项目之间的相关性。具体步骤如下：
选择初始项目：从边缘概率中选择具有最大值的项目作为初始项目。
选择下一个项目：对于已选择的项目集合，计算每个候选项目的得分。得分由候选项目与已选择项目之间的相关性以及候选项目的自相似性决定。根据得分选择一个候选项目作为下一个项目，并将其添加到已选择的项目集合中。
重复选择：重复上述步骤，直到生成完整的推荐列表或达到指定的推荐列表长度。
    """
    kernel_matrix=mat_similarity
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
    print("dpp len={}".format(len(selected_items)),end=' ')
    if len(selected_items) < max_length:
        selected_items += random.sample(
            list(set(range(item_size)) - set(selected_items)),
            max_length - len(selected_items),
        )

    return selected_items


def diversity(mat_candidate,dataset_name, seed,mat_candidate_rel_score,div_weight=0.5, K=20):
    #mat_candidate_rel_score [userNum,C=1000]
    # if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_div_list_res.npy")):
    #     list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_div_list_res.npy"))
    #     return list_res
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_item /= np.linalg.norm(emb_item, axis=1, keepdims=True)

    list_res = []
    factor=1e6
    alpha = (1 - div_weight) / (2*div_weight)
    print("\ndiv alpha=", alpha)
    for uind in tqdm(range(len(mat_candidate))):
        cand_set=mat_candidate[uind]#[1000]
        candidate_rel_score=mat_candidate_rel_score[uind]#[1000]
        # candidate_rel_score=(candidate_rel_score - np.min(candidate_rel_score)) / (np.max(candidate_rel_score) - np.min(candidate_rel_score))
        mat_similarity = np.dot(emb_item[cand_set], emb_item[cand_set].T)
        mat_similarity=(1+mat_similarity)/2
        L_mat=np.diag(np.exp(alpha*candidate_rel_score/factor))*mat_similarity*np.diag(np.exp(alpha*candidate_rel_score/factor))
        divers_dpp_items_idx=dpp(L_mat, K)
        list_res.append(np.array(cand_set)[divers_dpp_items_idx])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), K)  # [userNum,K]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_div_list_res.npy"), list_res)
    return list_res

def difference(mat_candidate, dataset_name, seed, K=20,regenerate=False):
    # load data
    if os.path.exists(os.path.join("data", dataset_name, "rec", str(seed), "rec_dif_list_res.npy")) and not(regenerate):
        list_res=np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_dif_list_res.npy"))
        return list_res
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    mat_dis = np.dot(emb_item, emb_item.T)
    max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)
    list_res = []
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    groups_train = df_train.groupby("userInd")
    for (_, group_train),uind in zip(groups_train,range(len(mat_candidate))):
        list_rec=mat_candidate[uind]
        emb_rec = emb_item[list_rec]#[1000,dim]
        list_train=group_train["itemInd"].values.tolist()#[num]
        emb_train = emb_item[list_train]#[num,dim]
        dif = 1 - (np.max(np.dot(emb_rec, emb_train.T), axis=-1) - min_dis) / (max_dis - min_dis)#[1000,]
        list_res.append(np.reshape(dif, (len(list_rec),)   ))
        # list_res.append(pool.apply_async(sub_argpartition, (-similarity_t, np.array(mat_candidate[uind]), K)))
    cand_num = len(mat_candidate[0])
    list_res = np.concatenate([np.array(sublist) for sublist in list_res], axis=0)
    list_res = list_res.reshape(len(mat_candidate), cand_num)  # [userNum.candNum]
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_dif_list_res.npy"), list_res)
    return list_res

def create_candidates_stratification_sub(df_item, K_c):
    '''
    如果 df_item 的长度小于 K_c（候选项数量），则将 K_c 更新为 df_item 的长度，以确保不会超出可用的候选项数量。
    df_item:
    itemInd  label
    '''
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

    return df_item["itemInd"].values.tolist()


def create_candidates_stratification(dataset_name, seed, K_c=1000, num_fold=10, epsilon=0.1):
    path_candidate = os.path.join("data", dataset_name, "rec", str(seed), "candidate.npy")
    path_list_res = os.path.join("data", dataset_name, "rec", str(seed), "list_res.pickle")
    print("load emb_item/user...")
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))
    if os.path.exists(path_candidate):
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

    import concurrent.futures
    print("list_res file exist? ={}".format(os.path.exists(path_list_res)))
    if not os.path.exists(path_list_res) or True:
        # 多进程
        random.seed(seed)
        with concurrent.futures.ProcessPoolExecutor(num_pool) as executor:
            list_res = []
            for list_label, (uind, group) in zip(mat_label, group_train):
                list_no_train = list(set(range(num_item)) - set(group["itemInd"].values.tolist()))
                list_label = list_label[list_no_train].tolist()
                df_item = pd.DataFrame({"itemInd": list_no_train, "label": list_label}, dtype=np.int32)
                future = executor.submit(create_candidates_stratification_sub, df_item, K_c)
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
    # list_test_iind = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))["itemInd"].values.tolist()
    list_test_uid2iinds = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))\
                            .groupby('userInd')['itemInd'].apply(list).to_dict()
    print(list_test_uid2iinds)
    # print(list_test_iind)
    mat_candidate = dict()
    for uind, res in enumerate(list_res):
        mat_candidate[uind] = res
        # print(mat_candidate[uind])
        # mat_candidate[uind].append(list_test_iind[uind])#在candi候选集尾部添加一个测试物品
        # 2023 10 24 lisong update
        mat_candidate[uind].extend(list_test_uid2iinds[uind])
    np.save(path_candidate, mat_candidate)
    return mat_candidate


def create_user(dataset_name,regenerate=False):
    if os.path.exists(os.path.join("data", dataset_name, "user.csv")) and not(regenerate):
        return
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    group_train = df_train.groupby("userInd")
    list_num_item, list_num_category = [], []
    for uind, group in group_train:
        list_num_item.append(len(group))

    df_user = pd.DataFrame({"num_item": list_num_item})
    df_user.to_csv(os.path.join("data", dataset_name, "user.csv"), index=False)

def recommend_combination(mat_candidate,scores_mat_of_diff_rec,
                          factor_names,factor_wei_list,dataset_name,seed,K=20):
    '''
    scores_list_of_diff_rec是不同的推荐策略产生的所有用户的推荐候选集合的推荐分数
    factor_wei是长度为len(scores_list_of_diff_rec)的因素权重列表
    len(cand_items)==len(scores_list_of_diff_rec[0])
    '''
    rec_comb_name = "_".join(
        ["{}{}".format(factor_wei, factor_name) for factor_name, factor_wei in zip(factor_names, factor_wei_list)])
    # if os.path.exists(
    # os.path.join("data", dataset_name, "rec", str(seed), "rec_{}.npy".format(rec_comb_name))
    # ):return
    userNum=scores_mat_of_diff_rec[0].shape[0]
    print("recommend_combination userNum={}".format(userNum))
    cand_items_num=len(mat_candidate[0])
    scores_list_of_diff_rec_combination=np.zeros(shape=(userNum,cand_items_num))#[userNum,1000 e.g.canditemNum]
    for scores_mat,weit in zip(scores_mat_of_diff_rec,factor_wei_list):
        #scores_list:[userNum,1000 e.g.canditemNum]
        wei_scores_list=scores_mat*weit
        scores_list_of_diff_rec_combination=scores_list_of_diff_rec_combination+wei_scores_list
    if "div" in factor_names:
        div_weight=factor_wei_list[factor_names.index('div')]
        mat_rec=diversity(mat_candidate, dataset_name,seed,scores_list_of_diff_rec_combination, div_weight=div_weight, K=20)
        # [userNum,K]
    else:
        pool = Pool(num_pool)
        list_res = [
            pool.apply_async(
                sub_argpartition,
                (
                    scores_list_of_diff_rec_combination[uind],
                    np.array(mat_candidate[uind]),
                    K,
                ),
            )
            for uind in range(len(mat_candidate))
        ]
        pool.close()
        pool.join()
        mat_rec = np.concatenate([res.get() for res in list_res])
    print("np.save=".format(os.path.join("data", dataset_name, "rec", str(seed), "rec_{}.npy".format(rec_comb_name))))
    np.save(os.path.join("data", dataset_name, "rec", str(seed), "rec_{}.npy".format(rec_comb_name)), mat_rec)

def rec_factor_merge(mat_candidate, dataset_name, seed,factor_wei_list,factor_rec_strategy_names):
    print("random nov unpop highQua rec")
    scores_mat_of_factor_rec_strategy=[]

    for factor_rec_strategy in factor_rec_strategy_names:
        if factor_rec_strategy=="rand":
            scores_mat_of_factor_rec_strategy.append(random_(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="nov":
            scores_mat_of_factor_rec_strategy.append(novelty(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="dif":
            scores_mat_of_factor_rec_strategy.append(difference(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="acc":
            scores_mat_of_factor_rec_strategy.append(accuracy_cf(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="ela":
            scores_mat_of_factor_rec_strategy.append(elasticity_item(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="pop":
            scores_mat_of_factor_rec_strategy.append(unpopularity(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy=="qua":
            scores_mat_of_factor_rec_strategy.append(high_quality(mat_candidate,dataset_name,seed))
        elif factor_rec_strategy == "div":
            continue
    recommend_combination(mat_candidate,scores_mat_of_factor_rec_strategy,
                  factor_rec_strategy_names,factor_wei_list,dataset_name,seed,K=20)

def recommend(list_dataset_name, list_seed):
    for dataset_name in list_dataset_name:
        create_user(dataset_name)
        for seed in tqdm(list_seed):
            path_seed = os.path.join("data", dataset_name, "rec", str(seed))
            if not os.path.exists(path_seed):
                os.makedirs(path_seed)
            print("begin gen mat candi")
            mat_candidate = create_candidates_stratification(dataset_name, seed)

            difference(mat_candidate, dataset_name, seed)
            accuracy_cf(mat_candidate, dataset_name, seed)
            # unpopularity(mat_candidate, dataset_name, seed)
            # high_quality(mat_candidate, dataset_name, seed)
            # print("ela rec")
            # elasticity_item(mat_candidate, dataset_name, seed)
            # print("difference rec")
            # difference(mat_candidate, dataset_name, seed)
            # print("accuracy_cf rec")
            # accuracy_cf(mat_candidate, dataset_name, seed)

