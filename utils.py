"""
Two Serendipity Metrics and Six Factor-based Mtrics
"""
import os
import pickle
from multiprocessing import Pool
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import exp_analysis

num_pool = 32


def sub_argpartition(list_score, list_candidate, K):
    # descend
    return list_candidate[np.argpartition(list_score, -K)[-K:]].reshape(1, K)


def ser1_sub(dataset_name, max_dis, min_dis, list_rec, list_train, list_test):
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_rec = emb_item[list_rec]
    try:
        emb_train = emb_item[list_train]
    except:
        print("===ser1_sub error")
        print(emb_item.shape)
        print(dataset_name)
        print(list_train)
    emb_test = emb_item[list_test]
    acc = (np.max(np.dot(emb_rec, emb_test.T), axis=-1) - min_dis) / (max_dis - min_dis)#rec [|R|,dim]  emb_test[|Test|,dim]
    dif = 1 - (np.max(np.dot(emb_rec, emb_train.T), axis=-1) - min_dis) / (max_dis - min_dis)
    ser = 2 * acc * dif / (acc + dif)

    return np.mean(acc), np.mean(dif), np.mean(ser),acc,dif


def ser1(dataset_name, mat_rec, max_dis, min_dis):
    """
    the balance between accuracy and difference
    """
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    df_test = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))
    groups_train = df_train.groupby("userInd")
    groups_test = df_test.groupby("userInd")

    list_acc, list_dif, list_ser,org_list_acc_arr,org_list_dif_arr = [], [], [], [], []
    for (_, group_train), (_, group_test), list_rec in zip(groups_train, groups_test, mat_rec):
        acc, dif, ser,org_acc_arr,org_dif_arr = ser1_sub(
            dataset_name,
            max_dis,
            min_dis,
            list_rec.astype(int),
            group_train["itemInd"].values.tolist(),
            group_test["itemInd"].values.tolist(),
        )
        list_acc.append(acc)
        list_dif.append(dif)
        list_ser.append(ser)
        org_list_acc_arr.append(org_acc_arr)
        org_list_dif_arr.append(org_dif_arr)


    return np.mean(list_acc), np.mean(list_dif), np.mean(list_ser),np.array(org_list_acc_arr),np.array(org_list_dif_arr)


def create_pm(mat_candidate, dataset_name, seed, K=200):
    '''
    根据候选物品的评分和流行度信息生成推荐结果，并将结果保存到文件中
    '''
    path_pm = os.path.join("data", dataset_name, "rec", str(seed), "pm.npy")
    if os.path.exists(path_pm) :
        return np.load(path_pm)

    df_rating = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"), usecols=["itemInd", "rating"])
    list_mean_rating = df_rating.groupby("itemInd")["rating"].mean().values
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))
    list_count = df_item["count"].values#popularity

    pool = Pool(num_pool)
    list_res_qua = [
        pool.apply_async(
            sub_argpartition,
            (
                list_mean_rating[mat_candidate[i]],
                np.array(mat_candidate[i]),
                int(K / 2),
            ),
        )
        for i in range(len(mat_candidate))
    ]
    list_res_pop = [
        pool.apply_async(
            sub_argpartition,
            (
                list_count[mat_candidate[i]],
                np.array(mat_candidate[i]),
                int(K / 2),
            ),
        )
        for i in range(len(mat_candidate))
    ]
    pool.close()
    pool.join()

    mat_qua = np.concatenate([res.get() for res in list_res_qua])
    mat_pop = np.concatenate([res.get() for res in list_res_pop])
    mat_pm = np.concatenate([mat_qua, mat_pop], axis=1)
    np.save(path_pm, mat_pm)

    return mat_pm


def ser2_sub(emb_train, emb_rec, min_dis):
    if emb_rec.shape[0] == 0:
        return min_dis
    return np.mean(np.max(np.dot(emb_rec, emb_train.T), axis=-1))


def ser2(dataset_name, mat_rec, max_dis, min_dis, seed):
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    df_train = pd.read_csv(
        os.path.join("data", dataset_name, "rating_train.csv"),
        usecols=["userInd", "itemInd"],
    )
    groups_train = df_train.groupby("userInd")
    mat_pm = np.load(os.path.join("data", dataset_name, "rec", str(seed), "pm.npy"))

    list_res = []
    for (_, group_train), list_rec, list_acc in zip(groups_train, mat_rec, mat_pm):
        list_res.append(
            ser2_sub(
                emb_item[group_train["itemInd"].values.tolist()],
                emb_item[list(set(list_rec.tolist()) - set(list_acc.tolist()))],
                min_dis,
            ),
        )

    return (np.mean([res for res in list_res]) - min_dis) / (max_dis - min_dis)

def HR_ser_sub(list_rec, list_test_iind_ser,list_test_score_ser,only_pos_ser=False,glb_ser_score=0.0):
    #判断两个list有无交集，有返回1,无返回0
    if only_pos_ser:
        return np.any(np.in1d(list_rec, list_test_iind_ser))#只使用正的惊喜度样本，但相当稀疏
    else:
        #对负惊喜样本仍赋予一个[0,1]之间的超参惊喜度得分，因为非惊喜样本实际上仍然是点击过的正样本。
        res=0.0
        for iind,serLabel in zip(list_test_iind_ser,list_test_score_ser):
            if iind in list_rec:
                if serLabel>0.9999:#pos label
                    return 1
                else:
                    res=glb_ser_score
        return res

def HR_ser(dataset_name, mat_rec):
    """
    the balance between accuracy and difference
    """
    df_test = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))
    groups_test = df_test.groupby("userInd")#positive serend label

    list_HR_ser=[]
    for userInd, group_test_pos_ser in groups_test:
        list_rec=mat_rec[userInd]
        # print("userind={}, rec list is {}".format(userInd,list_rec))
        HR_ser_val = HR_ser_sub(
            list_rec.astype(int),
            group_test_pos_ser["itemInd"].values.tolist(),
            group_test_pos_ser["serLabel"].values.tolist(),
        )
        list_HR_ser.append(HR_ser_val)
    return np.mean(list_HR_ser)



def calculate_dcg(relevance):
    dcg = 0
    for i, rel in enumerate(relevance):
        gain = 2**rel - 1
        discount = np.log2(i + 2)
        dcg += gain / discount
    return dcg
def calculate_idcg(relevance):
    sorted_relevance = sorted(relevance, reverse=True)
    return calculate_dcg(sorted_relevance)

def calculate_ndcg(rec_list, test_list,relevance):
    dcg = calculate_dcg(relevance)
    idcg = calculate_idcg(relevance)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def HR_ser_sub(list_rec, list_test_iind_ser,list_test_score_ser,only_pos_ser=False,glb_ser_score=0.0):
    #判断两个list有无交集，有返回1,无返回0
    if only_pos_ser:
        return np.any(np.in1d(list_rec, list_test_iind_ser))#只使用正的惊喜度样本，但相当稀疏
    else:
        #对负惊喜样本仍赋予一个[0,1]之间的超参惊喜度得分，因为非惊喜样本实际上仍然是点击过的正样本。
        res=0.0
        for iind,serLabel in zip(list_test_iind_ser,list_test_score_ser):
            if iind in list_rec:
                if serLabel>0.9999:#pos label
                    return 1
                else:
                    res=glb_ser_score
        return res

def NDCG_ser_sub(list_rec, list_test_ser,list_test_score_ser,only_pos_ser=False,glb_ser_score=0.0):
    if only_pos_ser:
        relevance = [1 if item in list_test_ser else 0 for item in list_rec]
        return calculate_ndcg(list_rec,list_test_ser,relevance)
    else:
        relevance = []
        for iind in list_rec:
            rel=0.0
            for iind_test,ser_score_test in zip(list_test_ser,list_test_score_ser):
                if iind==iind_test:
                    rel=1 if ser_score_test>0.9999 else glb_ser_score
                    break
            relevance.append(rel)
        return calculate_ndcg(list_rec, list_test_ser, relevance)
def NDCG_ser(dataset_name, mat_rec):
    """
    the balance between accuracy and difference
    """
    df_test = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))
    # groups_test = df_test[df_test["serLabel"]>0.5].groupby("userInd")#positive serend label
    groups_test = df_test.groupby("userInd")  # positive serend label
    list_NDCG_ser=[]
    for userInd, group_test_pos_ser in groups_test:
        list_rec=mat_rec[userInd]
        # print("userind={}, rec list is {}".format(userInd,list_rec))
        NDCG_ser_val = NDCG_ser_sub(
            list_rec.astype(int),
            group_test_pos_ser["itemInd"].values.tolist(),
            group_test_pos_ser["serLabel"].values.tolist(),
        )
        list_NDCG_ser.append(NDCG_ser_val)
    return np.mean(list_NDCG_ser)

def novelty(dataset_name, mat_rec):
    list_date = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))["date"].values
    score_novelty = (list_date - list_date.min()) / (list_date.max() - list_date.min())
    return score_novelty[mat_rec.flatten()].mean()


def unpopularity(dataset_name, mat_rec):
    list_date = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))["count"].values
    score_unpopularity = 1 - (list_date - list_date.min()) / (list_date.max() - list_date.min())
    return score_unpopularity[mat_rec.flatten()].mean()


def quality(dataset_name, mat_rec):
    list_mean_rating = (
        pd.read_csv(os.path.join("data", dataset_name, "rating.csv")).groupby("itemInd")["rating"].mean().values
    )
    score_quality = (list_mean_rating - list_mean_rating.min()) / (list_mean_rating.max() - list_mean_rating.min())
    return score_quality[mat_rec.flatten()].mean()


def diversity_sub(dataset_name, list_rec, max_dis, min_dis):
    emb_item_ = np.load(os.path.join("data", dataset_name, "emb_item.npy"))[list_rec]
    return 1 - (np.mean(np.dot(emb_item_, emb_item_.T)) - min_dis) / (max_dis - min_dis)


def diversity(dataset_name, mat_rec, max_dis, min_dis):
    pool = Pool(num_pool)
    list_res = [
        pool.apply_async(
            diversity_sub,
            (
                dataset_name,
                mat_rec[uind],
                max_dis,
                min_dis,
            ),
        )
        for uind in range(mat_rec.shape[0])
    ]
    pool.close()
    pool.join()

    return np.mean([res.get() for res in list_res])


def sum_res_all_seed(list_dataset_name, list_seed,compDisentDeg=False,merge_factor_rec_strategy=None):
    for dataset_name in list_dataset_name:
        list_res = []
        list_disent_jcd,list_disent_diff=[],[]
        for seed in list_seed:
            if merge_factor_rec_strategy is None:
                path_res = os.path.join("data", dataset_name, "rec", str(seed), "single_factor_res.npy")
            else:
                path_res = os.path.join("data", dataset_name, "rec", str(seed),
                                        "{}_res.npy".format("_".join(merge_factor_rec_strategy)))
            mat_res = np.load(path_res)
            list_res.append(np.expand_dims(mat_res, axis=-1))
            if compDisentDeg:
                if merge_factor_rec_strategy is None:
                    # mat_disent_jcd = np.load(
                    #     os.path.join("data", dataset_name, "rec", str(seed), "single_factor_disent_jcd_mat.npy"))
                    mat_disent_diff = np.load(
                        os.path.join("data", dataset_name, "rec", str(seed), "single_factor_disent_dif_mat.npy"))
                    # list_disent_jcd.append(np.expand_dims(mat_disent_jcd, axis=-1))
                    list_disent_diff.append(np.expand_dims(mat_disent_diff, axis=-1))
                else:
                    # mat_disent_jcd = np.load(
                    #     os.path.join("data", dataset_name, "rec", str(seed), "{}_disent_jcd_mat.npy".format("_".join(merge_factor_rec_strategy))))
                    mat_disent_diff = np.load(
                        os.path.join("data", dataset_name, "rec", str(seed), "{}_disent_dif_mat.npy".format("_".join(merge_factor_rec_strategy))))
                    # list_disent_jcd.append(np.expand_dims(mat_disent_jcd, axis=-1))
                    list_disent_diff.append(np.expand_dims(mat_disent_diff, axis=-1))


        mat_res = np.concatenate(list_res, axis=-1)
        mat_mean = np.mean(mat_res, axis=-1)
        mat_std = np.std(mat_res, axis=-1)
        if merge_factor_rec_strategy is None:
            np.save(os.path.join("data", dataset_name, "res", "single_factor_res_mean.npy"), mat_mean)
            np.save(os.path.join("data", dataset_name, "res", "single_factor_res_std.npy"), mat_std)
        else:
            np.save(os.path.join("data", dataset_name, "res", "{}_res_mean.npy".format("_".join(merge_factor_rec_strategy))), mat_mean)
            np.save(os.path.join("data", dataset_name, "res", "{}_res_std.npy".format("_".join(merge_factor_rec_strategy))), mat_std)


        if compDisentDeg:
            # mat_disent_jcd = np.concatenate(list_disent_jcd, axis=-1)
            mat_disent_diff = np.concatenate(list_disent_diff,axis=-1)
            # mat_disent_jcd_mean=np.mean(mat_disent_jcd, axis=-1)
            mat_disent_diff_mean=np.mean(mat_disent_diff, axis=-1)
            if merge_factor_rec_strategy is None:
                # np.save(os.path.join("data", dataset_name, "res", "single_factor_disent_jcd_mat.npy"), mat_disent_jcd_mean)
                np.save(os.path.join("data", dataset_name, "res", "single_factor_disent_dif_mat.npy"), mat_disent_diff_mean)
            else:
                # np.save(os.path.join("data", dataset_name, "res", "{}_disent_jcd_mat.npy".format("_".join(merge_factor_rec_strategy))
                #                      ), mat_disent_jcd_mean)
                np.save(os.path.join("data", dataset_name, "res", "{}_disent_dif_mat.npy".format("_".join(merge_factor_rec_strategy))
                                     ), mat_disent_diff_mean)

def jaccard_dis(a, b):
    intersection = intersect1d(a, b)
    union = np.union1d(a, b)
    similarity = len(intersection) / len(union)
    return 1 - similarity
def sum_linkage(setA,node,distance):
    linkage=0
    for item_a in setA:
        linkage+=distance[item_a,node]
    return linkage

def find_max_average_linkage(A, I, K,item_item_pair_dis):
    N = len(I)
    dp = np.zeros((N+1, K+1))
    best_R = np.zeros((K,), dtype=int)
    # path[n][k]=np.zeros((N+1, K+1))
    for n in range(1, N+1):#==>I[0] ~ I[N-1]     n=1=>idx=n-1=0
        for k in range(1, min(n, K)+1):
            dp[n][k] = max(dp[n-1][k], dp[n-1][k-1]+sum_linkage(A,I[n-1],item_item_pair_dis))
    return dp[N][K]/K

def find_max_min_cosDiff(A, I, K,item_item_pair_dis):
    N = len(I)
    min_diff= np.full((N+1, K+1, K),1e6)
    dp = np.zeros((N+1, K+1))
    # path[n][k]=np.zeros((N+1, K+1))
    for n in range(1, N+1):#==>I[0] ~ I[N-1]     n=1=>idx=n-1=0
        for k in range(1, min(n, K)+1):
            Inmin1_A_dis=np.array([item_item_pair_dis[item_a,I[n-1]] for item_a in A ])#[K]
            min_diff_last_state=min_diff[n - 1][k - 1]#[K]
            nextState = np.minimum(Inmin1_A_dis,min_diff_last_state)
            if dp[n-1][k]>np.sum(nextState):
                dp[n][k] = dp[n-1][k]
                min_diff[n,k,:]=min_diff[n-1,k,:]
            else:
                dp[n][k] = np.sum(nextState)
                min_diff[n, k, :] = nextState

    return dp[N][K]/K

def disentanglementDegree(dataset_name,recMatA, recMatB, norm_emb_item_cosine_max_dis, norm_emb_item_cosine_min_dis,
                          metric='jaccardDis',eps = 1e-4,idealMaxDistNorm=False,userMaxDistNorm=True):
    '''
    metric: jaccardDis  cosDis_diff cosDis_avgLink
    '''
    '''
    shape=[userNum,K]的推荐结果矩阵
    '''

    userNum = len(recMatA)
    K=recMatA.shape[1]
    if metric == 'jaccardDis':
        avg_jaccard_dis = 0.0
        for recResA, recResB in zip(recMatA, recMatB):
            avg_jaccard_dis += jaccard_dis(recResA, recResB)
        disentanglementDegree = avg_jaccard_dis / userNum
    if metric.startswith("cosDis"):
        #num,k1,dim  num,dim,k2
        emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
        # 计算归一化余弦相似度
        max_dis,min_dis=norm_emb_item_cosine_max_dis, norm_emb_item_cosine_min_dis
        recMatA_emd,recMatB_emd=emb_item[recMatA],emb_item[recMatB]
        recMatB_emd_t=recMatB_emd.transpose((0, 2, 1))
        recMatAB_emd_dot=np.matmul(recMatA_emd,recMatB_emd_t)#[num,k1,k2]
        # 计算矩阵的范数
        norm_matrix1 = np.linalg.norm(recMatA_emd, axis=-1)
        norm_matrix2 = np.linalg.norm(recMatB_emd, axis=-1)
        # 计算余弦距离
        cosine_distance = 1 - recMatAB_emd_dot / (norm_matrix1[:, :, None] * norm_matrix2[:, None, :])#[num,k1,k2]
        cosine_distance=cosine_distance/2#   [0.,2]==>[0,1]
        cosine_distance = np.where(np.isclose(cosine_distance, 0, atol=eps), 0, cosine_distance)#[-eps,eps]=>0.0
        cosine_distance=(cosine_distance-min_dis)/(max_dis-min_dis)
        if metric.endswith("diff"):
            if idealMaxDistNorm:
                # 计算点积
                dot_product = np.dot(emb_item, emb_item.T)
                # 计算归一化余弦相似度
                norm_emb_item = np.linalg.norm(emb_item, axis=1)
                norm_emb_item_cosine_dis = (1 - dot_product / np.outer(norm_emb_item, norm_emb_item)) / 2  # [num,num]
                max_dis, min_dis = np.max(norm_emb_item_cosine_dis), np.min(norm_emb_item_cosine_dis)
                norm_emb_item_cosine_dis = (norm_emb_item_cosine_dis - min_dis) / (max_dis - min_dis)
                idealMaxDisentRecA, idealMaxDisentRecB = [], []
                disentAB = np.mean(np.min(cosine_distance, axis=-1),axis=-1) # #[num,k1,k2]->[num,k1]->[num]
                disentBA = np.mean(np.min(cosine_distance, axis=1),axis=-1) # #[num,k1,k2]->[num,k2]->[num]
                for recA_oneUser in recMatA:
                    idealMaxDisentRecA.append(find_max_min_cosDiff(recA_oneUser, range(emb_item.shape[0]), K,
                                                                       norm_emb_item_cosine_dis))  # [num]

                for recB_oneUser in recMatB:
                    idealMaxDisentRecB.append(find_max_min_cosDiff(recB_oneUser, range(emb_item.shape[0]), K,
                                                                       norm_emb_item_cosine_dis))  # [num]
                disentanglementDegree = (disentAB / np.array(idealMaxDisentRecA) +
                                         disentBA / np.array(idealMaxDisentRecB)) / 2  # [num]
                disentanglementDegree = np.mean(disentanglementDegree)
            elif userMaxDistNorm:
                disentAB = np.mean(np.min(cosine_distance, axis=-1), axis=-1,keepdims=True)  # #[num,k1,k2]->[num,k1]->[num,1]
                disentBA = np.mean(np.min(cosine_distance, axis=1), axis=-1,keepdims=True)  # #[num,k1,k2]->[num,k2]->[num,1]

                # disentAB = np.zeros_like(disentAB) if np.max(disentAB)<eps else disentAB / np.max(disentAB)
                # disentBA = np.zeros_like(disentBA) if np.max(disentBA)<eps else disentBA / np.max(disentBA)
                #
                # disentanglementDegree = (disentAB +
                #                          disentBA ) / 2  # [num]
                # disentanglementDegree = np.mean(disentanglementDegree)
                return    np.concatenate([disentAB,disentBA],axis=-1)
            else:
                dif = (np.min(cosine_distance, axis=-1) + np.min(cosine_distance, axis=1)) / 2  # [num,k1,k2]->[num,k]
                disentanglementDegree = np.mean(dif)
        elif metric.endswith("avgLink"):
            if not idealMaxDistNorm:
                disentanglementDegree = np.mean(cosine_distance)#[num,k1,k2]->[,]
            else:
                idealMaxDisentRecA,idealMaxDisentRecB=[],[]
                disentAB=np.mean(cosine_distance,axis=(1,2))#[num]
                disentBA=disentAB
                for recA_oneUser in recMatA:
                    idealMaxDisentRecA.append(find_max_average_linkage(recA_oneUser,range(emb_item.shape[0]),K,
                                                                norm_emb_item_cosine_dis))#[num]

                for recB_oneUser in recMatB:
                    idealMaxDisentRecB.append(find_max_average_linkage(recB_oneUser,range(emb_item.shape[0]),K,
                                                                norm_emb_item_cosine_dis))#[num]
                disentanglementDegree=(disentAB/np.array(idealMaxDisentRecA)+
                                       disentBA/np.array(idealMaxDisentRecB))/2#[num]
                disentanglementDegree=np.mean(disentanglementDegree)

    return disentanglementDegree
def exclude_outliers(data):
    # 计算四分位数和四分位距
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    # 计算上下边界
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # 排除异常点
    filtered_data = [value for value in data if lower_bound <= value <= upper_bound]
    return filtered_data
def evaluate(list_dataset_name, list_seed, list_method,compDisentDeg=False,merge_factor_rec_strategy=None,mem_enough=False):
    for dataset_name in list_dataset_name:
        print("evaluate dataset={}".format(dataset_name))
        emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
        norm_emb_item = np.linalg.norm(emb_item, axis=1)
        if mem_enough:
            mat_dis = np.dot(emb_item, emb_item.T)
            max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)
            # emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))
            # mat_dis_ui = np.dot(emb_user, emb_item.T)
            # max_dis_ui, min_dis_ui = np.max(mat_dis_ui), np.min(mat_dis_ui )
            if compDisentDeg:
                method_acc_list, method_dif_list = defaultdict(list), defaultdict(list)
                print("norm_emb_item = np.linalg.norm(emb_item, axis=1)")
                norm_emb_item_cosine_dis = (1 - mat_dis / np.outer(norm_emb_item,
                                                                       norm_emb_item)) / 2  # [num,num]
                norm_emb_item_cosine_max_dis, norm_emb_item_cosine_min_dis= np.max(norm_emb_item_cosine_dis), np.min(norm_emb_item_cosine_dis)
            mat_dis=None
        else:
            # 定义块的大小
            block_size = 100
            # 获取emb_item矩阵的形状
            num_items, emb_dim = emb_item.shape
            # 初始化结果列表
            norm_emb_item_cosine_dis = []
            # 按块处理数据
            norm_emb_item_cosine_max_dis,norm_emb_item_cosine_min_dis=-1e4,1e4
            max_dis = float('-inf')  # 最大值初始化为负无穷
            min_dis = float('inf')  # 最小值初始化为正无穷

            for i in range(0, num_items, block_size):
                # 获取当前块的索引范围
                start_idx = i
                end_idx = min(i + block_size, num_items)
                # 计算当前块的部分结果
                block_emb_item = emb_item[start_idx:end_idx]#block row vec
                block_mat_dis = np.dot(block_emb_item, emb_item.T)
                max_dis_block = np.max(block_mat_dis)  # 当前块的最大值
                min_dis_block = np.min(block_mat_dis)  # 当前块的最小值
                max_dis = max(max_dis, max_dis_block)  # 更新最大值
                min_dis = min(min_dis, min_dis_block)  # 更新最小值
                if compDisentDeg:
                    block_norm_emb_item = np.linalg.norm(block_emb_item, axis=1)
                    block_norm_emb_item_cosine_dis = (1 - block_mat_dis / np.outer(block_norm_emb_item, norm_emb_item)) / 2
                    # 将当前块的结果添加到列表中
                    norm_emb_item_cosine_max_dis=norm_emb_item_cosine_max_dis if norm_emb_item_cosine_max_dis>np.max(block_norm_emb_item_cosine_dis) else np.max(block_norm_emb_item_cosine_dis)
                    norm_emb_item_cosine_min_dis=norm_emb_item_cosine_min_dis if norm_emb_item_cosine_min_dis<np.min(block_norm_emb_item_cosine_dis) else np.min(block_norm_emb_item_cosine_dis)
                    # norm_emb_item_cosine_dis.append(block_norm_emb_item_cosine_dis)
        print("mat_dis=None")
        for seed in tqdm(list_seed):
            if merge_factor_rec_strategy is None:
                path_res = os.path.join("data", dataset_name, "rec", str(seed), "single_factor_res.npy")
            else:
                path_res = os.path.join("data", dataset_name, "rec", str(seed), "{}_res.npy".format("_".join(merge_factor_rec_strategy)))
            method_acc_inseed_list, method_dif_inseed_list = dict(), dict()
            if (not os.path.exists(path_res)) or True:
                mat_candidate = np.load(
                    os.path.join("data", dataset_name, "rec", str(seed), "candidate.npy"),
                    allow_pickle=True,
                ).item()
                create_pm(mat_candidate, dataset_name, seed)
                if os.path.exists(path_res) :
                    mat_res=np.load(path_res)
                else:
                    mat_res = np.zeros((len(list_method),8))
                mat_rec_list=[]
                for i_m, method in enumerate(list_method):
                    print("i_m ={}, method= {}".format(i_m, method),end="; ")
                    mat_rec = np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_" + method + ".npy")).astype(
                        int
                    )
                    mat_rec_list.append(mat_rec)
                if compDisentDeg:
                    if merge_factor_rec_strategy is None:
                        path_disent_jcd_file = os.path.join("data", dataset_name, "rec", str(seed),
                                                            "single_factor_disent_jcd_mat.npy")
                        path_disent_dif_file = os.path.join("data", dataset_name, "rec", str(seed),
                                                            "single_factor_disent_dif_mat.npy")
                    else:
                        path_disent_jcd_file = os.path.join("data", dataset_name, "rec", str(seed),
                                                            "{}_disent_jcd_mat.npy".format("_".join(merge_factor_rec_strategy)))
                        path_disent_dif_file = os.path.join("data", dataset_name, "rec", str(seed),
                                                            "{}_disent_dif_mat.npy".format("_".join(merge_factor_rec_strategy)))

                    # if os.path.exists(path_disent_jcd_file):disentJaccardDis_mat=np.load(path_disent_jcd_file)
                    # if os.path.exists(path_disent_dif_file):disentDifference_mat=np.load(path_disent_dif_file)
                    # disentJaccardDis_list,disentDifference_list=[],[]
                    userNum = len(mat_rec_list[0])
                    disentDifference_mat=np.zeros(shape=(len(list_method),len(list_method),userNum))
                    for i_m, method_i in tqdm(enumerate(list_method)):
                        for j_m, method_j in enumerate(list_method):
                            if j_m<i_m:
                                continue
                            # if method_i !='div' and method_j !='div':
                            #     disentJaccardDis_list.append(disentJaccardDis_mat[i_m][j_m])
                            #     disentDifference_list.append(disentDifference_mat[i_m][j_m])
                            #     continue
                            index_m_l,index_m_r=i_m,j_m
                            # disentJaccardDis=disentanglementDegree(dataset_name,mat_rec_list[index_m_l],mat_rec_list[index_m_r],
                            #                                        norm_emb_item_cosine_max_dis, norm_emb_item_cosine_min_dis,
                            #                                        metric='jaccardDis')
                            disentDifference=disentanglementDegree(dataset_name,mat_rec_list[index_m_l],mat_rec_list[index_m_r],
                                                                   norm_emb_item_cosine_max_dis, norm_emb_item_cosine_min_dis,
                                                                   metric='cosDis_diff')
                            disentDifference_mat[index_m_l,index_m_r,:]=disentDifference[:,0]
                            disentDifference_mat[index_m_r,index_m_l,:]=disentDifference[:,1]

                            # disentJaccardDis_list.append(disentJaccardDis)
                            # disentDifference_list.append(disentDifference)
                    max_values = np.max(disentDifference_mat, axis=1)  # 沿第二个维度计算最大值  maxval[i,u]
                    tmp_mat=disentDifference_mat / max_values[:, np.newaxis, :] / 2#[i,j,u]
                    # for i in range(disentDifference_mat.shape(0)):
                    #     for j in range(disentDifference_mat.shape(1)):
                    #         disentDifference_res[i,j]=np.sum(tmp_mat[i,j,:])+np.sum(tmp_mat[j,i,:])
                    disentDifference_res = np.sum(tmp_mat, axis=2) + np.sum(tmp_mat.transpose((1, 0, 2)), axis=2)
                    disentDifference_res /= disentDifference_mat.shape[2]

                    # disentDifference_res[i,j,u]= newMat[i,j,u]+ newMat[j,i,u]

                    #disentDifference_res=np.zeros_like(shape=((len(list_method),len(list_method))))
                    # for i in range(disentDifference_mat.shape(0)):
                    #     for j in range(disentDifference_mat.shape(1)):
                    #         disentDifference_res[i,j]=0
                    #         for u in range(disentDifference_mat.shape(2)):
                    #             disentDifference_res[i, j]+=disentDifference_mat[i,j,u]/max(disentDifference_mat[i,:,u])/2+
                    #             disentDifference_mat[j,i,u]/max(disentDifference_mat[j,:,u])/2
                    #         disentDifference_res[i, j]=disentDifference_res[i,j]/disentDifference_mat.shape(2)

                    # disentJaccardDis_mat=np.array(disentJaccardDis_list).reshape((len(list_method),len(list_method)))
                    # disentDifference_mat=np.array(disentDifference_list).reshape((len(list_method),len(list_method)))
                    print("np.save(path_disent_jcd_file,disentJaccardDis_mat)={}".format(path_disent_dif_file))
                    # np.save(path_disent_jcd_file,disentJaccardDis_mat)
                    np.save(path_disent_dif_file,disentDifference_res)
                    continue
                if not mem_enough:
                    acc_max = 0
                    dif_max = 0
                for i_m, method in enumerate(list_method):
                    print("i_m ={}, method= {}".format(i_m, method),end="; ")
                    # mat_rec = np.load(os.path.join("data", dataset_name, "rec", str(seed), "rec_" + method + ".npy")).astype(
                    #     int
                    # )
                    mat_rec=mat_rec_list[i_m]
                    # print("metric= {}".format("novelty unpop qua"),end="; ")
                    # mat_res[i_m, 0] = novelty(dataset_name, mat_rec)
                    # mat_res[i_m, 1] = unpopularity(dataset_name, mat_rec)
                    # print("metric= {}".format("qua diversity"),end="; ")
                    # mat_res[i_m, 2] = quality(dataset_name, mat_rec)
                    # mat_res[i_m, 5] = diversity(dataset_name, mat_rec, max_dis, min_dis)
                    print("metric= {}".format("acc dif ser1"),end="; ")
                    acc, dif, ser,org_list_acc_arr,org_list_dif_arr = ser1(dataset_name, mat_rec, max_dis, min_dis)
                    if mem_enough:
                        method_acc_inseed_list[method]=org_list_acc_arr
                        method_dif_inseed_list[method]=org_list_dif_arr
                        method_acc_list[method].append(org_list_acc_arr)
                        method_dif_list[method].append(org_list_dif_arr)
                    else:
                        ex_acc_max=exclude_outliers(org_list_acc_arr.flatten())
                        ex_dif_max=exclude_outliers(org_list_dif_arr.flatten())
                        acc_max = acc_max if acc_max>np.max(ex_acc_max) else np.max(ex_acc_max)
                        dif_max = dif_max if dif_max>np.max(ex_dif_max) else np.max(ex_dif_max)

                    mat_res[i_m, 3] = acc
                    mat_res[i_m, 4] = dif
                    mat_res[i_m, 6] = ser
                    print("metric= {}".format("ser2"),end="; ")
                    mat_res[i_m, 7] = ser2(dataset_name, mat_rec, max_dis, min_dis, seed)

                    '''
                    20231030 add new serend metrics from "Wisdom of Crowds and Fine-Grained Learning 
                    for Serendipity Recommendations", names HR_ser and NDCG_ser
                    '''
                    #
                    # mat_res[i_m, 8] = HR_ser(dataset_name,mat_rec)
                    # mat_res[i_m, 9] = NDCG_ser(dataset_name,mat_rec)
                if mem_enough:
                    method_acc_inseed,method_dif_inseed=[],[]
                    for i_m, method in enumerate(list_method):
                        method_acc_inseed.extend(exclude_outliers(method_acc_inseed_list[method].flatten()))
                        method_dif_inseed.extend(exclude_outliers(method_dif_inseed_list[method].flatten()))
                    acc_max=np.max(method_acc_inseed)
                    dif_max=np.max(method_dif_inseed)
                    print("=====acc max",acc_max)
                    print("=====dif_max",dif_max)
                    for i_m, method in enumerate(list_method):
                        acc_max_norm=method_acc_inseed_list[method]/acc_max#[user,K]
                        dif_max_norm=method_dif_inseed_list[method]/dif_max#[user,K]
                        mat_res[i_m, 3] = np.mean(acc_max_norm)
                        mat_res[i_m, 4] = np.mean(dif_max_norm)
                        mat_res[i_m, 6] = np.mean(2 * acc_max_norm * dif_max_norm / (acc_max_norm + dif_max_norm))#[user,K]
                else:
                    for i_m, method in enumerate(list_method):
                        print("mem_not_enough i_m ={}, method= {}".format(i_m, method), end="; ")
                        mat_rec = mat_rec_list[i_m]
                        print("metric= {}".format("acc dif ser1"), end="; ")
                        acc, dif, ser, org_list_acc_arr, org_list_dif_arr = ser1(dataset_name, mat_rec, max_dis,
                                                                                 min_dis)
                        acc_max_norm = org_list_acc_arr / acc_max  # [user,K]
                        dif_max_norm = org_list_dif_arr / dif_max  # [user,K]
                        mat_res[i_m, 3] = np.mean(acc_max_norm)
                        mat_res[i_m, 4] = np.mean(dif_max_norm)
                        mat_res[i_m, 6] = np.mean(
                            2 * acc_max_norm * dif_max_norm / (acc_max_norm + dif_max_norm))  # [user,K]
                print()
                np.save(path_res, mat_res)
                # print(method_acc_list)
                # print(type(method_acc_list))
        # if not compDisentDeg:
        #     path_res_acc = os.path.join("data", dataset_name, "res", "method_acc_list.npy")
        #     path_res_diff = os.path.join("data", dataset_name, "res", "method_dif_list.npy")
        #     with open(path_res_acc, 'wb') as file:
        #         pickle.dump(method_acc_list, file)
        #     with open(path_res_diff, 'wb') as file:
        #         pickle.dump(method_dif_list, file)

    sum_res_all_seed(list_dataset_name, list_seed,compDisentDeg=compDisentDeg,merge_factor_rec_strategy=merge_factor_rec_strategy)

