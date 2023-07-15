"""
Two Serendipity Metrics and Six Factor-based Mtrics
"""
import os
from multiprocessing import Pool
from collections import Counter

import pandas as pd
import numpy as np

num_pool = 32


def load_emb(dataset_name, is_user=True, emb_type="lgn", root_dir="./data"):
    emb_path = os.path.join(root_dir,
                            dataset_name,
                            f"emb_{'user' if is_user else 'item'}{'_w2v' if emb_type == 'w2v' else ''}.npy")
    return np.load(emb_path)


def sub_argpartition(list_score, list_candidate, K):
    # descend
    return list_candidate[np.argpartition(list_score, -K)[-K:]].reshape(1, K)


def accuracy_score(y_true, y_pred):
    """
    计算推荐列表的 accuracy
    """
    return len(np.intersect1d(y_true, y_pred)) / (len(y_pred)+1e-5)


def ser1_sub(dataset_name, max_dis, min_dis, list_rec, list_train, list_test):
    emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))
    emb_rec = emb_item[list_rec]
    emb_train = emb_item[list_train]
    emb_test = emb_item[list_test]
    acc = (np.max(np.dot(emb_rec, emb_test.T), axis=-1) - min_dis) / (max_dis - min_dis)
    dif = 1 - (np.max(np.dot(emb_rec, emb_train.T), axis=-1) - min_dis) / (max_dis - min_dis)
    ser = 2 * acc * dif / (acc + dif)
    return np.mean(acc), np.mean(dif), np.mean(ser)


def ser1(dataset_name, mat_rec, max_dis, min_dis):
    """
    the balance between accuracy and difference
    """
    df_train = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))
    df_test = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))
    groups_train = df_train.groupby("userInd")
    groups_test = df_test.groupby("userInd")

    list_acc, list_dif, list_ser = [], [], []
    for (_, group_train), (_, group_test), list_rec in zip(groups_train, groups_test, mat_rec):
        acc, dif, ser = ser1_sub(
            dataset_name,
            max_dis,
            min_dis,
            list_rec.astype(np.int),
            group_train["itemInd"].values.tolist(),
            group_test["itemInd"].values.tolist(),
        )
        list_acc.append(acc)
        list_dif.append(dif)
        list_ser.append(ser)

    return np.mean(list_acc), np.mean(list_dif), np.mean(list_ser)


def create_pm(mat_candidate, dataset_name, seed, K=200, topK=20):
    path_pm = os.path.join("data", dataset_name, f"rec{topK}", str(seed), "pm.npy")
    if os.path.exists(path_pm):
        return np.load(path_pm)

    df_rating = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"), usecols=["itemInd", "rating"])
    list_mean_rating = df_rating.groupby("itemInd")["rating"].mean().values
    df_item = pd.read_csv(os.path.join("data", dataset_name, "item.csv"))
    list_count = df_item["count"].values

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


def sum_res_all_seed(list_dataset_name, list_seed, K, corrected=True):
    print("Summarizing all results...")
    for dataset_name in list_dataset_name:
        list_res = []
        for seed in list_seed:
            mat_res = np.load(os.path.join("data", dataset_name, f"rec{K}", str(seed), "res.npy"))
            list_res.append(np.expand_dims(mat_res, axis=-1))
        mat_res = np.concatenate(list_res, axis=-1)
        mat_mean = np.mean(mat_res, axis=-1)
        mat_std = np.std(mat_res, axis=-1)

        save_dir = os.path.join('data', dataset_name, f'res{K}')
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        np.save(os.path.join(save_dir, "mean.npy"), mat_mean)
        np.save(os.path.join(save_dir, "std.npy"), mat_std)
        
        
def evaluate(list_dataset_name, list_seed, list_method, K):
    print("evaluating...")
    for dataset_name in list_dataset_name:
        print(f"evaluating {dataset_name}...")
        if os.path.exists(os.path.join("data", dataset_name, f"rec{K}", str(list_seed[0]), "res.npy")):
            print("Skip")
            continue
        emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy")).astype('float16')
        emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy")).astype('float16')
        mat_dis = np.dot(emb_item, emb_item.T)
        max_dis, min_dis = np.max(mat_dis), np.min(mat_dis)
        # mat_dis_ui = np.dot(emb_user, emb_item.T)
        # max_dis_ui, min_dis_ui = np.max(mat_dis_ui), np.min(mat_dis_ui)
        for seed in list_seed:
            path_res = os.path.join("data", dataset_name, f"rec{K}", str(seed), "res.npy")
            if not os.path.exists(path_res):
                mat_candidate = np.load(
                    os.path.join("data", dataset_name, f"rec{K}", str(seed), "candidate.npy"),
                    allow_pickle=True,
                ).item()
                create_pm(mat_candidate, dataset_name, seed, topK=K)
                mat_res = np.zeros((len(list_method), 8))
                for i_m, method in enumerate(list_method):
                    mat_rec = np.load(os.path.join("data", dataset_name, f"rec{K}", str(seed), "rec_" + method + ".npy")).astype(
                        np.int
                    )
                    mat_res[i_m, 0] = novelty(dataset_name, mat_rec)
                    mat_res[i_m, 1] = unpopularity(dataset_name, mat_rec)
                    mat_res[i_m, 2] = quality(dataset_name, mat_rec)
                    mat_res[i_m, 5] = diversity(dataset_name, mat_rec, max_dis, min_dis)
                    acc, dif, ser = ser1(dataset_name, mat_rec, max_dis, min_dis)
                    mat_res[i_m, 3] = acc
                    mat_res[i_m, 4] = dif
                    mat_res[i_m, 6] = ser
                    mat_res[i_m, 7] = ser2(dataset_name, mat_rec, max_dis, min_dis, seed)
                np.save(path_res, mat_res)
        print(f"evaluating {dataset_name} finished...")
    sum_res_all_seed(list_dataset_name, list_seed, K, corrected=False)
    print("Evaluating finished...")
