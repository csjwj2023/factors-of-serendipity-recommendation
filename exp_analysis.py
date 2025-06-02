"""
Two Serendipity Metrics and Six Factor-based Mtrics
"""
import os
import pickle
from random import random

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def embedAnalyz(list_dataset_name):
    for dataset_name in list_dataset_name:
        emb_item_xueqi = np.load(os.path.join("data", dataset_name, "emb_item_xueqi.npy"))  # [numI,dim]
        emb_user_xueqi = np.load(os.path.join("data", dataset_name, "emb_user_xueqi.npy"))  # [numU,dim]
        emb_item_leesong = np.load(os.path.join("data", dataset_name, "emb_user_leesong_1000.npy"))  # [numI,dim]
        emb_user_leesong = np.load(os.path.join("data", dataset_name, "emb_user_leesong_1000.npy"))  # [numU,dim]
        # 将数据转换为 DataFrame
        df = pd.DataFrame(emb_user_xueqi)
        df.to_csv(os.path.join("data", dataset_name, "emb_user_xueqi.csv"), index=False)
        # 将数据转换为 DataFrame
        df = pd.DataFrame(emb_user_leesong)
        df.to_csv(os.path.join("data", dataset_name, "emb_user_leesong.csv"), index=False)
        # 计算每列的均值和标准差
        mean = np.mean(emb_user_leesong, axis=0)
        std = np.std(emb_user_leesong, axis=0)
        # 归一化嵌入矩阵
        # normalized_emb_user_leesong = (emb_user_leesong - mean) / std#zscore
        factor=np.mean(emb_user_leesong/emb_user_xueqi)
        # factor=10
        normalized_emb_user_leesong = emb_user_leesong/factor

        print("====np.min(emb_user")
        print(np.min(emb_user_xueqi,axis=0))
        print(np.min(emb_user_leesong,axis=0))
        print(np.min(normalized_emb_user_leesong,axis=0))

        print("====np.max(emb_user")
        print(np.max(emb_user_xueqi,axis=0))
        print(np.max(emb_user_leesong,axis=0))
        print(np.max(normalized_emb_user_leesong,axis=0))


def plot_disent_mat(list_dataset_name,list_method,eps=1e-6,merge_factor_rec_strategy=None,needNorm=False):
    for dataset_name in list_dataset_name:
        if merge_factor_rec_strategy is None:
            mat_mean_file_name="single_factor_disent_dif_mat.npy"
        else:
            mat_mean_file_name="{}_disent_dif_mat.npy".format("_".join(merge_factor_rec_strategy))

        mat_mean_path = os.path.join("data", dataset_name, "res", mat_mean_file_name)

        disentDegreeMat=np.load(mat_mean_path)
        # 保留小数
        disentDegreeMat = np.round(disentDegreeMat, decimals=2)
        # 移除第五列和第五行
        disentDegreeMat = np.delete(disentDegreeMat, 4, axis=1)  # 移除第五列 elasticity
        disentDegreeMat = np.delete(disentDegreeMat, 4, axis=0)  # 移除第五行 elasticity
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 创建一个示例的二维数组
        # 绘制热力图
        convert_to_zero = np.vectorize(lambda x: 0.0 if (x<eps and x>-eps) else x)
        print("before convert -0:\n", disentDegreeMat)
        disentDegreeMat = convert_to_zero(disentDegreeMat)
        print("after convert -0:\n", disentDegreeMat)
        # plt.figure(figsize=(9, 7))
        plt.rcParams.update({'font.size': 12})

        # 创建上三角掩码
        # disentDegreeMat=disentDegreeMat+disentDegreeMat.T
        # disentDegreeMat=disentDegreeMat
        if needNorm:
            disentDegreeMat=disentDegreeMat/np.max(disentDegreeMat)
        mask = np.triu(np.ones_like(disentDegreeMat),k=1)

        ax = sns.heatmap(disentDegreeMat, cmap='YlGnBu', mask=mask,cbar=False, annot=True)
        # plt.imshow(mat_mean, cmap='YlGnBu', interpolation='nearest')
        print("disentDegreeMat: ", disentDegreeMat)
        plt.xticks(np.array(range(len(disentDegreeMat[0]))) + 0.5, list_method)
        plt.yticks(np.array(range(len(disentDegreeMat))) + 0.5, list_method)
        plt.xticks(rotation=55)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        # 设置坐标轴标签
        # 显示图形
        if merge_factor_rec_strategy is None:
            fig_file_save_name="single_factor_disent_dif_mat"
        else:
            fig_file_save_name="{}_disent_dif_mat".format("_".join(merge_factor_rec_strategy))
        plt.savefig("./fig/{}_{}.jpg".format(dataset_name,fig_file_save_name), format='jpg')
        plt.savefig("./fig/{}_{}.pdf".format(dataset_name,fig_file_save_name), format='pdf')
        plt.show()

def displayMetricBounds(list_dataset_name,list_method,list_metric=['dif','acc']):
    for dataset_name in list_dataset_name:
        for metric in list_metric:
            print("==============displayMetric={} Bounds=====".format(metric))
            pathmethod=os.path.join("data", dataset_name, "res", "method_{}_list.npy".format(metric))
            with open(pathmethod, 'rb') as file:
                method_metric_list = pickle.load(file)
            print(method_metric_list)
            print(type(method_metric_list))
            for i_m, method in enumerate(list_method):
                metric_arr=np.stack(method_metric_list[method])#[seed,user,recNum]
                print(metric_arr)
                print(metric_arr.shape)
                print("========")

def plot_res_mat(list_dataset_name,list_method,list_metric,merge_factor_rec_strategy=None,decimals=2,eps=1e-6):
    for dataset_name in list_dataset_name:
        if merge_factor_rec_strategy is None:
            mat_mean_file_name="single_factor_res_mean.npy"
        else:
            mat_mean_file_name="{}_res_mean.npy".format("_".join(merge_factor_rec_strategy))
        mat_mean_path = os.path.join("data", dataset_name, "res", mat_mean_file_name)
        mat_mean=np.load(mat_mean_path)
        mat_mean=np.delete(mat_mean, 4, axis=0)
        #保留两位小数
        print(mat_mean)
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 创建一个示例的二维数组
        # 绘制热力图
        convert_to_zero = np.vectorize(lambda x: 0.0 if x < eps and x>-eps else x)
        print("before convert -0:\n",mat_mean)
        mat_mean = convert_to_zero(mat_mean)
        print("after convert -0:\n",mat_mean)
        # 计算每列的最大值
        max_values = np.max(mat_mean, axis=0)
        # 对数组的每一列进行除法操作
        mat_mean = mat_mean / max_values
        mat_mean = np.round(mat_mean, decimals=decimals)

        plt.rcParams.update({'font.size': 12})
        # plt.figure(figsize=(7, 4))

        ax=sns.heatmap(mat_mean, cmap='YlGnBu', cbar=False,annot=True)


        # plt.imshow(mat_mean, cmap='YlGnBu', interpolation='nearest')
        print("mat mean[0]: ",mat_mean[0])
        print("xticks: ",np.array(range(len(mat_mean[0])))+0.5)
        print("yticks: ",np.array(range(len(mat_mean)))+0.5)
        plt.xticks(np.array(range(len(mat_mean[0])))+0.5,list_metric)
        plt.yticks(np.array(range(len(mat_mean)))+0.5,list_method)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        # 设置坐标轴标签
        # 显示图形
        plt.savefig("./fig/{}_{}.jpg".format(dataset_name,mat_mean_file_name), format='jpg')
        plt.savefig("./fig/{}_{}.pdf".format(dataset_name,mat_mean_file_name), format='pdf')

        plt.show()
def interaction_num_distb(df_data):
    # 按照 'userId' 分组并统计每组的交互数
    import matplotlib.pyplot as plt
    grouped_by_userId = df_data.groupby('userId').size().reset_index(name='count_by_userId')
    print(grouped_by_userId['count_by_userId'].min())
    count_distribution = grouped_by_userId['count_by_userId'].value_counts().sort_index()
    # print(count_distribution)
    plt.bar(count_distribution.index, count_distribution.values)
    plt.xlabel('Interaction Num')
    plt.ylabel('Frequency')
    plt.title('Interaction Num Distribution of grouped_by_userId')
    plt.show()
    # 按照 'itemId' 分组并统计每组的交互数
    grouped_by_itemId = df_data.groupby('itemId').size().reset_index(name='count_by_itemId')
    # print(grouped_by_itemId['count_by_itemId'].min())

    count_distribution = grouped_by_itemId['count_by_itemId'].value_counts().sort_index()
    plt.bar(count_distribution.index, count_distribution.values)
    plt.xlabel('Interaction Num')
    plt.ylabel('Frequency')
    plt.title('Interaction Num Distribution of grouped_by_itemId')
    plt.show()
def timeCheck(df_data):
    sorted_df = df_data.groupby('userInd').apply(lambda x: x.sort_values('timestamp'))
    # print(sorted_df[sorted_df['userInd']==0].head(3))

def exclude_outliers(data):
    # 绘制箱线图
    # plt.boxplot(data)
    # plt.show()
    # 计算四分位数和四分位距
    print("begin exclude_outliers")
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    # 计算上下边界
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    # 排除异常点
    # filtered_data = [value for value in data if lower_bound <= value <= upper_bound]
    condition = np.logical_and(data >= lower_bound, data <= upper_bound)
    filtered_data = data[condition]
    return filtered_data

def StatiSimiEmbed(list_dataset_name,useData="train",withSerLabel=False,epsilon=0.1,needMinMax=False,
                   linearScale=False,sigmoid=False,ex_outliers=False,emd_norm=False,mem_enough=False):
    '''
    嵌入空间相似性统计。
    '''
    def mat_MinMaxNorm(mat,needMinMax=needMinMax,linearScale=linearScale,sigmoid=sigmoid,
                                ex_outliers=ex_outliers):
        if ex_outliers:
            mat=exclude_outliers(np.array(mat.flatten().tolist()))
        if not needMinMax:
            return mat
        max_dis, min_dis = np.max(mat) + epsilon, np.min(mat)
        if linearScale:
            mat_norm = (mat - min_dis) / (max_dis - min_dis)*max_dis
        elif sigmoid:
            if mem_enough:
                mat_norm = 1 / (1 + np.exp(-mat))
            else:
                if isinstance(mat, np.ndarray):
                    if mat.ndim == 1:
                        # 处理一维数组
                        for i in range(mat.shape[0]):
                            mat[i] = 1 / (1 + np.exp(-mat[i]))
                    elif mat.ndim == 2:
                        # 处理二维矩阵
                        for i in range(mat.shape[0]):
                            mat[i, :] = 1 / (1 + np.exp(-mat[i,:]))

                elif isinstance(mat, list):
                    for i in range(len(mat)):
                        mat[i] = 1 / (1 + np.exp(-mat[i]))
                else:
                    raise TypeError("数据类型不支持")
                mat_norm = mat
        else:
            mat_norm = (mat - min_dis) / (max_dis - min_dis)
        return mat_norm
    print_res=[]
    for dataset_name in list_dataset_name:
        emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))#[numI,dim]
        # print("(np.mean(emb_item, axis=0)) ")
        # print(np.mean(emb_item,axis=0))
        # print(np.linalg.norm(emb_item, axis=1))
        emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))#[numU,dim]
        if emd_norm:
            emb_item = emb_item / np.linalg.norm(emb_item, axis=1, keepdims=True)
            emb_user = emb_user / np.linalg.norm(emb_user, axis=1, keepdims=True)
        # $\mu_{u,i}$,$sigma_{u,i}$ on rated items (pos serend+neg serend)
        # print("df_all_data = pd.read_csv  begin")
        if useData=="train":
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))

        elif useData=="test":
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))

        else:
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"))
        # interaction_num_distb(df_all_data)
        # timeCheck(df_all_data)
        # $\mu_{u,i}$,$sigma_{u,i}$ on all items
        print("emb_ui_dot = np.dot(emb_user, emb_item.T)  begin")
        emb_ui_dot = np.dot(emb_user, emb_item.T)
        emb_ui_dot=mat_MinMaxNorm(emb_ui_dot)
        mu_ui_all = np.mean(emb_ui_dot)
        sigma_ui_all = np.std(emb_ui_dot)
        emb_ui_dot=None
        # $\mu_{i,i}$,$sigma_{i,i}$ on all items
        print("emb_ii_dot = np.dot(emb_item, emb_item.T)  begin")
        emb_ii_dot = np.dot(emb_item, emb_item.T)  # [n,dim][dim,n]=>[n,n]
        print("emb_ii_dot = np.dot(emb_item, emb_item.T)  end")
        emb_ii_dot=mat_MinMaxNorm(emb_ii_dot)
        mu_ii_all = np.mean(emb_ii_dot)
        sigma_ii_all = np.std(emb_ii_dot)
        emb_ii_dot=None
        group_all_data = df_all_data.groupby("userInd")
        # print("group_all_data = df_all_data.groupby(userInd)  end")

        #userInd,itemInd,rating,timestamp,userId,itemId,serLabel
        # $\mu_{u,i}$,$sigma_{u,i}$,$\mu_{i,i}$,$sigma_{i,i}$ on serend items
        emb_u_i_rated_pos_ser_dot_list,emb_u_i_rated_neg_ser_dot_list=[],[]
        emb_ii_rated_pos_ser_dot_list, emb_ii_rated_neg_ser_dot_list = [], []
        emb_u_i_rated_dot_list,emb_ii_rated_dot_list=[],[]
        for uind,one_user_all_data in tqdm(group_all_data):
            emd_u=emb_user[uind]#[dim]
            emd_u=np.reshape(emd_u,(1,len(emd_u)))
            if withSerLabel:
                rated_pos_serend_iind=one_user_all_data[one_user_all_data['serLabel']>0.999]['itemInd'].values.tolist()
                rated_neg_serend_iind=one_user_all_data[one_user_all_data['serLabel']<0.999]['itemInd'].values.tolist()
                emb_i_rated_pos_serend=emb_item[rated_pos_serend_iind]#[n,dim]
                emb_u_i_rated_pos_ser_dot = mat_MinMaxNorm(np.dot(emd_u, emb_i_rated_pos_serend.T)).flatten().tolist()#[n,]

                emb_u_i_rated_pos_ser_dot_list.extend(emb_u_i_rated_pos_ser_dot)
                emb_ii_rated_pos_ser_dot=mat_MinMaxNorm(np.dot(emb_i_rated_pos_serend, emb_i_rated_pos_serend.T)).flatten().tolist()#[n,]

                emb_ii_rated_pos_ser_dot_list.extend(emb_ii_rated_pos_ser_dot)
                # mu_u_i_rated_pos_ser_dot=np.mean(emb_u_i_rated_pos_ser_dot)
                # sigma_u_i_rated_pos_ser_dot=np.std(emb_u_i_rated_pos_ser_dot)

                emb_i_rated_neg_serend=emb_item[rated_neg_serend_iind]#[n,dim]
                emb_u_i_rated_neg_ser_dot = np.dot(emd_u, emb_i_rated_neg_serend.T)
                emb_u_i_rated_neg_ser_dot = mat_MinMaxNorm(emb_u_i_rated_neg_ser_dot)
                emb_u_i_rated_neg_ser_dot=emb_u_i_rated_neg_ser_dot.flatten().tolist()  # [n,]
                emb_u_i_rated_neg_ser_dot_list.extend(emb_u_i_rated_neg_ser_dot)
                emb_ii_rated_neg_ser_dot=np.dot(emb_i_rated_neg_serend, emb_i_rated_neg_serend.T)
                emb_ii_rated_neg_ser_dot = mat_MinMaxNorm(emb_ii_rated_neg_ser_dot)
                emb_ii_rated_neg_ser_dot=emb_ii_rated_neg_ser_dot.flatten().tolist()#[n,]
                emb_ii_rated_neg_ser_dot_list.extend(emb_ii_rated_neg_ser_dot)
                # mu_u_i_rated_neg_ser_dot=np.mean(emb_u_i_rated_neg_ser_dot)
                # sigma_u_i_rated_neg_ser_dot=np.std(emb_u_i_rated_neg_ser_dot)


            rated_iind = one_user_all_data['itemInd'].values.tolist()
            emb_i_rated=emb_item[rated_iind]
            emb_u_i_rated_dot=mat_MinMaxNorm(np.dot(emd_u, emb_i_rated.T)).flatten().tolist()#[n,]

            emb_u_i_rated_dot_list.extend(emb_u_i_rated_dot)
            emb_ii_rated_dot=mat_MinMaxNorm(np.dot(emb_i_rated, emb_i_rated.T)).flatten().tolist()#[n,]
            emb_ii_rated_dot_list.extend(emb_ii_rated_dot)
        if withSerLabel:
            mu_u_i_rated_pos_ser_dot=np.mean(emb_u_i_rated_pos_ser_dot_list)
            sigma_u_i_rated_pos_ser_dot=np.std(emb_u_i_rated_pos_ser_dot_list)
            mu_u_i_rated_neg_ser_dot=np.mean(emb_u_i_rated_neg_ser_dot_list)
            sigma_u_i_rated_neg_ser_dot=np.std(emb_u_i_rated_neg_ser_dot_list)

        mu_u_i_rated_dot = np.mean(emb_u_i_rated_dot_list)
        sigma_u_i_rated_dot = np.std(emb_u_i_rated_dot_list)
        if withSerLabel:
            mu_ii_rated_pos_ser_dot=np.mean(emb_ii_rated_pos_ser_dot_list)
            sigma_ii_rated_pos_ser_dot=np.std(emb_ii_rated_pos_ser_dot_list)
            mu_ii_rated_neg_ser_dot=np.mean(emb_ii_rated_neg_ser_dot_list)
            sigma_ii_rated_neg_ser_dot=np.std(emb_ii_rated_neg_ser_dot_list)
        mu_ii_rated_dot = np.mean(emb_ii_rated_dot_list)
        sigma_ii_rated_dot = np.std(emb_ii_rated_dot_list)

        print("Statistics on Similarity in Embedding Space of dataset {}:".format(dataset_name))
        if withSerLabel:
            print("on the rated_pos_serend items: ")
            print("mu_ui={}, sigma_ui={}, mu_ii={}, sigma_ii={}".format(mu_u_i_rated_pos_ser_dot,sigma_u_i_rated_pos_ser_dot,
                                                                         mu_ii_rated_pos_ser_dot,sigma_ii_rated_pos_ser_dot))
            print("on the rated_neg_serend items: ")
            print("mu_ui={}, sigma_ui={}, mu_ii={}, sigma_ii={}".format(mu_u_i_rated_neg_ser_dot,
                                                                     sigma_u_i_rated_neg_ser_dot,
                                                                     mu_ii_rated_neg_ser_dot,
                                                                     sigma_ii_rated_neg_ser_dot))
        # print("on the rated items: ")
        # print("mu_ui={}, sigma_ui={}, mu_ii={}, sigma_ii={}".format(mu_u_i_rated_dot,
        #                                                             sigma_u_i_rated_dot,
        #                                                             mu_ii_rated_dot,
        #                                                             sigma_ii_rated_dot))
        #
        # print("on the all items: ")
        # print("mu_ui={}, sigma_ui={}, mu_ii={}, sigma_ii={}".format(mu_ui_all,
        #                                                              sigma_ui_all,
        #                                                              mu_ii_all,
        #                                                              sigma_ii_all))
        print_res.append("\t".join(map(str, [mu_u_i_rated_dot,sigma_u_i_rated_dot,mu_ii_rated_dot,sigma_ii_rated_dot,mu_ui_all,sigma_ui_all,mu_ii_all,sigma_ii_all])))

    for info in print_res:
        print(info)

def StatiSimiEmbedCos(list_dataset_name,useData="train",emd_norm=False,block_size = 100):
    '''
    嵌入空间相似性统计。
    '''
    print_res=[]
    for dataset_name in list_dataset_name:
        emb_item = np.load(os.path.join("data", dataset_name, "emb_item.npy"))#[numI,dim]
        # print("(np.mean(emb_item, axis=0)) ")
        # print(np.mean(emb_item,axis=0))
        # print(np.linalg.norm(emb_item, axis=1))
        emb_user = np.load(os.path.join("data", dataset_name, "emb_user.npy"))#[numU,dim]
        if emd_norm:
            emb_item = emb_item / np.linalg.norm(emb_item, axis=1, keepdims=True)
            emb_user = emb_user / np.linalg.norm(emb_user, axis=1, keepdims=True)
        # $\mu_{u,i}$,$sigma_{u,i}$ on rated items (pos serend+neg serend)
        # print("df_all_data = pd.read_csv  begin")
        if useData=="train":
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating_train.csv"))

        elif useData=="test":
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating_test.csv"))

        else:
            df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"))
        # interaction_num_distb(df_all_data)
        # timeCheck(df_all_data)
        # $\mu_{u,i}$,$sigma_{u,i}$ on all items
        print("emb_ui_dot = np.dot(emb_user, emb_item.T)  begin")
        emb_ui_dot = np.dot(emb_user, emb_item.T)
        # 获取矩阵的维度
        n_users, dim = emb_user.shape
        n_items = emb_item.shape[0]
        # 初始化结果矩阵
        emb_ui_dot = np.zeros((n_users, n_items))
        # 分块计算余弦相似度
        for i in range(0, n_users, block_size):
            emb_user_block = emb_user[i:i + block_size]
            emb_ui_dot_block = (1-cosine_similarity(emb_user_block, emb_item))/2
            emb_ui_dot[i:i + block_size] = emb_ui_dot_block
        mu_ui_all = np.mean(emb_ui_dot)
        sigma_ui_all = np.std(emb_ui_dot)
        emb_ui_dot=None
        # $\mu_{i,i}$,$sigma_{i,i}$ on all items
        print("emb_ii_dot = np.dot(emb_item, emb_item.T)  begin")
        # 分块计算余弦相似度
        emb_ii_dot = np.zeros((n_items, n_items))
        for i in range(0, n_users, block_size):
            emb_item_block = emb_item[i:i + block_size]
            emb_ii_dot_block = (1-cosine_similarity(emb_item_block, emb_item))/2
            emb_ii_dot[i:i + block_size] = emb_ii_dot_block
        mu_ii_all = np.mean(emb_ii_dot)
        sigma_ii_all = np.std(emb_ii_dot)
        emb_ii_dot=None
        group_all_data = df_all_data.groupby("userInd")
        # print("group_all_data = df_all_data.groupby(userInd)  end")
        #userInd,itemInd,rating,timestamp,userId,itemId,serLabel
        # $\mu_{u,i}$,$sigma_{u,i}$,$\mu_{i,i}$,$sigma_{i,i}$ on serend items
        emb_u_i_rated_pos_ser_dot_list,emb_u_i_rated_neg_ser_dot_list=[],[]
        emb_ii_rated_pos_ser_dot_list, emb_ii_rated_neg_ser_dot_list = [], []
        emb_u_i_rated_dot_list,emb_ii_rated_dot_list=[],[]
        for uind,one_user_all_data in tqdm(group_all_data):
            emd_u=emb_user[uind]#[dim]
            emd_u=np.reshape(emd_u,(1,len(emd_u)))

            rated_iind = one_user_all_data['itemInd'].values.tolist()
            emb_i_rated=emb_item[rated_iind]
            emb_u_i_rated_dot=(1-cosine_similarity(emd_u, emb_i_rated))/2
            emb_u_i_rated_dot=emb_u_i_rated_dot.flatten().tolist()#[n,]
            emb_u_i_rated_dot_list.extend(emb_u_i_rated_dot)
            emb_ii_rated_dot=(1-cosine_similarity(emb_i_rated, emb_i_rated))/2
            emb_ii_rated_dot=emb_ii_rated_dot.flatten().tolist()#[n,]
            emb_ii_rated_dot_list.extend(emb_ii_rated_dot)
        mu_u_i_rated_dot = np.mean(emb_u_i_rated_dot_list)
        sigma_u_i_rated_dot = np.std(emb_u_i_rated_dot_list)
        mu_ii_rated_dot = np.mean(emb_ii_rated_dot_list)
        sigma_ii_rated_dot = np.std(emb_ii_rated_dot_list)
        print("Statistics on Similarity in Embedding Space of dataset {}:".format(dataset_name))
        print_res.append("\t".join(map(str, [mu_u_i_rated_dot,sigma_u_i_rated_dot,mu_ii_rated_dot,sigma_ii_rated_dot,mu_ui_all,sigma_ui_all,mu_ii_all,sigma_ii_all])))

    for info in print_res:
        print(info)

def ImpactsOfFactorsOnSerendipity(list_dataset_name,list_method,merge_factor_rec_strategy=None,
                                  ser_col_idxs=[-2,-1],ser_col_name=["ser1","ser2"],drop_method_idxs = [4]):
    method_index_array = [i for i in range(len(list_method)) if i not in drop_method_idxs]
    for dpidx in drop_method_idxs:
        list_method.pop(dpidx)
    ser_rank_list = [[] for _ in range(len(ser_col_idxs))]
    if merge_factor_rec_strategy is None:
        mat_mean_file_name = "single_factor_res_mean.npy"
    else:
        mat_mean_file_name = "{}_res_mean.npy".format("_".join(merge_factor_rec_strategy))
    for dataset_name in list_dataset_name:
        mat_mean_path = os.path.join("data", dataset_name, "res", mat_mean_file_name)
        mat_mean = np.load(mat_mean_path)[method_index_array,:8]
        for ser_rank_list_idx,ser_col_idx in enumerate(ser_col_idxs):
            ser_col_data=mat_mean[:,ser_col_idx]
            sorted_indices = np.argsort(-ser_col_data)# 降序排序索引
            ranks=np.zeros_like(sorted_indices)
            for idx,sorted_indice in enumerate(sorted_indices):
                ranks[sorted_indice]=idx+1
            ser_rank_list[ser_rank_list_idx].append(ranks)#[dataset,method]

    def ImpactsOfFactorsOnSerendipityPlot(list_dataset_name,list_method, file_name,ser_rank_mat,ser_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 绘制热力图
        plt.rcParams.update({'font.size': 13})
        ax = sns.heatmap(ser_rank_mat, cmap='YlGnBu', cbar=False, annot=True)
        # plt.title(ser_name)
        print(ser_rank_mat)
        plt.xticks(np.array(range(len(ser_rank_mat[0]))) + 0.5, list_dataset_name)
        plt.yticks(np.array(range(len(ser_rank_mat))) + 0.5, list_method)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=55)

        plt.tight_layout()
        plt.savefig("./fig/{}.jpg".format(ser_name), format='jpg')
        plt.savefig("./fig/{}.pdf".format(ser_name), format='pdf')
        plt.show()
    ser_rank_comb=np.zeros_like(np.array(ser_rank_list[0]).T)#[method,dataset]
    for ser_rank_list_idx,_ in enumerate(ser_col_idxs):
        ser_rank_mat=np.array(ser_rank_list[ser_rank_list_idx]).T#[method,dataset]
        ser_rank_comb+=ser_rank_mat
        ser_name=ser_col_name[ser_rank_list_idx]
        ImpactsOfFactorsOnSerendipityPlot(list_dataset_name, list_method, "ser{}".format(ser_rank_list_idx), ser_rank_mat, ser_name)

    for col_idx in range(ser_rank_comb.shape[1]):
        col_data=ser_rank_comb[:,col_idx]
        sorted_indices = np.argsort(col_data)  # 升序排序索引
        ranks = np.zeros_like(sorted_indices)
        for idx, sorted_indice in enumerate(sorted_indices):
            ranks[sorted_indice] = idx + 1
        ser_rank_comb[:,col_idx]=ranks
    ImpactsOfFactorsOnSerendipityPlot(list_dataset_name, list_method, mat_mean_file_name, ser_rank_comb, "ser3")


def topk_single_factor(list_dataset_name,list_method,Ks=[5,10,15,20],decimals=2,eps=1e-6):
    for dataset_name in list_dataset_name:
        metrics_idxs=[-1,-2]
        metrics_data_ks = [[] for _ in range(len(metrics_idxs))]
        print(metrics_data_ks)
        for k_val in Ks:
            mat_mean_file_name = "single_factor_res_mean.npy"
            mat_mean_path = os.path.join("data", "{}_top{}".format(dataset_name,k_val), "res", mat_mean_file_name)
            mat_mean = np.load(mat_mean_path)
            mat_mean = np.delete(mat_mean, 4, axis=0)
            # 保留两位小数
            # print(mat_mean)
            import matplotlib.pyplot as plt
            import seaborn as sns
            # 创建一个示例的二维数组
            # 绘制热力图
            convert_to_zero = np.vectorize(lambda x: 0.0 if x < eps and x > -eps else x)
            # print("before convert -0:\n", mat_mean)
            mat_mean = convert_to_zero(mat_mean)
            # print("after convert -0:\n", mat_mean)
            # 计算每列的最大值
            max_values = np.max(mat_mean, axis=0)
            # 对数组的每一列进行除法操作
            mat_mean = mat_mean / max_values
            mat_mean = np.round(mat_mean, decimals=decimals)
            for idx,metrics_idx in enumerate(metrics_idxs):
                print("idx ",idx)
                print("metrics_idx ",metrics_idx)
                metrics_data_ks[idx].append(mat_mean[:,metrics_idx])

        plt.rcParams.update({'font.size': 12})

        for idx, metrics_idx in enumerate(metrics_idxs):
            # 遍历每行数据，绘制折线图并设置图例标注
            data=np.concatenate(metrics_data_ks[idx])
            data=data.reshape((len(Ks),len(list_method))).T
            print(data)
            for i, row in enumerate(data):
                if i==1 or i==len(data)-1:
                    print(row)
                    if metrics_idx==-1:
                        row=row-0.01
                    elif metrics_idx==-2:
                        row = row - 0.002
                    print(row)
                plt.plot(row, label=list_method[i],marker='o')
            plt.xticks(range(len(Ks)), Ks)
            if metrics_idx==-2:
                plt.ylim(0.6, 1.02)
            plt.xlabel("k")
            if metrics_idx==-1:
                # 添加图例
                plt.legend(loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1)
            else:
                # 添加图例
                plt.legend()
            # 显示图形
            metrics_names={-2:"ser1",-1:"ser2"}
            fig_file_save_name="topk_single_factor_{}".format(metrics_names[metrics_idx])
            print(fig_file_save_name)

            plt.savefig("./fig/{}_{}.jpg".format(dataset_name, fig_file_save_name), format='jpg')
            plt.savefig("./fig/{}_{}.pdf".format(dataset_name, fig_file_save_name), format='pdf')
            plt.show()

def interactions_num_freq(list_dataset_name):
    for dataset_name in list_dataset_name:
        df_all_data = pd.read_csv(os.path.join("data", dataset_name, "rating.csv"))
        for k in range(1,11):
            item_counts = df_all_data.groupby("itemId").size()
            # item_counts = df_all_data.groupby("userId").size()

            less_than_10_items = (item_counts <=k).sum()
            less_than_10_items_percentage = (less_than_10_items / len(item_counts)) * 100
            less_than_10_items_percentage=np.around(less_than_10_items_percentage, decimals=2)
            # print("less_than_10_items= {}%".format(less_than_10_items_percentage))
            print("{}%".format(less_than_10_items_percentage),end="\t")
        print()


#topk  parm sensitivity exp

list_dataset_name = ["kindle"]
list_method=["random", "novelty", "unpopularity", "high quality", "relevance", "difference", "diversity"]
import numpy as np

top5kindle = np.array([
    [0.91, 0.98, 0.83, 0.69, 0.92, 0.99, 0.88, 0.76],
    [1, 1, 0.85, 0.66, 0.96, 1, 0.84, 0.69],
    [1, 1, 0.85, 0.66, 0.96, 1, 0.84, 0.69],
    [0.93, 0.99, 1, 0.68, 0.94, 0.99, 0.86, 0],
    [0.9, 0.97, 0.81, 1, 0.78, 0.93, 1, 1],
    [0.93, 0.98, 0.83, 0.64, 1, 0.97, 0.84, 0.62],
    [0.91, 0.98, 0.83, 0.69, 0.92, 0.99, 0.88, 0.76]
])

top10kindle = np.array([
    [0.91, 0.98, 0.83, 0.76, 0.93, 0.99, 0.92, 0.78],
    [1, 1, 0.85, 0.72, 0.97, 1, 0.91, 0.71],
    [1, 1, 0.85, 0.72, 0.97, 1, 0.91, 0.71],
    [0.93, 0.99, 1, 0.74, 0.95, 0.99, 0.92, 0],
    [0.9, 0.97, 0.81, 1, 0.8, 0.93, 1, 1],
    [0.93, 0.98, 0.83, 0.7, 1, 0.97, 0.91, 0.66],
    [0.91, 0.98, 0.83, 0.76, 0.93, 0.99, 0.92, 0.78]
])

top15kindle = np.array([
    [0.91, 0.98, 0.83, 0.79, 0.94, 0.99, 0.95, 0.8],
    [1, 1, 0.85, 0.74, 0.97, 1, 0.93, 0.72],
    [1, 1, 0.85, 0.74, 0.97, 1, 0.93, 0.72],
    [0.93, 0.99, 1, 0.77, 0.95, 0.99, 0.93, 0],
    [0.9, 0.97, 0.81, 1, 0.82, 0.93, 1, 1],
    [0.93, 0.98, 0.83, 0.73, 1, 0.97, 0.92, 0.68],
    [0.91, 0.98, 0.83, 0.79, 0.94, 0.99, 0.95, 0.8]
])

top20kindle = np.array([
    [0.91, 0.98, 0.83, 0.81, 0.94, 0.99, 0.95, 0.8],
    [1, 1, 0.85, 0.76, 0.97, 1, 0.94, 0.72],
    [1, 1, 0.85, 0.76, 0.97, 1, 0.94, 0.72],
    [0.93, 0.99, 1, 0.79, 0.95, 0.99, 0.95, 0],
    [0.9, 0.97, 0.81, 1, 0.82, 0.93, 1, 1],
    [0.93, 0.98, 0.83, 0.74, 1, 0.97, 0.94, 0.68],
    [0.91, 0.98, 0.83, 0.81, 0.94, 0.99, 0.95, 0.8]
])
mat_mean_arr=[top5kindle,top10kindle,top15kindle,top20kindle]
def topk_single_factor(list_dataset_name,list_method,Ks=[5,10,15,20],decimals=2,eps=1e-6):
    for dataset_name in list_dataset_name:
        metrics_idxs=[-1,-2]
        metrics_data_ks = [[] for _ in range(len(metrics_idxs))]
        print(metrics_data_ks)
        for k_idx,k_val in enumerate(Ks):
            # mat_mean_file_name = "single_factor_res_mean.npy"
            # mat_mean_path = os.path.join("data", "{}_top{}".format(dataset_name,k_val), "res", mat_mean_file_name)
            # mat_mean = np.load(mat_mean_path)
            # mat_mean = np.delete(mat_mean, 4, axis=0)
            mat_mean=mat_mean_arr[k_idx]
            # 保留两位小数
            # print(mat_mean)
            import matplotlib.pyplot as plt
            import seaborn as sns
            # 创建一个示例的二维数组
            # 绘制热力图
            convert_to_zero = np.vectorize(lambda x: 0.0 if x < eps and x > -eps else x)
            # print("before convert -0:\n", mat_mean)
            mat_mean = convert_to_zero(mat_mean)
            # print("after convert -0:\n", mat_mean)
            # 计算每列的最大值
            max_values = np.max(mat_mean, axis=0)
            # 对数组的每一列进行除法操作
            mat_mean = mat_mean / max_values
            mat_mean = np.round(mat_mean, decimals=decimals)
            for idx,metrics_idx in enumerate(metrics_idxs):
                print("idx ",idx)
                print("metrics_idx ",metrics_idx)
                metrics_data_ks[idx].append(mat_mean[:,metrics_idx])

        plt.rcParams.update({'font.size': 12})

        for idx, metrics_idx in enumerate(metrics_idxs):
            # 遍历每行数据，绘制折线图并设置图例标注
            data=np.concatenate(metrics_data_ks[idx])
            data=data.reshape((len(Ks),len(list_method))).T
            print(data)
            for i, row in enumerate(data):
                if i==1 or i==len(data)-1:
                    print(row)
                    if metrics_idx==-1:
                        row=row-0.01
                    elif metrics_idx==-2:
                        row = row - 0.003
                    print(row)
                plt.plot(row, label=list_method[i],marker='D')
            plt.xticks(range(len(Ks)), Ks)
            if metrics_idx==-2:
                plt.ylim(0.6, 1.02)
            plt.xlabel("k")
            if metrics_idx==-1:
                # 添加图例
                plt.legend(loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1)
            else:
                # 添加图例
                plt.legend()
            # 显示图形
            metrics_names={-2:"ser1",-1:"ser2"}
            fig_file_save_name="topk_single_factor_{}".format(metrics_names[metrics_idx])
            print(fig_file_save_name)

            plt.savefig("./fig/{}_{}.jpg".format(dataset_name, fig_file_save_name), format='jpg')
            plt.savefig("./fig/{}_{}.pdf".format(dataset_name, fig_file_save_name), format='pdf')
            plt.show()

topk_single_factor(list_dataset_name,list_method,Ks=[5,10,15,20],decimals=2,eps=1e-6)

