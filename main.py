import os

import recommend
import utils
import exp_analysis
import numpy as np
def method2namelist(methodlist,method2nameDict,method2name_all):
    methods=method2nameDict.keys()
    methodsName=[]
    for method in methodlist:
        methodName=method
        for m in methods:
            if m==methodName:
                methodName = method2name_all[m]
                break
            if m in methodName:
                methodName = methodName.replace(m, method2nameDict[m])
        methodsName.append(methodName)
    return methodsName
if __name__ == "__main__":
    print("2023年10月24日")
    # list_dataset_name = ["home", "electronics","clothing"]#,"ser_bk","ser_mv"]
    # list_dataset_name = ["electronics"]
    # list_dataset_name = ["electronics"]
    list_dataset_name = ["mlls", "tool", "beauty","kindle","sport","home","electronics","clothing","ser_bk","ser_mv"]
    # list_dataset_name = ["electronics","clothing"]#,"ser_bk"]
    list_dataset_name = ["kindle","clothing","electronics","ser_bk","home","ser_mv"]
    list_dataset_name = [ "ser_bk","ser_mv"]
    list_dataset_name = [ "kindle","clothing","electronics","ser_bk","sport","tool", "beauty","home","ser_mv"]
    list_dataset_name = [ "kindle","electronics","home","clothing","tool", "beauty","sport","ser_bk","ser_mv"]


    # list_dataset_name = [ "kindle"]

    # emd_norm = False
    # needMinMax = True
    # linearScale = False
    # sigmoid = False
    # ex_outliers = True
    # exp_analysis.StatiSimiEmbedCos(list_dataset_name, useData="train", block_size=999999)
    # exp_analysis.StatiSimiEmbedCos(list_dataset_name, useData="test", block_size=999999)
    # exp_analysis.StatiSimiEmbedCos(list_dataset_name, useData="all", block_size=999999)

    # exp_analysis.StatiSimiEmbed(list_dataset_name,useData="train",needMinMax=needMinMax,linearScale=linearScale,sigmoid=sigmoid,
    #                             ex_outliers=ex_outliers,emd_norm=emd_norm)
    # exp_analysis.StatiSimiEmbed(list_dataset_name, useData="test", needMinMax=needMinMax, linearScale=linearScale,
    #                             sigmoid=sigmoid,
    #                             ex_outliers=ex_outliers, emd_norm=emd_norm)
    # exp_analysis.StatiSimiEmbed(list_dataset_name, useData="all", needMinMax=needMinMax, linearScale=linearScale,
    #                             sigmoid=sigmoid,
    #                             ex_outliers=ex_outliers, emd_norm=emd_norm)
    exp_analysis.interactions_num_freq(list_dataset_name)
    exit(0)
    # list_method = ["div"]#rec Strategy
    list_method = ["rand", "nov", "pop", "qua", "ela", "acc", "dif", "div"]#rec Strategy
    list_method_name = ["rand", "nov", "unpop", "qua", "ela", "acc", "dif", "div"]#rec Strategy
    list_method_name_all=["random", "novelty", "unpopularity", "high quality", "elasticity", "relevance", "difference", "diversity"]
    method2name={method:name for method,name in zip(list_method,list_method_name)}
    method2name_all={method:name for method,name in zip(list_method,list_method_name_all)}
    # list_method=[]
    # merge_factor_rec_strategy = None
    # merge_factor_wei_list = None
    merge_factor_rec_strategy_list=[["acc", "dif"],["acc", "rand"],["acc", "nov"],["acc", "pop"],["acc", "qua"],["acc", "div"]]
    # merge_factor_rec_strategy_list=[["acc", "dif"],["acc", "rand"],["acc", "nov"],["acc", "pop"],["acc", "qua"]]
    # merge_factor_rec_strategy_list=[["acc", "qua"],["acc", "div"]]
    merge_factor_rec_strategy_list=[["acc", "dif"]]
    for merge_factor_rec_strategy in merge_factor_rec_strategy_list:
        merge_factor_wei_list = [[round(i/10, 1), round(1-i/10, 1)] for i in range(1,10)]#[0,1] 10 avg itv
        list_method_merge=list_method.copy()
        for wei in merge_factor_wei_list:
            name = ""
            for i in range(len(merge_factor_rec_strategy)):
                name += f"{wei[i]}{merge_factor_rec_strategy[i]}_"
            name = name[:-1]  # 去除最后一个下划线
            list_method_merge.append(name)
        list_method_name=method2namelist(list_method_merge, method2name,method2name_all)
        list_metric=["nov","unpop","qua","acc","diff","div","ser1","ser2"]#,"hr_s","ndcg_s"]#rec Metric
        print(list_metric)
        list_dataset_name = list_dataset_name[:]
        list_seed = [777, 7777, 77777, 73, 79, 83, 89, 97, 101, 103][:5]
        # list_seed=[777]
        list_method_merge=list_method
        list_method_name=method2namelist(list_method_merge, method2name,method2name_all)
        merge_factor_rec_strategy=None
        merge_factor_wei_list=None

        # exp_analysis.ImpactsOfFactorsOnSerendipity(list_dataset_name, list_method_name,
        #                                            merge_factor_rec_strategy=merge_factor_rec_strategy, )
        # exit(0)

        # recommend.recommend(list_dataset_name, list_seed,merge_factor_rec_strategy,merge_factor_wei_list)
        # utils.evaluate(list_dataset_name, list_seed, list_method_merge,compDisentDeg=False,
        #                merge_factor_rec_strategy=merge_factor_rec_strategy,mem_enough=False)
        # exp_analysis.topk_single_factor(list_dataset_name,[method for method in list_method_name if method != "elasticity"],Ks=[5,10,15,20],decimals=2,eps=1e-6)
        # exp_analysis.plot_res_mat(list_dataset_name,[method for method in list_method_name if method != "elasticity"],list_metric,merge_factor_rec_strategy=merge_factor_rec_strategy)
        # exp_analysis.plot_disent_mat(list_dataset_name,[name for name in list_method_name if name != "elasticity"],merge_factor_rec_strategy=merge_factor_rec_strategy)
    # exp_analysis.displayMetricBounds(list_dataset_name,list_method)
    # exp_analysis.StatiSimiEmbed(list_dataset_name,useData="train")
    # exp_analysis.StatiSimiEmbed(list_dataset_name,useData="test")
    # exit(0)