import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


fontd = {'fontsize': 16}
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', '.', '*', 'O']
markers = ['.', 'o', 'v', '^', '>', '<', '*', 'D']
edgecolors = ['k', 'g', 'b', 'c', 'darkorange', 'm', 'r', 'y']
barcolors1 = ['#A1A9D0', '#F0988C', '#B883D4', '#9E9E9E', '#CFEAF1', '#C4A5DE', '#F6CAE5', '#96CCCB']
barcolors2 = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#96CCCB']
barcolors3 = ['#934B43', '#F1D77E', '#9394E7', '#D76364', '#B1CE46', '#5F97D2', '#EF7A6D', '#63E398', '#9DC3E7']
barcolors4 = ['#14517C', '#2F7FC1', '#E7EFFA', '#96C37D', '#F3D266', '#D8383A', '#F7E1ED', '#F8F3F9', '#C497B2', '#A9B8C6']
barcolors5 = ['#00a8e1', '#99cc00', '#e30039', '#fcd300', '#800080', '#00994e',  '#ff6600', '#808000', '#db00c2', '#008080', '#0000ff', '#c8cc00']
barcolors6 = ['#3b6291', '#943c39', '#779043', '#624c7c', '#388498', '#bf7334', '#3f6899', '#9c403d', '#7d9847', '#675083', '#3b8ba1', '#c97937']
barcolors7 = ['#194f97', '#555555', '#bd6b08', '#00686b', '#c82d31', '#625ba1', '#898989', '#9c9800', '#007f54', '#a195c5', '#103667', '#f19272']
barcolors = ['#0e72cc', '#6ca30f', '#f59311', '#fa4343', '#A9B8C6', '#16afcc', '#db00c2', '#844bb3', '#d12a60', '#5f6694']
barcolors9 = ['#3682be', '#45a776', '#f05326', '#eed777', '#334f65', '#b3974e', '#38cb7d', '#ddae33', '#844bb3', '#93c555', '#5f6694', '#df3881']
barcolors10 = ['#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac', '#95a2ff']


def gen_rank_of_factors(root_dir="./results"):
    list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]
    indexes = ['random', 'novelty', 'unpopularity', 'high quality', 'elasticity', 'accuracy',
               'difference', 'diversity']
    columns = ['nov', 'unpop', 'qua', 'acc', 'dif', 'div', 'ser1', 'ser2']
    s1 = {}
    s2 = {}
    sc = {}

    for dn in list_dataset_name:
        df = pd.read_csv(os.path.join(root_dir, f"{dn}_cmp.csv"), index_col=0)
        s1[dn] = df['ser1'].values.argsort().argsort()
        s2[dn] = df['ser2'].values.argsort().argsort()
        sc[dn] = (s1[dn]+s2[dn]).argsort().argsort()#(0.5 * df['ser1'].values + 0.5 * df['ser2'].values).argsort().argsort()  # 

    df1 = len(indexes) - 1 - pd.DataFrame(data=s1, columns=list_dataset_name, index=indexes)
    df2 = len(indexes) - 1 - pd.DataFrame(data=s2, columns=list_dataset_name, index=indexes)
    dfc = len(indexes) - 1 - pd.DataFrame(data=sc, columns=list_dataset_name, index=indexes)

    #df1.to_csv(os.path.join(root_dir, "ser1_rank.csv"))
    #df2.to_csv(os.path.join(root_dir, "ser2_rank.csv"))
    dfc.to_csv(os.path.join(root_dir, "ser-combined_rank.csv"))
    

def gen_score_of_datasets(res_dir="res", save_dir="./results/"):
    """
    将各个数据集在不同策略上的结果保存成 csv 文件
    """
    list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]
    indexes = ['random', 'novelty', 'unpopularity', 'high quality', 'elasticity', 'accuracy',
               'difference', 'diversity']
    columns = ['nov', 'unpop', 'qua', 'acc', 'dif', 'div', 'ser1', 'ser2']
    base = './data'
    rank_ser1 = dict()
    rank_ser2 = dict()
    rank_comb = dict()

    for dn in list_dataset_name:
        path = os.path.join(base, dn, res_dir, 'mean.npy')
        if not os.path.exists(path):
            print(f"{path} doesn't exist...")
            continue
        mean = np.load(path)
        df = pd.DataFrame(data=mean, columns=columns, index=indexes)
        save_path = os.path.join(save_dir, f"{dn}_cmp.csv")
        df.to_csv(save_path)
        rank_ser1[dn] = df['ser1'].values.argsort().argsort()
        rank_ser2[dn] = df['ser2'].values.argsort().argsort()
        rank_comb[dn] = ((rank_ser1[dn] + rank_ser2[dn]) / 2).argsort().argsort()

    df_ser1 = len(indexes) - 1 - pd.DataFrame(data=rank_ser1, columns=list_dataset_name, index=indexes)
    df_ser1.to_csv(os.path.join(save_dir, 'ser1_rank.csv'))
    df_ser2 = len(indexes) - 1 - pd.DataFrame(data=rank_ser2, columns=list_dataset_name, index=indexes)
    df_ser2.to_csv(os.path.join(save_dir, 'ser2_rank.csv'))
    df_ser_comb = len(indexes) - 1 - pd.DataFrame(data=rank_comb, columns=list_dataset_name, index=indexes)
    df_ser_comb.to_csv(os.path.join(save_dir, 'ser-combined_rank.csv'))


def plot_stack_bar(path, ax=None, show=True, legend=True, show_x_label=True, show_y_label=True, use_hatch=True, **kwargs):
    """
    可视化不同策略在不同指标下的得分。
    """
    df = pd.read_csv(path, index_col=0)
    # df = df.T
    interval = 1.5
    x = np.arange(0, interval*df.shape[1], interval)
    indexes = df.index
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    w = 0.15
    for i in range(df.shape[0]):
        ax.bar(x+i*w, df.iloc[i, :], label=indexes[i] if indexes[i] != 'accuracy' else 'relevance', width=w, color=barcolors[i], ec=edgecolors[0], hatch=hatches[i] if use_hatch else None)

    if show_x_label:
        ax.set_xticks(x+w*(df.shape[0]//2))
        columns = df.columns.tolist()
        if 'acc' in columns:
            columns[columns.index('acc')] = 'rel'
        ax.set_xticklabels(columns, rotation=30, **kwargs)

    if not show_y_label:
        ax.axes.yaxis.set_visible(False)

    if legend:
        if fig:
            fig.legend(bbox_to_anchor=(0.95, 0.88) , loc="upper right", borderaxespad=0)
        else:
            ax.legend()
    if show:
        plt.show()


def plot_all_datasets(root_dir="./results"):
    """
    把所有数据集的结果画在一张图上
    """
    datasets = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]
    nrow = 3
    ncol = len(datasets) // nrow
    fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=False)
    axes = axes.flatten()


    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 10,}

    idxs = 'abcdefghi'
    for i, dn in enumerate(datasets):
        path = os.path.join(root_dir, f"{dn}_cmp.csv")
        show_x_label = True #if i > 5 else False
        show_y_label = True #if i % 3 == 0 else False
        plot_stack_bar(path, axes[i], show=False, legend=False, show_x_label=show_x_label, show_y_label=show_y_label, use_hatch=False, font=font)
        axes[i].set_title(f"({idxs[i]}) {dn}", y=-0.35, fontdict=fontd)

    labels = ['random', 'novelty', 'unpopularity', 'high quality', 'elasticity', 'relevance', 'difference', 'diversity']#['nov', 'unpop', 'qua', 'acc', 'dif', 'div', 'ser1', 'ser2']
    fig.legend(labels=labels)  # , bbox_to_anchor=(1, 0))  # , loc=3, borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_stack_box(path, ax=None, use_hatch=True):
    df = pd.read_csv(path, index_col=0)
    if ax is None:
        fig, ax = plt.subplots()

    x = np.arange(df.shape[1])
    y = np.array([1]*len(x))
    indexes = df.index
    width = 0.5

    for i in range(df.shape[0]):
        bottom = 8-df.iloc[i, :] + 0.5
        ax.bar(x, y, width=width, bottom=bottom, label=indexes[i] if indexes[i] != 'accuracy' else 'relevance', color=barcolors[i], ec=edgecolors[0], hatch=hatches[i] if use_hatch else None)

    ax.set_xticks(x)
    yticklabels = [f"No.{i}" for i in range(9, 0, -1)]
    ax.set_yticklabels(yticklabels)
    columns = df.columns
    if 'acc' in columns:
        columns[columns.index('acc')] = 'rel'
    ax.set_xticklabels(df.columns, rotation=30)
    # ax.legend()
    # plt.show()


def plot_all_metrics(root_dir="./results"):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    axes = axes.flatten()

    titles = ['ser1', 'ser2', 'ser-combined']
    idxs = 'abc'
    for i in range(len(titles)):
        path = os.path.join(root_dir, f"{titles[i]}_rank.csv")
        plot_stack_box(path, axes[i], use_hatch=False)
        axes[i].set_title(f"({idxs[i]}) {titles[i]}", y=-0.13, fontdict=fontd)
        if i > 0:
            axes[i].axes.yaxis.set_visible(False)

    axes[0].set_ylabel("Rank of Strategies", fontdict=fontd)
    labels = ['random', 'novelty', 'unpopularity', 'high quality', 'elasticity', 'relevance', 'difference', 'diversity']
    fig.legend(labels, prop={'size': 12})
    plt.tight_layout()
    plt.show()


def plot_topK(dataset_name, list_topK=[5, 10, 15, 20], root_dir="./results"):
    x = np.arange(len(list_topK))
    xtls = sorted(list_topK)
    metrics=['ser1', 'ser2']
    fig, axes = plt.subplots(1, len(metrics))
    axes = axes.flatten()
    idxs = 'abcdefghijk'

    for idx, metric in enumerate(metrics):
        df = pd.DataFrame()
        for K in sorted(list_topK):
            path = os.path.join(root_dir, f"top{K}" if K != 20 else "old", f"{dataset_name}_cmp.csv")
            tmp = pd.read_csv(path, index_col=0)
            df[f"top{K}"] = tmp[metric]
        for i, index in enumerate(df.index):
            axes[idx].plot(x, df.loc[index, :], label=index if index != 'accuracy' else 'relevance', color=barcolors[i], marker=markers[i])
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(xtls, fontdict={"fontsize": 18})
            axes[idx].set_xlabel('k', fontdict={'fontsize': 18})
            axes[idx].set_title(f"({idxs[idx]}) {metric}", y=-.15, fontdict={'fontsize': 20})
            axes[idx].tick_params(axis='both', labelsize=18)#set_yticks(ticks=axes[idx].get_yticks(), size=15)

    indexes = df.index.tolist()
    if 'accuracy' in indexes:
        indexes[indexes.index('accuracy')] = 'relevance'
    fig.legend(indexes, prop={'size': 15})
    plt.show()


def plot_different_embedding_method(dataset_name, K=20, root_dir="./results"):
    emb_methods = ['Word2Vec', 'LightGCN']
    metrics = ['ser1', 'ser2']

    interval = 1.5
    w = 0.15
    x = np.arange(0, interval*len(metrics), interval)
    
    fig, axes = plt.subplots(1, len(emb_methods))
    axes = axes.flatten()

    idxs = 'abcedfghi'

    for midx, metric in enumerate(metrics):
        df = pd.DataFrame()
        for em in emb_methods:
            if em == 'LightGCN':
                path = os.path.join(root_dir, f"top{K}", f"{dataset_name}_cmp.csv")
            else:
                path = os.path.join(root_dir, f"top{K}_w2v_2", f"{dataset_name}_cmp.csv")
            tmp = pd.read_csv(path, index_col=0)
            df[em] = tmp[metric]
        indexes = df.index
        for i in range(df.shape[0]):
            axes[midx].bar(x+i*w, df.iloc[i, :], label=indexes[i], width=w, color=barcolors[i], ec=edgecolors[0])

        axes[midx].set_xticks(x+w*(df.shape[0]//2))
        axes[midx].set_xticklabels(df.columns, rotation=0, fontdict={"fontsize": 12})
        axes[midx].set_title(f"({idxs[midx]}) {metric}", y=-0.1, fontdict=fontd)

    indexes = df.index.tolist()
    if 'accuracy' in indexes:
        indexes[indexes.index('accuracy')] = 'relevance'
    fig.legend(indexes, prop={'size': 15})

    plt.show()


if __name__ == "__main__":
    K = 20
    # 生成数据
    save_dir = f"./results/top{K}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f"K={K}\tsave_dir={save_dir}")
    # gen_score_of_datasets(res_dir=f"res_corrected", save_dir=save_dir)
    # path = "./results/old/mlls_cmp.csv"

    # 可视化策略在指标上的得分
    # plot_stack_bar(path, use_hatch=False, show_y_label=False)
    root_dir = f"./results/old"
    #gen_rank_of_factors(root_dir)
    #plot_all_datasets(root_dir)

    # 可视化策略在各个指标上的排名
# 	# path = "./results/ser-combined_rank.csv"
# 	# plot_stack_box(path)
    #plot_all_metrics(root_dir)

    # 展示 topK 的影响
    dataset_name = "mlls"
    plot_topK(dataset_name)

    # 展示嵌入方法的影响
    #plot_different_embedding_method(dataset_name)
