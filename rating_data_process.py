import os
import re
from collections import defaultdict

import pandas as pd
import random

from tqdm import tqdm


def filter_k_core_fast(record, k_core, filtered_column, count_column):
    stat = record[[filtered_column, count_column]] \
        .groupby(filtered_column) \
        .count() \
        .reset_index() \
        .rename(index=str, columns={count_column: 'count'})

    stat = stat[stat['count'] >= k_core]
    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])
    return record
def filter_k_core_precise(df, k_core, source_col, target_col):
    print("filter_k_core_precise begin...")
    # 创建边缘列表
    edges = [(row[source_col], row[target_col]) for _, row in df.iterrows()]
    # 创建初始节点字典
    nodes = {}
    for edge in tqdm(edges):
        source_node, target_node = edge
        if source_node not in nodes:
            nodes[source_node] = {'neighbors': set(), 'degree': 0}
        if target_node not in nodes:
            nodes[target_node] = {'neighbors': set(), 'degree': 0}
        nodes[source_node]['neighbors'].add(target_node)
        nodes[source_node]['degree'] += 1
        nodes[target_node]['neighbors'].add(source_node)
        nodes[target_node]['degree'] += 1

    # 迭代删除度小于k的节点
    del_idx=0
    while True:
        del_idx+=1
        if del_idx%10==0:
            print("del_idx={};\n".format(del_idx))
        else:
            print("del_idx={}; ".format(del_idx))
        # print("node[deg]={}".format( [neb_deg['degree'] for _, neb_deg in nodes.items()] ))
        removed_nodes = []
        for node, data in nodes.items():
            if len(data['neighbors']) < k_core:
                removed_nodes.append(node)
                for neighbor in data['neighbors']:
                    nodes[neighbor]['neighbors'].remove(node)
                    nodes[neighbor]['degree']=len(nodes[neighbor]['neighbors'])


        if not removed_nodes:
            break
        for node in removed_nodes:
            del nodes[node]
    # 过滤数据框
    filtered_rows = [(row[source_col] in nodes) and (row[target_col] in nodes) for _, row in df.iterrows()]
    filtered_df = df[filtered_rows].copy()

    return filtered_df

def filter_k_core_serend(df,k_core,mode="precise"):
    if mode=="precise":
        filtered_df = filter_k_core_precise(df, k_core, 'item_id', 'user_id')
    elif mode=="fast":
        filtered_df = filter_k_core_fast(df, k_core, 'item_id', 'user_id')
        filtered_df = filter_k_core_fast(filtered_df, k_core, 'user_id', 'item_id')
    return filtered_df


def stat_dataset(df,dataName):
    # 统计用户数
    num_users = df["user_id"].nunique()
    # 统计物品数
    num_items = df["item_id"].nunique()
    # 统计交互数
    num_interactions = len(df)
    # 计算数据集密度
    density = num_interactions / (num_users * num_items)
    # 打印统计结果
    print("数据集-{}的统计特征:".format(dataName))
    print("1.用户数：", num_users,end="; ")
    print("2.物品数：", num_items,end="; ")
    print("3.交互数：", num_interactions,end="; ")
    print("4.数据集密度：", density)
    print("\t".join(map(str, [num_users,num_items,density,num_interactions])))


# 定义一个函数来对每个分组进行随机打乱
def shuffle_group(group):
    return group.sample(frac=1, random_state=2023)  # 使用frac=1来保留所有行并进行打乱
def serData2proNE(data_src_path,data_tgt_path):
    # item_list,user_list  org_id remap_id
    # 提取itemInd, itemId列构建items_remapid
    colName = ["userInd", "itemInd", "rating", "timestamp", "user_id", "item_id"]
    # ['userInd','itemInd','rating','timestamp','user_id','item_id']
    all_data, train_data, test_data = pd.read_csv(os.path.join(data_src_path, "rating.csv"), header=0, names=colName), \
                                      pd.read_csv(os.path.join(data_src_path, "rating_train.csv"), header=0,
                                                  names=colName), \
                                      pd.read_csv(os.path.join(data_src_path, "rating_test.csv"), header=0,
                                                  names=colName)
    proNEedges=train_data[['userInd', "itemInd"]]
    proNEedges['itemGid']=proNEedges["itemInd"]+max(proNEedges["userInd"])+1
    proNEedges[['userInd', 'itemGid']].to_csv(os.path.join(data_tgt_path, "proNE_ui_graph.txt"), sep=' ', index=False, header=False)
def serData2Lightgcn(data_src_path,data_tgt_path):
    # 产生lightGCN需要的文件格式
    # item_list,user_list  org_id remap_id
    # 提取itemInd, itemId列构建items_remapid
    colName=["userInd","itemInd","rating","timestamp","user_id","item_id"]
    #['userInd','itemInd','rating','timestamp','user_id','item_id']
    all_data,train_data,test_data=pd.read_csv(os.path.join(data_src_path,"rating.csv"),header=0,names=colName),\
                                  pd.read_csv(os.path.join(data_src_path, "rating_train.csv"),header=0, names=colName),\
                                  pd.read_csv(os.path.join(data_src_path, "rating_test.csv"),header=0,names=colName)

    items_remapid = all_data[['itemInd', 'item_id']].drop_duplicates().set_index('item_id').to_dict()['itemInd']
    # 提取userInd, userId列构建users_remapid
    users_remapid = all_data[['userInd', 'user_id']].drop_duplicates().set_index('user_id').to_dict()['userInd']
    items_remapid=dict(sorted(items_remapid.items(), key=lambda x: x[1]))
    users_remapid=dict(sorted(users_remapid.items()))

    print(items_remapid)
    print(users_remapid)
    #uid itemlist
    # 按照 'user_id' 分组，并将每组的 'item_id' 放入数组

    def groupdf2list(data_df):
        print(data_df)
        grouped_df_agg = data_df.groupby('user_id')['item_id'].agg(list).reset_index()
        # 创建新的 DataFrame，包含 'user_id' 和 'item_id' 数组
        user_itemlist = pd.DataFrame({
            'user_id': grouped_df_agg['user_id'],
            'item_array': grouped_df_agg['item_id']
        })

        print(user_itemlist)
        data_lines=[]
        for userName,itemNameArray in zip(grouped_df_agg['user_id'],grouped_df_agg['item_id']):
            train_items = itemNameArray
            if len(train_items)<1:# or len(test_items)<1:
                print("len(train_items)<1 or len(test_items)<1")
                exit(0)
            # 输出训练集和测试集
            # print("训练集：", train_items)
            # print("测试集：", test_items)
            dataOneLine=[users_remapid[userName]]
            for itemName in train_items:
                dataOneLine.append(items_remapid[itemName])
            data_lines.append(dataOneLine)
        return data_lines

    trainData=groupdf2list(train_data)
    testData=groupdf2list(test_data)

    dataSetDir=data_tgt_path
    fileName=["item_list.txt","user_list.txt","train.txt","test.txt"]
    with open(os.path.join(dataSetDir,fileName[0]),'w') as file:
        file.write("org_id remap_id\n")
        for itemName,itemid in items_remapid.items():
            file.write("{} {}\n".format(itemName,itemid))

    with open(os.path.join(dataSetDir,fileName[1]),'w') as file:
        file.write("org_id remap_id\n")
        for userName,userid in users_remapid.items():
            file.write("{} {}\n".format(userName,userid))
    with open(os.path.join(dataSetDir,fileName[2]),'w') as file:
        for trainOneLine in trainData:
            file.write(' '.join(str(item) for item in trainOneLine)+"\n")
    with open(os.path.join(dataSetDir,fileName[3]),'w') as file:
        for testOneLine in testData:
            file.write(' '.join(str(item) for item in testOneLine)+"\n")

def generateLightgcn(data_df,data_path,sort_way="rand",hasSerendLabel=False):
    #sort_way决定分组中的排序方式，"timestamp" "rand"
    # 产生lightGCN需要的文件格式
    # item_list,user_list  org_id remap_id
    items= sorted(data_df["item_id"].unique())
    users= sorted(data_df["user_id"].unique())
    items_remapid,users_remapid=defaultdict(int),defaultdict(int)
    index=0
    for itemsName in items:
        items_remapid[itemsName]=index
        index+=1
    index = 0
    for userName in users:
        users_remapid[userName] = index
        index += 1
    #uid itemlist
    # 按照 'user_id' 分组，并将每组的 'item_id' 放入数组
    if sort_way=="timestamp":
        grouped_df=data_df.sort_values(by='timestamp').groupby('user_id')
        for group_key, group_data in grouped_df:
            print("sort_way==timestamp:")
            print(group_key, group_data,group_data['timestamp'])
            break
    else:
        # 按照 'userId' 分组，并对每个分组内的数据进行随机打乱
        grouped_df = data_df.groupby('user_id').apply(shuffle_group).reset_index(drop=True).groupby('user_id')
    for group_key, group_data in grouped_df:
        print(group_key, group_data)
        break
    if hasSerendLabel:
        grouped_df_test = data_df[data_df['label'] >0.99].groupby('user_id').apply(
            lambda x: x.loc[x['timestamp'].idxmax()])
        # 重置索引
        test_data = grouped_df_test.reset_index(drop=True)
        # 合并测试数据的 'user_id' 列和 'timestamp' 列作为条件
        condition = test_data[['user_id', 'timestamp']]
        # 根据条件筛选训练数据
        train_data = pd.merge(data_df, condition, on=['user_id', 'timestamp'], how='left', indicator=True)
        train_data = train_data[train_data['_merge'] == 'left_only']
        grouped_df_train = train_data.sort_values(by='timestamp').groupby('user_id')
        grouped_df_test = test_data.sort_values(by='timestamp').groupby('user_id')

        grouped_df_agg_train = grouped_df_train['item_id'].agg(list).reset_index()
        grouped_df_agg_test  = grouped_df_test['item_id'].agg(list).reset_index()

        trainData, testData = [], []
        for userName, itemNameArray in zip(grouped_df_agg_train['user_id'], grouped_df_agg_train['item_id']):
            trainOneLine = [users_remapid[userName]]
            for itemName in itemNameArray:
                trainOneLine.append(items_remapid[itemName])
            trainData.append(trainOneLine)
        for userName, itemNameArray in zip(grouped_df_agg_test['user_id'], grouped_df_agg_test['item_id']):
            testOneLine = [users_remapid[userName]]
            for itemName in itemNameArray:
                testOneLine.append(items_remapid[itemName])
            testData.append(testOneLine)
        print("trainData:\n",trainData)
        print("testData:\n",testData)
    else:
        grouped_df_agg = grouped_df['item_id'].agg(list).reset_index()
        # 创建新的 DataFrame，包含 'user_id' 和 'item_id' 数组
        print(grouped_df_agg)
        user_itemlist = pd.DataFrame({
            'user_id': grouped_df_agg['user_id'],
            'item_array': grouped_df_agg['item_id']
        })
        print(user_itemlist)
        # print(grouped_df['user_id'])
        # print(grouped_df['item_id'])
        trainData,testData=[],[]
        for userName,itemNameArray in zip(grouped_df_agg['user_id'],grouped_df_agg['item_id']):
            # 随机打乱数组
            # random.shuffle(itemNameArray)
            # 计算分割比例
            train_ratio = TRAIN_TEST_RATIO
            # 计算分割点
            split_index = max(1, int(len(itemNameArray) * train_ratio))
            # 分割数组
            train_items = itemNameArray[:split_index]
            test_items = itemNameArray[split_index:]
            if len(train_items)<1:# or len(test_items)<1:
                print("len(train_items)<1 or len(test_items)<1")
                exit(0)
            # 输出训练集和测试集
            # print("训练集：", train_items)
            # print("测试集：", test_items)
            trainOneLine=[users_remapid[userName]]
            for itemName in train_items:
                trainOneLine.append(items_remapid[itemName])
            trainData.append(trainOneLine)
            if len(test_items)>=1:
                testOneLine=[users_remapid[userName]]
                for itemName in test_items:
                    testOneLine.append(items_remapid[itemName])
                testData.append(testOneLine)

    dataSetDir=data_path
    fileName=["item_list.txt","user_list.txt","train.txt","test.txt"]
    with open(os.path.join(dataSetDir,fileName[0]),'w') as file:
        file.write("org_id remap_id\n")
        for itemName in items:
            file.write("{} {}\n".format(itemName,items_remapid[itemName]))

    with open(os.path.join(dataSetDir,fileName[1]),'w') as file:
        file.write("org_id remap_id\n")
        for userName in users:
            file.write("{} {}\n".format(userName,users_remapid[userName]))
    with open(os.path.join(dataSetDir,fileName[2]),'w') as file:
        for trainOneLine in trainData:
            file.write(' '.join(str(item) for item in trainOneLine)+"\n")
    with open(os.path.join(dataSetDir,fileName[3]),'w') as file:
        for testOneLine in testData:
            file.write(' '.join(str(item) for item in testOneLine)+"\n")

    return items_remapid,users_remapid,grouped_df

def generateSerData(items_remapid,users_remapid
                    ,data_df,data_path,hasSerendLabel=False):
    '''
    input:
        data_df   "user_id","item_id","timestamp","rating","label"
    output:
    item.csv    itemInd,date,count,category,itemId
    user.csv    num_item,dot
    rating.csv rating_test.csv rating_train.csv   userInd,itemInd,rating,timestamp,userId,itemId,"label"
    '''
    if hasSerendLabel:
        columns=["userInd","itemInd","rating","timestamp","userId","itemId","serLabel"]
        column_rename = {'user_id': 'userId', 'item_id': 'itemId','label': 'serLabel'}

    else:
        columns=["userInd","itemInd","rating","timestamp","userId","itemId"]
        column_rename = {'user_id': 'userId', 'item_id': 'itemId'}


    rating,rating_train,rating_test=pd.DataFrame(columns=columns),pd.DataFrame(columns=columns),pd.DataFrame(columns=columns)
    groupdf_train_list,groupdf_test_list=[rating_train],[rating_test]
    for userid,groupdf in tqdm(data_df):
        groupdf['itemInd'] = groupdf['item_id'].map(items_remapid)
        groupdf['userInd'] = groupdf['user_id'].map(users_remapid)
        groupdf = groupdf.rename(columns=column_rename)
        # print("groupdf before reindx: \n",groupdf)
        groupdf = groupdf.reindex(columns=columns)
        # print("groupdf after reindx: \n",groupdf)
        # 计算分割比例
        train_test_ratio = TRAIN_TEST_RATIO
        # 计算分割点
        split_index = max(1, int(len(groupdf) * train_test_ratio))
        # 全数据集 rating
        rating = pd.concat([rating, groupdf], ignore_index=True)
        if hasSerendLabel:
            groupdf_test=groupdf[groupdf['serLabel']>0.99].tail(1)#ser item
            groupdf_test_index = groupdf[groupdf['serLabel'] > 0.99].index[-1]

            groupdf_train=groupdf.drop(groupdf_test_index)
            groupdf_train_list.append(groupdf_train)
            if len(groupdf_test)>=1:
                groupdf_test_list.append(groupdf_test)
                # rating_test = pd.concat([rating_test, groupdf_test], ignore_index=True)
            else:
                print("if len(groupdf_test)<1:")
                exit(0)
        else:
            #训练集和测试集分割 0.8比例
            groupdf_train = groupdf.iloc[:split_index,:]
            groupdf_test = groupdf.iloc[split_index:, :]
            groupdf_train_list.append(groupdf_train)
            # rating_train=pd.concat([rating_train, groupdf_train], ignore_index=True)
            if len(groupdf_test)>=1:
                groupdf_test= groupdf.iloc[split_index:,:]
                groupdf_test_list.append(groupdf_test)
                # rating_test = pd.concat([rating_test, groupdf_test], ignore_index=True)
            else:
                print("if len(groupdf_test)<1:")
                exit(0)
    '''
    从训练集中产生item.csv    itemInd,date,count,category,itemId
    date:在论文中是release time, 猜测为min(time of item)
    count: 应该是交互用户数
    '''
    rating_train=pd.concat(groupdf_train_list)
    rating_test=pd.concat(groupdf_test_list)
    print(rating_test)
    # 使用 groupby 和聚合函数进行统计
    df_agg = rating_train.groupby('itemId').agg({'timestamp': 'min', 'itemId': 'count'}).rename(
        columns={'timestamp': 'date', 'itemId': 'count'})
    # 重置索引，将 itemId 列变为新的列
    df_agg = df_agg.reset_index()

    # 选择需要的列
    item_df = df_agg[['itemId', 'date', 'count']]
    item_df['itemInd']=item_df['itemId'].map(items_remapid)
    item_df_train = item_df.reindex(columns=['itemInd','date','count','itemId'])
    print("item_df_train: ",item_df_train)
    print()
    item_df_all=pd.DataFrame(sorted(rating['itemId'].unique()),columns=['itemId'])
    item_df_all['itemInd']=item_df_all['itemId'].map(items_remapid)
    # 合并DataFrame
    item_df_all = pd.merge(item_df_all, item_df_train, on=['itemInd', 'itemId'], how='left')
    # 填补空缺的'date'属性为全局最大的日期
    max_date = item_df_train['date'].max()
    item_df_all['date'].fillna(max_date, inplace=True)
    # 填补缺失的'count'属性为0
    item_df_all['count'].fillna(0, inplace=True)
    item_df_all = item_df_all.reindex(columns=['itemInd','date','count','itemId'])
    #保存数据集文件
    dataSetDir = data_path
    fileName = ["item.csv", "rating.csv", "rating_train.csv", "rating_test.csv"]
    item_df_all.to_csv(os.path.join(dataSetDir, fileName[0]),index=False)
    print(rating[['userInd','timestamp']].head(20))
    rating.to_csv(os.path.join(dataSetDir, fileName[1]),index=False)
    rating_train.to_csv(os.path.join(dataSetDir, fileName[2]), index=False)
    rating_test.to_csv(os.path.join(dataSetDir, fileName[3]), index=False)

def parse_dataset(file_path):
    with open(file_path, 'r') as file:
        dataset = file.read()

    entries = re.split(r'\n\n', dataset)
    parsed_entries = []
    errNum=0
    errFlag=False
    for entry in tqdm(entries):
        parsed_entry = {}
        lines = entry.split('\n')
        for line in lines:
            key_value = line.split(': ',1)
            if len(key_value)!=2:
                errNum+=1
                errFlag=True
                break
            key,value=key_value
            key = key.split('/')[1]
            parsed_entry[key] = value
        if errFlag:
            errFlag = False
        else:
            parsed_entries.append(parsed_entry)
    print("errNum={}".format(errNum))
    return parsed_entries


def convert_to_dataframe(parsed_entries):
    dataframe = []

    for entry in tqdm(parsed_entries):
        product_id = entry['productId']
        user_id = entry['userId']
        score = float(entry['score'])
        time = int(entry['time'])

        dataframe.append({
            'item_id': product_id,
            'user_id': user_id,
            'rating': score,
            'timestamp': time
        })
    df = pd.DataFrame(dataframe)

    return df

TRAIN_TEST_RATIO=0.8
K_CORE=10
if __name__ == '__main__':

    dataType="home"#["ser_bk","ser_mv","clothing","electronics","home","kindle"]



    if dataType=="ser_bk":
        dataSetDir,dataOrgFile,dataOrgHasHeader,dataDfHeadCol=\
            "./data/{}".format(dataType),"SerenLens_Books.csv",0, \
            ["user_id","item_id","timestamp","review","rating","label"]
    elif dataType=="ser_mv":
        dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "SerenLens_Movies.csv",0, \
            ["user_id","item_id","timestamp","review","rating","label"]
    elif dataType=="clothing":
            dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "ratings_Clothing_Shoes_and_Jewelry.csv",None, \
            ["user_id", "item_id", "rating", "timestamp"]
    elif dataType=="sport":
            dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "ratings_Sports_and_Outdoors.csv",None, \
            ["user_id", "item_id", "rating", "timestamp"]
    elif dataType=="cloth_5core_onlyRate":
            dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
                "./data/{}".format(dataType), "Clothing_Shoes_and_Jewelry.csv",None, \
                ["item_id", "user_id", "rating", "timestamp"]
    elif dataType=="home_5core_onlyRate":
            dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
                "./data/{}".format(dataType), "Home_and_Kitchen.csv",None, \
                ["item_id", "user_id", "rating", "timestamp"]
    elif dataType=="home_5core_onlyRate2013":
            dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
                "./data/{}".format(dataType), "Home_and_Kitchen.csv",0, \
                ["item_id", "user_id", "rating", "timestamp"]
            orgReviewDataFile="Home_&_Kitchen.txt"
    elif dataType=="elec_5core_onlyRate":
            dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
                "./data/{}".format(dataType), "Electronics.csv",None, \
                ["item_id", "user_id", "rating", "timestamp"]
    elif dataType=="kindle_5core_onlyRate":
            dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
                "./data/{}".format(dataType), "Kindle_Store.csv",None, \
                ["item_id", "user_id", "rating", "timestamp"]
    elif dataType == "sport_5core_onlyRate":
        dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "Sports_and_Outdoors.csv", None, \
            ["item_id", "user_id", "rating", "timestamp"]


    elif dataType=="electronics":
        dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "ratings_Electronics.csv",None, \
             ["user_id", "item_id", "rating", "timestamp"]
    elif dataType=="home":
        dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "ratings_Home_and_Kitchen.csv",None, \
             ["user_id", "item_id", "rating", "timestamp"]
    elif dataType=="kindle":
        dataSetDir, dataOrgFile,dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "ratings_Kindle_Store.csv",None, \
                                                  ["user_id", "item_id", "rating", "timestamp"]
    elif dataType=="beauty" or dataType=="mlls" or dataType=="mlls_cp" or dataType=="tool"or dataType=="tool_cp":
        dataSetDir, dataOrgFile, dataOrgHasHeader, dataDfHeadCol = \
            "./data/{}".format(dataType), "rating.csv", 0, \
            ['userInd','itemInd','rating','timestamp','user_id','item_id']
    # serData2proNE(dataSetDir,dataSetDir)
    # serData2Lightgcn(dataSetDir,dataSetDir)
    # exit(0)
    #["ser_bk",""ser_mv","clothing"]
    #user_id,item_id,timestamp,review,rating,label

    if  os.path.exists(os.path.join(dataSetDir,"rating_10core.csv")) and False:
        data_10core_df=pd.read_csv(os.path.join(dataSetDir,"rating_10core.csv"))
    else:
        if dataType.endswith("2013"):
            if not os.path.exists(os.path.join(dataSetDir,dataOrgFile)):
                parsed_entries = parse_dataset(os.path.join(dataSetDir,orgReviewDataFile))
                dataframe = convert_to_dataframe(parsed_entries)
                print(dataframe.head())
                dataframe.to_csv(os.path.join(dataSetDir,dataOrgFile),index=False)

        data_df=pd.read_csv(os.path.join(dataSetDir,dataOrgFile),usecols=dataDfHeadCol,header=dataOrgHasHeader,names=dataDfHeadCol)
        if 'review' in dataDfHeadCol:
            data_df = data_df.drop('review', axis=1)
            data_df['timestamp'] = pd.to_numeric(data_df['timestamp'], errors='coerce')
            data_df['timestamp'] = data_df['timestamp'].astype('Int64')
            data_df = data_df.dropna(subset=['timestamp'])

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(data_df.head(2))
        print(data_df.columns)
        print("the length of {} before k core filter: ".format(dataType),len(data_df))
        stat_dataset(data_df,dataType)
        # exit(0)
        data_10core_df=filter_k_core_serend(data_df,k_core=K_CORE,mode="fast")
        data_10core_df.to_csv(os.path.join(dataSetDir, "rating_10core.csv"), index=False)
    print("the length of {} after k core filter: ".format(dataType),len(data_10core_df))
    stat_dataset(data_10core_df,"{}_10core".format(dataType))
    data_10core_df['item_id'] = data_10core_df['item_id'].astype(str)
    data_10core_df['user_id'] = data_10core_df['user_id'].astype(str)
    print(data_10core_df.head())#item_id  user_id  rating   timestamp
    exit(0)
    # 假设df是包含交互记录的DataFrame，列名为item_id、user_id、rating和timestamp（这里假设为df）
    # 统计每个用户的交互数
    user_interaction_counts = data_10core_df.groupby('user_id').size().reset_index(name='interaction_count')
    # 打印每个用户的交互数及其用户ID
    for index, row in user_interaction_counts.iterrows():
        if row['interaction_count']<K_CORE:
            uid = row['user_id']
            count = row['interaction_count']
            print(f"用户ID: {uid}, 交互数: {count}")
    item_user_counts = data_10core_df.groupby('item_id')['user_id'].nunique()
    # 统计每个用户交互过的物品数
    user_item_counts = data_10core_df.groupby('user_id')['item_id'].nunique()
    # 打印交互数小于10的物品ID
    print("交互数小于10的物品ID：")
    print(item_user_counts[item_user_counts < K_CORE].index.tolist())
    # 打印交互数小于10的用户ID
    print("交互数小于10的用户ID：")
    print(user_item_counts[user_item_counts < K_CORE].index.tolist())
    data_10core_path=dataSetDir
    print("generateLightgcn begin...")
    items_remapid,users_remapid,grouped_df=generateLightgcn(data_10core_df,data_10core_path,sort_way='timestamp',hasSerendLabel=False)
    print("generateSerData begin...")
    generateSerData(items_remapid, users_remapid,grouped_df,data_10core_path,hasSerendLabel=False)
    print("rating_process_end")
    exit(0)


