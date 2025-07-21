import pandas as pd

# 读取聚类结果数据
data1 = pd.read_excel("../data/data1_DBSCAN_Kmeans.xlsx")

# 按照聚类标签升序排序（-1 的散点在前）
data1.sort_values(by="DBSCAN", inplace=True)
data1.reset_index(drop=True, inplace=True)

# 计算每个聚类标签对应的数量
cluster_sizes = data1["DBSCAN"].value_counts().to_dict()

# 添加新列，表示该任务所在聚类的大小
data1["cluster_size"] = data1["DBSCAN"].map(cluster_sizes)

data1.loc[data1["DBSCAN"] == -1, "cluster_size"] = 1

data1.to_excel("../data/data1_sorted.xlsx", index=False)
