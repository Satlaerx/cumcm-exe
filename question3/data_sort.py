import pandas as pd

# 读取聚类结果数据
data1 = pd.read_excel("../data/data1_all_with_DBSCAN.xlsx")
distance1 = pd.read_excel("../data/task_to_member_distance.xlsx")

# 按照聚类标签升序排序（-1 的散点在前）
data1.sort_values(by="DBSCAN", inplace=True)
data1.reset_index(drop=True, inplace=True)

# 计算每个聚类标签对应的数量
cluster_sizes = data1["DBSCAN"].value_counts().to_dict()

# 添加新列，表示该任务所在聚类的大小
data1["cluster_size"] = data1["DBSCAN"].map(cluster_sizes)

data1.loc[data1["DBSCAN"] == -1, "cluster_size"] = 1

# 提取当前排序后的 task_id 顺序
sorted_task_ids = data1["task_id"].tolist()

# 设置 task_id 为索引（以便用 .loc 按顺序重排）
distance1.set_index("task_id", inplace=True)

# 按照新的顺序重排
distance1_sorted = distance1.loc[sorted_task_ids].reset_index()

data1.to_excel("../data/data1_new.xlsx", index=False)
distance1_sorted.to_excel("../data/task_to_member_distance_new.xlsx", index=False)
