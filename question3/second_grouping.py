import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

data1 = pd.read_excel("../data/data1_new.xlsx")
clusters = [8, 14, 41, 51, 58, 66, 68, 70, 77, 83, 85, 109, 125, 176]
clusters_max = 167  # 当前最大标号，初始化为 167

for cluster in clusters:
    data1_grouped = data1.loc[data1['DBSCAN'] == cluster].copy()

    # 根据成员数量决定聚类数
    k = 3 if len(data1_grouped) > 10 else 2

    # 提取需要聚类的特征
    X_scaled = data1_grouped[["gps_0", "gps_1"]]

    # 执行 KMeans 聚类
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

    # 初始化 new_clusters 数组，存储每个数据点的最终聚类标签
    new_clusters = np.zeros(len(data1_grouped))

    # 对于第一小组，使用原大组的标号
    first_cluster_label = cluster
    first_group_indices = np.where(kmeans.labels_ == 0)[0]
    new_clusters[first_group_indices] = first_cluster_label

    # 对于其他小组，使用 clusters_max + 1 开始重新编号
    for i in range(1, k):
        group_indices = np.where(kmeans.labels_ == i)[0]
        new_clusters[group_indices] = clusters_max + 1
        clusters_max += 1  # 更新 clusters_max

    # 更新原数据的DBSCAN列
    data1.loc[data1['DBSCAN'] == cluster, 'DBSCAN'] = new_clusters

print("二次聚类之后的分组标号的最大值为:", clusters_max)

# 将处理后的数据保存为新的文件
data1.to_excel("../data/data1_DBSCAN_and_Kmeans.xlsx",index=False)
