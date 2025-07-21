import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# 读取数据
data1 = pd.read_excel("../data/data1_all_with_DBSCAN.xlsx")
clusters_max = int(data1['DBSCAN'].max())  # 初始最大组号
print("初始最大组号:", clusters_max)

# 忽略掉 DBSCAN == -1 的散点
iteration = 0
while True:
    print(f"\n========= 第 {iteration + 1} 轮细分聚类 =========")
    # 统计每组的样本数量，忽略掉 -1 的
    valid_data = data1[data1['DBSCAN'] != -1]
    group_counts = valid_data['DBSCAN'].value_counts()

    # 找出需要进一步聚类的组（样本数量 > 5）
    large_groups = group_counts[group_counts > 5].index.tolist()
    print(f"样本数大于 5 的组数量: {len(large_groups)}")

    if not large_groups:
        print("所有组的样本数都不超过 5，结束迭代。")
        break

    for cluster in large_groups:
        data1_grouped = data1.loc[data1['DBSCAN'] == cluster].copy()

        # 根据成员数量决定聚类数
        k = 3 if len(data1_grouped) > 10 else 2

        X_scaled = data1_grouped[["gps_0", "gps_1"]]
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

        new_clusters = np.zeros(len(data1_grouped))

        # 第一小组继续使用原始编号
        first_group_indices = np.where(kmeans.labels_ == 0)[0]
        new_clusters[first_group_indices] = cluster

        # 其余小组用新编号
        for i in range(1, k):
            group_indices = np.where(kmeans.labels_ == i)[0]
            clusters_max += 1
            new_clusters[group_indices] = clusters_max

        # 更新原数据
        data1.loc[data1['DBSCAN'] == cluster, 'DBSCAN'] = new_clusters

    iteration += 1

# 输出最终聚类编号最大值
print("\n最终聚类分组标号最大值为:", clusters_max)

# 输出最终每组样本数量
final_counts = data1[data1['DBSCAN'] != -1]['DBSCAN'].value_counts().sort_index()
print("\n最终每组样本数量如下：")
print(final_counts)

# 保存文件
data1.to_excel("../data/data1_DBSCAN_KMeans.xlsx", index=False)
