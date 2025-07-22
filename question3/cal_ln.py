import pandas as pd
import numpy as np
from itertools import combinations

# 读取数据
data1 = pd.read_excel("../data/data1_sorted.xlsx")

# 第一步：提取 cluster_size 为 2~5 的数据
df_2 = data1[data1["cluster_size"] == 2].copy()
df_3 = data1[data1["cluster_size"] == 3].copy()
df_4 = data1[data1["cluster_size"] == 4].copy()
df_5 = data1[data1["cluster_size"] == 5].copy()


# 第二步：插入 haversine 距离函数
def haversine_distance(lat1, lon1, lat2, lon2):
    phi1, lambda1 = np.radians(float(lat1)), np.radians(float(lon1))
    phi2, lambda2 = np.radians(float(lat2)), np.radians(float(lon2))

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    term1 = np.sin(delta_phi / 2) ** 2
    term2 = np.cos(phi1) * np.cos(phi2)
    term3 = np.sin(delta_lambda / 2) ** 2
    a = term1 + term2 * term3

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c  # 地球半径为 6371 公里


# 第三步：编写一个函数，处理任意 df，计算其组内距离平均值
def compute_avg_group_distance(df, group_size):
    distances = []
    n_rows = len(df)

    # 每 group_size 行为一组（已排序）
    for i in range(0, n_rows, group_size):
        group = df.iloc[i:i + group_size]
        coords = list(zip(group["gps_0"], group["gps_1"]))  # [(lat1, lon1), (lat2, lon2), ...]

        # 计算组内所有点之间的两两距离
        for (lat1, lon1), (lat2, lon2) in combinations(coords, 2):
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)

    return np.mean(distances)


# 第四步：对每个 df 计算平均距离
avg_dist_2 = compute_avg_group_distance(df_2, 2)
avg_dist_3 = compute_avg_group_distance(df_3, 3)
avg_dist_4 = compute_avg_group_distance(df_4, 4)
avg_dist_5 = compute_avg_group_distance(df_5, 5)

# 打印结果
print("cluster_size = 2 的平均组内距离：", avg_dist_2)
print("cluster_size = 3 的平均组内距离：", avg_dist_3)
print("cluster_size = 4 的平均组内距离：", avg_dist_4)
print("cluster_size = 5 的平均组内距离：", avg_dist_5)
