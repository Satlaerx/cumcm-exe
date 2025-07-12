import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['figure.dpi'] = 500

data1 = pd.read_excel("../data/data1_with_city.xlsx")

X_scaled = data1[["gps_0", "gps_1"]]

# 确定最佳K值
inertias = []
for k in range(1, 50):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 50), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 计算DBI和CH
scores = []
for k in range(2, 50):
    kmeans = KMeans(n_clusters=k).fit(X_scaled)
    scores.append({
        'K': k,
        'Davies-Bouldin':
            davies_bouldin_score(X_scaled, kmeans.labels_),  # 越小越好
        'Calinski-Harabasz':
            calinski_harabasz_score(X_scaled, kmeans.labels_)  # 越大越好
    })

df_scores = pd.DataFrame(scores).set_index('K')
df_scores.to_excel("../data/K-Means_scores.xlsx")

k = 46
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

# 将聚类结果添加到原始DataFrame
data1['cluster'] = kmeans.labels_
# 按聚类分组计算condition列的平均值(每组计算一次)
cluster_means = data1.groupby('cluster')['condition'].mean()

# 使用映射将每组平均值添加到对应行的rate列
data1['rate'] = data1['cluster'].map(cluster_means)

data1.to_excel("../data/data1_with_city&cluster.xlsx", index=False)

# 计算每个聚类的完成率（condition列的平均值）
cluster_stats = data1.groupby('cluster').agg({
    'gps_0': 'mean',  # 聚类中心纬度
    'gps_1': 'mean',  # 聚类中心经度
    'condition': 'mean'  # 完成率
}).reset_index()

# 重命名列
cluster_stats.columns = ['cluster', 'center_lat', 'center_lng', 'completion_rate']

# 创建结果DataFrame
result_df = pd.DataFrame({
    'cluster_id': range(k),
    'latitude': kmeans.cluster_centers_[:, 0],  # 纬度（gps_0）
    'longitude': kmeans.cluster_centers_[:, 1],  # 经度（gps_1）
    'completion_rate': cluster_stats['completion_rate']
})

# 保存结果
result_df.to_csv("../data/cluster_centers_with_rates.csv",
                 index=False, encoding='utf-8')
print("聚类中心点数据已保存，包含46个点的经纬度和完成率：")
print(result_df.head())

# 可选：验证聚类中心与分组计算的一致性
print("\n验证聚类中心与分组均值的差异：")
print("纬度平均差异:", np.mean(np.abs(result_df['latitude'] - cluster_stats['center_lat'])))
print("经度平均差异:", np.mean(np.abs(result_df['longitude'] - cluster_stats['center_lng'])))
