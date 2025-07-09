import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

data1=pd.read_excel("./data/data1.xls",names=["task_id","gps_0","gps_1","pricing","condition"])

X_scaled=data1[["gps_0","gps_1","condition"]]
# pass
# # 确定最佳K值
# inertias = []
# for k in range(1, 50):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertias.append(kmeans.inertia_)
#
# plt.plot(range(1, 50), inertias, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()
#
# # 计算DBI和CH
# scores = []
# for k in range(2, 50):
#     kmeans = KMeans(n_clusters=k).fit(X_scaled)
#     scores.append({
#         'K': k,
#         'Davies-Bouldin':
#             davies_bouldin_score(X_scaled, kmeans.labels_),  # 越小越好
#         'Calinski-Harabasz':
#             calinski_harabasz_score(X_scaled, kmeans.labels_)  # 越大越好
#     })
#
# df_scores = pd.DataFrame(scores).set_index('K')
# df_scores.to_excel("./data/K-means_scores.xlsx")
# print(df_scores)
k=44
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

# 将聚类结果添加到原始DataFrame
data1['cluster'] = kmeans.labels_


data1.to_csv("./data/data1_with_cluster.csv", index=False,encoding='utf-8')

