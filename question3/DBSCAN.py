from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN

# 图像显示参数
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
task_to_task_distance = pd.read_excel("../data/task_to_task_distance.xlsx")
distances = task_to_task_distance.iloc[:, 1:].to_numpy(dtype=float)
data1 = pd.read_excel("../data/data1_all.xlsx")

# 聚类
db = DBSCAN(eps=1.0, min_samples=2, metric='precomputed')
labels = db.fit_predict(distances)
data1["DBSCAN"] = labels
data1.to_excel("../data/data1_all_with_DBSCAN.xlsx", index=False)

# 构建聚类结果字典
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# 统计聚类大小（将每个散点看作一个独立聚类）
cluster_sizes = []
for label, members in clusters.items():
    if label == -1:
        cluster_sizes.extend([1] * len(members))
    else:
        cluster_sizes.append(len(members))

size_counter = Counter(cluster_sizes)

# 输出个体数量大于 5 的聚类标号
larger_than_5_clusters = [label for label, members in clusters.items() if len(members) > 5]
print("聚类中个体数量大于 5 的类标号:", larger_than_5_clusters)

# 画柱状图
sizes = sorted(size_counter)
counts = [size_counter[s] for s in sizes]

plt.figure(figsize=(10, 6))
bars = plt.bar(sizes, counts, width=0.6, color='skyblue', edgecolor='black')

# 添加柱子顶部的数字标签
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(count),
             ha='center', va='bottom', fontsize=14)

# 设置坐标轴标题和刻度字号
plt.xlabel("聚类大小（任务数量）", fontsize=16, weight='bold')
plt.ylabel("该大小的聚类个数", fontsize=16, weight='bold')
plt.title("不同聚类大小的分布（将每个散点单独作为大小为1的聚类）", fontsize=18, weight='bold')
plt.xticks(ticks=range(1, max(sizes) + 1), fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('DBSCAN.pdf', dpi=500, bbox_inches='tight', facecolor='white')
plt.show()
