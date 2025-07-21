import pandas as pd
import numpy as np

data1 = pd.read_excel("../data/data1_merged.xlsx")
data2 = pd.read_excel("../data/data2_with_city.xlsx")


def haversine_distance(lat1, lon1, lat2, lon2):
    # 使用 numpy 的弧度和三角函数
    phi1, lambda1 = np.radians(float(lat1)), np.radians(float(lon1))
    phi2, lambda2 = np.radians(float(lat2)), np.radians(float(lon2))

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    term1 = np.sin(delta_phi / 2) ** 2
    term2 = np.cos(phi1) * np.cos(phi2)
    term3 = np.sin(delta_lambda / 2) ** 2
    a = term1 + term2 * term3

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


# 1. 计算任务和成员之间的距离矩阵（pandas.DataFrame类型存储）
# 提取任务和成员的坐标
tasks = data1[['task_id', 'gps_0', 'gps_1']].copy()
members = data2[['mem_id', 'gps_0', 'gps_1']].copy()

# 设置索引以便于匹配
tasks.set_index('task_id', inplace=True)
members.set_index('mem_id', inplace=True)

# 创建空的距离矩阵
distance1 = pd.DataFrame(
    index=tasks.index,
    columns=members.index,
    dtype=float
)

# 填充距离矩阵
for task_id, task_row in tasks.iterrows():
    for mem_id, mem_row in members.iterrows():
        # 提取经纬度
        task_lat, task_lon = task_row['gps_0'], task_row['gps_1']
        mem_lat, mem_lon = mem_row['gps_0'], mem_row['gps_1']

        # 计算距离
        distance = haversine_distance(task_lat, task_lon, mem_lat, mem_lon)

        # 填充矩阵
        distance1.at[task_id, mem_id] = distance

distance1.to_excel("../data/q3_task_to_member_distances.xlsx")

# 2. 计算任务与任务之间的距离矩阵
# 创建空的距离矩阵
distance2 = pd.DataFrame(
    index=tasks.index,
    columns=tasks.index,
    dtype=float
)

# 填充任务与任务之间的距离矩阵
for task_id1, task_row1 in tasks.iterrows():
    for task_id2, task_row2 in tasks.iterrows():
        # 如果是同一个任务，距离为0
        if task_id1 == task_id2:
            distance2.at[task_id1, task_id2] = 0
            continue

        # 提取经纬度
        task1_lat, task1_lon = task_row1['gps_0'], task_row1['gps_1']
        task2_lat, task2_lon = task_row2['gps_0'], task_row2['gps_1']

        # 计算距离
        distance = haversine_distance(task1_lat, task1_lon, task2_lat, task2_lon)

        # 填充矩阵
        distance2.at[task_id1, task_id2] = distance

# 以同样的格式输出到Excel文件
distance2.to_excel("../data/q3_task_to_task_distances.xlsx")

# 3. 计算只考虑5km内的任务到成员的距离矩阵
# 创建空的距离矩阵
distance3 = pd.DataFrame(
    index=tasks.index,
    columns=members.index,
    dtype=float
)

# 填充距离矩阵
for task_id, task_row in tasks.iterrows():
    for mem_id, mem_row in members.iterrows():
        # 提取经纬度
        task_lat, task_lon = task_row['gps_0'], task_row['gps_1']
        mem_lat, mem_lon = mem_row['gps_0'], mem_row['gps_1']

        # 计算距离
        distance = haversine_distance(task_lat, task_lon, mem_lat, mem_lon)

        # 距离大于5km时填充-1，否则填充实际距离
        if distance > 5:
            distance3.at[task_id, mem_id] = -1
        else:
            distance3.at[task_id, mem_id] = distance

distance3.to_excel("../data/q3_task_to_member_distance_5km.xlsx")
