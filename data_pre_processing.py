import pandas as pd
import numpy as np

data1=pd.read_excel("./data/data1.xls",names=["task_id","gps_0","gps_1","pricing","condition"])
data2=pd.read_excel("./data/data2.xlsx",names=["mem_id", "gps", "task_limit", "start_time", "credit"])
data3=pd.read_excel("./data/data3.xls",names=["task_id","gps_0","gps_1"])

split_data = data2['gps'].str.split(' ', n=1, expand=True)
split_data.columns = ['gps_0', 'gps_1']
gps_idx = data2.columns.get_loc('gps')
data2 = pd.concat([
    data2.iloc[:, :gps_idx],     # 之前的列
    split_data,                  # 新分割的两列
    data2.iloc[:, gps_idx+1:]    # 之后的列
], axis=1)

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
    print(1)
    return 6371.0 * c

# 计算任务和成员之间的距离矩阵（pandas.DataFrame类型存储）
# 提取任务和成员的坐标
tasks = data1[['task_id', 'gps_0', 'gps_1']].copy()
members = data2[['mem_id', 'gps_0', 'gps_1']].copy()

# 设置索引以便于匹配
tasks.set_index('task_id', inplace=True)
members.set_index('mem_id', inplace=True)

# 创建空的距离矩阵
distance_matrix = pd.DataFrame(
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
        distance_matrix.at[task_id, mem_id] = distance

distance_matrix.to_excel("./data/distance_matrix.xlsx")