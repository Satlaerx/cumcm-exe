import pandas as pd
import numpy as np

sigma=2.5

data1 = pd.read_excel("./data/data1_all.xlsx",
                     names=["task_id", "gps_0", "gps_1", "pricing", "condition",
                            "difficulty", "difficulty_d", "city"])

distances=pd.read_excel("./data/distance_matrix.xlsx")
distances2=pd.read_excel("./data/task_to_task_distance_matrix.xlsx")
data2 = pd.read_excel("./data/data2.xlsx",
                      names=["mem_id", "gps", "task_limit", "start_time", "credit"])

E=[1,1,1,1] # 依次为广东、深圳、东莞和佛山

# 忽略不在这个4个市中的1个任务和18个成员
data1=data1[data1['city']!="未知"]
data2=data2[data2['city']!="未知"]

def gaussian_decay(x):
    """高斯衰减函数"""
    return np.exp(-x**2 / (2 * sigma**2))

# 初始化MD和TD
n_tasks = data1.shape[0]
MD = np.zeros(n_tasks)
TD = np.zeros(n_tasks)

# 假设用户最多接10个任务
data2['task_limit'] = data2['task_limit'].clip(upper=10)
task_limits = data2['task_limit'].values

# 计算MD特征
values = distances.iloc[:, 1:].values  # 所有行和列都是数据
for i in range(values.shape[0]):  # 行循环
    for j in range(values.shape[1]):  # 列循环
        value = values[i, j]
        MD[i] += gaussian_decay(value) * task_limits[j]

# 计算TD特征
values2 = distances2.iloc[:, 1:].values
for i in range(values2.shape[0]):
    for j in range(values2.shape[1]):
        value = values2[i, j]  # 修正变量名
        TD[i] += gaussian_decay(value)

data1["MD"] = MD
data1["TD"] = TD

# 求alpha, beta的值