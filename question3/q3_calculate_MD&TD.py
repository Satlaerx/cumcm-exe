import numpy as np
import pandas as pd

sigma = 2.5
mu = 0.5

data1 = pd.read_excel("../data/data1_merged.xlsx")
distances = pd.read_excel("../data/q3_task_to_member_distances.xlsx")
data2 = pd.read_excel("../data/data2_with_city.xlsx")
distances2 = pd.read_excel("../data/q3_task_to_task_distances.xlsx")


# 忽略不在这个4个市中的1个任务和18个成员
# data1=data1[data1['city']!="未知"]
# data2=data2[data2['city']!="未知"]

def gaussian_decay(x):
    """高斯衰减函数"""
    return np.exp(-x ** 2 / (2 * sigma ** 2))


# 初始化MD和TD, 与第一问的区别在于认为用户最多接10个任务
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

data1.to_excel("../data/data1_fix_MD_TD.xlsx", index=False)
