import pandas as pd

# 读取数据
df = pd.read_excel('../data/task_to_member_distance.xlsx')  # 第一行是列名，第0列开始是数据
data2 = pd.read_excel("../data/data2.xlsx",
                      names=["mem_id", "gps", "task_limit", "start_time", "credit"])

# 定义区间范围和初始化统计字典
intervals = [
    (0, 1),  # 0-1 (闭区间)
    (1, 2),  # 1-2 [1, 2)
    (2, 3),  # 2-3 [2, 3)
    (3, 5),  # 3-5 [3, 5)
    (5, 10),  # 5-10 [5, 10)
    (10, 20),  # 10-20 [10, 20)
    (20, 30),  # 20-30 [20, 30)
    (30, float('inf'))  # 30以上 [30, ∞)
]

# 初始化统计结果字典
interval_stats = {f"{low}-{high}": 0 for low, high in intervals}

# 预先获取所有task_limit值
task_limits = data2['task_limit'].values

# 遍历距离矩阵 - 现在第0列开始就是数据
values = df.iloc[:, 1:].values  # 所有行和列都是数据
for i in range(values.shape[0]):  # 行循环
    for j in range(values.shape[1]):  # 列循环
        value = values[i, j]

        # 判断值所在的区间
        for low, high in intervals:
            if (low == 0 and value == 0) or (low <= value < high):
                # 直接通过列索引j获取对应的task_limit值
                if j < len(task_limits):  # 确保索引不越界
                    task_limit_value = task_limits[j]
                    interval_key = f"{low}-{high}" if high != float('inf') else "30-inf"
                    interval_stats[interval_key] += task_limit_value

                break  # 找到区间后跳出循环

# 将字典转换为DataFrame
result_df = pd.DataFrame.from_dict(interval_stats, orient='index', columns=['value'])
result_df.index.name = 'section'  # 设置索引名称
result_df = result_df.reset_index()  # 将索引转换为列

result_df['value/835'] = result_df['value'] / 835

print(result_df)
