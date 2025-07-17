import pandas as pd

data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city.xlsx")
distances = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

mu = 1


def satisfied(pricing, distance, difficulty):
    return pricing / (distance + mu * difficulty)


# 创建空的满意度矩阵（行：任务，列：成员）
satisfaction_matrix = pd.DataFrame(
    index=data1['task_id'],  # 使用task_id作为索引
    columns=data2['mem_id'],  # 使用mem_id作为列名
    dtype=float
)

distance_values = distances.values[:, 1:]
pricing = data1["pricing"].values

s_max = -float('inf')
s_min = float('inf')

difficulty = data1["difficulty_d"].values
for i in range(distances.shape[0]):
    for j in range(distances.shape[1] - 1):
        if distance_values[i, j] == -1:
            satisfaction_matrix.iloc[i, j] = -1
            continue
        satisfaction = pricing[i] / (distance_values[i, j] + mu * difficulty[i])
        satisfaction_matrix.iloc[i, j] = satisfaction

        if satisfaction > s_max:
            s_max = satisfaction
        if satisfaction < s_min:
            s_min = satisfaction

# 打印结果
satisfaction_matrix.to_excel("../data/satisfaction_cal_alpha_beta.xlsx", index=False)

print(s_max, s_min)

alpha = 1 / (s_max - s_min)  # 0.017224862626975857
beta = -1 * (s_min / (s_max - s_min))  # -0.13781535718863075

print(alpha, beta)
