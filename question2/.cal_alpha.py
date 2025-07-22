import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ==================== 读取数据 ====================
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

# ==================== 数据准备 ====================
y_actual = data1["condition"].values
sum_actual = y_actual.sum()
city = data1["city"].tolist()
pricing = data1["pricing"].values
d_level = data1["difficulty_d"].values
distance_matrix = distance3.iloc[:, 1:].values.astype(float)
task_limits = data2["task_limit"].values

# ==================== 常量设置 ====================
mu = 1  # 距离难度系数

# ==================== 模型函数 ====================
def model_function(city, pricing, d_level, distance_matrix, task_limits, alpha):
    n_tasks = distance_matrix.shape[0]
    predictions = np.zeros(n_tasks)

    for i in range(n_tasks):
        log_sum = 0.0

        for j in range(distance_matrix.shape[1]):
            d = distance_matrix[i, j]
            if d == -1:
                continue

            # 根据城市调整偏差
            E = {"广州": 0, "深圳": 5, "东莞": -5, "佛山": 0}.get(city[i], 0)
            denom = d + mu * d_level[i]

            # 计算p并确保其在数值范围内
            p = alpha * ((pricing[i] - E) / denom)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 防止 log(0)

            # 累加任务限制与概率的对数
            log_sum += task_limits[j] * np.log(1 - p)

        predictions[i] = 1 - np.exp(log_sum)

    return predictions

# 定义方程
def equation(alpha, city, pricing, d_level, distance_matrix, task_limits, sum_actual):
    # 使用模型函数计算预测值
    predictions = model_function(city, pricing, d_level, distance_matrix, task_limits, alpha)
    predictions_sum = predictions.sum()

    # 输出每次计算的预测总和，查看方程如何变化
    print(f"Alpha: {alpha}, Predictions Sum: {predictions_sum}, Target Sum Actual: {sum_actual}")

    # 方程：我们希望 predictions_sum 接近 sum_actual，所以我们返回 predictions_sum - sum_actual
    return predictions_sum - sum_actual

# 初始猜测的 alpha 值
initial_alpha = 0.1

# 使用 fsolve 函数来求解方程
optimal_alpha, = fsolve(equation, initial_alpha, args=(city, pricing, d_level, distance_matrix, task_limits, sum_actual))

# 输出最优的 alpha 值
print(f"The optimal value for alpha is: {optimal_alpha}")
