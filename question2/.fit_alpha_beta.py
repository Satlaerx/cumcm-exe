import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ========== 读取数据 ==========
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

# ========== 常数 ==========
mu = 1  # 距离难度系数


# ========== 拟合模型函数 ==========
def model_function(city, pricing, d_level, distance_matrix, task_limits, alpha, beta):
    n_tasks = distance_matrix.shape[0]
    predictions = np.zeros(n_tasks)

    for i in range(n_tasks):
        log_sum = 0.0  # 改为对数求和

        for j in range(distance_matrix.shape[1]):
            d = distance_matrix[i, j]
            if d == -1:
                continue

            E = {"广州": 0, "深圳": 5, "东莞": -5, "佛山": 0}.get(city[i], 0)
            denom = d + mu * d_level[i]
            p = alpha * ((pricing[i] - E) / denom) + beta
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 防止 log(0)

            log_sum += task_limits[j] * np.log(1 - p)

        predictions[i] = 1 - np.exp(log_sum)

    return predictions


# ========== 包装成 curve_fit 支持的形式 ==========
def fit_func(x, alpha, beta):
    return model_function(
        city=city,
        pricing=pricing,
        d_level=d_level,
        distance_matrix=distance_matrix,
        task_limits=task_limits,
        alpha=alpha,
        beta=beta
    )


# ========== 数据准备 ==========
y_actual = data1["condition"].values
city = data1["city"].tolist()
pricing = data1["pricing"].values
d_level = data1["difficulty_d"].values
distance_matrix = distance3.iloc[:, 1:].values.astype(float)
task_limits = data2["task_limit"].values

# ========== 拟合参数 ==========
initial_guess = [0.01, 0.01]

params_opt, params_cov = curve_fit(
    f=fit_func,
    xdata=np.arange(len(y_actual)),  # 模型中不使用，但是必须传
    ydata=y_actual,
    p0=initial_guess,
    maxfev=10000
)

# ========== 输出拟合结果 ==========
alpha, beta = params_opt
print("拟合参数结果:")
print(f"alpha = {alpha:.6f}")
print(f"beta = {beta:.6f}")

# ========== 计算 R² ==========
y_pred = fit_func(np.arange(len(y_actual)), alpha, beta)
ss_res = np.sum((y_actual - y_pred) ** 2)
ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"\n拟合优度 R² = {r_squared:.4f}")
