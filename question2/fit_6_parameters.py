import pandas as pd
from scipy.optimize import curve_fit
import numpy as np

# 加载数据
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

mu = 0.5  # 距离难度系数

def model_function(params, df_tasks, distance_matrix, data2):
    """
    非线性拟合模型函数
    params: 参数数组 [m1, m2, m3, m4, alpha, beta]
    df_tasks: 任务DataFrame
    distance_matrix: 距离矩阵(numpy数组)
    data2: 成员DataFrame
    """
    m1, m2, m3, m4, alpha, beta = params

    predictions = np.zeros(len(df_tasks))

    for i in range(len(df_tasks)):
        task = df_tasks.iloc[i]
        city_name = task['city']
        pricing = task['pricing']
        difficulty = task['difficulty_d']

        count = 1.0

        # 遍历所有成员(距离矩阵的列)
        for j in range(distance_matrix.shape[1]):
            d_ij = distance_matrix[i, j]  # 直接使用numpy数组索引

            if d_ij == -1:  # 跳过无效距离
                continue

            # 确定城市系数
            if city_name == '广州':
                x = m1
            elif city_name == '深圳':
                x = m2
            elif city_name == '东莞':
                x = m3
            elif city_name == '佛山':
                x = m4
            else:
                x = m1

            # 计算接受概率
            p = alpha * (pricing / (d_ij + mu * difficulty) * (1 / x)) + beta
            p = np.clip(p, 0, 1)  # 限制概率范围

            # 获取成员的目标限制
            m_val = data2['task_limit'].iloc[j]

            # 更新累积概率
            count *= (1 - p) ** m_val

        predictions[i] = 1 - count

    return predictions


# 准备数据
initial_guess = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.01])  # 初始参数猜测
y_actual = data1["condition"].values  # 实际值(numpy数组)
distance_fix = distance3.iloc[:, 1:].values  # 距离矩阵转为numpy数组


# 关键修正：正确的curve_fit调用方式
def fit_func(x, *params):
    """包装函数以适应curve_fit的调用约定"""
    return model_function(params, data1, distance_fix, data2)


# 执行拟合
params_opt, params_cov = curve_fit(
    f=fit_func,  # 使用包装函数
    xdata=np.arange(len(y_actual)),  # x数据（虽然不使用，但必须提供）
    ydata=y_actual,  # y数据
    p0=initial_guess,  # 初始猜测
    maxfev=10000  # 最大迭代次数
)

# 输出结果
m1, m2, m3, m4, alpha, beta = params_opt
print("拟合参数结果:")
print(f"m1 = {m1:.6f}")
print(f"m2 = {m2:.6f}")
print(f"m3 = {m3:.6f}")
print(f"m4 = {m4:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"beta = {beta:.6f}")

# 计算拟合优度
y_pred = model_function(params_opt, data1, distance_fix, data2)
ss_res = np.sum((y_actual - y_pred) ** 2)
ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\n拟合优度 R² = {r_squared:.4f}")