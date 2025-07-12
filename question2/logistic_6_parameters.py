import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import pandas as pd


# 安全版本的sigmoid和logit函数
def safe_sigmoid(x):
    x = np.clip(x, -500, 500)  # 防止溢出
    return 1 / (1 + np.exp(-x))


def safe_logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)  # 避免边界值
    return np.log(p / (1 - p))


def logistic_model_function(params, df_tasks, distance_matrix, data2):
    m1, m2, m3, m4, alpha, beta = params

    probabilities = np.zeros(len(df_tasks))

    for i in range(len(df_tasks)):
        task = df_tasks.iloc[i]
        city_name = task['city']
        pricing = task['pricing']
        difficulty = task['difficulty_d']

        log_odds = 0
        valid_count = 0  # 记录有效贡献数

        for j in range(distance_matrix.shape[1]):
            d_ij = distance_matrix[i, j]

            if d_ij == -1:
                continue

            # 安全处理分母
            denominator = d_ij + mu * difficulty
            if denominator <= 0:
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

            if x <= 0:
                continue

            # 安全计算贡献
            contribution = alpha * (pricing / denominator) * (1 / x) + beta
            p = safe_sigmoid(contribution)
            m_val = data2['task_limit'].iloc[j]

            # 累加log odds
            log_odds += m_val * safe_logit(p)
            valid_count += 1

        # 如果有有效贡献则计算概率
        if valid_count > 0:
            probabilities[i] = safe_sigmoid(log_odds / valid_count)  # 平均处理
        else:
            probabilities[i] = 0.5  # 默认中性概率

    return probabilities


# 定义带安全检查的损失函数
def objective_function(params):
    prob = logistic_model_function(params, data1, distance_fix, data2)
    if np.any(np.isnan(prob)):
        print("警告: 检测到NaN值，参数:", params)
        return 1e10  # 返回大数值惩罚
    return log_loss(y_actual, prob)


# 数据加载和预处理
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")
mu = 0.5

# 数据清洗
distance_fix = distance3.iloc[:, 1:].fillna(-1).values
y_actual = data1["condition"].values

# 检查数据
print("缺失值检查:")
print("距离矩阵NaN:", np.isnan(distance_fix).sum())
print("y_actual NaN:", np.isnan(y_actual).sum())

# 参数优化
initial_guess = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.01])
bounds = [(1e-3, 10)] * 4 + [(-10, 10), (-5, 5)]  # 更严格的边界

result = minimize(
    objective_function,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 10000, 'disp': True}
)

# 结果处理
if result.success:
    params_opt = result.x
    print("优化成功! 参数:")
    print(f"m1={params_opt[0]:.4f}, m2={params_opt[1]:.4f}")
    print(f"m3={params_opt[2]:.4f}, m4={params_opt[3]:.4f}")
    print(f"alpha={params_opt[4]:.4f}, beta={params_opt[5]:.4f}")

    # 评估
    y_prob = logistic_model_function(params_opt, data1, distance_fix, data2)
    print("最小log loss:", result.fun)
    print("预测概率范围:", np.min(y_prob), "to", np.max(y_prob))
else:
    print("优化失败:", result.message)