import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 图像显示参数
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.unicode_minus'] = False

# ==================== 读取数据 ====================
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

# ==================== 常量设置 ====================
mu = 0.1  # 距离难度系数


# ==================== 模型函数 ====================
def model_function(city, pricing, d_level, distance_matrix, task_limits, alpha, beta):
    n_tasks = distance_matrix.shape[0]
    predictions = np.zeros(n_tasks)

    for i in range(n_tasks):
        log_sum = 0.0

        for j in range(distance_matrix.shape[1]):
            d = distance_matrix[i, j]
            if d == -1:
                continue

            E = {"广州": 0, "深圳": 5, "东莞": -5, "佛山": 0}.get(city[i], 0)
            denom = d + mu * d_level[i]
            s_ij = (pricing[i] - E) / denom
            p = alpha * s_ij + beta
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 防止 log(0)

            log_sum += task_limits[j] * np.log(1 - p)

        predictions[i] = 1 - np.exp(log_sum)

    return predictions


# ==================== 数据准备 ====================
y_actual = data1["condition"].values
city = data1["city"].tolist()
pricing = data1["pricing"].values
d_level = data1["difficulty_d"].values
distance_matrix = distance3.iloc[:, 1:].values.astype(float)
task_limits = data2["task_limit"].values


# ==================== 定义目标函数（负对数似然） ====================
def neg_log_likelihood(params):
    alpha, beta = params
    preds = model_function(city, pricing, d_level, distance_matrix, task_limits, alpha, beta)
    preds = np.clip(preds, 1e-10, 1 - 1e-10)
    loss = -np.sum(y_actual * np.log(preds) + (1 - y_actual) * np.log(1 - preds))
    return loss


# ==================== 参数拟合（极大似然估计） ====================
initial_guess = [0.001, 0.001]
bounds = [(0, 1), (-1, 1)]

result = minimize(
    fun=neg_log_likelihood,
    x0=initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'disp': True}
)

alpha_opt, beta_opt = result.x

# ==================== 输出拟合结果 ====================
print("极大似然估计结果:")
print(f"alpha = {alpha_opt:.6f}")
print(f"beta  = {beta_opt:.6f}")

# ==================== 拟合优度指标 ====================
# 计算预测值
y_pred = model_function(city, pricing, d_level, distance_matrix, task_limits, alpha_opt, beta_opt)
y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

# 平均交叉熵损失
cross_entropy = -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))
print(f"\n平均交叉熵损失 = {cross_entropy:.6f}")

# McFadden 伪 R²
log_likelihood_model = -neg_log_likelihood([alpha_opt, beta_opt])
p_mean = np.mean(y_actual)
beta_null = np.log(p_mean / (1 - p_mean))
log_likelihood_null = -neg_log_likelihood([0.0, beta_null])
pseudo_r2 = 1 - log_likelihood_model / log_likelihood_null
print(f"McFadden 伪 R² = {pseudo_r2:.4f}")

# ==================== 结果可视化 ====================
fpr, tpr, _ = roc_curve(y_actual, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('MLE_ROC.pdf', dpi=500, bbox_inches='tight', facecolor='white')
plt.show()
