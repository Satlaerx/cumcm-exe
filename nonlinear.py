from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置图片清晰度
# 设置中文字体和图片清晰度
plt.rcParams["font.family"] = ["Times New Roman","SimHei"]
plt.rcParams['figure.dpi'] = 500

# 加载数据
sigma = 2.5

data1 = pd.read_excel("./data/data1.xls", names=["task_id", "gps_0", "gps_1", "pricing", "condition"])
distances = pd.read_excel('./data/distance_matrix.xlsx')
data2 = pd.read_excel("./data/data2.xlsx", names=["mem_id", "gps", "task_limit", "start_time", "credit"])
distances2 = pd.read_excel('./data/task_to_task_distance_matrix.xlsx')

# 定义模型函数
def gaussian_decay(x):
    """高斯衰减函数"""
    return np.exp(-x**2 / (2 * sigma**2))

def model(x, a, b, p0):
    """目标拟合函数
    x: 包含MD和TD的元组或列表
    a, b, p0: 待拟合参数
    """
    MD, TD = x
    return p0 * (1 / (1 + a * MD)) * (1 / (1 + b * TD))

# 初始化x1和x2
n_tasks = data1.shape[0]
x1 = np.zeros(n_tasks)
x2 = np.zeros(n_tasks)

# 预先获取所有task_limit值
task_limits = data2['task_limit'].values

# 计算x1特征
values = distances.iloc[:, 1:].values  # 所有行和列都是数据
for i in range(values.shape[0]):  # 行循环
    for j in range(values.shape[1]):  # 列循环
        value = values[i, j]
        x1[i] += gaussian_decay(value) * task_limits[j]

# 计算x2特征
values2 = distances2.iloc[:, 1:].values
for i in range(values2.shape[0]):
    for j in range(values2.shape[1]):
        value = values2[i, j]  # 修正变量名
        x2[i] += gaussian_decay(value)

# 获取目标变量
y_exp = data1["pricing"].values

# 使用 curve_fit 进行非线性最小二乘拟合
# 将x1和x2组合成一个元组传递给func2
popt, pcov = curve_fit(model, (x1, x2), y_exp, p0=[0.1, 0.1, 100])
# p0是参数初始猜测值，需要根据实际问题调整

# 计算参数的标准误差
perr = np.sqrt(np.diag(pcov))

# 打印拟合结果
print("拟合参数:")
print(f"a = {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"b = {popt[1]:.4f} ± {perr[1]:.4f}")
print(f"p0 = {popt[2]:.4f} ± {perr[2]:.4f}")

# 计算拟合优度 R²
y_pred = model((x1, x2), *popt)
residuals = y_exp - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\n拟合优度 R²: {r_squared:.4f}")

# 可视化拟合结果
plt.figure(figsize=(12, 5))

# 散点图：预测值 vs 实际值
plt.subplot(1, 2, 1)
plt.scatter(y_exp, y_pred, alpha=0.6)
plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], 'r--')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('预测值 vs 实际值')
plt.grid(True)

# 残差图
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True)

plt.tight_layout()
plt.show()