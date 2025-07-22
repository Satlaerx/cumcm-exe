import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, freeze_support
import random

# === 全局常量 ===
MU = 1
ALPHA = 0.000479
BETA = -0.000731
PR_TOTAL_THRESHOLD = 57707.5

# === 读取数据 ===
data1 = pd.read_excel("../data/data1_all.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distances3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

d_matrix = distances3.iloc[:, 1:].to_numpy(dtype=float)
credit = data2["task_limit"].values

# === E 映射 ===
city_e_map = {
    "广州": 0,
    "深圳": 5,
    "东莞": -5,
    "佛山": 0
}
data1["E"] = data1["city"].map(city_e_map)

# === 提取变量 ===
MD = data1["MD"].values
TD = data1["TD"].values
h_d = data1["difficulty_d"].values
h_values = data1["difficulty"].values
e_values = data1["E"].values

# === 遗传算法函数 ===
import numpy as np


# 常量 ALPHA, BETA, PR_TOTAL_THRESHOLD 假设在全局已经定义
# ALPHA = 0.000479
# BETA = -0.000731
# PR_TOTAL_THRESHOLD = 57707.5

def fitness_function(params, mu, h_values, h_d, d, credit, MD, TD, e):
    """
    遗传算法的适应度函数，计算给定参数 params 对目标的评分。

    参数:
    - params: (a, b, P0) 三元组，浮点数
    - mu: float
    - h_values: ndarray, 任务数长度的一维数组
    - h_d: ndarray, 任务数长度的一维数组
    - d: ndarray, 形状 (n_tasks, n_members)，任务与成员距离矩阵，-1表示无效
    - credit: ndarray, 成员额度，长度为 n_members
    - MD: ndarray, 任务数长度的一维数组
    - TD: ndarray, 任务数长度的一维数组
    - e: ndarray, 任务数长度的一维数组

    返回:
    - tuple 包含一个元素，表示适应度值 (total_score,)
      若参数非法或超过阈值则返回大负值惩罚 (-1e10,)
    """
    a, b, P0 = params

    # 参数非正数时惩罚
    if a <= 0 or b <= 0 or P0 <= 0:
        return (-1e300,)

    n_tasks, n_members = d.shape

    # 生成有效成员掩码矩阵，True 表示有效
    mask = (d >= 0)  # shape: (n_tasks, n_members)

    # 计算任务的基础概率 Pr_i，向量化计算
    # 形状为 (n_tasks,)
    Pr = P0 / ((1 + a * MD) * (1 + b * TD)) + h_values + e

    # 准备广播的分子和分母，用于计算 S_ij
    numerator = (Pr - e)[:, None]  # shape (n_tasks, 1)，广播到成员维度
    denominator = mu * h_d[:, None] + d  # shape (n_tasks, n_members)

    # 对无效成员位置避免除以无效距离，用 np.where 替换为 np.nan，后续处理
    denominator_masked = np.where(mask, denominator, np.nan)

    # 计算 S_ij = (Pr_i - e_i) / (mu * h_d_i + d_ij)
    S = numerator / denominator_masked  # 无效成员为 nan

    # 计算 P_ij = ALPHA * S_ij + BETA，且限制在 [0, 1-1e-10]
    P_ij = ALPHA * S + BETA
    P_ij = np.clip(P_ij, 0, 1 - 1e-10)

    # 对无效成员，将 P_ij 设置为0，避免影响后续计算
    P_ij = np.where(mask, P_ij, 0)

    # credit 是成员额度，shape (n_members,)
    # 需要广播成 shape (1, n_members)，再和 (n_tasks, n_members) 的 log1p(-P_ij) 相乘
    credit_broadcast = credit[None, :]

    # 计算 credit_j * log1p(-P_ij) 对每个成员
    log_terms = credit_broadcast * np.log1p(-P_ij)

    # 对无效成员对应的 log_terms 设置为0，避免影响 sum
    log_terms = np.where(mask, log_terms, 0)

    # 任务维度求和，得到每个任务的 log_prod
    log_prod = np.sum(log_terms, axis=1)  # shape (n_tasks,)

    # 计算任务完成概率 prob_i = exp(log_prod)
    prob_i = np.exp(log_prod)

    # 过滤数值异常（nan 或过小）
    prob_i = np.where(np.isnan(prob_i) | (prob_i < 1e-300), 0, prob_i)

    # 计算目标总分，等价于原函数中的 total_score
    total_score = np.sum(1 - prob_i)

    # 计算 Pr_i 总和，若超阈值惩罚
    total_Pr = np.sum(Pr)
    if total_Pr > PR_TOTAL_THRESHOLD:
        return (-1e300,)

    # 返回适应度值元组
    return (total_score,)


# === 个体生成函数 ===
def create_individual():
    return [
        random.uniform(0.01, 10),  # a
        random.uniform(0.01, 10),  # b
        random.uniform(0.1, 10)  # P₀
    ]


# === 注册 DEAP 工具 ===
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function,
                 mu=MU, h_values=h_values, h_d=h_d, d=d_matrix,
                 credit=credit, MD=MD, TD=TD, e=e_values)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# === 主程序 ===
if __name__ == "__main__":
    freeze_support()
    print("开始遗传算法优化...")

    with Pool() as pool:
        toolbox.register("map", pool.map)

        population = toolbox.population(n=100)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(5)

        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=0.8, mutpb=0.3,
            ngen=250,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        best = tools.selBest(population, 1)[0]
        print("最优参数组合（保留 10 位小数）:\n a = {:.10f}, b = {:.10f}, P₀ = {:.10f}".format(*best))
