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
L = np.array([1, 1, 1, 1, 1])

# === 读取数据 ===
data1 = pd.read_excel("../data/data1_fix_MD_TD.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")
distances3 = pd.read_excel("../data/q3_task_to_member_distance_5km.xlsx")

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
k_array = data1["cluster_size"].values

# d_value 是你函数中常数项，这里设为0，可以改成你需要的值
d_value = 0


def fitness_function(params, mu, h_values, h_d, d, credit, MD, TD, e, k_array, L):
    """
    遗传算法目标函数（向量化版本）

    参数:
    - params: 长度为15的参数向量，每3个一组，代表(a, b, P)，共5组
    - mu: 模型常数
    - h_d: 每个任务的历史维度
    - d: 任务到成员的距离矩阵，形状 (n_tasks, n_members)，无效值为负数
    - credit: 每个成员的额度，一维数组，长度 n_members
    - MD, TD: 每个任务的两种特征，一维数组，长度 n_tasks
    - e: 每个任务的 e_i 值
    - k_array: 每个任务的类别编号（取值为1~5）
    - h_values: 每个任务的历史得分
    - L: 每个类别对应的常数，长度为5

    返回:
    - 单元素元组：(total_score,)，若违反限制条件则返回惩罚分数 (-1e10,)
    """
    n_tasks, n_members = d.shape

    # ==================== 1. 参数合法性检查 ====================
    param_groups = np.array(params).reshape(5, 3)
    if np.any(param_groups <= 0):
        return (-1e10,)  # 非法参数惩罚

    a = param_groups[:, 0]
    b = param_groups[:, 1]
    P = param_groups[:, 2]

    # ==================== 2. 获取每个任务的对应参数组 ====================
    k_index = k_array.astype(int) - 1  # 转为 0~4 的索引

    a_k = a[k_index]  # 每个任务对应的 a_k
    b_k = b[k_index]  # 每个任务对应的 b_k
    P_k = P[k_index]  # 每个任务对应的 P_k
    L_k = L[k_index]  # 每个任务对应的 L_k

    # ==================== 3. 计算每个任务的 C_i ====================
    C = P_k * (1 / (1 + a_k * MD)) * (1 / (1 + b_k * TD)) + h_values + k_array * e

    # ==================== 4. 构造分母矩阵 mu * h_d_i + d_ij + L_k * (k_i - 1) ====================
    # d: (n_tasks, n_members)
    h_d_broadcast = h_d[:, None]  # (n_tasks, 1)
    e_broadcast = e[:, None]  # (n_tasks, 1)
    k_broadcast = k_array[:, None]  # (n_tasks, 1)
    L_k_broadcast = L_k[:, None]  # (n_tasks, 1)

    denominator = mu * h_d_broadcast + d + L_k_broadcast * (k_broadcast - 1)

    # 构造分子：C_i - k_i * e_i
    numerator = (C - (k_array * e))[:, None]  # (n_tasks, 1)

    # 计算 S_ij
    S = numerator / denominator  # (n_tasks, n_members)

    # 屏蔽非法 d < 0 的项（直接设置为 0，避免后续概率计算）
    mask = d >= 0
    S = np.where(mask, S, 0)

    # 计算 P_ij 并裁剪
    P_matrix = np.clip(ALPHA * S + BETA, 0, 1 - 1e-10)

    # 计算 log(1 - P_ij)，避免 log(0) 出现 nan
    log1mP = np.log1p(-P_matrix)

    # 乘上成员额度 credit（广播乘法）
    weighted_log = log1mP * credit  # (n_tasks, n_members)

    # 对每个任务累加所有成员贡献
    log_prod = np.sum(weighted_log * mask, axis=1)  # (n_tasks,)

    # === 5. 概率值 prob_i = exp(log_prod)，极小值/溢出处理
    prob = np.exp(log_prod)
    prob = np.where(np.isnan(prob) | (prob < 1e-300), 0, prob)

    # === 6. 累计结果
    total_score = np.sum(1 - prob)
    total_Pr = np.sum(C)

    if total_Pr > PR_TOTAL_THRESHOLD:
        return (-1e10,)

    return (total_score,)


# === 个体生成函数 ===
def create_individual():
    individual = []
    for _ in range(5):
        individual.append(random.uniform(0.01, 10))  # a
        individual.append(random.uniform(0.01, 10))  # b
        individual.append(random.uniform(0.1, 10))  # P
    return individual


# === DEAP工具注册 ===
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function,
                 mu=MU, h_values=h_values, h_d=h_d, d=d_matrix, credit=credit,
                 MD=MD, TD=TD, e=e_values, k_array=k_array, L=L)

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
        print("最优参数组合（保留10位小数）：")
        for i in range(5):
            a = best[3 * i]
            b = best[3 * i + 1]
            P = best[3 * i + 2]
            print(f"组{i + 1}: a = {a:.10f}, b = {b:.10f}, P = {P:.10f}")
