import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, freeze_support
import random

# === 全局常量 ===
MU = 1
ALPHA = 0.000651007440380953
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


# === 遗传算法目标函数 ===
def fitness_function(params, mu, h_values, h_d, d, credit, MD, TD, e):
    a, b, P0 = params

    if a <= 0 or b <= 0 or P0 <= 0:
        return (-1e10,)  # 惩罚非法值

    total_score = 0
    total_Pr = 0

    n_tasks, n_members = d.shape

    for i in range(n_tasks):
        Pr_i = P0 / ((1 + a * MD[i]) * (1 + b * TD[i])) + h_values[i] + e[i]
        log_prod = 0

        for j in range(n_members):
            if d[i][j] == -1:
                continue

            S_ij = (Pr_i - e[i]) / (mu * h_d[i] + d[i][j])
            P_ij = ALPHA * S_ij
            P_ij = np.clip(P_ij, 0, 1 - 1e-10)

            log_prod += credit[j] * np.log1p(-P_ij)

        prob_i = np.exp(log_prod)
        prob_i = 0 if np.isnan(prob_i) or prob_i < 1e-300 else prob_i

        total_score += (1 - prob_i)
        total_Pr += Pr_i

    if total_Pr > PR_TOTAL_THRESHOLD:
        return (-1e10,)  # 惩罚非法值

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
