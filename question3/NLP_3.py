import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, freeze_support
import random

# 固定参数
mu = 1
alpha = 0.007786111538245771
beta = -6.592993037312056e-05

# 数据读取
data1 = pd.read_excel("../data/data1_new.xlsx")
distances3 = pd.read_excel("../data/task_to_member_distance_new.xlsx")
d = distances3.iloc[:, 1:].to_numpy(dtype=float)
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")

# E值映射
city_e_values = {
    "广州": 7.5896320020684875,
    "深圳": 53.20182709198384,
    "东莞": 1.0,
    "佛山": 0.3665363288363272
}
data1["E"] = data1["city"].map(city_e_values)

# 提取数据
MD = data1["MD"].values
TD = data1["TD"].values
h_d = data1["difficulty_d"].values
h_values = data1["difficulty"].values
e_values = data1["E"].values
credit = data2["task_limit"].values
i_k_limit = data1["cluster_size"].values

n = 628  # DBSCAN打包之后任务数
w = data2.shape[0]


# 遗传算法目标函数
def equation_to_solve(params, mu, h_values, h_d, d, n, w,
                      credit, MD, TD, e, i_k_limit):
    a, b, P_0 = params

    # 非法值惩罚
    if a <= 0 or b <= 0 or P_0 <= 0:
        return (-1e10,)

    index = 0  # 数据DataFrame中当前所在行的索引
    sum_term = 0
    C_total = 0

    for i in range(n):
        sum_C_i_k=0
        min_d
        for k in range(i_k_limit[i]):
            C_i_k = (P_0 / (1 + a * MD[index + k]) *
                     (1 + b * TD[index + k]) + h_values[index + k]) * e[index + k]
            sum_C_i_k += C_i_k


        log_prod_term = 0

        for j in range(w):
            if d[i][j] == -1:
                continue

            S_ij = C_i / ((mu * h_d[i] + d[i][j]) * e[i])
            P_ij = alpha * S_ij + beta
            P_ij = np.clip(P_ij, 0, 1 - 1e-10)  # 避免 log1p(-1)

            log_prod_term += credit[j] * np.log1p(-P_ij)

        prob = np.exp(log_prod_term)
        if np.isnan(prob) or prob < 1e-300:
            prob = 0

        sum_term += (1 - prob)
        C_total += C_i

    if C_total > 57707.5:
        return (sum_term - 1000 * (C_total - 57707.5),)

    return (sum_term,)


# 遗传算法设置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def create_individual():
    return [random.uniform(0.01, 10),  # a
            random.uniform(0.01, 10),  # b
            random.uniform(0.1, 10)]  # P₀


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", equation_to_solve,
                 mu=mu, h_values=h_values, h_d=h_d, d=d,
                 n=n, w=w,
                 credit=credit, MD=MD, TD=TD, e=e_values)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 并行运行主程序
if __name__ == "__main__":
    freeze_support()

    print("并行优化开始")

    # 注册 map 映射函数以启用并行
    with Pool() as pool:
        toolbox.register("map", pool.map)  # 关键一步：DEAP会并行所有 evaluate 的个体

        population = toolbox.population(n=100)

        # 每代记录输出
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=0.7, mutpb=0.2,
            ngen=100,
            stats=stats,
            halloffame=None,
            verbose=True
        )

        best_individual = tools.selBest(population, 1)[0]
        print("最佳参数组合：", best_individual)
