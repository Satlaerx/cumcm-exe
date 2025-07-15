import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

# 读取数据
data1 = pd.read_excel("../data/data1_all.xlsx")
distances3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")
d = distances3.values[:, 1:]
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")

mu = 0.5
alpha = 0.007786111538245771
beta = -6.592993037312056e-05

# 城市与E_i值的映射字典
city_e_values = {
    "广州": 7.5896320020684875,
    "深圳": 53.20182709198384,
    "东莞": 1.0,
    "佛山": 0.3665363288363272
}

# 添加 E_i 列
data1["E"] = data1["city"].map(city_e_values)

MD = data1["MD"].values
TD = data1["TD"].values
h_values = data1["difficulty_d"].values
credit = data2["task_limit"].values
e_values = data1["E"].values


def equation_to_solve(params, mu, h_values, d, n, w, credit, MD, TD, e):
    a, b, P_0 = params

    if a <= 0 or b <= 0 or P_0 <= 0:
        return (-1e10,)

    sum_term = 0
    Pr_total = 0

    for i in range(n):
        prod_term = 1
        for j in range(w):
            if d[i][j] == -1:
                continue
            Pr_i = (P_0 / (1 + a * MD[i]) * (1 + b * TD[i]) + h_values[i]) * e[i]
            S_ij = Pr_i / ((mu * h_values[i] + d[i][j]) * e[i])
            P_ij = alpha * S_ij + beta
            prod_term *= (1 - P_ij) ** credit[j]

        sum_term += (1 - prod_term)
        Pr_total += Pr_i

    if Pr_total > 57707.5:
        return (sum_term - 1000 * (Pr_total - 57707.5),)

    return (sum_term,)


# === 遗传算法设定 ===

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
                 mu=mu, h_values=h_values, d=d,
                 n=data1.shape[0], w=data2.shape[0],
                 credit=credit, MD=MD, TD=TD, e=e_values)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# === 执行遗传算法 ===

population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                    ngen=100, stats=None, halloffame=None)

best_individual = tools.selBest(population, 1)[0]
print("最佳参数组合：", best_individual)
