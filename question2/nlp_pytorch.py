import numpy as np
import pandas as pd
import torch  # 导入PyTorch
from deap import base, creator, tools, algorithms
import random

# 读取数据
data1 = pd.read_excel("../data/data1_all.xlsx")
distances3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")
d = torch.tensor(distances3.values[:, 1:], dtype=torch.float32).cuda()  # 将数据移至GPU
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

# 在data1中添加E列，根据城市匹配E_i值
data1["E"] = data1["city"].map(city_e_values)

MD = torch.tensor(data1["MD"].values, dtype=torch.float32).cuda()  # 转为PyTorch张量并移至GPU
TD = torch.tensor(data1["TD"].values, dtype=torch.float32).cuda()  # 转为PyTorch张量并移至GPU
h_values = torch.tensor(data1["difficulty_d"].values, dtype=torch.float32).cuda()  # 转为PyTorch张量并移至GPU
credit = torch.tensor(data2["task_limit"].values, dtype=torch.float32).cuda()  # 转为PyTorch张量并移至GPU
e_values = torch.tensor(data1["E"].values, dtype=torch.float32).cuda()  # 转为PyTorch张量并移至GPU


# 遗传算法中目标函数
def calculate_S(mu, h_i, d, Pr_i, e_i):
    if d == -1:
        return 0  # 跳过该项
    S_ij = Pr_i / ((mu * h_i + d) * e_i)  # 直接计算S_ij
    return S_ij


def calculate_P(S, alpha, beta):
    return alpha * S + beta


def equation_to_solve(params, mu, h_values, d, n, w, credit, MD, TD, e):
    a, b, P_0 = params
    sum_term = 0
    Pr_total = 0  # 用来计算Pr_i的总和

    # 计算目标函数的值和Pr_i总和
    for i in range(n):
        prod_term = 1
        for j in range(w):
            # 计算 S_ij 和 Pr_i
            Pr_i = (P_0 / (1 + a * MD[i]) * (1 + b * TD[i]) + h_values[i]) * e[i]
            S_ij = calculate_S(mu=mu, h_i=h_values[i], d=d[i][j], Pr_i=Pr_i, e_i=e[i])
            if S_ij == 0:
                continue
            P_ij = calculate_P(S_ij, alpha, beta)
            prod_term *= (1 - P_ij) ** credit[j]
        sum_term += (1 - prod_term)
        Pr_total += Pr_i

    # 加入约束惩罚
    if abs(Pr_total - 57707.5) > 0.1:  # 允许小的浮动
        sum_term += 1000 * abs(Pr_total - 57707.5)  # 惩罚不满足的情况

    return (sum_term,)  # 返回目标函数值


# 设定遗传算法的基本参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 我们要最小化目标函数
creator.create("Individual", list, fitness=creator.FitnessMin)


# 生成个体时，确保a, b, P_0 > 0
def create_individual():
    return [random.uniform(0.01, 10),  # a的范围 > 0
            random.uniform(0.01, 10),  # b的范围 > 0
            random.uniform(0.1, 10)]  # P_0的范围 > 0


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate,
                 creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 目标函数，用于计算适应度
toolbox.register("evaluate", equation_to_solve, mu=mu, h_values=h_values,
                 d=d, n=data1.shape[0], w=data2.shape[0],
                 credit=credit, MD=MD, TD=TD, e=e_values)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # 变异操作
toolbox.register("select", tools.selTournament, tournsize=3)  # 选择操作

# 遗传算法主函数
population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                    ngen=100, stats=None, halloffame=None)

# 获取最佳个体
best_individual = tools.selBest(population, 1)[0]
print("最佳参数组合：", best_individual)
