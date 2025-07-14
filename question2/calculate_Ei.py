import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# 读取数据
data1 = pd.read_excel("../data/data1_all.xlsx")
distances3 = pd.read_excel("../data/city_distance.xlsx", sheet_name=["广州", "深圳", "东莞", "佛山"])  # 读取所有城市的距离数据
data2 = pd.read_excel("../data/data2_with_city.xlsx")

# 城市列表和对应的k值
cities = ["广州", "深圳", "东莞", "佛山"]
k_values = {
    "广州": 194,
    "深圳": 34,
    "东莞": 178,
    "佛山": 115
}

# 已知常量
mu = 0.5
alpha = 0.007786111538245771
beta = -6.592993037312056e-05


def calculate_S(pr, mu, h, d, e, i, j):
    if d[i][j] == -1:
        return 0  # 跳过该项
    return pr[i] / ((mu * h[i] + d[i][j]) * e)


def calculate_P(S, alpha, beta):
    return alpha * S + beta


def equation_to_solve(e, pr, mu, h, d, alpha, beta, k, n, w, credit):
    sum_term = 0
    for i in range(n):
        prod_term = 1
        for j in range(w):
            S_ij = calculate_S(pr, mu, h, d, e, i, j)
            if S_ij == 0:
                continue
            P_ij = calculate_P(S_ij, alpha, beta)
            prod_term *= (1 - P_ij) ** credit[j]
        sum_term += (1 - prod_term)
    return sum_term - k


def solve_for_city(city, data1, distances3, data2, k):
    city_data1 = data1[data1["city"] == city]
    city_distances = distances3[city].values[:, 1:]  # 从对应的 sheet 读取距离数据
    city_data2 = data2[data2["city"] == city]

    pr = city_data1["pricing"].values
    h = city_data1["difficulty_d"].values
    task_limit = city_data2["task_limit"].values

    n = city_data1.shape[0]
    w = city_data2.shape[0]

    e_initial_guess = 1  # 初始猜测值
    e_solution = fsolve(equation_to_solve, e_initial_guess,
                        args=(pr, mu, h, city_distances, alpha, beta, k, n, w, task_limit))

    return e_solution[0]


# 计算每个城市的 e 值
city_e_values = {}

for city in cities:
    k = k_values[city]  # 获取当前城市对应的k值
    e_value = solve_for_city(city, data1, distances3, data2, k)
    city_e_values[city] = e_value
    print(f"城市 {city} 的 e 值为：{e_value}")
