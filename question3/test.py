import numpy as np
import pandas as pd

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

import numpy as np


def new(params, mu, h_d, d, credit, MD, TD, e, k_array, h_values, L):
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


def old(params, mu, h_d, d, credit, MD, TD, e, k_array, h_values, L):
    # 拆分15个参数为5组，每组(a, b, P)
    param_groups = [
        (params[0], params[1], params[2]),
        (params[3], params[4], params[5]),
        (params[6], params[7], params[8]),
        (params[9], params[10], params[11]),
        (params[12], params[13], params[14]),
    ]

    # 参数合法性检查
    for (a, b, P) in param_groups:
        if a <= 0 or b <= 0 or P <= 0:
            return (-1e10,)  # 非法参数惩罚

    total_score = 0
    total_Pr = 0

    n_tasks, n_members = d.shape

    for i in range(n_tasks):
        k_i = k_array[i]
        k_index = int(k_i) - 1

        a_k, b_k, P_k = param_groups[k_index]
        MD_i = MD[i]
        TD_i = TD[i]
        e_i = e[i]
        h_d_i = h_d[i]
        L_k = L[k_index]

        C_i = P_k * (1 / (1 + a_k * MD_i)) * (1 / (1 + b_k * TD_i)) + h_values[i] + k_i * e_i

        log_prod = 0
        for j in range(n_members):
            d_ij = d[i][j]
            if d_ij < 0:
                continue

            S_ij = (C_i - k_i * e_i) / (mu * h_d_i + d_ij + L_k * (k_i - 1))
            P_ij = ALPHA * S_ij + BETA
            P_ij = np.clip(P_ij, 0, 1 - 1e-10)

            log_prod += credit[j] * np.log1p(-P_ij)

        prob_i = np.exp(log_prod)
        prob_i = 0 if np.isnan(prob_i) or prob_i < 1e-300 else prob_i

        total_score += (1 - prob_i)
        total_Pr += C_i

    if total_Pr > PR_TOTAL_THRESHOLD:
        return (-1e10,)

    return (total_score,)

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

params = np.random.uniform(0.5, 2.0, size=15)
out1 = old(params, MU, h_d, d_matrix, credit, MD, TD, e_values, k_array, h_values, L)
out2 = new(params, MU, h_d, d_matrix, credit, MD, TD, e_values, k_array, h_values, L)
print(out1, out2)
assert np.allclose(out1, out2, atol=1e-12), "两个函数结果不一致"
