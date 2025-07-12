import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.sparse import csr_matrix
from skopt import gp_minimize
from skopt.space import Real

# 假设数据已加载（示例结构）
data1 = pd.DataFrame({
    'MD_i': [1.0, 1.5, ...],  # 834行，每个i对应的MD_i
    'TD_i': [0.8, 1.2, ...],  # 834行，每个i对应的TD_i
    'l_i':  [0.5, 0.3, ...],  # 834行，l_i参数
    'w_i':  [10, 20, ...]     # 每个i的w_i值（j的上限）
})

distance = pd.DataFrame({
    0: [1.0, 2.0, ...],       # 第1列d_i1
    1: [1.5, 3.0, ...],       # 第2列d_i2
})  # 共834行，每行长度可能不同

# 转换为CSR稀疏格式（跳过-1值）
data = []
indices = []
indptr = [0]

for i in range(834):
    row = distance.iloc[i].values
    valid_mask = row != -1
    valid_data = row[valid_mask]

    data.extend(valid_data)
    indices.extend(np.where(valid_mask)[0])
    indptr.append(len(data))

d_ij_sparse = csr_matrix((data, indices, indptr), shape=(834, distance.shape[1]))


@jit(nopython=True, parallel=True)
def objective_function(P0, a, b, alpha, MD_i, TD_i, l_i, w_i, d_ij_data, d_ij_indices, d_ij_indptr):
    """
    参数说明:
        P0, a, b: 待优化变量
        alpha: 常数
        MD_i, TD_i, l_i, w_i: 长度为834的数组
        d_ij_data, d_ij_indices, d_ij_indptr: CSR格式稀疏矩阵
    """
    total = 0.0
    for i in prange(834):
        sum_i = 1.0
        w_i_current = w_i[i]
        row_start = d_ij_indptr[i]
        row_end = d_ij_indptr[i + 1]

        # 计算核心项分母
        denominator = (1 + a * MD_i[i]) * (1 + b * TD_i[i])
        core = (P0 / denominator) + l_i[i]

        # 遍历有效d_ij
        for idx in range(row_start, row_end):
            j = d_ij_indices[idx]
            if j >= w_i_current:
                continue  # 跳过超出w_i范围的j
            d_ij = d_ij_data[idx]
            term = 1 - alpha * core / (d_ij + l_i[i])
            sum_i *= term

        total += (1 - sum_i)
    return -total  # 取负是因为原问题是最大化

def wrapped_objective(x):
    P0, a, b = x
    return objective_function(
        P0, a, b, alpha=0.1,  # 假设alpha=0.1（根据实际修改）
        MD_i=data1['MD_i'].values,
        TD_i=data1['TD_i'].values,
        l_i=data1['l_i'].values,
        w_i=data1['w_i'].values,
        d_ij_data=d_ij_sparse.data,
        d_ij_indices=d_ij_sparse.indices,
        d_ij_indptr=d_ij_sparse.indptr
    )

# 定义搜索空间
space = [
    Real(0.1, 10.0, name="P0"),   # P0范围
    Real(0.01, 1.0, name="a"),    # a范围
    Real(0.01, 1.0, name="b")     # b范围
]

# 运行优化
res = gp_minimize(
    wrapped_objective,
    space,
    n_calls=50,                   # 总评估次数
    n_initial_points=20,          # 初始采样点
    acq_func="EI",                # 采集函数
    random_state=42,
    verbose=True
)

# 输出结果
print("最优解 (P0, a, b):", res.x)
print("最优目标值:", -res.fun)      # 转换回最大值