import pandas as pd

# 读取数据（确保列名正确）
data1 = pd.read_excel('./data/data1_with_difficulty.xlsx',
                     names=["task_id", "gps_0", "gps_1", "pricing", "condition", "difficulty"])

# 计算极差和区间宽度
difficulty_min = data1["difficulty"].min()
difficulty_max = data1["difficulty"].max()
range_width = (difficulty_max - difficulty_min) / 5

# 生成区间边界（前4个前闭后开，第5个闭区间）
bins = [difficulty_min + i * range_width for i in range(5)] + [difficulty_max]
labels = [f"{i+1}" for i in range(5)]  # 区间标签可自定义

# 使用 pd.cut 划分区间（right=False 控制前闭后开，最后一个区间单独处理）
data1["difficulty_d"] = pd.cut(
    data1["difficulty"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True  # 确保第一个区间包含最小值
)

# 验证划分结果（可选）
print("区间边界:", bins)
print("\n各区间的数量分布:")
print(data1["difficulty_d"].value_counts().sort_index())

# 保存结果（可选）
data1.to_excel("./data/data1_with_difficulty_levels.xlsx", index=False)