import pandas as pd

# 将同一个包内的多个任务合成 1 个，
# 计算中心经纬度，合并 difficulty 值和 difficulty_d 的值

data1 = pd.read_excel("../data/data1_sorted.xlsx")
data2 = pd.read_excel("../data/data2_with_city_10.xlsx")

i = 0  # 当前行数字索引
while i < len(data1):
    cluster_size = data1.loc[i, "cluster_size"]
    if cluster_size == 1:
        i += 1
        continue
    else:
        data1_cluster = data1.iloc[i:i + cluster_size, :].copy()
        # 计算中心位置
        gps_0_center = data1_cluster["gps_0"].mean()
        gps_1_center = data1_cluster["gps_1"].mean()
        difficulty_sum = data1_cluster["difficulty"].sum()
        difficulty_d_sum = data1_cluster["difficulty_d"].sum()

        # 修改data1
        data1.loc[i, "gps_0"] = gps_0_center
        data1.loc[i, "gps_1"] = gps_1_center
        data1.loc[i, "difficulty"] = difficulty_sum
        data1.loc[i, "difficulty_d"] = difficulty_d_sum

        # 删掉被合并的其他数据
        data1.drop(index=range(i + 1, i + cluster_size), inplace=True)
        data1.reset_index(drop=True, inplace=True)

        i += 1

data1.reset_index(drop=True, inplace=True)

data1.to_excel("../data/data1_merged.xlsx", index=False)
