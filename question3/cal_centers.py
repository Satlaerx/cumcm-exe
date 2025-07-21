import pandas as pd

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

        # 修改data1
        data1.loc[i, "gps_0"] = gps_0_center
        data1.loc[i, "gps_1"] = gps_1_center

        # 删掉被合并的其他数据
        data1.drop(index=range(i + 1, i + cluster_size), inplace=True)
        data1.reset_index(drop=True, inplace=True)

        i += 1

data1.reset_index(drop=True, inplace=True)

data1.to_excel("../data/data1_merged.xlsx", index=False)
