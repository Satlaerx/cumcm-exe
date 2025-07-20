import pandas as pd

# 读取数据（确保列名正确）
data1 = pd.read_excel("../data/data1_all.xlsx",)

# 筛选出目标城市（排除“未知”）
target_cities = ["广州", "深圳", "佛山", "东莞"]
filtered_data = data1[data1["city"].isin(target_cities)]

# 计算各城市condition的平均值
city_condition_sum=filtered_data.groupby("city")["condition"].sum()
city_condition_means = filtered_data.groupby("city")["condition"].mean()
city_pricing_means=filtered_data.groupby("city")["pricing"].mean()

pricing_min=data1["pricing"].min()
pricing_max=data1["pricing"].max()

condition_rate=data1["condition"].mean()

# 打印结果（保留4位小数）
print("各城市condition的和:")
print(city_condition_sum)

print("总完成率:")
print(condition_rate)

print("各城市condition的平均值:")
print(city_condition_means.round(4))

print("各城市pricing的平均值:")
print(city_pricing_means.round(4))

print("最低定价",pricing_min)
print("最高定价",pricing_max)