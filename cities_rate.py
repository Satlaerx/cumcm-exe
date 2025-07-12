import pandas as pd

# 读取数据（确保列名正确）
data1 = pd.read_excel("./data/data1_all.xlsx",
                     names=["task_id", "gps_0", "gps_1", "pricing", "condition",
                            "difficulty", "difficulty_d", "city"])

# 筛选出目标城市（排除“未知”）
target_cities = ["广州", "深圳", "佛山", "东莞"]
filtered_data = data1[data1["city"].isin(target_cities)]

# 计算各城市condition的平均值
city_condition_means = filtered_data.groupby("city")["condition"].mean()

# 打印结果（保留4位小数）
print("各城市condition的平均值:")
print(city_condition_means.round(4))