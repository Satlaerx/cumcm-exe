import pandas as pd

# 假设数据已经加载
data1 = pd.read_excel("../data/data1_all.xlsx")
distance3 = pd.read_excel("../data/task_to_member_distance_5km.xlsx")

# 获取各个城市的索引
cities = ['广州', '深圳', '东莞', '佛山']
city_dfs = {}

# 分别获取各个城市对应的DataFrame
for city in cities:
    city_indices = data1[data1['city'] == city].index
    city_dfs[city] = distance3.loc[city_indices]

# 输出为Excel文件
with pd.ExcelWriter("../data/city_distance.xlsx") as writer:
    for city in cities:
        # 将每个城市的distance和satisfaction数据分别写入不同的sheet
        city_dfs[city].to_excel(writer, sheet_name=f'{city}', index=False)
