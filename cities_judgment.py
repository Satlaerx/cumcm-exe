import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 加载城市边界数据（完全保持您的变量名）
guangzhou = gpd.read_file('./data/广州市.json')
shenzhen = gpd.read_file('./data/深圳市.json')
dongguan = gpd.read_file('./data/东莞市.json')
foshan = gpd.read_file('./data/佛山市.json')

# 读取数据（保持您的原始列名和结构）
data1=pd.read_excel("./data/data1_with_difficulty_levels.xlsx",
                    names=["task_id","gps_0","gps_1","pricing","condition", "difficulty", "difficulty_d"])
data2 = pd.read_excel("./data/data2.xlsx",
                      names=["mem_id", "gps", "task_limit", "start_time", "credit"])

# 分割gps列（保持您的split逻辑，仅添加类型转换）
split_data = data2['gps'].str.split(' ', n=1, expand=True)
split_data.columns = ['gps_0', 'gps_1']  # 保持您的原始列名

# 关键修复：将字符串坐标转为float（不改变您的变量名）
split_data['gps_0'] = pd.to_numeric(split_data['gps_0'], errors='coerce')  # 纬度
split_data['gps_1'] = pd.to_numeric(split_data['gps_1'], errors='coerce')  # 经度
data1['gps_0']=pd.to_numeric(data1['gps_0'], errors='coerce')
data1['gps_1']=pd.to_numeric(data1['gps_1'], errors='coerce')

# 合并数据（完全保持您的原始拼接方式）
gps_idx = data2.columns.get_loc('gps')
data2 = pd.concat([
    data2.iloc[:, :gps_idx],
    split_data,
    data2.iloc[:, gps_idx + 1:]
], axis=1)


# 3. 城市判断函数（保持您的逻辑和变量名）
def determine_city(lon, lat):
    # 防御性编程：检查是否为有效数字
    if pd.isna(lon) or pd.isna(lat):
        return '坐标无效'

    point = Point(float(lon), float(lat))  # 显式转换为float

    if guangzhou.geometry.contains(point).any():
        return '广州'
    elif shenzhen.geometry.contains(point).any():
        return '深圳'
    elif dongguan.geometry.contains(point).any():
        return '东莞'
    elif foshan.geometry.contains(point).any():
        return '佛山'
    else:
        return '未知'


# 4. 应用函数（保持您的apply调用方式）
data1['city'] = data1.apply(
    lambda row: determine_city(row['gps_1'], row['gps_0']),  # 保持您的坐标顺序
    axis=1)
data2['city'] = data2.apply(
    lambda row: determine_city(row['gps_1'], row['gps_0']),  # 保持您的坐标顺序
    axis=1)

# 5. 保存结果
data1.to_excel('./data/data1_all.xlsx',index=False)
data2.to_excel('./data/data2_with_city.xlsx', index=False)
print("处理完成！无效坐标数量:", data2['city'].eq('坐标无效').sum())