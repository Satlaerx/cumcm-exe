import folium
from folium.plugins import FastMarkerCluster, MarkerCluster
import pandas as pd

# 函数：安全加载经纬度数据
def prepare_data(df):
    df = df[['gps_0', 'gps_1']].copy()
    df['lat'] = pd.to_numeric(df['gps_0'], errors='coerce')
    df['lng'] = pd.to_numeric(df['gps_1'], errors='coerce')
    return df.dropna(subset=['lat', 'lng'])

data1=pd.read_excel("./data/data1.xls",names=["task_id","gps_0","gps_1","pricing","condition"])
data2=pd.read_excel("./data/data2.xlsx",names=["mem_id", "gps", "task_limit", "start_time", "credit"])

split_data = data2['gps'].str.split(' ', n=1, expand=True)
split_data.columns = ['gps_0', 'gps_1']
gps_idx = data2.columns.get_loc('gps')
data2 = pd.concat([
    data2.iloc[:, :gps_idx],     # 之前的列
    split_data,                  # 新分割的两列
    data2.iloc[:, gps_idx+1:]    # 之后的列
], axis=1)
# 准备数据（完全不影响原始DataFrame）
df_tasks = prepare_data(data1)  # 任务数据
df_users = prepare_data(data2)  # 用户数据

# 创建地图（使用高德地图中文版）
m = folium.Map(
    location=[23.135, 113.33],
    zoom_start=10,
    tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
    attr='高德地图'
)

# 创建两个独立的图层组（方便控制显示）
task_layer = MarkerCluster(name="任务数据").add_to(m)
user_layer = MarkerCluster(name="用户数据").add_to(m)

# 标记任务数据（红色）
for _, row in df_tasks.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lng']],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup="任务ID: XXX"  # 可替换为实际数据列
    ).add_to(task_layer)

# 标记用户数据（蓝色）
for _, row in df_users.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lng']],
        radius=3,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup="用户ID: YYY"  # 可替换为实际数据列
    ).add_to(user_layer)

# 添加图层控制（右上角复选框）
folium.LayerControl(collapsed=False).add_to(m)

# 保存地图
m.save('map1.html')

print(f"任务数据点: {len(df_tasks)}个 | 用户数据点: {len(df_users)}个")