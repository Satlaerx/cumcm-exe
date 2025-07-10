import folium
import pandas as pd

# 数据准备函数
def prepare_data(df):
    df = df[['gps_0', 'gps_1']].copy()
    df['lat'] = pd.to_numeric(df['gps_0'], errors='coerce')
    df['lng'] = pd.to_numeric(df['gps_1'], errors='coerce')
    return df.dropna(subset=['lat', 'lng'])

# 加载数据
data1 = pd.read_excel("./data/data1.xls", names=["task_id","gps_0","gps_1","pricing","condition"])
data2 = pd.read_excel("./data/data2.xlsx", names=["mem_id", "gps", "task_limit", "start_time", "credit"])

# 处理data2的GPS数据
split_data = data2['gps'].str.split(' ', n=1, expand=True)
split_data.columns = ['gps_0', 'gps_1']
gps_idx = data2.columns.get_loc('gps')
data2 = pd.concat([
    data2.iloc[:, :gps_idx],
    split_data,
    data2.iloc[:, gps_idx+1:]
], axis=1)

# 准备清洗后的数据
df_tasks = prepare_data(data1)  # 任务数据
df_users = prepare_data(data2)  # 用户数据

# 创建高德地图底图
m = folium.Map(
    location=[23.135, 113.33],
    zoom_start=11,
    # width=800,
    # height=600,
    tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
    attr='高德地图'
)

# 设置点半径
LARGE_RADIUS = 300

# 添加任务数据点（红色）
for _, row in df_tasks.iterrows():
    folium.Circle(
        location=[row['lat'], row['lng']],
        radius=LARGE_RADIUS,
        color='#e41a1c',
        fill=True,
        fill_opacity=0.7,
        weight=0
    ).add_to(m)

# 添加用户数据点（蓝色）
for _, row in df_users.iterrows():
    folium.Circle(
        location=[row['lat'], row['lng']],
        radius=LARGE_RADIUS,
        color='#377eb8',
        fill=True,
        fill_opacity=0.7,
        weight=0
    ).add_to(m)

# 优化后的图例（右下角+紧凑尺寸）
legend_html = '''
<div style="position: fixed; 
     bottom: 20px; right: 20px; 
     border:1px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity:0.9; padding: 4px 8px;
     font-family: Arial, sans-serif;
     line-height: 1.2;">
     <div style="display: flex; align-items: center; margin: 2px 0;">
          <div style="width:12px; height:12px; background:#e41a1c; margin-right:6px;"></div>
          <span>任务点</span>
     </div>
     <div style="display: flex; align-items: center; margin: 2px 0;">
          <div style="width:12px; height:12px; background:#377eb8; margin-right:6px;"></div>
          <span>用户点</span>
     </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 保存地图
m.save('map2.html')