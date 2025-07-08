import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 构造数据（示例经纬度）
data = pd.DataFrame({
    'longitude': [113.9, 113.8, 114.0, 113.7],  # 珠三角经度
    'latitude': [22.5, 22.6, 22.4, 22.7]       # 珠三角纬度
})

# 转为GeoDataFrame
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

# 转换为Web墨卡托（底图通用投影）
gdf_web = gdf.to_crs(epsg=3857)

# 绘图
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制点位
gdf.plot(ax=ax, color='red', markersize=50, alpha=0.8, label='采样点')

# 添加腾讯地图底图（国内稳定）
ctx.add_basemap(
    ax,
    # 腾讯地图瓦片地址，{x}/{y}/{z} 为标准占位符
    source="https://rt{s}.map.gtimg.com/tile?z={z}&x={x}&y={y}&styleid=1&version=117",
    crs=gdf_web.crs,
    zoom=12,  # 地图缩放级别（12-16 适合珠三角）
    subdomains=["0", "1", "2", "3"]  # 腾讯地图多域名负载
)

# 调整地图范围（自动适配点位）
ax.set_xlim(gdf_web.total_bounds[[0, 2]])
ax.set_ylim(gdf_web.total_bounds[[1, 3]])

# 标题与标签
ax.set_title('珠三角地区点位分布', fontsize=15)
ax.set_xlabel('经度', fontsize=12)
ax.set_ylabel('纬度', fontsize=12)
ax.legend()

plt.tight_layout()
plt.show()