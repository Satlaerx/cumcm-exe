import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.interpolate import Rbf
from scipy.spatial import KDTree

# ==================== 配置参数 ====================
effective_radius = 0.1

# ==================== 字体设置 ====================
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['figure.dpi'] = 500
rcParams['axes.unicode_minus'] = False

# ==================== 数据加载 ====================
cluster_data = pd.read_csv("../data/cluster_centers_with_rates.csv")
cluster_centers = list(zip(cluster_data['latitude'], cluster_data['longitude']))
completion_rates = cluster_data['completion_rate'].values

# 计算显示范围（扩展10%缓冲）
lats = np.array([c[0] for c in cluster_centers])
lngs = np.array([c[1] for c in cluster_centers])
lat_buffer = (max(lats) - min(lats)) * 0.1
lng_buffer = (max(lngs) - min(lngs)) * 0.1
extent = [min(lngs) - lng_buffer, max(lngs) + lng_buffer,
          min(lats) - lat_buffer, max(lats) + lat_buffer]

# ==================== 创建地图 ====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3, edgecolor='blue')
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff', alpha=0.2)

# ==================== 改进的电势图 ====================
# 1. 创建KDTree用于最近邻搜索
points = np.column_stack([lngs, lats])
tree = KDTree(points)

# 2. 创建插值网格
grid_size = 300
xgrid = np.linspace(extent[0], extent[1], grid_size)
ygrid = np.linspace(extent[2], extent[3], grid_size)
X, Y = np.meshgrid(xgrid, ygrid)

# 3. 计算每个网格点到最近数据点的距离
grid_points = np.column_stack([X.ravel(), Y.ravel()])
distances, _ = tree.query(grid_points, k=1)
distance_mask = (distances <= effective_radius).reshape(X.shape)

# 4. 径向基函数插值
rbf = Rbf(lngs, lats, completion_rates,
          function='gaussian',
          epsilon=0.1)
Z = rbf(X, Y)
Z = np.clip(Z, 0, 1)

# 5. 应用距离掩码
Z[~distance_mask] = np.nan  # 超出有效半径的区域设为透明

# 6. 绘制电势图
contour = ax.contourf(
    X, Y, Z,
    levels=100,
    cmap='RdYlGn',
    transform=ccrs.PlateCarree(),
    alpha=0.7,
    zorder=5,
    antialiased=True,
    extend='neither'  # 不扩展颜色范围
)

# 7. 添加原始数据点
sc = ax.scatter(
    lngs, lats,
    c=completion_rates,
    cmap='RdYlGn',
    s=30,
    edgecolor='k',
    linewidth=0.5,
    transform=ccrs.PlateCarree(),
    zorder=10,
    alpha=0.8
)

# ==================== 城市标记 ====================
cities = {
    '广州': (23.13, 113.26),
    '深圳': (22.54, 114.05),
    '佛山': (23.02, 113.12),
    '东莞': (23.05, 113.75),
    '珠海': (22.27, 113.57)
}
for city, (lat, lng) in cities.items():
    if (extent[0] <= lng <= extent[1]) and (extent[2] <= lat <= extent[3]):
        ax.plot(lng, lat, 'o', color='black', markersize=5,
                transform=ccrs.PlateCarree(), zorder=15)
        ax.text(lng + 0.02, lat, city,
                transform=ccrs.PlateCarree(),
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'),
                zorder=15)

# ==================== 其他元素 ====================

# 添加颜色条
cbar = plt.colorbar(sc, fraction=0.025, pad=0.01)
cbar.set_label('任务完成率', rotation=270, labelpad=20, fontsize=16)
cbar.ax.tick_params(labelsize=14)

# 网格线
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.3,
    linestyle='--'
)
gl.top_labels = False
gl.right_labels = False

# 设置刻度标签的字体大小
gl.xlabel_style = {'size': 14}  # 经度刻度字体大小
gl.ylabel_style = {'size': 14}  # 纬度刻度字体大小

# 添加半径说明
ax.text(0.02, 0.02, f'有效插值半径: {effective_radius}度 (约{int(effective_radius * 111)}公里)',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, pad=5, edgecolor='none'),
        zorder=20)

plt.title('珠三角地区任务完成率分布\n(仅显示数据点{effective_radius}度范围内的插值)'.format(
    effective_radius=effective_radius),
    fontsize=16, pad=18)

# 保存
plt.savefig('map4.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
