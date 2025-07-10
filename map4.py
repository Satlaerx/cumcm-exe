import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.tri as tri
from matplotlib import rcParams

# ==================== 字体设置 ====================
font_path = '/Users/liumingxin/Library/Fonts/FangZhengHeiTiJianTi-1.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [font_prop.get_name()]
rcParams['axes.unicode_minus'] = False

# ==================== 数据加载 ====================
cluster_data = pd.read_csv("./data/cluster_centers_with_rates.csv")
cluster_centers = list(zip(cluster_data['latitude'], cluster_data['longitude']))
completion_rates = cluster_data['completion_rate'].values

# 计算显示范围（扩展5%缓冲）
lats = [c[0] for c in cluster_centers]
lngs = [c[1] for c in cluster_centers]
lat_buffer = (max(lats) - min(lats)) * 0.05
lng_buffer = (max(lngs) - min(lngs)) * 0.05
extent = [min(lngs)-lng_buffer, max(lngs)+lng_buffer,
          min(lats)-lat_buffer, max(lats)+lat_buffer]

# ==================== 创建地图 ====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3, edgecolor='blue')
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff', alpha=0.2)

# ==================== 电势图核心代码 ====================
from scipy.interpolate import Rbf

# 1. 创建插值网格
grid_size = 300  # 网格密度
xgrid = np.linspace(extent[0], extent[1], grid_size)
ygrid = np.linspace(extent[2], extent[3], grid_size)
X, Y = np.meshgrid(xgrid, ygrid)

# 2. 使用径向基函数插值（高斯核）
rbf = Rbf(lngs, lats, completion_rates,
          function='gaussian',
          epsilon=0.1)  # 控制平滑度
Z = rbf(X, Y)

# 确保插值结果在[0,1]范围内
Z = np.clip(Z, 0, 1)

# 3. 绘制插值后的电势图
contour = ax.contourf(
    X, Y, Z,
    levels=100,
    cmap='RdYlGn',
    transform=ccrs.PlateCarree(),
    alpha=0.7,
    zorder=5,
    antialiased=True
)

# 4. 添加原始数据点（带完成率颜色）
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

# 5. 调整显示范围（比数据范围扩大10%）
ax.set_extent([extent[0]-lng_buffer*2, extent[1]+lng_buffer*2,
               extent[2]-lat_buffer*2, extent[3]+lat_buffer*2],
              crs=ccrs.PlateCarree())
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
# 颜色条
cbar = plt.colorbar(contour, fraction=0.025, pad=0.01)
cbar.set_label('任务完成率', rotation=270, labelpad=20, fontsize=16)
cbar.ax.tick_params(labelsize=10)

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

# 标题
plt.title('珠三角地区任务完成率电势图\n完成率范围: {:.2f}~{:.2f}'.format(
    min(completion_rates), max(completion_rates)),
    fontsize=16, pad=18)

# 保存
plt.savefig('potential_map.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()