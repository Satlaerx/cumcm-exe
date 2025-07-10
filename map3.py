import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# 指定方正黑体的绝对路径
font_path = '/Users/liumingxin/Library/Fonts/FangZhengHeiTiJianTi-1.ttf'

# 注册字体到Matplotlib
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)  # 全局添加字体

# 设置全局默认字体（通过字体的PostScript名称或家族名）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [font_prop.get_name()]  # 自动获取字体注册名
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
cluster_data = pd.read_csv("./data/cluster_centers_with_rates.csv")
cluster_centers = list(zip(cluster_data['latitude'], cluster_data['longitude']))
completion_rates = cluster_data['completion_rate'].values

# 自动计算最佳显示范围（扩展5%的缓冲区域）
lats = [c[0] for c in cluster_centers]
lngs = [c[1] for c in cluster_centers]
lat_buffer = (max(lats) - min(lats)) * 0.05
lng_buffer = (max(lngs) - min(lngs)) * 0.05

extent = [
    min(lngs) - lng_buffer,  # 西经
    max(lngs) + lng_buffer,  # 东经
    min(lats) - lat_buffer,  # 南纬
    max(lats) + lat_buffer   # 北纬
]

# 创建地图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加地理特征（简化版，避免遮挡数据点）
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3, edgecolor='blue')
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff', alpha=0.2)

# 绘制热力图（增大点尺寸）
sc = ax.scatter(
    x=lngs,
    y=lats,
    c=completion_rates,
    s=100,  # 增大点尺寸
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree(),
    edgecolors='k',
    linewidths=0.5,
    alpha=0.9,
    zorder=10
)

# 设置动态显示范围
ax.set_extent(extent, crs=ccrs.PlateCarree())

# 添加颜色条
cbar = plt.colorbar(sc, fraction=0.025, pad=0.01)
cbar.set_label('任务完成率', rotation=270, labelpad=20, fontsize=16)
cbar.ax.tick_params(labelsize=10)

# 添加主要城市标记（可选）
cities = {
    '广州': (23.13, 113.26),
    '深圳': (22.54, 114.05),
    '佛山': (23.02, 113.12),
    '东莞': (23.05, 113.75),
    '珠海': (22.27, 113.57)
}
for city, (lat, lng) in cities.items():
    if (extent[0] <= lng <= extent[1]) and (extent[2] <= lat <= extent[3]):
        # 标记点（黑色圆点）
        ax.plot(lng, lat, 'o',
                color='black',
                markersize=4,
                transform=ccrs.PlateCarree(),
                zorder=15)  # 提高层级

        # 城市名称（显示在标记点右方，水平对齐）
        ax.text(lng + 0.02, lat, city,  # x坐标增加0.02度偏移
                transform=ccrs.PlateCarree(),
                fontsize=12,
                color='black',
                ha='left',   # 水平左对齐（从标记点向右延伸）
                va='center', # 垂直居中
                zorder=15)

# 添加网格线
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.3,
    linestyle='--'
)
gl.top_labels = False  # 关闭顶部标签
gl.right_labels = False  # 关闭右侧标签

# 添加标题
plt.title('珠三角地区任务完成率热力图（46个聚类中心）\n完成率范围: {:.2f}~{:.2f}'.format(
    min(completion_rates), max(completion_rates)),
    fontsize=16, pad=18)

# 保存图片
plt.savefig('map3.pdf',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')

plt.show()