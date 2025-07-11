import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和图片清晰度
plt.rcParams["font.family"] = ["Times New Roman","SimHei"]
plt.rcParams['figure.dpi'] = 1000

def fun(d, sigma):
    return np.exp(-d**2 / (2*sigma**2))

d_values = np.linspace(0, 20, 1000)
sigmas = [1, 2, 2.5, 3, 4, 5]

for sigma in sigmas:
    y = fun(d_values, sigma)  # 变量名改为y更符合惯例
    plt.plot(d_values, y, label=f'σ={sigma}')

plt.grid(True, linestyle='--', alpha=0.7)

# 修正x轴刻度
plt.xticks(np.arange(0, 21, 1))  # 从0到20，间隔2

# 添加图表元素
plt.title('不同σ值的高斯函数曲线')
plt.xlabel('距离 d')
plt.ylabel('函数值')
plt.legend()  # 显示图例

plt.tight_layout()  # 优化布局
plt.show()