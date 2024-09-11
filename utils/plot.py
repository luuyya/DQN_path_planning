import matplotlib.pyplot as plt
import numpy as np

def plot_map(grid, start, end):
    """
    绘制地图，显示起点和终点
    :param grid: 生成的地图
    :param start: 起点坐标
    :param end: 终点坐标
    """
    size = grid.shape[0]

    # 创建一个颜色映射表，0 -> 白色，1 -> 黑色（障碍物）
    cmap = plt.cm.get_cmap('Greys', 2)

    plt.figure(figsize=(8, 8))

    # 画出地图的障碍物（黑色）和空白区域（白色）
    plt.imshow(grid, cmap=cmap, origin="upper", extent=[0, size, 0, size])

    # 画出网格线
    plt.grid(True, which='both', color='black', linewidth=1)

    # 设置坐标轴的刻度以适应方格
    plt.xticks(np.arange(0, size + 1, 1))
    plt.yticks(np.arange(0, size + 1, 1))

    plt.fill_between([start[1], start[1] + 1], start[0], start[0] + 1, color='green', label="Start")
    plt.fill_between([end[1], end[1] + 1], end[0], end[0] + 1, color='red', label="End")
    plt.legend(loc='upper right')

    plt.show()