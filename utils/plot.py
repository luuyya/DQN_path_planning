import matplotlib.pyplot as plt
import numpy as np

def plot_map(grid, start, end):
    """
    绘制地图，显示起点、终点、障碍物和空白区域
    :param grid: 生成的地图
    :param start: 起点坐标
    :param end: 终点坐标
    """
    size = grid.shape[0]
    
    plt.figure(figsize=(8, 8))
    
    # 创建颜色映射
    color_map = np.zeros((size, size, 3))  # 初始化RGB颜色数组

    # 设置普通方格为白色
    color_map[grid == 0] = [1, 1, 1]  # 白色
    # 设置障碍物为黑色
    color_map[grid == 1] = [0, 0, 0]  # 黑色
    # 设置起点和终点的颜色
    color_map[start[0], start[1]] = [0, 1, 0]  # 起点颜色
    color_map[end[0], end[1]] = [1, 0, 0]  # 终点颜色

    plt.imshow(color_map, origin="upper", extent=[0, size, 0, size])
    
    # 画出网格线
    plt.grid(True, which='both', color='black', linewidth=1)
    
    # 设置坐标轴的刻度以适应方格
    plt.xticks(np.arange(0, size + 1, 1))
    plt.yticks(np.arange(0, size + 1, 1))

    # 添加起点和终点标注，注意调整 y 坐标
    plt.text(start[1] + 0.5, size - start[0] - 0.5, 'Start', ha='center', va='center', color='black')
    plt.text(end[1] + 0.5, size - end[0] - 0.5, 'End', ha='center', va='center', color='black')

    plt.title("Map Visualization")
    
    plt.show()

def plot_path(env, path):
    """
    绘制地图并显示路径。

    参数：
    - env: Map 类的实例，包含地图信息
    - path: list，包含路径上的坐标
    """
    # 创建地图的副本以进行绘制
    grid_map = np.array(env.map)

    # 绘制路径
    for position in path:
        grid_map[position[0], position[1]] = 2  # 用值 2 表示路径

    # 设置绘图
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_map, cmap='gray_r')
    plt.title('Path Planning Result')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    plt.grid(True, which='both', color='black', linestyle='--', linewidth=0.5)

    # 显示起点和终点
    start, end = env.start, env.end
    plt.scatter(start[1], start[0], color='green', s=100, label='Start')
    plt.scatter(end[1], end[0], color='red', s=100, label='End')

    plt.legend()
    plt.show()