import matplotlib.pyplot as plt
import numpy as np
# from env import Map

def plot_map(env, path=None):
    """
    绘制地图，显示起点、终点、障碍物、空白区域，并可选地显示路径（蓝色方格）。
    :param env: 环境对象，包含地图、起点和终点信息
    :param path: 路径列表，包含路径上的坐标（可选）
    """
    grid = env.get_grid()
    size = grid.shape[0]
    
    plt.figure(figsize=(8, 8))
    
    # 创建颜色映射
    color_map = np.zeros((size, size, 3))  # 初始化RGB颜色数组

    # 设置普通方格为白色
    color_map[grid == 0] = [1, 1, 1]  # 白色
    # 设置障碍物为黑色
    color_map[grid == 1] = [0, 0, 0]  # 黑色
    
    # 如果提供了路径，路径显示为蓝色
    if path:
        for position in path:
            color_map[position[0], position[1]] = [0, 0, 1]  # 蓝色表示路径

    # 设置起点和终点的颜色
    start = env.start
    end = env.end
    color_map[start[0], start[1]] = [0, 1, 0]  # 起点绿色
    color_map[end[0], end[1]] = [1, 0, 0]  # 终点红色

    plt.imshow(color_map, origin="lower", extent=[0, size, 0, size])
    
    # 画出网格线
    plt.grid(True, which='both', color='black', linewidth=1)
    
    # 设置坐标轴的刻度以适应方格
    plt.xticks(np.arange(0, size + 1, 1))
    plt.yticks(np.arange(0, size + 1, 1))

    # 调整 y 轴的方向，使其从上到下递增
    plt.gca().invert_yaxis()

    # 添加起点和终点标注
    plt.text(start[1] + 0.5, start[0] + 0.5, 'Start', ha='center', va='center', color='black')
    plt.text(end[1] + 0.5, end[0] + 0.5, 'End', ha='center', va='center', color='black')

    plt.title("Map with Path Visualization")
    plt.show()


# if __name__ == '__main__':
#     env = Map(size=20, obstacle_ratio=0.1, seed=40)
#     env.create_random_map()
#     env.initialize_start_end()
#     print(f"Start: {env.start}, End: {env.end}")
#     plot_map(env)