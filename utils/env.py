import numpy as np

class Map:
    def __init__(self, size=8, obstacle_ratio=0.1, seed=None):
        """
        初始化 Map 类
        :param size: 地图大小（默认 8x8）
        :param obstacle_ratio: 障碍物占比（默认为 10%）
        :param seed: 随机种子（可选）
        """
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.seed = seed
        self.grid = None

    def create_random_map(self):
        """
        创建一个随机地图，并在地图上随机放置障碍物
        :return: 生成的地图
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.grid = np.zeros((self.size, self.size))

        obstacle_num = int(self.size * self.size * self.obstacle_ratio)

        obstacle_coords = np.random.randint(0, self.size, size=(obstacle_num, 2))

        print(obstacle_coords)

        for coord in obstacle_coords:
            self.grid[coord[0], coord[1]] = 1

    def initialize_start_end(self):
        """
        初始化起点和终点，确保它们不在障碍物上
        :return: 起点和终点的坐标
        """
        if self.grid is None:
            raise ValueError("请先创建地图。")

        if self.seed is not None:
            np.random.seed(self.seed)

        start, end = None, None
        available_coords = np.argwhere(self.grid == 0)  # 获取所有可用坐标

        if len(available_coords) < 2:
            raise ValueError("可用的起点和终点坐标不足。")

        start_index = np.random.choice(len(available_coords))
        start = available_coords[start_index]

        available_coords = np.delete(available_coords, start_index, axis=0)  # 删除已选择的起点
        end_index = np.random.choice(len(available_coords))
        end = available_coords[end_index]

        return start, end
    
    def get_grid(self):
        return self.grid
