import copy
import numpy as np

CURRENT_POSITION = 2
END_POSITION = 3

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
        self.start = None
        self.end = None
        self.cur = None
        self.depth = 0
        self.episode_rewards = []  # 用于记录每一回合的累积奖励
        self.current_episode_reward = 0  # 当前回合奖励

        if self.seed is not None:
            np.random.seed(self.seed)

    def create_random_map(self):
        """
        创建一个随机地图，并在地图上随机放置障碍物
        :return: 生成的地图
        """
        self.grid = np.zeros((self.size, self.size))

        obstacle_num = int(self.size * self.size * self.obstacle_ratio)
        total_cells = self.size * self.size

        # 无放回地选择障碍物位置
        obstacle_indices = np.random.choice(total_cells, obstacle_num, replace=False)
        obstacle_coords = np.unravel_index(obstacle_indices, (self.size, self.size))

        self.grid[obstacle_coords] = 1

    def initialize_start_end(self):
        """
        初始化起点和终点，确保它们不在障碍物上
        :return: 起点和终点的坐标
        """
        if self.grid is None:
            raise ValueError("请先创建地图。")

        available_coords = np.argwhere(self.grid == 0)  # 获取所有可用坐标

        if len(available_coords) < 2:
            raise ValueError("可用的起点和终点坐标不足。")

        start_index = np.random.choice(len(available_coords))
        self.start = available_coords[start_index]

        available_coords = np.delete(available_coords, start_index, axis=0)  # 删除已选择的起点
        end_index = np.random.choice(len(available_coords))
        self.end = available_coords[end_index]

        self.cur = self.start

    def get_grid(self):
        return self.grid

    def get_total_depth(self):
        return self.depth

    def step(self, action):
        #todo: need to do some modify
        """
        根据动作更新当前状态，并返回当前状态，动作，奖励，下一个状态，以及是否结束
        :param action: 移动的方向，0=上，1=下，2=左，3=右
        :return: 当前状态，动作，奖励，下一个状态，done
        """
        # 定义移动方向
        directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }

        if action not in directions:
            raise ValueError("无效的动作。")

        current_state = copy.deepcopy(self.grid)  # 保存当前状态
        current_state[self.cur[0], self.cur[1]] = CURRENT_POSITION
        current_state[self.end[0], self.end[1]] = END_POSITION

        # 计算下一个位置
        move = directions[action]
        next_position = (self.cur[0] + move[0], self.cur[1] + move[1])

        # 初始化下一个状态、奖励和done标志
        next_state = copy.deepcopy(self.grid)
        reward = -1  # 默认奖励
        done = False  # 默认未结束

        # 检查下一个位置是否在边界内
        if (0 <= next_position[0] < self.size) and (0 <= next_position[1] < self.size):
            # 检查下一个位置是否为障碍物
            if self.grid[next_position] == 0:
                # 有效移动，更新当前位置和深度
                self.cur = next_position
                self.depth += 1

                # 检查是否到达终点
                if np.array_equal(self.cur, self.end):
                    reward = 10  # 到达终点的奖励
                    done = True
            else:
                reward = -10  # 碰到障碍物的惩罚
        else:
            reward = -10  # 出界的惩罚

        # 更新当前回合的奖励
        self.current_episode_reward += reward

        next_state[self.cur[0]][self.cur[1]] = CURRENT_POSITION
        next_state[self.end[0]][self.end[1]] = END_POSITION
        
        return current_state, action, reward, next_state, done

    def reset(self):
        """
        重置环境，返回初始状态
        :return: 初始状态的网格
        """
        self.cur = self.start.copy()  # 重置当前位置为起点
        self.depth = 0  # 重置深度

        # 将当前回合的奖励记录到回合奖励列表中
        if self.current_episode_reward != 0:
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0  # 重置当前回合奖励

        # 创建初始状态的网格副本
        grid = self.grid.copy()
        grid[self.cur[0]][self.cur[1]] = CURRENT_POSITION  # 标记起点
        grid[self.end[0]][self.end[1]] = END_POSITION  # 标记终点

        return grid

    def get_episode_rewards(self):
        """
        获取所有回合的累积奖励
        :return: 每一回合的累积奖励列表
        """
        return self.episode_rewards

    def get_current_state(self):
        cur_state = self.grid.copy()
        cur_state[self.cur[0]][self.cur[1]] = 5
        cur_state[self.end[0]][self.end[1]] = 6

        return cur_state



if __name__ == '__main__':
    env = Map(size=8, obstacle_ratio=0.1, seed=34)
    env.create_random_map()
    env.initialize_start_end()
    print(f"Start: {env.start}, End: {env.end}")

    grid = env.get_grid()  # 获取网格数据

    print(env.get_total_depth())

    res = env.step(0)
    print(res[0])
    print(res[1])
    print(res[2])
    print(res[3])
    print(res[4])

    print(env.get_total_depth())

    