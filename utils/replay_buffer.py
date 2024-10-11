import numpy as np
import random

from utils.env import Map

class ReplayBuffer(object):
    """
    缓冲区，存放状态动作空间，一个循环队列
    存储的数据：
    当前状态
    行动
    回报
    判断因子：0正常 1到达终点 2碰到障碍物或者出界
    下一个状态
    """
    def __init__(self, size):
        self.size = size

        self.next_idx = 0
        self.num_in_buffer = 0

        self.cur_obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.next_obs = None

    def can_sample(self, batch_size):
        #判断是否能够进行采样
        return batch_size <= self.num_in_buffer

    # def encode_sample(self, indexes):
    #     obs_batch = np.array([self.obs[idx] for idx in indexes])
    #     act_batch = self.action[indexes]
    #     rew_batch = self.reward[indexes]
    #     next_obs_batch = np.array([self.obs[(idx + 1) % self.size] for idx in indexes])
    #     done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in indexes], dtype=np.float32)
    #
    #     return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample_batch(self, batch_size):
        #在buffer中进行采样，得到batch_size大小的数据
        assert self.can_sample(batch_size), "buffer未存储到足够的数据"

        indices=list(range(self.num_in_buffer))
        indexes=random.sample(indices,batch_size)

        cur_obs_batch = self.cur_obs[indexes]
        action_batch = self.action[indexes]
        r_batch = self.reward[indexes]
        next_obs_batch = self.next_obs[indexes]
        done_batch = self.done[indexes]

        return cur_obs_batch, action_batch, r_batch, next_obs_batch, done_batch

    # def encode_recent_observation(self):
    #     assert self.num_in_buffer > 0,"buffer未存储数据"
    #     return self.obs[(self.next_idx - 1) % self.size]

    def store_frame(self, cur, a, r, d, next):
        if self.cur_obs is None:
            self.cur_obs = np.empty([self.size]+list(cur.shape), dtype=np.int32)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.int32)
            self.next_obs=np.empty([self.size]+list(cur.shape), dtype=np.int32)

        self.cur_obs[self.next_idx] = cur
        self.action[self.next_idx] = a
        self.reward[self.next_idx] = r
        self.done[self.next_idx] = d
        self.next_obs[self.next_idx] = next

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    # def store_effect(self, idx, action, reward, done):
    #     self.action[idx] = action
    #     self.reward[idx] = reward
    #     self.done[idx] = done

def main():
    # 假设有四个方向的动作，分别是：上、下、左、右
    ACTIONS = [0, 1, 2, 3]  # 0: 上, 1: 下, 2: 左, 3: 右
    
    # 初始化地图大小为10x10，障碍物比例为0.2
    size = 10
    obstacle_ratio = 0.2
    env = Map(size=size, obstacle_ratio=obstacle_ratio)
    env.create_random_map()
    env.initialize_start_end()  # 初始化起点和终点

    # 初始化ReplayBuffer，大小为100
    buffer_size = 100
    replay_buffer = ReplayBuffer(size=buffer_size)

    # 模拟10次路径规划
    for episode in range(10):
        env.reset()  # 重置环境
        state = env.get_current_state()  # 获取当前状态，假设有cur_state函数返回坐标
        
        for t in range(20):  # 每个episode最多20步
            action = random.choice(ACTIONS)  # 随机选择一个动作
            
            # 执行动作，假设move函数返回新的状态，奖励，结束标志和额外信息
            _, reward, done, next_state = env.step(action)

            # 存储当前经历到ReplayBuffer
            replay_buffer.store_frame(state, action, reward, done, next_state)

            # 更新状态
            state = next_state
            
            # 如果到达终点或发生碰撞，结束当前episode
            if done == 1 or done == 2:
                print(f"Episode {episode} ends at step {t} with done flag {done}.")
                break

    # 从缓冲区中采样
    batch_size = 5
    if replay_buffer.can_sample(batch_size):
        cur_obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample_batch(batch_size)
        print(type(done_batch))
        
        print("Sampled a batch from ReplayBuffer:")
        print("Current observations:", cur_obs_batch)
        print("Actions:", action_batch)
        print("Rewards:", reward_batch)
        print("Next observations:", next_obs_batch)
        print("Done flags:", done_batch)
    else:
        print("Not enough data in the buffer to sample a batch.")

if __name__ == "__main__":
    main()