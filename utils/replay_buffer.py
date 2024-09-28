import numpy as np
import random

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    """缓冲区，存放状态动作空间"""
    def __init__(self, size):
        self.size = size
        # self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        # obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        obs_batch      = np.array([self.obs[idx] for idx in idxes])
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        # next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        next_obs_batch = np.array([self.obs[(idx + 1) % self.size] for idx in idxes])
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        # return self._encode_observation((self.next_idx - 1) % self.size)
        return self.obs[(self.next_idx - 1) % self.size]

    def _encode_observation(self, idx):
        # 不用那么复杂，直接返回状态
        return self.obs[idx]

    def store_frame(self, frame):
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


if __name__ == '__main__':
    # 创建一个大小为100的ReplayBuffer
    replay_buffer = ReplayBuffer(100)

    # 定义网格地图大小
    grid_size = (5, 5)

    # 定义初始位置
    agent_position = [0, 0]

    # 定义动作空间：0-上，1-下，2-左，3-右
    action_space = [0, 1, 2, 3]

    # 定义目标位置
    goal_position = [4, 4]

    # 运行一定的步骤来模拟路径规划
    for _ in range(200):
        # 存储当前状态（位置）
        state = np.array(agent_position)
        idx = replay_buffer.store_frame(state)

        # 随机选择一个动作
        action = random.choice(action_space)

        # 根据动作更新位置
        next_position = agent_position.copy()
        if action == 0 and agent_position[0] > 0:
            next_position[0] -= 1  # 上
        elif action == 1 and agent_position[0] < grid_size[0] - 1:
            next_position[0] += 1  # 下
        elif action == 2 and agent_position[1] > 0:
            next_position[1] -= 1  # 左
        elif action == 3 and agent_position[1] < grid_size[1] - 1:
            next_position[1] += 1  # 右
        # 如果动作导致出界，位置保持不变

        # 定义奖励：到达目标位置奖励1，否则为-0.1
        if next_position == goal_position:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        # 存储效果
        replay_buffer.store_effect(idx, action, reward, done)

        # 更新位置
        agent_position = next_position

        # 如果到达目标位置，重置位置
        if done:
            agent_position = [0, 0]

    # 检查是否可以采样
    if replay_buffer.can_sample(5):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(5)
        print("obs_batch:", obs_batch)
        print("act_batch:", act_batch)
        print("rew_batch:", rew_batch)
        print("next_obs_batch:", next_obs_batch)
        print("done_mask:", done_mask)
    else:
        print("无法采样，缓冲区中的数据不足。")