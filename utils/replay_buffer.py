import numpy as np
import random
from collections import deque

from utils.env import Map
from segment_tree import SumSegmentTree,MinSegmentTree

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
    def __init__(self, size, obs_dim, n_step=1, gamma=0.9):
        self.size = size

        self.next_idx = 0
        self.num_in_buffer = 0

        self.cur_obs = np.empty([self.size, obs_dim], dtype=np.int32)
        self.action = np.empty([self.size], dtype=np.int32)
        self.reward = np.empty([self.size], dtype=np.float32)
        self.done = np.empty([self.size], dtype=np.int32)
        self.next_obs = np.empty([self.size, obs_dim], dtype=np.int32)

        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def can_sample(self, batch_size):
        #判断是否能够进行采样
        return batch_size <= self.num_in_buffer

    def sample_batch(self, batch_size):
        #在buffer中进行采样，得到batch_size大小的数据
        if not self.can_sample(batch_size):
            raise ValueError("buffer未存储到足够的数据")

        indices=list(range(self.num_in_buffer))
        indexes=random.sample(indices,batch_size)

        cur_obs_batch = self.cur_obs[indexes]
        action_batch = self.action[indexes]
        r_batch = self.reward[indexes]
        next_obs_batch = self.next_obs[indexes]
        done_batch = self.done[indexes]

        return cur_obs_batch, action_batch, r_batch, next_obs_batch, done_batch

    def sample_batch_index(self,batch_size):
        if not self.can_sample(batch_size):
            raise ValueError("buffer未存储到足够的数据")

        indices=list(range(self.num_in_buffer))
        indexes=random.sample(indices,batch_size)
        return indexes

    def sample_batch_from_indexes(self, indexes):
        cur_obs_batch = self.cur_obs[indexes]
        action_batch = self.action[indexes]
        r_batch = self.reward[indexes]
        next_obs_batch = self.next_obs[indexes]
        done_batch = self.done[indexes]

        return cur_obs_batch, action_batch, r_batch, next_obs_batch, done_batch

    def store_frame(self, cur, a, r, d, next):
        transition = (cur, a, r, d, next)
        self.n_step_buffer.append(transition)

        r, d, next = self._get_n_step_info(self.n_step_buffer, self.gamma)
        cur, a = self.n_step_buffer[0][:3]

        self.cur_obs[self.next_idx] = cur
        self.action[self.next_idx] = a
        self.reward[self.next_idx] = r
        self.done[self.next_idx] = d
        self.next_obs[self.next_idx] = next

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def _get_n_step_info(self, n_step_buffer, gamma):
        def calculate_d(done):
            if done==1 or done==2:
                return False
            else:
                return True

        rew, done, next = n_step_buffer[-1][2:]

        # n_step_buffer[0]是cur_obs，n_step_buffer[-1]是next_obs(在之前未更新地图的情况下)
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, d, n = transition[2:]
            d=calculate_d(d)

            rew = r + gamma * rew * (1 - d)
            next, done = (n, d) if d else (next, done)

        return rew, done, next

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self, size, obs_dim, n_step=1, gamma=0.9, alpha=0.6):
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(size, obs_dim, n_step, gamma)
        self.max_priority = 0
        self.tree_ptr = 1.0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store_frame(self, cur, a, r, d, next):
        transition = super().store_frame(cur, a, r, d, next)
        self.n_step_buffer.append(transition)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.size

    def sample_batch(self, batch_size, beta=0.4):
        if not self.can_sample(batch_size):
            raise ValueError("buffer未存储到足够的数据")
        if not beta>0:
            raise ValueError("beta设置异常")

        indexes = self._sample_proportional()

        cur_obs_batch = self.cur_obs[indexes]
        action_batch = self.action[indexes]
        r_batch = self.reward[indexes]
        next_obs_batch = self.next_obs[indexes]
        done_batch = self.done[indexes]
        weights = np.array([self._calculate_weight(i, beta) for i in indexes])

        return cur_obs_batch, action_batch, r_batch, next_obs_batch, done_batch, weights

    def update_priorities(self, indexes, priorities):
        """Update priorities of sampled transitions."""
        assert len(indexes) == len(priorities)

        for idx, priority in zip(indexes, priorities):
            assert priority > 0
            # assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        indexes = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indexes.append(idx)

        return indexes

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight

        return weight
    
