import random
import numpy as np

from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
from utils.schedules import LinearSchedule
from collections import namedtuple
import os
import sys

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA变量
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

LOG_EVERY_N_STEPS = 1000
SAVE_MODEL_EVERY_N_STEPS = 10000
STOP_CONDITION = 500
PRIOR_EPS = 1e-6
V_MIN = -1
V_MAX = 100
NUM_FRAMES = 100000


def calculate_loss(replay_buffer_one,
                   replay_buffer_n,
                   prioritized_buffer, 
                   n_step, 
                   batch_size,
                   beta, 
                   gamma,
                   Q,
                   Q_target,
                   double_dqn):
    indexes=replay_buffer_one.sample_batch_index(batch_size)
    # print(indexes)+
    weights = None
    if prioritized_buffer:
        cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, weights = replay_buffer_one.sample_batch_from_indexes(indexes, beta)
        weights = torch.FloatTensor(weights.reshape(-1, 1)).type(dtype)
    else:
        cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer_one.sample_batch_from_indexes(indexes)
    
    samples = (cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)
    elementwise_loss = compute_dqn_loss(samples, gamma, Q, Q_target, double_dqn)

    if prioritized_buffer:
        loss = torch.mean(elementwise_loss * weights)
    else:
        loss = elementwise_loss

    if n_step > 1:
        gamma = gamma ** n_step
        samples = replay_buffer_n.sample_batch_from_indexes(indexes)
        elementwise_loss_n_loss = compute_dqn_loss(samples, gamma, Q, Q_target, double_dqn)
        elementwise_loss += elementwise_loss_n_loss

        loss = torch.mean(elementwise_loss * weights)

    new_priorities = None

    if prioritized_buffer:
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + PRIOR_EPS
    
    return loss, indexes, new_priorities

def compute_dqn_loss(samples, 
                     gamma, 
                     Q, 
                     Q_target, 
                     double_dqn):
    state = torch.FloatTensor(samples[0]).type(dtype)
    next_state = torch.FloatTensor(samples[3]).type(dtype)
    action = torch.LongTensor(samples[1]).type(dlongtype)
    reward = torch.FloatTensor(samples[2]).type(dtype)
    done = torch.LongTensor(samples[4]).type(dlongtype)

    Q_values = Q(state.unsqueeze(1))
    Q_c_a = Q_values.gather(1, action.unsqueeze(1))  # 选取指定action的q值
    Q_c_a = Q_c_a.squeeze()

    if double_dqn:
        # Double DQN
        Q_n_values = Q(next_state.unsqueeze(1)).detach()
        _, a_index = Q_n_values.max(1)

        # 选取Q_target中在Q中最大的动作
        Q_target_n_values = Q_target(next_state.unsqueeze(1)).detach()
        Q_target_a_index = Q_target_n_values.gather(1, a_index.unsqueeze(1))
        Q_target_a_index = Q_target_a_index.squeeze()

        # 将进入死状态的obs的Q_target设置为0
        judgement = torch.from_numpy(np.where(done.cpu().numpy() == 0, 1, 0)).type(dtype)
        Q_target_a_index = judgement * Q_target_a_index

        # 使用smooth L1损失
        target = reward + gamma * Q_target_a_index
        loss = F.smooth_l1_loss(target, Q_c_a, reduction='none')  # 不求和，返回每个样本的损失

    else:
        # 常规DQN
        Q_n_values = Q(next_state.unsqueeze(1)).detach()
        Q_n_a_index, a_index = Q_n_values.max(1)

        # 将进入死状态的obs的Q_target设置为0
        judgement = torch.from_numpy(np.where(done.cpu().numpy() == 0, 1, 0)).type(dtype)
        Q_n_a_index = judgement * Q_n_a_index

        loss = F.smooth_l1_loss(reward + gamma * Q_n_a_index, Q_c_a, reduction='none')  # 不求和，返回每个样本的损失

    return loss  # 返回每个样本的损失

def dqn_learning(
          env,
          q_func,
          optimizer_spec,
          reset_num,
          restart_depth,
          prioritized_buffer,
          replay_buffer_size,
          n_step,
          batch_size,
          gamma,
          learning_starts,
          learning_freq,
          target_update_freq,
          double_dqn,
          input_channels,
          nums_actions,
          seed
        ):
    
    # 保证随机选择动作的可重复性
    random.seed(seed)
    np.random.seed(seed)

    # 定义网络
    Q = q_func(input_channels, nums_actions).type(dtype)
    Q_target = q_func(input_channels, nums_actions).type(dtype)

    # 初始化优化器
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # 初始化回放缓冲区
    replay_buffer_one=None
    replay_buffer_n=None

    if not prioritized_buffer:
        replay_buffer_one = ReplayBuffer(replay_buffer_size,[env.size,env.size],1,0.9)
    else:
        replay_buffer_one = PrioritizedReplayBuffer(replay_buffer_size,[env.size,env.size],1,0.9,0.6)

    if n_step >1:
        replay_buffer_n = ReplayBuffer(replay_buffer_size, [env.size, env.size], n_step, 0.9)

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    current_obs = env.restart()
    map_nums=0
    restart_nums=0
    invalid_map_nums=0

    score=0
    scores=[]

    actions_block=[0,1,2,3]

    exploration = LinearSchedule(100000, 0.1)
    t = 0
    beta = 0.6

    while True:
        # # 迭代停止条件
        # if map_nums>STOP_CONDITION:
        #     break
        #
        # # 当有一定的到达次数后，进行地图的reset
        # if len(actions_block)==0 or env.get_arrive_nums() > reset_num:
        #     if len(actions_block)==0:
        #         invalid_map_nums+=0
        #     map_nums+=1
        #     restart_nums=0
        #     env.reset()
        #     exploration = LinearSchedule(100000, 0.1)
        #     t = 0
        #     beta = 0.6

        if env.get_arrive_nums() > STOP_CONDITION:
            break

        # 在得到一定的数据之前进行随机游走
        if t < learning_starts:
            action = np.random.choice(actions_block)
        else:
            # 贪心的探索
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                x = torch.from_numpy(current_obs).unsqueeze(0).type(dtype) # 感觉不用除 / 255.0
                Q_all_actions = Q(x).cpu() # 调用模型
                action = ((Q_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(nums_actions)]])[0][0]
            action = action.item()
            # print(action)
            if action not in actions_block:
                action = np.random.choice(actions_block)

        action, reward, done, next_state = env.step(action)

        # 存储信息
        if replay_buffer_n == None:
            replay_buffer_one.store_frame(current_obs, action, reward, done, next_state)
        else:
            replay_buffer_one.store_frame(current_obs, action, reward, done, next_state)
            replay_buffer_n.store_frame(current_obs, action, reward, done, next_state)

        if prioritized_buffer:
            fraction = min(t / NUM_FRAMES, 1.0)
            beta = beta + fraction * (1.0 - beta)

        score += reward

        if done == 1:
            next_state = env.restart()
            restart_nums += 1
            scores.append(score)
            score=0
        elif done == 2:
            actions_block.remove(action)
            score += 10
        else:
            actions_block = [0,1,2,3]

        if env.get_total_depth() > restart_depth:
            next_state = env.restart()
            restart_nums += 1
            scores.append(score)
            score=0

        current_obs = next_state

        if (t > learning_starts and t % learning_freq == 0 and replay_buffer_one.can_sample(batch_size)): # 模型训练
            loss,indexes,e_loss = calculate_loss(replay_buffer_one,
                                                 replay_buffer_n,
                                                 prioritized_buffer,
                                                 n_step,
                                                 batch_size,
                                                 beta,
                                                 gamma,
                                                 Q,
                                                 Q_target,
                                                 double_dqn
                                                 )
            # 限制误差区间
            if e_loss is not None:
                replay_buffer_one.update_priorities(indexes, e_loss)
            clipped_error = -1.0 * loss.clamp(V_MIN, V_MAX)
            optimizer.zero_grad()
            loss.backward(clipped_error.data)
            optimizer.step()
            num_param_updates += 1

            # 更新参数
            if double_dqn and num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            os.makedirs("models", exist_ok=True)
            add_str = ''
            if double_dqn:
                add_str = 'double'
            else:
                add_str = 'regular'
            if Q.name != 'DQN':
                add_str += '_dueling'
            if prioritized_buffer:
                add_str += '_prioritized'
            # model_save_path = f"models/{add_str}_{n_step}_{map_nums}_{t}_{seed}.model"
            model_save_path = f"models/{add_str}_{n_step}_{t}_{seed}.model"
            torch.save(Q.state_dict(), model_save_path)

        episode_rewards = env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print(f"Timestep {t}")
            print(f"learning started? {t > learning_starts}")
            if len(episode_rewards) > 0:
                print(f"mean reward (100 episodes) {mean_episode_reward:.6f}")
                print(f"best mean reward {best_mean_episode_reward:.6f}")
            else:
                print("mean reward (100 episodes) -")
                print("best mean reward -")
            # print(f"map_nums {map_nums}")
            # print(f"invalid_map_nums {invalid_map_nums}")
            print(f"restart_nums {restart_nums}")
            print(f"arrive_nums {env.get_arrive_nums()}")
            print(f"exploration {exploration.value(t):.6f}")
            print(f"learning_rate {optimizer_spec.kwargs['lr']:.6f}")
            sys.stdout.flush()

        t+=1