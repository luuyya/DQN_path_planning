import torch
from torch.autograd import Variable
import numpy as np
import random
import itertools
from collections import namedtuple
from utils.replay_buffer import ReplayBuffer
from utils.schedules import LinearSchedule
from logger import Logger
import time
import os
import sys

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA变量
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# 设置 Logger
logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy()

def dqn_learning(env,
          q_func,
          optimizer_spec,
          exploration,
          stopping_num,
          replay_buffer_size,
          batch_size,
          gamma,
          learning_starts,
          learning_freq,
          # frame_history_len=4,
          target_update_freq,
          double_dqn
        ):


    in_channels = 1 # 输入通道数
    num_actions = 4 # 决策数
    
    # 定义网络
    Q = q_func(in_channels, num_actions).type(dtype)
    Q_target = q_func(in_channels, num_actions).type(dtype)

    # 初始化优化器
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # 初始化回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size)

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 1000

    for t in itertools.count():
        # todo:停止迭代条件
        if env.get_total_depth() > stopping_num:
            break

        last_stored_frame_idx = replay_buffer.store_frame(last_obs) # 存入状态
        observations = replay_buffer.encode_recent_observation() # 得到前一个状态

        # 在得到一定的数据之前进行随机游走
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # 贪心的探索
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                input = torch.from_numpy(observations).unsqueeze(0).type(dtype) # 感觉不用除 / 255.0
                q_value_all_actions = Q(input).cpu() # 调用模型
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]
            action = action.item()
            # print(action)

        obs, action, reward, next_state, done = env.step(action)
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done) #存储其他信息

        if done:
            obs = env.reset()

        last_obs = obs

        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):# 模型训练

            # obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            obs_batch = torch.from_numpy(obs_batch).type(dtype)
            act_batch = torch.from_numpy(act_batch).type(dlongtype)
            rew_batch = torch.from_numpy(rew_batch).type(dtype)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype)
            done_mask = torch.from_numpy(done_mask).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            # print(obs_batch.unsqueeze(0).shape)
            q_values = Q(obs_batch.unsqueeze(1)).cpu()
            
            q_s_a = q_values.gather(1, act_batch.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            if (double_dqn):
                # Double dqn
                q_tp1_values = Q(next_obs_batch.unsqueeze(1)).detach()
                _, a_prime = q_tp1_values.max(1)

                q_target_tp1_values = Q_target(next_obs_batch.unsqueeze(1)).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime 

                error = rew_batch + gamma * q_target_s_a_prime - q_s_a
            else:
                # regular DQN
                q_tp1_values = Q_target(next_obs_batch.unsqueeze(1)).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)
                q_s_a_prime = (1 - done_mask) * q_s_a_prime 

                # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                error = rew_batch + gamma * q_s_a_prime - q_s_a

            # clip the error and flip 
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()

            # print(clipped_error.data.shape)
            q_s_a.backward(clipped_error.data)

            # update
            optimizer.step()
            num_param_updates += 1

            # 更新参数
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####
        #
        # ### 4. Log progress

        #模型保存
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            os.makedirs("models", exist_ok=True)
            add_str = ''
            if double_dqn:
                add_str = 'double'
            if Q.name != 'DQN':
                add_str = 'dueling'
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            model_save_path = f"models/{add_str}_{t}_{timestamp}.model"
            torch.save(Q.state_dict(), model_save_path)

        # print(f"epoch {t}")

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
            print(f"episodes {len(episode_rewards)}")
            print(f"exploration {exploration.value(t):.6f}")
            print(f"learning_rate {optimizer_spec.kwargs['lr']:.6f}")
            sys.stdout.flush()

            # ============ TensorBoard logging ============#
            def log_info(info_dict, step):
                for tag, value in info_dict.items():
                    logger.scalar_summary(tag, value, step)

            # (1) Log the scalar values
            info = {
                'learning_started': t > learning_starts,
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }
            log_info(info, t + 1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }
                log_info(info, t + 1)

            if best_mean_episode_reward != -float('inf'):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward,
                }
                log_info(info, t + 1)