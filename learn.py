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

LOG_EVERY_N_STEPS = 1000
SAVE_MODEL_EVERY_N_STEPS = 5000

# 设置 Logger
logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy()

def dqn_learning(
          env,
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
          double_dqn,
          input_channels,
          nums_actions
        ):
    
    # 定义网络
    Q = q_func(input_channels, nums_actions).type(dtype)
    Q_target = q_func(input_channels, nums_actions).type(dtype)

    # 初始化优化器
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # 初始化回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size)

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    # current_obs = env.restart()

    actions_block=[0,1,2,3]

    for t in itertools.count():
        # todo:当有一定的到达次数后，进行地图的reset
        if env.get_total_depth() > stopping_num:
            break

        # last_stored_frame_idx = replay_buffer.store_frame(last_obs) # 存入状态
        # observations = replay_buffer.encode_recent_observation() # 得到前一个状态

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
        replay_buffer.store_frame(current_obs, action, reward, done, next_state) #存储信息

        if done==1:
            next_state = env.restart()
        elif done==2:
            actions_block.remove(action)
        else:
            actions_block = [0,1,2,3]

        current_obs = next_state

        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)): # 模型训练

            cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample_batch(batch_size)
            #修改各个变量的类型
            cur_obs_batch = torch.from_numpy(cur_obs_batch).type(dtype)
            act_batch = torch.from_numpy(act_batch).type(dlongtype)
            rew_batch = torch.from_numpy(rew_batch).type(dtype)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype)
            done_batch = torch.from_numpy(done_batch).type(dtype)

            Q_values = Q(cur_obs_batch.unsqueeze(1))
            Q_c_a = Q_values.gather(1, act_batch.unsqueeze(1)) #选取指定action的q值

            Q_c_a = Q_c_a.squeeze()

            if (double_dqn):
                # Double dqn
                Q_n_values = Q(next_obs_batch.unsqueeze(1)).detach()
                _, a_index = Q_n_values.max(1)

                # 选取Q_target中在Q中最大的的动作
                Q_target_n_values = Q_target(next_obs_batch.unsqueeze(1)).detach()
                Q_target_a_index = Q_target_n_values.gather(1, a_index.unsqueeze(1))
                Q_target_a_index = Q_target_a_index.squeeze()

                # 将进入死状态的obs的Q_target设置为0
                judgement=np.where(done_batch==0,1,0)
                Q_target_a_index = judgement * Q_target_a_index

                error = rew_batch + gamma * Q_target_a_index - Q_c_a
            else:
                # regular DQN
                # todo:考虑是否要更改为针对Q的操作
                Q_n_values = Q(next_obs_batch.unsqueeze(1)).detach()
                Q_n_a_index, a_index = Q_n_values.max(1)

                # 将进入死状态的obs的Q_target设置为0
                judgement=torch.from_numpy(np.where(done_batch==0,1,0)).type(dtype)
                Q_n_a_index = judgement * Q_n_a_index

                error = rew_batch + gamma * Q_n_a_index - Q_c_a

            # 限制误差区间
            clipped_error = -1.0 * error.clamp(-1, 1)

            optimizer.zero_grad()
            Q_c_a.backward(clipped_error.data)
            optimizer.step()
            num_param_updates += 1

            # 更新参数
            if double_dqn and num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # todo:需要进行一定的修改完善
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