import torch
from torch.autograd import Variable
import numpy as np
import random
import itertools
from collections import namedtuple
from utils.replay_buffer import ReplayBuffer
from utils.schedules import LinearSchedule
# from logger import Logger
import time
from utils.env import Map
import os

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
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          double_dqn=False,
          dueling_dqn=False):


    in_channels = 4
    num_actions = 4
    
    # 定义网络
    Q = q_func(in_channels, num_actions).type(dtype)
    Q_target = q_func(in_channels, num_actions).type(dtype)

    # 初始化优化器
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # todo：初始化回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 100000

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        obs, reward, done, info = env.step(action)

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()

        # update last_obs
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
            act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
            rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
            obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            if (double_dqn):
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = Q(obs_tp1).detach()
                _, a_prime = q_tp1_values.max(1)

                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(obs_tp1).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime 

                error = rew_t + gamma * q_target_s_a_prime - q_s_a
            else:
                # ---------------
                #   regular DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime 

                # Compute Bellman error
                # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip 
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error.data.unsqueeze(1))

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path)

        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)