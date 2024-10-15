import random
import numpy as np
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch
from utils.schedules import LinearSchedule

# CUDA变量
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

LOG_EVERY_N_STEPS = 1000
SAVE_MODEL_EVERY_N_STEPS = 10000
STOP_CONDITION = 500
beta = 0.6

def calculate_loss(prioritized_buffer, n_step):
    pass

def compute_dqn_loss():
    pass

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

    while True:
        # 迭代停止条件
        if map_nums>STOP_CONDITION:
            break

        # 当有一定的到达次数后，进行地图的reset
        if len(actions_block)==0 or env.get_arrive_nums() > reset_num:
            if len(actions_block)==0:
                invalid_map_nums+=0
            map_nums+=1
            restart_nums=0
            env.reset()
            exploration = LinearSchedule(100000, 0.1)
            t = 0
            
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

        # todo:计算beta
        if prioritized_buffer:
            f = min()
            beta = beta + f * (1.0 - beta)

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
            if not prioritized_buffer:
                if n_step==1:
                    cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch,weights = replay_buffer_one.sample_batch(batch_size,beta)



                else:
                    indexes=replay_buffer_one.sample_batch_index(batch_size)
                    cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, weights = replay_buffer_one.sample_batch_from_indexes(indexes,beta)
                    cur_obs_batch_n, act_batch_n, rew_batch_n, next_obs_batch_n, done_batch_n = replay_buffer_n.sample_batch_from_indexes(indexes)




            else:
                cur_obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer_one.sample_batch(batch_size,beta)


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
                # judgement=np.where(done_batch==0,1,0)
                judgement = torch.from_numpy(np.where(done_batch.cpu().numpy() == 0, 1, 0)).type(dtype)
                Q_target_a_index = judgement * Q_target_a_index

                error = rew_batch + gamma * Q_target_a_index - Q_c_a
            else:
                # regular DQN
                Q_n_values = Q(next_obs_batch.unsqueeze(1)).detach()
                Q_n_a_index, a_index = Q_n_values.max(1)

                # 将进入死状态的obs的Q_target设置为0
                # judgement=torch.from_numpy(np.where(done_batch==0,1,0)).type(dtype)
                judgement = torch.from_numpy(np.where(done_batch.cpu().numpy() == 0, 1, 0)).type(dtype)
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