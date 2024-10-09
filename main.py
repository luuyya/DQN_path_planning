import argparse
import torch
import torch.optim as optim
import numpy as np

from model import DQN, Dueling_DQN
from learn import dqn_learning, OptimizerSpec
from utils.env import Map
from utils.schedules import *
from utils import plot

BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4 #每个状态输入包含最近的四帧信息
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99 # 未来奖励折扣因子
LEARNING_FREQ = 4 #四个环境交互步骤（例如，每执行四次动作），模型才会更新一次
LEARNING_RATE = 0.00025
ALPHA = 0.95 #计算优先经验重放的参数，控制经验重放的优先级
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)# 1000000：表示在训练的前100万步内，探索概率将逐渐降低。0.1：表示最终探索概率的下限，即在经过设定的步数后，探索概率将稳定在10%。
LEARNING_STARTS = 10000 #开始训练前所需的初始经验数量

def grid_map_learn(env, num_timesteps, double_dqn, dueling_dqn):

    #todo:
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop, # RMSprop 作为优化算法
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS) # 传递给优化器的关键字参数
    )

    if dueling_dqn:
        dqn_learning(
            env=env,
            q_func=Dueling_DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_num=num_timesteps,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            # frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn
        )
    else:
        dqn_learning(
            env=env,
            q_func=DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_num=num_timesteps,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            # frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn = double_dqn
        )

def main():
    parser = argparse.ArgumentParser(description='Path Planning with DQN and Dueling DQN, choose Train and Test Module')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    #train
    #各种参数设置
    train_parser = subparsers.add_parser("train", help="Train an RL agent for grid map path planning")
    train_parser.add_argument("--map-size", type=int, default=100, help="Size of the grid map")
    train_parser.add_argument("--obstacle-ratio", type=float, default=0.1, help="Ratio of obstacles in the grid map")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed for environment")
    #训练时间步长
    train_parser.add_argument("--num-timesteps", type=int, default=10000, help="Number of timesteps to run the training")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--double-dqn", type=int, default=0, help="Use Double DQN - 0 = No, 1 = Yes")
    train_parser.add_argument("--dueling-dqn", type=int, default=0, help="Use Dueling DQN - 0 = No, 1 = Yes")

    #todo:test
    #各种参数设置
    test_parser = subparsers.add_parser("test", help="Test an RL agent for grid map path planning")
    test_parser.add_argument("--map-size", type=int, default=100, help="Size of the grid map")
    test_parser.add_argument("--obstacle-ratio", type=float, default=0.1, help="Ratio of obstacles in the grid map")
    test_parser.add_argument("--seed", type=int, default=None, help="Random seed for environment")
    #训练时间步长
    # test_parser.add_argument("--num-timesteps", type=int, default=10000, help="Number of timesteps to run the training")
    test_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    test_parser.add_argument("--double-dqn", type=int, default=0, help="Use Double DQN - 0 = No, 1 = Yes")
    test_parser.add_argument("--dueling-dqn", type=int, default=0, help="Use Dueling DQN - 0 = No, 1 = Yes")

    args = parser.parse_args()

    # GPU Setup
    if args.gpu is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"CUDA Device: {torch.cuda.current_device()}")

    # Create the grid map environment
    env = Map(size=args.map_size, obstacle_ratio=args.obstacle_ratio, seed=args.seed)
    env.create_random_map()
    env.initialize_start_end()
    print(f"Start: {env.start}, End: {env.end}")

    # grid = env.get_grid()  # 获取网格数据
    # plot.plot_map(grid, env.start, env.end)  # 将网格数据传递给绘图函数      

    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    is_train = args.subcommand=="train"

    if is_train:
        # Run training
        print(f"Training with map size {args.map_size}, obstacle ratio {args.obstacle_ratio}, seed {args.seed}, double_dqn {double_dqn}, dueling_dqn {dueling_dqn}")
        grid_map_learn(env, num_timesteps=args.num_timesteps, double_dqn=double_dqn, dueling_dqn=dueling_dqn)

    else:
        # Run Test
        pass

if __name__ == '__main__':
    main()
