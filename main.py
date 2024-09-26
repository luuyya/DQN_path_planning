import argparse
import torch
import torch.optim as optim
import numpy as np

from model import DQN, Dueling_DQN
# from learn import dqn_learning, OptimizerSpec
from utils.env import Map
from utils.schedules import *
from utils import plot

BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4 #?
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4 #?
LEARNING_RATE = 0.00025
ALPHA = 0.95 #?
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
LEARNING_STARTS = 50000

# def grid_map_learn(env, num_timesteps, double_dqn, dueling_dqn):
#     def stopping_criterion(env, t):
#         """todo"""
#         return env.get_total_steps() >= num_timesteps

#     optimizer = OptimizerSpec(
#         constructor=optim.RMSprop,
#         kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
#     )

#     if dueling_dqn:
#         dqn_learning(
#             env=env,
#             q_func=Dueling_DQN,
#             optimizer_spec=optimizer,
#             exploration=EXPLORATION_SCHEDULE,
#             stopping_criterion=stopping_criterion,
#             replay_buffer_size=REPLAY_BUFFER_SIZE,
#             batch_size=BATCH_SIZE,
#             gamma=GAMMA,
#             learning_starts=LEARNING_STARTS,
#             learning_freq=LEARNING_FREQ,
#             frame_history_len=FRAME_HISTORY_LEN,
#             target_update_freq=TARGET_UPDATE_FREQ,
#             double_dqn=double_dqn,
#             dueling_dqn=dueling_dqn
#         )
#     else:
#         dqn_learning(
#             env=env,
#             q_func=DQN,
#             optimizer_spec=optimizer,
#             exploration=EXPLORATION_SCHEDULE,
#             stopping_criterion=stopping_criterion,
#             replay_buffer_size=REPLAY_BUFFER_SIZE,
#             batch_size=BATCH_SIZE,
#             gamma=GAMMA,
#             learning_starts=LEARNING_STARTS,
#             learning_freq=LEARNING_FREQ,
#             frame_history_len=FRAME_HISTORY_LEN,
#             target_update_freq=TARGET_UPDATE_FREQ,
#             double_dqn=double_dqn,
#             dueling_dqn=dueling_dqn
#         )

def main():
    parser = argparse.ArgumentParser(description='Path Planning with DQN and Dueling DQN')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")


    #各种参数设置
    train_parser = subparsers.add_parser("train", help="Train an RL agent for grid map path planning")
    train_parser.add_argument("--map-size", type=int, required=True, help="Size of the grid map")
    train_parser.add_argument("--obstacle-ratio", type=float, required=True, help="Ratio of obstacles in the grid map")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed for environment")
    #训练时间步长
    train_parser.add_argument("--num-timesteps", type=int, required=True, help="Number of timesteps to run the training")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--double-dqn", type=int, default=0, help="Use Double DQN - 0 = No, 1 = Yes")
    train_parser.add_argument("--dueling-dqn", type=int, default=0, help="Use Dueling DQN - 0 = No, 1 = Yes")

    args = parser.parse_args()

    # GPU Setup
    if args.gpu is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"CUDA Device: {torch.cuda.current_device()}")

    # Create the grid map environment
    map_instance = Map(size=args.map_size, obstacle_ratio=args.obstacle_ratio, seed=args.seed)
    map_instance.create_random_map()
    start, end = map_instance.initialize_start_end()
    print(f"Start: {start}, End: {end}")

    env = map_instance  # 使用 map_instance 作为环境对象
    grid = env.get_grid()  # 获取网格数据
    plot.plot_map(grid, start, end)  # 将网格数据传递给绘图函数      

    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    
    # Run training
    print(f"Training with map size {args.map_size}, obstacle ratio {args.obstacle_ratio}, seed {args.seed}, double_dqn {double_dqn}, dueling_dqn {dueling_dqn}")
    # grid_map_learn(env, num_timesteps=args.num_timesteps, double_dqn=double_dqn, dueling_dqn=dueling_dqn)

if __name__ == '__main__':
    main()
