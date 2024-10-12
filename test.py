import torch
import numpy as np
import os
from utils.env import Map
from utils.plot import plot_map
from model import DQN,Dueling_DQN

def get_newest_model(models_path):
    file_names = os.listdir(models_path)
    
    # 检查文件列表是否为空
    if not file_names:
        raise ValueError("未能找到模型")

    # 获取所有文件的完整路径
    full_paths = [os.path.join(models_path, f) for f in file_names]
    
    # 按最后修改时间排序，最新的文件在最后
    full_paths.sort(key=os.path.getmtime)
    
    # 返回最新的模型路径
    return full_paths[-1]

def dqn_testing(file_path, env, dueling_dqn, double_dqn, input_channels,nums_actions,test_size):
    # 检查模型路径是否存在
    assert os.path.exists(file_path), "Model file not found: {model_path}"

    # 加载模型
    model=None
    if dueling_dqn:
        model=Dueling_DQN(input_channels,nums_actions)
    else:
        model=DQN(input_channels,nums_actions)
    model_path=get_newest_model(file_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式

    succee_count = 0

    for i in range(test_size):
        # 获取起点和终点
        start, end = env.start, env.end
        current_position = start

        # 保存路径
        path = [current_position]

        # 模拟模型对路径的预测
        while True:
            if current_position[0] == end[0] and current_position[1]==end[1]:
                succee_count+=1
                break
            # 将当前状态转换为模型输入
            state=np.array(env.get_current_state(),dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0)

            # 获取动作（模型的输出）
            with torch.no_grad():
                q_values = model(state_tensor)
                # print(q_values)
                action = torch.argmax(q_values).item()

            # 执行动作并更新当前位置
            # todo:处理一下碰到障碍物的情况
            _, _,done,_ = env.step(action)
            current_position=env.cur
            path.append(current_position)

            print(current_position)

            # 检查是否陷入死循环（例如遇到障碍物无法前进）
            if len(path) > env.size*env.size:
                print("The agent seems to be stuck in a loop.")
                break

        # 绘制地图并显示路径
        # plot_map(env, path)
        print("Path found by the agent:", path)

        env.reset()

    print(f"successful ratio:{succee_count/test_size}")

# 示例调用
if __name__ == "__main__":
    folder_path="./models"

    env = Map(size=100, obstacle_ratio=0.1, seed=None)
    env.create_random_map()
    env.initialize_start_end()

    dqn_testing(folder_path, env, False,False,1,4,100)