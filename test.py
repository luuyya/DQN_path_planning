import torch
import numpy as np
import os
from utils.env import Map
from utils.plot import plot_map

def get_newest_model(models_path):
    file_names = os.listdir(models_path)
    model_path = file_names[-1]
    return torch.load(model_path)

def dqn_testing(models_path, env):
    # 检查模型路径是否存在
    assert os.path.exists(models_path), "Model file not found: {model_path}"

    # 加载模型
    model = get_newest_model(models_path)
    model.eval()  # 切换到评估模式

    # 获取起点和终点
    start, end = env.start, env.end
    current_position = start

    # 保存路径
    path = [current_position]

    # 模拟模型对路径的预测
    while current_position != end:
        # 将当前状态转换为模型输入
        state = np.array(current_position, dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0)

        # 获取动作（模型的输出）
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        # 执行动作并更新当前位置
        current_position, _,_,done = env.step(action)
        path.append(current_position)

        # 检查是否陷入死循环（例如遇到障碍物无法前进）
        if len(path) > env.size*env.size:
            print("The agent seems to be stuck in a loop.")
            break

    # 绘制地图并显示路径
    plot_path(env, path)

    print("Path found by the agent:", path)

# 示例调用
if __name__ == "__main__":
    folder_path="./models"

    env = Map(size=100, obstacle_ratio=0.1, seed=None)
    env.create_random_map()
    env.initialize_start_end()

    dqn_testing(folder_path, env)