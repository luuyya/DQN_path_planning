import torch
import numpy as np
import os
from utils.env import Map
from utils.plot import plot_map

def dqn_testing(model_path, map_size, obstacle_ratio, seed=None):
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # 加载模型
    model = torch.load(model_path)
    model.eval()  # 切换到评估模式

    # 初始化地图环境
    env = Map(size=map_size, obstacle_ratio=obstacle_ratio, seed=seed)
    env.initialize_start_end()

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
        current_position, _ = env.move(current_position, action)
        path.append(current_position)

        # 检查是否陷入死循环（例如遇到障碍物无法前进）
        if len(path) > map_size * map_size:
            print("The agent seems to be stuck in a loop.")
            break

    # 绘制地图并显示路径
    plot_map(env, path)

    print("Path found by the agent:", path)

# 示例调用
if __name__ == "__main__":
    model_path = "models/your_saved_model.pth"
    map_size = 10
    obstacle_ratio = 0.2
    dqn_testing(model_path, map_size, obstacle_ratio, seed=42)