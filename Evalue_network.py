import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueEvaluationNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ValueEvaluationNetwork, self).__init__()

        # 定义卷积层（卷积 + 批归一化 + ReLU）
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)  # 卷积核大小8x8，步幅4，填充0
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化
        
        # 计算卷积后的输出大小
        # 输入为 20x20，卷积核 8x8，步幅 4，无填充
        # Wout = (20 - 8) / 4 + 1 = 4
        conv_output_size = 4 * 4 * 32  # 卷积层输出 4x4 大小，通道数 32

        # 定义全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)  # 全连接层输入大小 256
        self.fc2 = nn.Linear(256, num_actions)  # 输出为动作的数量

    def forward(self, x):
        # 输入通过卷积层、批归一化和ReLU激活
        x = F.relu(self.bn1(self.conv1(x)))

        # 展平成向量
        x = x.view(x.size(0), -1)  # 展平
        
        # 输入全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出最终的Q值

        return x

# 初始化网络
input_channels = 32  # 假设输入有32个通道
num_actions = 4      # 假设有4个动作输出
model = ValueEvaluationNetwork(input_channels, num_actions)

# 打印网络结构
print(model)

# 测试网络，输入一个随机tensor
dummy_input = torch.randn(1, 32, 20, 20)  # batch_size=1，输入大小为20x20，通道数32
output = model(dummy_input)
print(output)
