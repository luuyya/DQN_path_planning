import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self,input_channel,num_actions):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=5184, out_features=512) # 修改输入时需要修改
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

        self.name="DQN"

    def forward(self, x):
        batch_size = x.size(0)
        # print(x.shape)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # print(x.shape)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, input_channel, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

        self.name="Dueling_DQN"

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.relu(self.conv2(x))
        print(x.shape)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        print(x.shape)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

if __name__=='__main__':
    from utils.env import Map

    env = Map(size=100, obstacle_ratio=0.3, seed=34)
    env.create_random_map()
    env.initialize_start_end()
    print(f"Start: {env.start}, End: {env.end}")
    print(env.cur)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    grid = env.get_grid()  # 获取网格数据
    grid=torch.from_numpy(grid).unsqueeze(0).type(dtype)
    print(grid)


    net=DQN(1,4).type(dtype)

    with torch.no_grad():
        output=net(grid)
        print(output)
