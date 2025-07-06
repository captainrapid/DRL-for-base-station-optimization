#  代码的公共部分
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from torch.nn import MultiheadAttention


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim): #输入维度，输出维度
        super(Actor, self).__init__() #调用module的构造函数
        self.fc1 = nn.Linear(state_dim, 100) #定义3层神经网络
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):  # 从状态s到行为a的传递
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 假设输出行为a的区间是-1到1


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, 200)  # 输入由actor和state维度共同决定
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, state, action):
        '''print(">> state.shape =", state.shape,
              "  expected self.state_dim =", self.state_dim)
        print(">> action.shape =", action.shape,
              " expected self.action_dim =", self.action_dim)'''
        #state = state.view(-1, self.state_dim)
        #action = action.view(-1, self.action_dim)
        x = torch.cat([state, action], dim=1)  # state和action被串在一起，维度为1
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 储存记忆的空间
class ReplayMemory:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) >= self.capacity

    def __len__(self):
        return len(self.memory)


# 目标网络软更新
def soft_update(target, source, tau): # tau是软更新因子，tau越大更新幅度越大
    with torch.no_grad(): # 不跟踪梯度，加快计算速度
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# actor网络根据状态生成动作
def select_action(state, actor, noise_scale=0.1):
    action = actor(state)
    noise = noise_scale * torch.randn_like(action)
    return action + noise  # 增加探索噪声

def to_tensor_list(data_list, device):
    """
    把一个可迭代的 numpy/Tensor 列表转为 Tensor 列表并搬到 device。
    """
    tensors = []
    for x in data_list:
        if not torch.is_tensor(x):
            t = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            t = x.to(device)
        tensors.append(t)
    return tensors


def to_numpy(tensor_list):
    """
    把一个 Tensor 列表转为 numpy 数组列表。
    """
    arrays = []
    for x in tensor_list:
        if torch.is_tensor(x):
            arrays.append(x.detach().cpu().numpy())
        else:
            arrays.append(np.array(x, dtype=np.float32))
    return arrays


# ------------------------
# AttentionCritic（带自注意力的 Critic）
# ------------------------
class AttentionCritic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, num_agents, local_state_dim):
        super().__init__()
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim

        # 自注意力层：对每个 agent 的局部状态做 attention
        # embed_dim = local_state_dim，heads 自行调节，这里用 1 头示例
        self.attn = MultiheadAttention(embed_dim=local_state_dim,
                                       num_heads=1,
                                       batch_first=True)

        # 注意：输入维度变为 (num_agents*local_state_dim + total_action_dim)
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, state, action):
        """
        state: (batch, num_agents*local_state_dim)
        action: (batch, total_action_dim)
        """
        batch_size = state.size(0)
        # 拆分并 reshape 为 (batch, num_agents, local_state_dim)
        state_seq = state.view(batch_size, self.num_agents, self.local_state_dim)
        # 自注意力：query=key=value=state_seq
        attn_out, _ = self.attn(state_seq, state_seq, state_seq)
        # 展平为 (batch, num_agents*local_state_dim)
        attn_flat = attn_out.reshape(batch_size, -1)

        # 拼接动作后做全连接
        x = torch.cat([attn_flat, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ------------------------
# initialize_agents：用 AttentionCritic 替换原 Critic
# ------------------------
def initialize_agents(num_agents, state_dim, action_dim, device):
    actors, critics, target_actors, target_critics = [], [], [], []
    total_s_dim = num_agents * state_dim
    total_a_dim = num_agents * action_dim

    for _ in range(num_agents):
        # Actor 保持不变
        actors.append(Actor(state_dim, action_dim).to(device))
        # Critic 改为 AttentionCritic
        critics.append(
            AttentionCritic(
                total_state_dim=total_s_dim,
                total_action_dim=total_a_dim,
                num_agents=num_agents,
                local_state_dim=state_dim
            ).to(device)
        )

    for i in range(num_agents):
        # 复制到 target 网络
        a_t = Actor(state_dim, action_dim).to(device)
        c_t = AttentionCritic(
            total_state_dim=total_s_dim,
            total_action_dim=total_a_dim,
            num_agents=num_agents,
            local_state_dim=state_dim
        ).to(device)

        # 直接硬拷贝参数
        soft_update(a_t, actors[i], tau=1.0)
        soft_update(c_t, critics[i], tau=1.0)

        target_actors.append(a_t)
        target_critics.append(c_t)

    return actors, critics, target_actors, target_critics


