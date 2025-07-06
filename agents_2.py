#agents_2.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from common_2 import (
    Actor, Critic, QNetwork,
    ReplayMemory, soft_update,
    select_action, to_tensor_list
)


# ------------------------
# DQN Agent
# ------------------------

class DQNAgent:
    def __init__(
        self, state_dim, action_dim, action_bound,
        gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
        tau=0.01, learning_rate=1e-4,
        memory_size=1000000, batch_size=128, device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        self.memory = ReplayMemory(capacity=memory_size)
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self._last_decision_time = 0

        # 初始化 target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 随机探索
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).clamp(0, self.action_dim-1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1) 计算 target Q
        with torch.no_grad():
            next_q_all = self.target_network(next_states)  # (batch, 3)
            next_q_max = next_q_all.max(dim=1, keepdim=True)[0]  # (batch, 1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_max  # (batch,1)

        # 2) 计算当前 Q
        q_all = self.q_network(states)  # (batch, 3)
        # 如果 actions 是 (batch,), reshape 成 (batch,1)
        acts = actions.view(-1, 1)  # (batch,1)
        # gather 对应动作的 Q 值
        current_q = q_all.gather(dim=1, index=acts)  # (batch,1)

        # 换用 Huber Loss，使极端误差不至于平方爆炸
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q, target_q)
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        # 裁剪梯度，max_norm=1.0
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 软更新 target network
        soft_update(self.target_network, self.q_network, self.tau)

        # epsilon 衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss


# ------------------------
# DDPG Agent
# ------------------------

class DDPGAgent:
    def __init__(
        self, state_dim, action_dim, action_bound,
        actor_lr=5e-5, critic_lr=5e-4,
        tau=0.01, gamma=0.99,
        memory_size=1000000, batch_size=128,
        noise_scale=0.2, device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.noise_scale = noise_scale
        self._last_decision_time = 0

        # Actor + target
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic + target
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.memory = ReplayMemory(capacity=memory_size)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor)
        if self.noise_scale > 0:
            action = action + self.noise_scale * torch.randn_like(action)
        action = torch.clamp(action, -1.0, 1.0)
        return (action * self.action_bound).detach().cpu().numpy().squeeze(0)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_action)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        # 用 Huber Loss + 梯度裁剪
        critic_loss_fn = nn.SmoothL1Loss()
        critic_loss = critic_loss_fn(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update
        self.actor_optimizer.zero_grad()
        # 获取当前动作
        a_pred = self.actor(states)
        # 最小化 –Q(s, a_pred)
        actor_loss = -self.critic(states, a_pred).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return critic_loss.item()


# ------------------------
# MADDPG Agent
# ------------------------

def initialize_agents(num_agents, state_dim, action_dim, device):
    actors, critics, target_actors, target_critics = [], [], [], []
    for _ in range(num_agents):
        actors.append(Actor(state_dim, action_dim).to(device))
        critics.append(Critic(num_agents*state_dim, num_agents*action_dim).to(device))

    for i in range(num_agents):
        a_t = Actor(state_dim, action_dim).to(device)
        c_t = Critic(num_agents*state_dim, num_agents*action_dim).to(device)
        soft_update(a_t, actors[i], tau=1.0)
        soft_update(c_t, critics[i], tau=1.0)
        target_actors.append(a_t)
        target_critics.append(c_t)

    return actors, critics, target_actors, target_critics


def create_optimizers(actors, critics, actor_lr=1e-3, critic_lr=1e-3):
    opts_a = [optim.Adam(a.parameters(), lr=actor_lr) for a in actors]
    opts_c = [optim.Adam(c.parameters(), lr=critic_lr) for c in critics]
    return opts_a, opts_c

class MADDPGAgent:
    def __init__(
        self, num_agents, state_dim, action_dim, action_bound,
        actor_lr=1e-4, critic_lr=5e-4,
        gamma=0.99, tau=0.01,
        memory_size=1000000, batch_size=128,
        noise_scale=0.2, device='cpu',
        use_attention=False
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.device = device
        self._last_decision_time = 0
        self._update_counter = 0

        self.memory = ReplayMemory(capacity=memory_size)
        self.actors, self.critics, self.target_actors, self.target_critics = \
            initialize_agents(num_agents, state_dim, action_dim, device)
        self.opt_a, self.opt_c = create_optimizers(
            self.actors, self.critics, actor_lr, critic_lr)
        # 缓存切片索引，减少循环开销
        self._state_slices = [
            slice(i * state_dim, (i + 1) * state_dim)
            for i in range(num_agents)
        ]
        self._action_slices = [
            slice(i * action_dim, (i + 1) * action_dim)
            for i in range(num_agents)
        ]

    def select_actions(self, states):
        states = to_tensor_list(states, self.device)
        actions = []
        for i, s in enumerate(states):
            if s.dim() == 1:
                s = s.unsqueeze(0)
            a = select_action(s, self.actors[i], noise_scale=self.noise_scale) * self.action_bound
            a = torch.clamp(a, -1.0, 1.0)
            actions.append(a.squeeze(0))
        return actions

    def select_action(self, joint_state: np.ndarray) -> np.ndarray:
        """
        给定一个拼好的全局状态向量 joint_state（shape = [n*state_dim]）,
        拆分成本地状态列表，调用 select_actions，然后拼回全局动作向量。
        """
        # 拆成 n 个本地 state
        # 假设 self.state_dim 是每个 agent 的局部状态维度
        states = [
            joint_state[i * self.state_dim:(i + 1) * self.state_dim]
            for i in range(self.num_agents)
        ]
        # select_actions 返回 List[Tensor]，每个 Tensor shape=(action_dim,)
        actions = self.select_actions(states)
        # 拼成全局动作向量并转回 numpy
        action_np = np.concatenate([a.detach().cpu().numpy() for a in actions], axis=0)
        return action_np

    def select_action_used(self, joint_state: np.ndarray) -> np.ndarray:
        """
        —— 修改：批量化前向，一次得到所有 agent 的动作 ——
        """
        # joint_state: shape=(n*state_dim,)
        batch = torch.FloatTensor(joint_state).view(self.num_agents, self.state_dim).to(self.device)
        # 并行前向
        outs = [actor(batch) for actor in self.actors]
        joint_a = torch.cat(outs, dim=1)
        if self.noise_scale > 0:
            joint_a = joint_a + self.noise_scale * torch.randn_like(joint_a)
        joint_a = torch.clamp(joint_a, -1.0, 1.0)
        return joint_a.detach().cpu().numpy().reshape(-1)

    def store_transition(self, joint_state, joint_action, joint_reward, joint_next_state):
        to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x, dtype=np.float32)
        self.memory.add((
            to_np(joint_state),
            to_np(joint_action),
            to_np(joint_reward),
            to_np(joint_next_state)
        ))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        self._update_counter += 1
        # 只有当调用次数能被 num_agents 整除时，才真正做一次更新
        i = self._update_counter % self.num_agents
        batch = self.memory.sample(self.batch_size)
        js, ja, jr, jns = zip(*batch)
        joint_states = torch.FloatTensor(np.array(js)).to(self.device)
        joint_actions = torch.FloatTensor(np.array(ja)).to(self.device)
        joint_rewards = torch.FloatTensor(np.array(jr)).to(self.device)
        joint_next_states = torch.FloatTensor(np.array(jns)).to(self.device)

        total_critic_loss = 0.0

        #for i in range(self.num_agents):
            # critic update
        with torch.no_grad():
            tgt_as = [
                self.target_actors[j](joint_next_states[:, sl]) * self.action_bound
                for j, sl in enumerate(self._state_slices)
            ]
            joint_tgt = torch.cat(tgt_as, dim=1)
            y = joint_rewards[:, i].unsqueeze(1) + self.gamma * self.target_critics[i](joint_next_states, joint_tgt)

        curr_q = self.critics[i](joint_states, joint_actions)
        # Huber + 梯度裁剪
        loss_c_fn = nn.SmoothL1Loss()
        loss_c = loss_c_fn(curr_q, y)
        self.opt_c[i].zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=1.0)
        self.opt_c[i].step()

        # actor update
        curr_as = []
        for j, sl in enumerate(self._state_slices):
            if j == i:
                curr_as.append(self.actors[j](joint_states[:, sl]) * self.action_bound)
            else:
                curr_as.append(joint_actions[:, self._action_slices[j]].detach())
        joint_curr_a = torch.cat(curr_as, dim=1)

        for p in self.critics[i].parameters(): p.requires_grad = False
        loss_a = -self.critics[i](joint_states, joint_curr_a).mean()
        self.opt_a[i].zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=1.0)
        self.opt_a[i].step()
        for p in self.critics[i].parameters(): p.requires_grad = True

        # soft update
        soft_update(self.target_actors[i], self.actors[i], self.tau)
        soft_update(self.target_critics[i], self.critics[i], self.tau)

        total_critic_loss += loss_c.item()
        return total_critic_loss / self.num_agents


class GreedyAgent:
    """
    每一步都尽量把当前电池中能量都用上（可再生优先），然后补满电网能量
    带宽也全部分配给用户，不做跨基站共享。
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.n = state_dim // 2
        # 与 BaseStationEnv 默认保持一致
        self.max_energy = 40.0
        self.max_renew = 20.0
        self.max_bw = 20e6
        self._last_decision_time = 0

    def select_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        n = self.n

        # 剩余能量与带宽（归一化）
        energy_norm = state[:n]
        bw_norm = state[n:2*n]

        # 计算实际可用物理量
        E_remain = energy_norm * self.max_energy
        B_remain = bw_norm * self.max_bw

        # 贪心：先用完可再生，再补满到 max_energy
        p_renew = np.minimum(E_remain, self.max_renew)
        p_grid  = self.max_energy - p_renew

        # 带宽全部分配
        bw_alloc = B_remain

        # 归一化到 [0,1]
        a_renew = p_renew / self.max_energy
        a_grid  = p_grid  / self.max_energy
        a_bw    = bw_alloc / self.max_bw

        # 不做共享
        share_dims = 2 * n * (n - 1)
        a_share = np.zeros(share_dims, dtype=np.float32)

        action01 = np.concatenate([a_renew, a_grid, a_bw, a_share])
        # map back to [-1,1]
        raw_action = action01 * 2.0 - 1.0
        return raw_action

    def store_transition(self, *args, **kwargs):
        # Greedy 不学习
        pass

    def update(self):
        # 返回 None，train_one_step 会跳过 loss 记录
        return None


class ConservativeAgent:
    def __init__(self, state_dim, action_dim):
        self.n = state_dim // 2
        self.max_energy = 40.0
        self.max_bw = 20e6
        self._last_decision_time = 0

    def select_action(self, state):
        # state 前 n 是能量比例，后 n 是带宽比例
        energy_norm = state[:self.n]
        bw_norm     = state[self.n:2*self.n]

        # 1) 先算出保证 b_min 所需的最小发射功率 p_req
        #    这里简单假设 p_req = 0.5 * max_energy （你也可以根据 _reward_qos 反推）
        p_req = 0.5 * self.max_energy
        # 全部用可再生（不触电网）
        p_renew = np.minimum(p_req, energy_norm*self.max_energy)
        p_grid = np.maximum(p_req - p_renew, 0.0)

        # 2) 带宽给最少保底：假定 b_req = 0.2 * max_bw
        b_req = 0.3 * self.max_bw
        bw_alloc = np.minimum(b_req, bw_norm*self.max_bw)

        # 归一化
        a_renew = p_renew / self.max_energy
        a_grid  = p_grid  / self.max_energy
        a_bw    = bw_alloc / self.max_bw

        a_share = np.zeros(2*self.n*(self.n-1), dtype=np.float32)
        action01 = np.concatenate([a_renew, a_grid, a_bw, a_share])
        return action01*2.0-1.0

    def store_transition(self, *args, **kwargs):
        # Greedy 不学习
        pass

    def update(self):
        # 返回 None，train_one_step 会跳过 loss 记录
        return None


class TD3Agent(DDPGAgent):
    def __init__(self, state_dim, action_dim, action_bound,
                 actor_lr=5e-5, critic_lr=5e-4, tau=0.005, gamma=0.99,
                 memory_size=1000000, batch_size=128,
                 noise_scale=0.2, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, device='cpu'):
        super().__init__(state_dim, action_dim, action_bound,
                         actor_lr, critic_lr, tau, gamma,
                         memory_size, batch_size,
                         noise_scale, device)
        # 双 Critic
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        # TD3 参数
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self._update_counter = 0

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1) 计算目标 Q（双 Critic + 平滑噪声）
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            q1_target = self.critic_target(next_states, next_action)
            q2_target = self.critic2_target(next_states, next_action)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(q1_target, q2_target)

        # 2) 更新 Critic1
        current_q1 = self.critic(states, actions)
        loss1 = nn.SmoothL1Loss()(current_q1, target_q)

        # 更新 Critic2
        current_q2 = self.critic2(states, actions)
        loss2 = nn.SmoothL1Loss()(current_q2, target_q)

        lambda_diff = 0.1  # 可以在 0.01～0.2 之间调试
        diff_loss = (current_q1 - current_q2).pow(2).mean()

        # 总的 Critic 损失
        total_critic_loss = loss1 + loss2 + lambda_diff * diff_loss

        # 一起反向传播
        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_optimizer.step()
        self.critic2_optimizer.step()

        # 3) 延迟更新 Actor 和 目标网络
        self._update_counter += 1
        if self._update_counter % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_optimizer.step()
            # 软更新所有 target 网络
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
        return loss1.item()