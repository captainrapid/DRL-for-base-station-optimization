#environment_2.py
import gym
from gym import spaces
import numpy as np
import random

class BaseStationEnv(gym.Env):
    """
    Action 格式（长度 = 3 * num_bs）：
      [ p_renew_0, …, p_renew_{n-1},
        p_grid_0,  …, p_grid_{n-1},
        b_alloc_0, …, b_alloc_{n-1} ]
    其中 p_renew/p_grid 是从可再生/电网购买的发射功率（归一化到 [0,1]），
    b_alloc 是分配给该基站下用户的带宽比例（归一化到 [0,1]）。
    """
    def __init__(
        self,
        num_base_stations: int = 2,
        max_energy: float = 40.0,        # 单基站最大发射功率 单位：W
        max_bandwidth: float = 20e6,     # 单基站最多可用频谱资源（5G），单位：Hz
        renewable_price: float = 0.0,    # 可再生价格，单位：元/kWh
        grid_price: float = 0.6,         # 电网价格，单位：元/kWh
        max_renew = 200.0,                # 可再生能源发电产量，单位：W
        channel_gain: np.ndarray = None, # shape (n,)
        min_rate: float = 20e6,           # 每个用户最低速率需求，单位：Mbps
        noise_power: float = 4e-15,      # 噪声功率谱密度（–174 dBm/Hz ≈ 4×10⁻²¹ W/Hz）
        reward_clip: tuple = (-1.5, 0.5),
        users_per_bs: int = 3
    ):
        super().__init__()
        self.n = num_base_stations
        self.max_energy = max_energy
        self.max_bw = max_bandwidth
        self.renewable_price = renewable_price
        self.grid_price = grid_price
        self.max_renew = max_renew
        self.reward_clip = reward_clip
        self.users_per_bs = users_per_bs
        self.p_static = 100.0

        # 如果用户没给，就假设增益和最低速率都为1维均匀
        self.gain = channel_gain if channel_gain is not None else np.ones(self.n)
        self.base_min_rate = min_rate
        self.b_min = np.full(self.n, min_rate, dtype=np.float32)
        self.N0 = noise_power
        self.t = 0
        self.fluct = 0.4  # 振幅：40%
        self.fluct_period = 1000  # 一个周期多少步

        # 状态：当前剩余能量 + 剩余带宽（均归一化到 [0,1]）
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2*self.n,), dtype=np.float32
        )

        # 动作空间：10 维（在 2 个 BS 情况时）
        # 每个基站自己有三个动作：购买电网，购买可再生，功率分配
        # 每个基站对于剩余 n-1 个基站有两个动作：分享能量，分享频谱
        # 内部 3*n + 2*(n*(n-1))  = 3*2 + 2*2 = 10
        dims = 3 * self.n + 2 * self.n * (self.n - 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dims,), dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.action_bound = 1.0

        # 内部记录：实际功率和带宽，以物理量表示
        self.state = np.zeros(2*self.n, dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        # 如果内部还用 torch.rand()、或者别的库，也都 seed 一下
        return [seed]

    def reset(self):
        self.b_min = self.base_min_rate * np.ones(shape=[self.n,])  # * np.random.uniform(0.8, 1.2, size=self.n)
        # 每个基站剩余能量/带宽随机初始化一下
        E = np.random.rand(self.n)    # 0~1 对应 0~max_energy
        B = np.random.rand(self.n)    # 0~1 对应 0~max_bandwidth
        self.state = np.concatenate([E, B]).astype(np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action):

        # 在 step() 里根据 self.t 决定 grid_price
        '''if not hasattr(self, "grid_state"):
            self.grid_state = 0  # 0=谷 1=峰 2=超峰
        P = np.array([[.8, .15, .05],  # 转移矩阵
                      [.2, .6, .2],
                      [.1, .2, .7]])
        self.grid_state = np.random.choice(3, p=P[self.grid_state])
        self.grid_price = [0.3, 0.8, 1.2][self.grid_state]'''

        '''self.users_per_bs = np.clip(
            self.users_per_bs + np.random.choice([-1, 0, 1]),
            2, 6
        )'''

        # 保存旧电池能量（物理量 W·s 或 Wh）
        old_E = self.state[:self.n] * self.max_energy

        # 1) 时间步+1
        self.t += 1
        self.gain = np.random.lognormal(mean=0., sigma=0.25, size=self.n)

        # 2) 更新最小需求 b_min：基于 base_min_rate 的 ±20% 正弦波
        fluct = 1.0 + self.fluct * np.sin(2 * np.pi * self.t / self.fluct_period)
        self.b_min = np.full(self.n, self.base_min_rate * fluct, dtype=np.float32)

        # 内部决策动作
        a01 = (action + 1) / 2
        p_renew = a01[0:self.n] * self.max_energy  # 可再生能量
        p_grid = a01[self.n:2*self.n] * self.max_energy  # 电网能量
        p_renew = np.clip(p_renew, 0, self.max_renew)  # 可再生购买不超过可再生产能

        # 1) 发射功率：可再生+p_grid 直接 clip
        actual_tx_power_total = np.clip(p_renew + p_grid,
                                        0.0, self.max_energy)
        # 2) 带宽分配：保持原逻辑
        bw_alloc_requested = a01[2*self.n:3*self.n] * self.max_bw
        current_B_available_actual = self.state[self.n:2*self.n] * self.max_bw
        bw_alloc_actual_for_users = np.clip(
            bw_alloc_requested,
            0.0,
            current_B_available_actual
        )


        # 共享决策动作（两个基站）
        offset = 3 * self.n
        share_len = 2 * self.n * (self.n - 1)
        share_actions = a01[offset: offset + share_len]
        # initialize per-BS vectors
        p_share = np.zeros(self.n, dtype=np.float32)
        bw_share = np.zeros(self.n, dtype=np.float32)
        # for each ordered pair (i→j), read [energy, bandwidth] and accumulate
        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if j == i:
                    continue
                # energy sent from BS i to BS j
                e_ij = share_actions[idx] * self.max_energy
                # bandwidth sent from BS i to BS j
                bw_ij = share_actions[idx + 1] * self.max_bw
                idx += 2
                # subtract from sender, add to receiver
            p_share[i] -= e_ij
            p_share[j] += e_ij
            bw_share[i] -= bw_ij
            bw_share[j] += bw_ij
        # now E_remain and B_remaining_after_user_alloc are both length-n
        E_remain = self.state[:self.n] * self.max_energy
        B_remaining_after_user_alloc = current_B_available_actual - bw_alloc_actual_for_users
        # update new resources
        E_new_pre = E_remain + p_share
        delta_E = E_new_pre - old_E
        B_new_pre = B_remaining_after_user_alloc + bw_share
        # 归一化回 [0,1]
        E_new = np.clip(E_new_pre / self.max_energy, 0.0, 1.0)
        B_new = np.clip(B_new_pre / self.max_bw, 0.0, 1.0)
        self.state = np.concatenate([E_new, B_new]).astype(np.float32)

        # 计算 reward
        r_qos = self._reward_qos(actual_tx_power_total, bw_alloc_actual_for_users)
        r_cost = -(p_grid*self.grid_price).sum()
        max_cost_est_val = (self.n * self.max_energy * self.grid_price)
        cost_norm = r_cost / (max_cost_est_val + 1e-9)
        qos_norm = r_qos / (self.b_min.sum() + 1e-9)

        raw_reward = 0.85*cost_norm + 0.15*qos_norm

        # 每次充放电给一个小负奖励
        cycle_penalty = -0.03 * np.sum(np.abs(delta_E))
        raw_reward += cycle_penalty

        #  奖励测量噪声
        raw_reward += np.random.normal(0.0, 0.05)

        reward = np.clip(raw_reward, self.reward_clip[0], self.reward_clip[1])

        info = {}
        info["raw_reward"] = raw_reward
        info["cost_norm"] = -cost_norm
        info["qos_norm"] = qos_norm
        # 假设 actual_tx_power_total 已算出，每个基站的带宽分配 bw_alloc_actual_for_users 也已有
        # 1) 计算每个基站的速率 rate_bs = bw_i * log2(1 + SINR_i)
        sinr = self.gain * actual_tx_power_total / (bw_alloc_actual_for_users * self.N0 + 1e-12)
        rate_bs = bw_alloc_actual_for_users * np.log2(1 + np.maximum(0, sinr))

        # 2) 总吞吐
        info["throughput"] = rate_bs.sum()

        # 3) 每基站吞吐
        info["per_bs_tp"] = rate_bs

        # 4) 单步总能耗
        info["energy_sum"] = (p_renew + p_grid).sum()

        return self.state.copy(), reward, False, info

    def _reward_qos(self, p, b, beta=0.1):
        """
        对每个基站，假设有 users_per_bs 个用户：
        - 相同的最小速率 self.b_min[i]
        - 用波动后的信道增益模拟用户差异
        """
        total_penalty = 0.0
        for i in range(self.n):
            # 平均带宽分给每个用户
            if self.users_per_bs > 0:
                bw_each = b[i] / self.users_per_bs
            else:
                bw_each = 0.0
            for _ in range(self.users_per_bs):
                # 模拟用户间的微小信道差异
                gain_user = self.gain[i]  # * np.random.uniform(0.8, 1.2)
                sinr = gain_user * p[i] / (bw_each * self.N0 + 1e-12)
                rate = bw_each * np.log2(1 + np.maximum(0, sinr))

                # 相同的最小速率需求
                diff = (self.b_min[i] - rate) / beta
                # Softplus 平滑惩罚
                penalty = beta * np.logaddexp(0.0, diff)
                total_penalty += penalty
        # 返回负惩罚作为 QoS 部分 reward
        return -total_penalty
