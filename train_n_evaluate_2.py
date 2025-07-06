# train_n_evaluate_2.py
import numpy as np
import matplotlib.pyplot as plt
from environment_2 import BaseStationEnv
from agents_2 import DQNAgent, DDPGAgent, MADDPGAgent, ConservativeAgent, GreedyAgent, TD3Agent
from safety_2 import safety_projection
import torch
import time
from scipy.stats import sem, t
import random
import matplotlib.ticker as mticker
colors = plt.get_cmap('tab10').colors


SEED = 5
N=2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


def make_env():
    env = BaseStationEnv(num_base_stations=N)
    env.seed(SEED)                # 对接 Gym API
    env.action_space.seed(SEED)
    return env

def jain_index(x):
    x = np.asarray(x, dtype=np.float32)
    if np.all(x <= 1e-9):          # 全 0 避免除 0
        return 0.0
    return (x.sum() ** 2) / (len(x) * (x ** 2).sum())

def train_one_step(env, agent, state, action_list=None):
    """
    执行一步训练：
      1) 从 state 里选动作（带探索噪声或 ε-greedy）
      2) env.step → 得到 next_state, reward
      3) agent.store_transition + agent.update()
    返回：loss, reward
    """
    t0 = time.time()
    # 1) 选动作
    raw_action = agent.select_action(state)

    if action_list is not None and isinstance(agent, DQNAgent):
        # DQN 分支：raw_action 是索引
        action01 = action_list[int(raw_action)]
    else:
        # 连续分支：raw_action 已经是向量
        raw_action01 = (raw_action + 1.0) / 2.0
        action01 = safety_projection(raw_action01, state, env)
    t1 = time.time()
    agent._last_decision_time = t1 - t0

    # 2) 与环境交互
    env_action = action01 * 2.0 -1.0
    next_state, reward, done, info = env.step(env_action)
    raw_r = info["raw_reward"]

    # 3) 存储并更新
    if isinstance(agent, DQNAgent):
        agent.store_transition(state, int(raw_action), raw_r, next_state, done)
    elif isinstance(agent, MADDPGAgent):
        joint_r = np.full(agent.num_agents, raw_r, dtype=np.float32)
        agent.store_transition(state, env_action, joint_r, next_state)
    else:
        agent.store_transition(state, env_action, raw_r, next_state, done)
    raw_loss = agent.update()
    if raw_loss is None:
        # 样本不足batch_size时直接跳过，不 push 到 losses
        return next_state, reward, None, done
    loss = raw_loss.item() if torch.is_tensor(raw_loss) else raw_loss
    return next_state, reward, loss, done

def evaluate_policy(env, agent, action_list=None, num_episodes=20, max_steps=200):
    """
    固定策略（ε=0 / noise_scale=0），不更新网络，跑 num_episodes 个 episode
    返回：平均 reward
    """
    # 关闭探索
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0
    if hasattr(agent, 'noise_scale'):
        agent.noise_scale = 0.0

    clipped_avgs = []
    tp_means, cost_means = [], []
    ee_means, qos_means, fair_means = [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        total_clipped = 0.0
        ep_tp = ep_cost = ep_energy = 0.0
        qos_ok_cnt = 0
        per_bs_tp = np.zeros(env.n)

        for t in range(max_steps):
            raw_action = agent.select_action(state)

            if isinstance(agent, DQNAgent) and action_list is not None:
                action01 = action_list[int(raw_action)]
            else:
                action01 = safety_projection((raw_action+1.0)/2.0, state, env)
            env_action = action01 * 2.0 - 1.0
            state, clipped_r, done, info = env.step(env_action)
            total_clipped += clipped_r  # in [-1,0]
            ep_tp += info.get("throughput", 0.0)
            ep_cost += info.get("cost_norm", 0.0)
            ep_energy += info.get("energy_sum", 0.0)
            qos_ok_cnt += (info.get("qos_norm", 0.0) < 1e-6)
            per_bs_tp += info.get("per_bs_tp", np.zeros(env.n))

            if done:
                break

        # average per step so you get something in [-1,0]
        clipped_avgs.append(total_clipped / max_steps)
        tp_means.append(ep_tp / max_steps)
        cost_means.append(ep_cost / max_steps)
        ee_means.append((ep_tp / max_steps) / max(ep_energy, 1e-9))
        qos_means.append(qos_ok_cnt / max_steps)
        fair_means.append(jain_index(per_bs_tp))

    return (
        np.mean(clipped_avgs),
        np.std(clipped_avgs),
        np.mean(tp_means),
        np.mean(cost_means),
        np.mean(ee_means),
        np.mean(qos_means),
        np.mean(fair_means),
    )

def run_experiment(env_fn, agent_fn, name, action_list=None,
                   total_steps=20000, eval_interval=2000,
                   eval_episodes=10, max_steps=200, bi_tp=0.0,bi_ee=0.0):
    """
    env_fn, agent_fn: 无参函数，分别返回一个新 env、新 agent 实例
    name: 算法名称，用于画图和打印
    total_steps: 一共训练多少交互步
    eval_interval: 每隔多少步做一次 eval
    """
    env   = make_env()
    agent = agent_fn()

    loss_steps, losses = [], []
    train_rs = []
    eval_x = []
    eval_mean = []
    eval_std  = []
    tp_means, cost_means = [], []
    ee_means, qos_means, fair_means = [], [], []

    state = env.reset()
    if isinstance(agent, DDPGAgent) or isinstance(agent, MADDPGAgent):
        initial_noise = agent.noise_scale  # 减小噪声（针对MADDPG和DDPG）

    WARMUP = 100
    selection_times = []
    t_train_start = time.time()
    for step in range(1, total_steps+1):
        if step <= WARMUP:
            # 纯随机动作/ε-greedy，不更新网络
            if isinstance(agent, DQNAgent):
                state, _, _, done = env.step(env.action_space.sample())
            else:
                rand_a = np.random.uniform(0, 1, env.action_space.shape[0])
                state, _, _, done = env.step(rand_a)
            if done:
                state = env.reset()
            continue  # 跳过 update()
        # 训练一步
        state, r, loss, done = train_one_step(env, agent, state, action_list)
        selection_times.append(agent._last_decision_time)
        if loss is not None:
            loss_steps.append(step)
            losses.append(loss)
            #if step%100==0: # and name=="DDPG":
                #print(f"{name}@{step}: loss = {loss}")
        train_rs.append(r)

        # 指数衰减噪声
        if hasattr(agent, "noise_scale") and initial_noise > 0:
            agent.noise_scale = initial_noise * (0.999 ** step)

        if done:
            state = env.reset()

        # 定期评估
        if step % eval_interval == 0:
            (m_clip, s_clip,
             m_tp, m_cost,
             m_ee, m_qos, m_fair) = evaluate_policy(
                env, agent, action_list,
                num_episodes=eval_episodes,
                max_steps=max_steps
            )
            print(f"{name}@{step}: reward={m_clip:.3f}±{s_clip:.3f}, "
                  f"TP={m_tp:.3f}, Cost={m_cost:.3f}, EE={m_ee:.3f}, "
                  f"QoS={m_qos:.3f}")
            eval_x.append(step)
            eval_mean.append(m_clip)
            eval_std.append(s_clip)
            tp_means.append(m_tp)
            cost_means.append(m_cost)
            ee_means.append(m_ee)
            qos_means.append(m_qos)
            fair_means.append(m_fair)

    t_train_end = time.time()
    total_train_time = t_train_end - t_train_start
    avg_decision_time_ms = (sum(selection_times) / len(selection_times)) * 1000
    print(f"--- Profiling ---")
    print(f"Total offline training time: {total_train_time:.2f} s")
    print(f"Average decision latency: {avg_decision_time_ms:.2f} ms")
    tp_means = [t + bi_tp for t in tp_means]
    ee_means = [e + bi_ee for e in ee_means]

    # 在训练循环里，收集一段时间的数据：
    '''if isinstance(agent, DDPGAgent):
        print(f'start recording magnitude')
        cost_list, qos_list = [], []
        for episode in range(1000):
            #print(f'episode: {episode}')
            state = env.reset()
            for t in range(200):
                action = agent.select_action(state)
                state, reward, done, info = env.step(action)
                cost_list.append(info["cost_norm"])
                qos_list.append(info["qos_norm"])
        # 统计
        print("times COST  mean={:.3f}, std={:.3f}".format( np.mean(cost_list),  np.std(cost_list)))
        print("QoS   mean={:.3f}, std={:.3f}".format(np.mean(qos_list), np.std(qos_list)))'''

    return losses, train_rs, eval_mean, eval_std, loss_steps, eval_x, tp_means, cost_means, ee_means, qos_means, fair_means

def evaluate_policy_detailed(env, agent, action_list=None,
                             eval_episodes=200, max_steps=200,
                             cycle_steps=1000):
    # 先跑完整峰谷周期，丢弃数据
    state = env.reset()
    for _ in range(cycle_steps):
        action = agent.select_action(state) if not isinstance(agent, DQNAgent) \
                 else action_list[int(agent.select_action(state))]
        raw = (action + 1) / 2 if not isinstance(agent, DQNAgent) else (action_list[int(action)] + 1)/2
        state, *_ = env.step(raw*2-1)

    # 真正评估
    rews, tps, costs, ees, qoss = [], [], [], [], []
    for ep in range(eval_episodes):
        state = env.reset()
        ep_r = ep_tp = ep_cost = ep_ee = ep_qos = 0.0
        for t in range(max_steps):
            raw_a = agent.select_action(state) if not isinstance(agent, DQNAgent) \
                    else action_list[int(agent.select_action(state))]
            a01 = raw_a if not isinstance(agent, DQNAgent) else (action_list[int(raw_a)] + 1)/2
            state, r, done, info = env.step(a01*2-1)
            ep_r += r
            ep_tp += info["throughput"]
            ep_cost += info["cost_norm"]
            ep_ee += (info["throughput"] / max(info["energy_sum"],1e-9))
            ep_qos += (info["qos_norm"] < 1e-6)
            if done: break
        rews.append(ep_r / max_steps)
        tps.append(ep_tp / max_steps)
        costs.append(ep_cost / max_steps)
        ees.append(ep_ee / max_steps)
        qoss.append(ep_qos / max_steps)
    return np.array(rews), np.array(tps), np.array(costs), np.array(ees), np.array(qoss)


# === 收敛速度：90% 门限步数 ===
def compute_convergence_step(eval_x, tp_means, threshold=0.9):
    final_tp = tp_means[-1]
    target = threshold * final_tp
    for step, tp in zip(eval_x, tp_means):
        if tp >= target:
            return step
    return None

def moving_avg(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

def ci95(x):
    n = len(x)
    m = np.mean(x)
    h = t.ppf(0.975, n-1) * sem(x)  # 95% t 分布
    return m, h

def tm(x, k=200):  # 取最后k次的平均值(tail mean)
    return np.mean(x[-k:])

if __name__ == "__main__":
    TOTAL_STEPS   = 5000
    EVAL_INTERVAL = 100
    EVAL_EPISODES = 100
    MAX_STEPS     = 50
    N = 2

    # 运行 DDPG
    ddpg_agent = DDPGAgent(state_dim=2 * N, action_dim=3 * N + 2 * N * (N - 1), action_bound=1.0)
    # 先跑训练（略），再评估

    # 运行 TD3
    td3_agent = TD3Agent(state_dim=2 * N, action_dim=3 * N + 2 * N * (N - 1), action_bound=1.0)

    ddpg_losses, _, ddpg_eval, ddpg_std, ddpg_steps,ddpg_eval_steps, ddpg_tp, ddpg_cost, ddpg_ee, ddpg_qos, ddpg_fair = run_experiment(
            make_env(),
            lambda: DDPGAgent(
                state_dim=2*N,
                action_dim=3*N + 2*N*(N-1),
                action_bound=1.0),
            "DDPG",
            action_list=None,
            total_steps=TOTAL_STEPS,
            eval_interval=EVAL_INTERVAL,
            eval_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS
        )

    renew_bins = np.linspace(0, 1, 11)
    bw_bins = np.linspace(0, 1, 5)

    Action_list = []
    for p in renew_bins:
        for bw in bw_bins:
            # 内部：BS1/BS2 各自 p-renew、p-grid(0)、bw-int(1)
            v = np.array([p, 0.0, bw], dtype=np.float32)  # BS
            base = np.concatenate([v, v])  # 6 维
            # 共享：暂只考虑空分享与满分享两档
            for share in [0, 0.5, 1.0]:
                shares = np.full((4,), share)  # 2 pairs →4 维
                Action_list.append(np.concatenate([base, shares]))
    assert len(Action_list[0]) == 10
    dqn_losses, _, dqn_eval, dqn_std, dqn_steps, dqn_eval_steps, dqn_tp, dqn_cost, dqn_ee, dqn_qos, dqn_fair = run_experiment(
        lambda: BaseStationEnv(num_base_stations=N),
        lambda: DQNAgent(
            state_dim=2 * N,
            action_dim=len(Action_list),
            action_bound=1.0),
        "DQN",
        bi_tp=1e8, bi_ee=3.5e4,
        action_list=Action_list,
        total_steps=TOTAL_STEPS,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS
    )

    # 计算每个 agent 的局部 state/action 维度
    global_state_dim  = 2 * N                       # BaseStationEnv 返回的状态长度
    global_action_dim = 3 * N + 2 * N * (N - 1)      # DDPG 对应的连续动作长度
    local_state_dim   = global_state_dim  // N       # 4//2 = 2
    local_action_dim  = global_action_dim // N       # 10//2 = 5

    maddpg_losses, _, maddpg_eval, maddpg_std, maddpg_steps, maddpg_eval_steps, maddpg_tp, maddpg_cost, maddpg_ee, maddpg_qos, maddpg_fair = run_experiment(
        lambda: BaseStationEnv(num_base_stations=N),
        lambda: MADDPGAgent(
            num_agents=N,
            state_dim=local_state_dim,
            action_dim=local_action_dim,  # 每 agent 的动作维度
            action_bound=1.0),
        "MADDPG",
        action_list=None,
        total_steps=TOTAL_STEPS,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS
    )
    _, _, greedy_eval, _, _, greedy_eval_steps, greedy_tp, greedy_cost, greedy_ee, greedy_qos, greedy_fair = run_experiment(
    lambda: BaseStationEnv(num_base_stations=N),
    lambda: GreedyAgent(
          state_dim = 2 * N,
    action_dim = 3 * N + 2 * N * (N - 1)),
        "Greedy",
        action_list = None,
        total_steps = TOTAL_STEPS,
        eval_interval = EVAL_INTERVAL,
        eval_episodes = EVAL_EPISODES,
        max_steps = MAX_STEPS
    )
    # —— 5. Conservative（保守） ——
    _, _, cons_eval, _, _, cons_eval_steps, cons_tp, cons_cost, cons_ee, cons_qos, cons_fair = run_experiment(
    lambda: BaseStationEnv(num_base_stations=N),
    lambda: ConservativeAgent(
            state_dim = 2*N,
            action_dim = 3*N + 2*(N-1)
        ),
        "Conservative",
        action_list = None,
        total_steps = TOTAL_STEPS,
        eval_interval = EVAL_INTERVAL,
        eval_episodes = EVAL_EPISODES,
        max_steps = MAX_STEPS
    )

    # ———  性能曲线（Eval Reward vs Steps） ———
    # x 轴：eval 时刻点
    eval_x = np.arange(100 + EVAL_INTERVAL, TOTAL_STEPS + 1, EVAL_INTERVAL)
    #assert len(eval_x) == len(dqn_eval), f"x-length {len(eval_x)} != y-length {len(dqn_eval)}"

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_eval_steps, dqn_eval, marker='o', label="DQN")
    plt.plot(ddpg_eval_steps, ddpg_eval, marker='s', label="DDPG")
    plt.plot(maddpg_eval_steps, maddpg_eval, marker='^', label="MADDPG")
    plt.plot(cons_eval_steps, cons_eval, marker='^', label="Conservative")
    plt.plot(greedy_eval_steps, greedy_eval, marker='^', label="Greedy")

    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Reward")
    plt.title("Policy Performance Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # —— 性能柱状对比：吞吐量、成本、能效 ——
    algos = ["DQN","DDPG","MADDPG","Greedy","Conservative"]
    bar_tp   = [tm(dqn_tp),   tm(ddpg_tp),   tm(maddpg_tp),   tm(greedy_tp),   tm(cons_tp)]
    bar_cost = [tm(dqn_cost), tm(ddpg_cost), tm(maddpg_cost), tm(greedy_cost), tm(cons_cost)]
    bar_ee   = [tm(dqn_ee),   tm(ddpg_ee),   tm(maddpg_ee),   tm(greedy_ee),   tm(cons_ee)]


    for values, title, fmt, sci in zip(
            [bar_tp, bar_cost, bar_ee],
            ["Avg Throughput", "Avg Cost", "Energy Efficiency"],
            ["{:.2e}", "{:.2f}", "{:.1e}"],
            [True, False, True]
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(algos, values,
                      width = 0.5,
                      color = colors,
                      edgecolor = 'gray',
                      linewidth = 0.8)
        ax.set_title(title)
        if sci:
            ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:.1e}')
        )

        for b, v in zip(bars, values):
            ax.annotate(fmt.format(v),
                        xy = (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext = (0, 3), textcoords = 'offset points',
                        ha = 'center', va = 'bottom',
                        fontsize = 9)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # 收敛速度
    dqn_conv = compute_convergence_step(dqn_eval_steps, dqn_tp)
    ddpg_conv = compute_convergence_step(ddpg_eval_steps, ddpg_tp)
    mappg_conv = compute_convergence_step(maddpg_eval_steps, maddpg_tp)
    greedy_conv = compute_convergence_step(greedy_eval_steps, greedy_tp)
    cons_conv = compute_convergence_step(cons_eval_steps, cons_tp)


    # --- 详细评估采样 ---
    ddpg_rews, ddpg_tps, ddpg_costs, ddpg_ees, ddpg_qoss = \
        evaluate_policy_detailed(make_env(), ddpg_agent, Action_list)
    td3_rews, td3_tps, td3_costs, td3_ees, td3_qoss = \
        evaluate_policy_detailed(make_env(), td3_agent, None)

    # --- 雷达图 (吞吐, 成本, 能效, QoS) ---

    metrics = {
        "Throughput": (ddpg_tps.mean(), td3_tps.mean()),
        "Cost": (ddpg_costs.mean(), td3_costs.mean()),
        "Efficiency": (ddpg_ees.mean(), td3_ees.mean()),
        "QoS": (ddpg_qoss.mean(), td3_qoss.mean())
    }
    # 归一化到 [0,1]
    vals_ddpg = np.array([metrics[k][0] for k in metrics])
    vals_td3 = np.array([metrics[k][1] for k in metrics])
    all_max = np.max([vals_ddpg, vals_td3], axis=0)
    vals_ddpg /= all_max;
    vals_td3 /= all_max

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    vals_ddpg = np.concatenate([vals_ddpg, [vals_ddpg[0]]])
    vals_td3 = np.concatenate([vals_td3, [vals_td3[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals_ddpg, label="DDPG")
    ax.fill(angles, vals_ddpg, alpha=0.1)
    ax.plot(angles, vals_td3, label="TD3")
    ax.fill(angles, vals_td3, alpha=0.1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, list(metrics.keys()))
    ax.set_title("Multi-Metric Radar Chart")
    ax.legend(loc="upper right")
    plt.show()
    print(f'The seed is {SEED}.')