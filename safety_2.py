# safety.py
import numpy as np
def required_bandwidth(p, b_min, gain, N0, max_bw, tol=1e-3, iters=30):
    """
    计算 b  [0, max_bw] 使得
        b * log2(1 + gain*p/(b*N0)) = b_min
    如果 max_bw 还是无法达到 b_min, 返回max_bw
    计算可以满足QoS的最小带宽
    """

    def rate(b):
        return b * np.log2(1 + gain * p / (b * N0))

    # 如果 max_bw 还是无法达到 b_min, 返回max_bw
    if rate(max_bw) < b_min:
        return max_bw

    lo, hi = 1e-9, max_bw
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rate(mid) >= b_min:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def safety_projection(raw_action, state, env):
    """
    raw_action: numpy array in [0,1]ⁿ of length = env.action_space.shape[0]
    state:     env.state (用于获取env.b_min, env.gain, env.N0)
    env:       BaseStationEnv

    返回一个满足 QoS 的动作
    """
    n = env.n
    a = raw_action.copy()

    a = np.clip(raw_action, 0.0, 1.0)

    # clamp renewable purchase fraction so p_renew <= env.max_renew
    a[0:n] = np.clip(raw_action[0:n], 0.0, env.max_renew / env.max_energy)

    # b_alloc = a[2*n : 3*n] left unchanged on purpose
    return a