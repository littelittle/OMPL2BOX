from __future__ import annotations

import math
import numpy as np
from typing import List, Sequence, Tuple, Optional, Dict, Any


QType = Sequence[float]
State = Tuple[QType, float]   # (q, yaw)
Layer = Sequence[State]
Q_RESET_SEEDS = {
    # 1) home
    "home": [0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.8],

    # 2) slightly left-biased
    "left_relaxed": [0.35, -0.85, 0.30, -2.20, 0.55, 1.75, 0.90],

    # 3) slightly right-biased
    "right_relaxed": [-0.35, -0.85, -0.30, -2.20, -0.55, 1.75, 0.90],

    # 4) elbow-out / shoulder-open (toward left branch)
    "left_elbow_out": [0.20, -1.05, 0.85, -2.00, 1.20, 1.60, 0.20],

    # 5) mirrored elbow-out (toward right branch)
    "right_elbow_out": [-0.20, -1.05, -0.85, -2.00, -1.20, 1.60, -0.20],

    # 6) more forward-reaching / less tucked
    "forward_mid": [0.00, -0.35, 0.00, -1.85, 0.00, 1.55, 0.80],

    # 7) wrist-twisted variant A
    "wrist_flip_a": [0.10, -0.75, 0.15, -2.30, 0.20, 2.40, -0.80],

    # 8) wrist-twisted variant B
    "wrist_flip_b": [-0.10, -0.75, -0.15, -2.30, -0.20, 2.40, 1.80],
}

def _weighted_l2_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    joint_weights: Optional[np.ndarray] = None,
) -> float:
    diff = q1 - q2
    if joint_weights is not None:
        diff = diff * joint_weights
    return float(np.linalg.norm(diff))


def dp_plan_yaw_path(
    feasible_by_step: Sequence[Layer],
    start_q: Optional[QType] = None,
    jump_threshold: float = 5000.0,
    hard_threshold: bool = True,
    big_penalty: float = 1e6,
    joint_weights: Optional[QType] = None,
) -> Optional[Dict[str, Any]]:
    """
    在分层候选状态图上做动态规划，找到总 q-space 距离最小的路径。

    参数
    ----
    feasible_by_step:
        feasible_by_step[i] = [(q, yaw), (q, yaw), ...]
    start_q:
        机器人执行第 0 步之前的当前配置；若提供，会把 start_q -> step0 的代价也算进去
    jump_threshold:
        相邻 step 间允许的最大 q-space 距离
    hard_threshold:
        True: 超过阈值的边直接禁掉
        False: 超过阈值的边允许，但加上大惩罚
    big_penalty:
        软约束时使用的惩罚系数
    joint_weights:
        关节权重，例如 np.array([1,1,1,1,1,1,1]) 或对某些关节赋更大权重

    返回
    ----
    若成功，返回 dict:
        {
            "path": [(q, yaw), ...],           # 最优路径
            "indices": [j0, j1, ...],          # 每层选中的候选下标
            "total_cost": float,               # 总 cost
            "dp_costs": [np.ndarray, ...],     # 每层每个候选的 cost-to-go
            "next_choice": [np.ndarray, ...],  # 回溯指针
        }
    若无可行路径，返回 None
    """

    n_steps = len(feasible_by_step)
    if n_steps == 0:
        return None

    # 预处理成 numpy
    proc_layers = []
    for i, layer in enumerate(feasible_by_step):
        if len(layer) == 0:
            # 某一层没有候选，必然无解
            return None

        qs = np.array([np.asarray(q, dtype=float) for q, _ in layer], dtype=float)
        yaws = np.array([float(yaw) for _, yaw in layer], dtype=float)
        proc_layers.append({
            "raw": list(layer),
            "qs": qs,
            "yaws": yaws,
        })

    if joint_weights is not None:
        joint_weights = np.asarray(joint_weights, dtype=float)

    if start_q is not None:
        start_q = np.asarray(start_q, dtype=float)

    # dp_costs[i][j] = 从第 i 层第 j 个状态出发，到最后一层的最小累计代价
    dp_costs: List[np.ndarray] = []
    dp_sums: List[np.ndarray] = []
    dp_max: List[np.ndarray] = []
    next_choice: List[np.ndarray] = []

    for layer in proc_layers:
        m = len(layer["raw"])
        dp_costs.append(np.full(m, np.inf, dtype=float))
        dp_sums.append(np.full(m, np.inf, dtype=float))
        dp_max.append(np.full(m, np.inf, dtype=float))
        next_choice.append(np.full(m, -1, dtype=int))

    # 最后一层：cost-to-go = 0
    last_idx = n_steps - 1
    dp_costs[last_idx][:] = 0.0
    dp_sums[last_idx][:] = 0.0
    dp_max[last_idx][:] = 0.0

    # 从后往前做 DP
    for i in range(n_steps - 2, -1, -1):
        curr_qs = proc_layers[i]["qs"]       # shape: [m, dof]
        next_qs = proc_layers[i + 1]["qs"]   # shape: [k, dof]
        next_dp = dp_costs[i + 1]            # shape: [k]
        next_sum_dp = dp_sums[i+1]           # shape: [k]
        next_max_dp = dp_max[i+1]            # shape: [k]

        for j, q_curr in enumerate(curr_qs):
            # 计算 q_curr 到下一层所有候选的距离
            diffs = next_qs - q_curr[None, :]
            # diffs[:-1] = diffs[:-1] % (2*math.pi)
            if joint_weights is not None:
                diffs = diffs * joint_weights[None, :]
            dists = np.linalg.norm(diffs, axis=1)  # [k]

            # 下一层本身必须可达
            valid = np.isfinite(next_dp)

            if hard_threshold:
                # 超过阈值直接禁掉
                valid = valid & (dists <= jump_threshold)
                if not np.any(valid):
                    continue

                total = np.full_like(dists, np.inf, dtype=float)
                # total[valid] = dists[valid] + next_dp[valid]
                total[valid] = np.maximum(dists[valid], next_max_dp[valid]) + 0.5*(dists[valid] + 0.5*next_sum_dp[valid])
                # total[valid] = dists[valid]**3 + next_dp[valid]
            else:
                # 软惩罚：超过阈值允许，但加大代价
                penalty = np.where(dists > jump_threshold, big_penalty, 0.0)
                total = dists + penalty + next_dp
                total[~valid] = np.inf

            best_k = int(np.argmin(total))
            best_cost = float(total[best_k])
            best_sum_cost = float(dists[best_k]+next_sum_dp[best_k])
            best_max_cost = max(dists[best_k], next_max_dp[best_k])

            if math.isfinite(best_cost):
                dp_costs[i][j] = best_cost
                dp_sums[i][j] = best_sum_cost
                dp_max[i][j] = best_max_cost
                next_choice[i][j] = best_k

    # 选择第 0 层起点
    first_dp = dp_costs[0]
    feasible_start_mask = np.isfinite(first_dp)

    if not np.any(feasible_start_mask):
        return None

    if start_q is None:
        start_total = first_dp.copy()
    else:
        first_qs = proc_layers[0]["qs"]
        diffs0 = first_qs - start_q[None, :]
        if joint_weights is not None:
            diffs0 = diffs0 * joint_weights[None, :]
        start_dists = np.linalg.norm(diffs0, axis=1)

        if hard_threshold:
            valid0 = feasible_start_mask & (start_dists <= jump_threshold)
            if not np.any(valid0):
                return None

            start_total = np.full_like(first_dp, np.inf, dtype=float)
            start_total[valid0] = start_dists[valid0] + first_dp[valid0]
        else:
            penalty0 = np.where(start_dists > jump_threshold, big_penalty, 0.0)
            start_total = start_dists + penalty0 + first_dp
            start_total[~feasible_start_mask] = np.inf

    start_idx = int(np.argmin(start_total))
    total_cost = float(start_total[start_idx])

    if not math.isfinite(total_cost):
        return None

    # 回溯整条路径
    indices = [start_idx]
    for i in range(n_steps - 1):
        nxt = int(next_choice[i][indices[-1]])
        if nxt < 0:
            return None
        indices.append(nxt)

    path: List[State] = []
    path_costs: List = []
    for i, j in enumerate(indices):
        q, yaw = proc_layers[i]["raw"][j]
        path.append((np.asarray(q, dtype=float), float(yaw)))
        if i > 0:
            diff = np.asarray(q) - np.asarray(former_q)
            if joint_weights is not None:
                diff = diff * joint_weights
            dist = np.linalg.norm(diff).item()
            path_costs.append(dist)
        former_q = q
    return {
        "path": path,
        "path_costs": path_costs,
        "indices": indices,
        "total_cost": total_cost,
        "dp_costs": dp_costs,
        "next_choice": next_choice,
    }