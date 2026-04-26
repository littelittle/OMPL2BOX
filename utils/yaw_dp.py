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

q_min = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
q_max = [2.9671,  1.8326,  2.9671, 0.0, 2.9671, 3.8223, 2.9671]

def uniform_q_sampling(num: int):
    q_min_np = np.asarray(q_min)
    q_max_np = np.asarray(q_max)

    rand = np.random.rand(num, len(q_min_np))

    samples = q_min_np + rand * (q_max_np - q_min_np)

    return samples.tolist()


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
    jump_threshold: float = 100.0,
    hard_threshold: bool = True,
    big_penalty: float = 1e6,
    joint_weights: Optional[QType] = None,
) -> Optional[Dict[str, Any]]:
    """
    在分层候选状态图上做动态规划，按字典序优化路径：
    1. 先最小化最大 edge cost
    2. 再最小化总 q-space 距离

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
            "total_cost": float,               # 总距离（第二优化目标）
            "max_edge_cost": float,            # 最大 edge cost（第一优化目标）
            "dp_costs": [np.ndarray, ...],     # 兼容旧接口，等同于 dp_sum_costs
            "dp_sum_costs": [np.ndarray, ...], # 每层每个候选的最优 suffix 总距离
            "dp_max_costs": [np.ndarray, ...], # 每层每个候选的最优 suffix 最大 edge
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

    # dp_sum_costs[i][j] / dp_max_costs[i][j] 表示从第 i 层第 j 个状态出发，
    # 到最后一层的最优 suffix 代价对 (max_edge, total_sum)。
    dp_sum_costs: List[np.ndarray] = []
    dp_max_costs: List[np.ndarray] = []
    next_choice: List[np.ndarray] = []

    for layer in proc_layers:
        m = len(layer["raw"])
        dp_sum_costs.append(np.full(m, np.inf, dtype=float))
        dp_max_costs.append(np.full(m, np.inf, dtype=float))
        next_choice.append(np.full(m, -1, dtype=int))

    # 最后一层：cost-to-go = 0
    last_idx = n_steps - 1
    dp_sum_costs[last_idx][:] = 0.0
    dp_max_costs[last_idx][:] = 0.0

    # 从后往前做 DP
    for i in range(n_steps - 2, -1, -1):
        curr_qs = proc_layers[i]["qs"]       # shape: [m, dof]
        next_qs = proc_layers[i + 1]["qs"]   # shape: [k, dof]
        next_sum_dp = dp_sum_costs[i + 1]    # shape: [k]
        next_max_dp = dp_max_costs[i + 1]    # shape: [k]

        for j, q_curr in enumerate(curr_qs):
            # 计算 q_curr 到下一层所有候选的距离
            diffs = next_qs - q_curr[None, :]
            # diffs[:-1] = diffs[:-1] % (2*math.pi)
            if joint_weights is not None:
                diffs = diffs * joint_weights[None, :]
            dists = np.linalg.norm(diffs, axis=1)  # [k]

            # 下一层本身必须可达
            valid = np.isfinite(next_sum_dp) & np.isfinite(next_max_dp)

            if hard_threshold:
                # 超过阈值直接禁掉
                valid = valid & (dists <= jump_threshold)
                if not np.any(valid):
                    continue
                edge_costs = dists
            else:
                # 软惩罚：超过阈值允许，但加大代价
                penalty = np.where(dists > jump_threshold, big_penalty, 0.0)
                edge_costs = dists + penalty
                if not np.any(valid):
                    continue

            valid_idx = np.flatnonzero(valid)
            cand_max = np.maximum(edge_costs[valid_idx], next_max_dp[valid_idx])
            cand_sum = edge_costs[valid_idx] + next_sum_dp[valid_idx]
            order = np.lexsort((cand_sum, cand_max))
            best_local = int(order[0])
            best_k = int(valid_idx[best_local])
            best_sum_cost = float(cand_sum[best_local])
            best_max_cost = float(cand_max[best_local])

            if math.isfinite(best_sum_cost) and math.isfinite(best_max_cost):
                dp_sum_costs[i][j] = best_sum_cost
                dp_max_costs[i][j] = best_max_cost
                next_choice[i][j] = best_k

    # 选择第 0 层起点
    first_sum_dp = dp_sum_costs[0]
    first_max_dp = dp_max_costs[0]
    feasible_start_mask = np.isfinite(first_sum_dp) & np.isfinite(first_max_dp)

    if not np.any(feasible_start_mask):
        return None

    if start_q is None:
        start_sum = first_sum_dp.copy()
        start_max = first_max_dp.copy()
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
            start_edge_costs = start_dists
        else:
            penalty0 = np.where(start_dists > jump_threshold, big_penalty, 0.0)
            valid0 = feasible_start_mask
            start_edge_costs = start_dists + penalty0

        start_sum = np.full_like(first_sum_dp, np.inf, dtype=float)
        start_max = np.full_like(first_max_dp, np.inf, dtype=float)
        start_sum[valid0] = start_edge_costs[valid0] + first_sum_dp[valid0]
        start_max[valid0] = np.maximum(start_edge_costs[valid0], first_max_dp[valid0])

    valid_start_idx = np.flatnonzero(feasible_start_mask if start_q is None else np.isfinite(start_sum) & np.isfinite(start_max))
    cand_start_max = start_max[valid_start_idx]
    cand_start_sum = start_sum[valid_start_idx]
    start_order = np.lexsort((cand_start_sum, cand_start_max))
    start_idx = int(valid_start_idx[int(start_order[0])])
    total_cost = float(start_sum[start_idx])
    max_edge_cost = float(start_max[start_idx])

    if not math.isfinite(total_cost) or not math.isfinite(max_edge_cost):
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
    start_edge_cost = None
    for i, j in enumerate(indices):
        q, yaw = proc_layers[i]["raw"][j]
        path.append((np.asarray(q, dtype=float), float(yaw)))
        if i > 0:
            path_costs.append(_weighted_l2_distance(np.asarray(q, dtype=float), np.asarray(former_q, dtype=float), joint_weights))
        former_q = q

    if start_q is not None:
        start_edge_cost = _weighted_l2_distance(path[0][0], start_q, joint_weights)

    return {
        "path": path,
        "path_costs": path_costs,
        "indices": indices,
        "total_cost": total_cost,
        "max_edge_cost": max_edge_cost,
        "start_edge_cost": start_edge_cost,
        "lexicographic_cost": (max_edge_cost, total_cost),
        "dp_costs": dp_sum_costs,
        "dp_sum_costs": dp_sum_costs,
        "dp_max_costs": dp_max_costs,
        "next_choice": next_choice,
    }


def dp_plan_yaw_path(
    feasible_by_step: Sequence[Layer],
    joint_weights: Optional[Sequence[float]] = None,
    state_costs: Optional[Sequence[Sequence[float]]] = None,
    edge_penalty_fn: Optional[
        Callable[[int, int, int, State, State], float]
    ] = None,
) -> Optional[Dict[str, Any]]:
    """
    在分层候选图上做动态规划，按字典序优化：

    1) 先最小化最大 q-space edge cost
    2) 再最小化总代价

    其中总代价 = 所有 edge distance 之和 + state_costs + edge_penalty_fn

    参数
    ----
    feasible_by_step:
        feasible_by_step[i] = [(q, yaw), (q, yaw), ...]

    joint_weights:
        关节权重，例如 np.ones(7)

    state_costs:
        state_costs[i][j] 表示选中第 i 层第 j 个状态的附加代价
        适合放 joint-limit barrier

    edge_penalty_fn:
        edge_penalty_fn(i, j, k, state_i, state_next) -> float
        表示从第 i 层第 j 个状态到第 i+1 层第 k 个状态的附加代价
        适合放 branch switching penalty

    返回
    ----
    {
        "path": [(q, yaw), ...],
        "indices": [j0, j1, ...],
        "path_costs": [edge01, edge12, ...],   # 纯 q-space jump，方便外层找 worst edge
        "total_cost": float,
        "max_edge_cost": float,
    }
    """

    raw_layers = [list(layer) for layer in feasible_by_step]
    if len(raw_layers) == 0:
        return None
    if any(len(layer) == 0 for layer in raw_layers):
        return None

    if joint_weights is not None:
        joint_weights = np.asarray(joint_weights, dtype=float)

    q_layers = [
        np.asarray([np.asarray(q, dtype=float) for q, _ in layer], dtype=float)
        for layer in raw_layers
    ]

    if state_costs is None:
        state_costs = [np.zeros(len(layer), dtype=float) for layer in raw_layers]
    else:
        state_costs = [np.asarray(costs, dtype=float) for costs in state_costs]
        if len(state_costs) != len(raw_layers):
            raise ValueError("state_costs length must match feasible_by_step")
        for costs, layer in zip(state_costs, raw_layers):
            if len(costs) != len(layer):
                raise ValueError("each state_costs[i] must match feasible_by_step[i]")

    n_steps = len(raw_layers)

    # dp_max[i][j]:
    #   从第 i 层第 j 个状态出发，到终点的最优 suffix 的“最大边长”
    # dp_sum[i][j]:
    #   在上述最大边长最优的前提下，对应的最小总代价
    dp_max = [np.full(len(layer), np.inf, dtype=float) for layer in raw_layers]
    dp_sum = [np.full(len(layer), np.inf, dtype=float) for layer in raw_layers]
    next_idx = [np.full(len(layer), -1, dtype=int) for layer in raw_layers]

    # 最后一层：没有后续边了
    dp_max[-1][:] = 0.0
    dp_sum[-1][:] = state_costs[-1]

    # backward DP
    for i in range(n_steps - 2, -1, -1):
        for j, q_curr in enumerate(q_layers[i]):
            best_pair = (np.inf, np.inf)  # (max_edge_cost, total_cost)
            best_k = -1

            for k, q_next in enumerate(q_layers[i + 1]):
                if not np.isfinite(dp_sum[i + 1][k]):
                    continue

                edge_dist = _weighted_l2_distance(q_curr, q_next, joint_weights)

                extra_edge_penalty = 0.0
                if edge_penalty_fn is not None:
                    extra_edge_penalty = float(
                        edge_penalty_fn(
                            i,
                            j,
                            k,
                            raw_layers[i][j],
                            raw_layers[i + 1][k],
                        )
                    )

                cand_max = max(edge_dist, dp_max[i + 1][k])
                cand_sum = (
                    state_costs[i][j]
                    + edge_dist
                    + extra_edge_penalty
                    + dp_sum[i + 1][k]
                )

                cand_pair = (cand_max, cand_sum)
                if cand_pair < best_pair:
                    best_pair = cand_pair
                    best_k = k

            if best_k >= 0:
                dp_max[i][j], dp_sum[i][j] = best_pair
                next_idx[i][j] = best_k

    if not np.any(np.isfinite(dp_sum[0])):
        return None

    start_idx = min(
        range(len(raw_layers[0])),
        key=lambda j: (dp_max[0][j], dp_sum[0][j]),
    )

    if not np.isfinite(dp_sum[0][start_idx]):
        return None

    # 回溯
    indices = [start_idx]
    for i in range(n_steps - 1):
        k = int(next_idx[i][indices[-1]])
        if k < 0:
            return None
        indices.append(k)

    path: List[State] = []
    path_costs: List[float] = []

    for i, j in enumerate(indices):
        q, yaw = raw_layers[i][j]
        q = np.asarray(q, dtype=float)
        path.append((q, float(yaw)))

        if i > 0:
            path_costs.append(
                _weighted_l2_distance(path[i - 1][0], q, joint_weights)
            )

    return {
        "path": path,
        "indices": indices,
        "path_costs": path_costs,
        "total_cost": float(dp_sum[0][start_idx]),
        "max_edge_cost": float(dp_max[0][start_idx]),
    }

def _state_close_to_center(state, center_state, joint_weights=None, q_radius=0.35):
    q, _ = state
    q0, _ = center_state
    q = np.asarray(q, dtype=float)
    q0 = np.asarray(q0, dtype=float)

    diff = q - q0
    if joint_weights is not None:
        diff = diff * joint_weights
    return float(np.linalg.norm(diff)) <= q_radius


def _rerun_blocking_worst_edge_neighborhood(
    feasible_by_step,
    best_pd,
    edge_idx: int,
    mode: str,   # "left" / "right" / "both"
    joint_weights=None,
    q_radius=0.35,
):
    left_center = best_pd["path"][edge_idx]
    right_center = best_pd["path"][edge_idx + 1]

    pruned_layers = []
    index_maps = []

    for i, layer in enumerate(feasible_by_step):
        keep = []
        for j, state in enumerate(layer):
            blocked = False

            if i == edge_idx and mode in ("left", "both"):
                if _state_close_to_center(
                    state, left_center,
                    joint_weights=joint_weights,
                    q_radius=q_radius,
                ):
                    blocked = True

            if i == edge_idx + 1 and mode in ("right", "both"):
                if _state_close_to_center(
                    state, right_center,
                    joint_weights=joint_weights,
                    q_radius=q_radius,
                ):
                    blocked = True

            if not blocked:
                keep.append(j)

        if len(keep) == 0:
            return None

        pruned_layers.append([layer[j] for j in keep])
        index_maps.append(keep)

    alt = dp_plan_yaw_path(
        feasible_by_step=pruned_layers,
        joint_weights=joint_weights,
    )
    if alt is None:
        return None

    # 映射回原始下标
    alt["indices"] = [index_maps[i][j] for i, j in enumerate(alt["indices"])]
    alt["blocked_mode"] = mode
    alt["blocked_edge_idx"] = edge_idx
    alt["blocked_radius"] = q_radius

    # 重点：看它在“同一个 edge 位置”上的代价有没有明显下降
    alt["same_edge_cost"] = float(alt["path_costs"][edge_idx])
    alt["same_edge_improve"] = float(best_pd["path_costs"][edge_idx] - alt["same_edge_cost"])

    return alt

def _lex_key(result):
    return (result["max_edge_cost"], result["total_cost"])


def _rerun_with_forbidden_state(
    feasible_by_step,
    forbid_layer: int,
    forbid_idx: int,
    joint_weights=None,
):
    """
    在第 forbid_layer 层禁止第 forbid_idx 个候选，
    然后重新跑一次 DP。

    返回的 indices 会被映射回“原始 feasible_by_step 的下标”。
    """
    pruned_layers = []
    index_maps = []

    for i, layer in enumerate(feasible_by_step):
        keep = [j for j in range(len(layer)) if not (i == forbid_layer and j == forbid_idx)]
        if len(keep) == 0:
            return None

        pruned_layers.append([layer[j] for j in keep])
        index_maps.append(keep)

    alt = dp_plan_yaw_path(
        pruned_layers,
        joint_weights=joint_weights,
    )
    if alt is None:
        return None

    local_indices = alt["indices"]
    orig_indices = [index_maps[i][j] for i, j in enumerate(local_indices)]

    alt["local_indices"] = local_indices
    alt["indices"] = orig_indices
    alt["forbid_layer"] = forbid_layer
    alt["forbid_idx"] = forbid_idx
    return alt


def path_difference_metrics(path_a, path_b, joint_weights=None, step_eps=0.2):
    """
    用来衡量两条路径差多少。
    step_eps: 某一层 q-space 差距超过这个阈值，就算“这一层真的不同”
    """
    if joint_weights is not None:
        joint_weights = np.asarray(joint_weights, dtype=float)

    diff_steps = 0
    q_dev_sum = 0.0
    yaw_dev_sum = 0.0

    for (q1, yaw1), (q2, yaw2) in zip(path_a, path_b):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        diff = q1 - q2
        if joint_weights is not None:
            diff = diff * joint_weights
        dq = float(np.linalg.norm(diff))

        q_dev_sum += dq
        yaw_dev_sum += abs(float(yaw1) - float(yaw2))

        if dq > step_eps:
            diff_steps += 1

    return {
        "diff_steps": diff_steps,
        "q_dev_sum": q_dev_sum,
        "yaw_dev_sum": yaw_dev_sum,
    }


def find_diverse_beam(
    feasible_by_step,
    q_source_trajectory,
    joint_weights=None,
    beam_width=2,
    min_diff_steps=2,
    q_eps=0.35,
):
    best = dp_plan_yaw_path(
        feasible_by_step,
        joint_weights=joint_weights,
    )
    if best is None:
        return []
    
    k = int(np.argmax(best["path_costs"]))   # worst edge: layer k -> k+1

    candidates = []
    for layer_idx, chosen_idx in enumerate(best["indices"]):
        alt = _rerun_with_forbidden_state(
            feasible_by_step,
            forbid_layer=layer_idx,
            forbid_idx=chosen_idx,
            joint_weights=joint_weights,
        )
        if alt is None:
            continue

        alt.update(
            path_difference_metrics(
                best['path'],
                alt['path'],
                step_eps=q_eps,
            )
        )

        # 必须“够不同”才进入候选池
        if alt["diff_steps"] >= min_diff_steps:
            candidates.append(alt)

    # 先按原目标代价排序：仍然优先选“低代价的备选”
    candidates.sort(key=lambda x: (x["max_edge_cost"], x["total_cost"]))

    selected = [best]
    for alt in candidates:
        too_similar = False
        for picked in selected:
            diff = path_difference_metrics(
                picked['path'],
                alt['path'],
                step_eps=q_eps,
            )
            if diff["diff_steps"] < min_diff_steps:
                too_similar = True
                break

        if not too_similar:
            selected.append(alt)

        if len(selected) >= beam_width:
            break

    return selected

def find_counterfactual_alt_for_worst_edge(
    feasible_by_step,
    best_pd,
    joint_weights=None,
    radius_list=(0.80, ),
):
    k = int(np.argmax(best_pd["path_costs"]))
    candidates = []

    for mode in ("left", "right"):
        for r in radius_list:
            alt = _rerun_blocking_worst_edge_neighborhood(
                feasible_by_step=feasible_by_step,
                best_pd=best_pd,
                edge_idx=k,
                mode=mode,
                joint_weights=joint_weights,
                q_radius=r,
            )
            if alt is not None:
                candidates.append(alt)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["same_edge_cost"],   # 先看原 bottleneck 在相同位置是否变小
            x["max_edge_cost"],    # 再看整条路径的最坏边
            x["total_cost"],       # 最后看总代价
        )
    )
    print("""candidates[0]["same_edge_cost"], candidates[0]["max_edge_cost"], candidates[0]["total_cost"]""")
    for i in range(5):
        print(candidates[i]["same_edge_cost"], candidates[i]["max_edge_cost"], candidates[i]["total_cost"])
    
    return [] #candidates[:1]

