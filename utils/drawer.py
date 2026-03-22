import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def joint_config_distance(q1, q2, angular_indices=None):
    """
    关节空间距离。
    angular_indices: 哪些 joint 需要按角度 wrap 到 [-pi, pi]
    对 Panda 通常可设为 range(7)
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    d = q2 - q1

    if angular_indices is not None:
        d = d.copy()
        for j in angular_indices:
            if j < len(d):
                d[j] = wrap_to_pi(d[j])

    return float(np.linalg.norm(d))

def greedy_match_step_to_step(step_a, step_b, angular_indices=None, one_to_one=True):
    """
    step_a, step_b: List[(q, yaw)]

    返回：
        matches = [(idx_a, idx_b, dist), ...]
    """
    if len(step_a) == 0 or len(step_b) == 0:
        return []

    # 版本1：每个 step_a 的点独立找最近邻（允许 many-to-one）
    if not one_to_one:
        matches = []
        for ia, (qa, _) in enumerate(step_a):
            best_ib = None
            best_dist = float("inf")
            for ib, (qb, _) in enumerate(step_b):
                dist = joint_config_distance(qa, qb, angular_indices=angular_indices)
                if dist < best_dist:
                    best_dist = dist
                    best_ib = ib
            matches.append((ia, best_ib, best_dist))
        return matches

    # 版本2：一对一贪心匹配
    candidates = []
    for ia, (qa, _) in enumerate(step_a):
        for ib, (qb, _) in enumerate(step_b):
            dist = joint_config_distance(qa, qb, angular_indices=angular_indices)
            candidates.append((dist, ia, ib))

    candidates.sort(key=lambda x: x[0])

    matches = []
    used_a = set()
    used_b = set()

    for dist, ia, ib in candidates:
        if ia in used_a or ib in used_b:
            continue
        matches.append((ia, ib, dist))
        used_a.add(ia)
        used_b.add(ib)

    return matches

def plot_feasible_yaw_evolution_greedy(
    q_trajectory,
    chosen_yaw_trajectory,
    save_path="feasible_yaw_evolution_current_config_greedy_qdist.png",
    show=True,
    use_degree=True,
    angular_indices=range(7),   # Panda 7 arm joints
    one_to_one=True,
):
    """
    q_trajectory: List[List[(q_goal, yaw)]]

    - 灰色点：该 step 所有 feasible yaw
    - 彩色连线：相邻两步之间按 q-space greedy matching 的对应关系
    - 红色星号：该 step 实际被采用的那个 yaw（也就是 step[0]）
    """

    if len(q_trajectory) == 0:
        print("[INFO] q_trajectory is empty, skip plotting.")
        return

    # 先把所有相邻 step 的匹配和距离算出来
    all_matches_per_step = []
    all_dists = []

    for i in range(len(q_trajectory) - 1):
        step_a = q_trajectory[i]
        step_b = q_trajectory[i + 1]

        matches = greedy_match_step_to_step(
            step_a,
            step_b,
            angular_indices=angular_indices,
            one_to_one=one_to_one,
        )
        all_matches_per_step.append(matches)
        all_dists.extend([dist for _, _, dist in matches])

    # 为连线距离准备 colormap
    if len(all_dists) > 0:
        dmin = min(all_dists)
        dmax = max(all_dists)
        if abs(dmax - dmin) < 1e-12:
            dmax = dmin + 1e-12
        norm = colors.Normalize(vmin=dmin, vmax=dmax)
        cmap = cm.viridis
    else:
        norm = None
        cmap = None

    fig, ax = plt.subplots(figsize=(12, 6))

    # 1) 先画所有 feasible yaw 的散点
    for i, step in enumerate(q_trajectory):
        ys = []
        for q, yaw in step:
            ys.append(np.rad2deg(yaw) if use_degree else yaw)

        if len(ys) > 0:
            ax.scatter(
                [i] * len(ys),
                ys,
                s=24,
                color="0.65",
                alpha=0.85,
                zorder=3,
            )

    # 2) 再画相邻两步之间的 greedy matching 连线，颜色映射为 q-space distance
    for i, matches in enumerate(all_matches_per_step):
        step_a = q_trajectory[i]
        step_b = q_trajectory[i + 1]

        for ia, ib, dist in matches:
            yaw_a = step_a[ia][1]
            yaw_b = step_b[ib][1]

            if use_degree:
                yaw_a = np.rad2deg(yaw_a)
                yaw_b = np.rad2deg(yaw_b)

            line_color = cmap(norm(dist)) if norm is not None else "0.3"

            ax.plot(
                [i, i + 1],
                [yaw_a, yaw_b],
                color=line_color,
                linewidth=1.6,
                alpha=0.9,
                zorder=2,
            )

    # 3) 最后把每个 step 真正执行的那个 yaw 标红
    for i, used_yaw in enumerate(chosen_yaw_trajectory):
        
        if use_degree:
            used_yaw = np.rad2deg(used_yaw)

        ax.scatter(
            i,
            used_yaw,
            s=150,
            color="red",
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
            label="executed yaw" if i == 0 else None,
        )

    # colorbar
    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("configuration-space distance")

    ax.set_xlabel("i")
    ax.set_ylabel("feasible yaw (deg)" if use_degree else "feasible yaw (rad)")
    ax.set_title("Feasible yaw evolution with greedy q-space matching")
    ax.set_xticks(range(len(q_trajectory)))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[INFO] Saved yaw evolution plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_feasible_yaw_evolution(q_trajectory, save_path="feasible_yaw_evolution.png", show=True, use_degree=True):
    """
    q_trajectory: List[List[(q_goal, yaw)]]
        外层 index = i
        内层是该步所有可行解 (q_goal, yaw)

    画法：
    - 横轴：i
    - 纵轴：该步所有可行 yaw
    - 散点：所有 feasible yaw
    - 折线：连接“每一步排序后第 k 个 yaw”，方便看分布演化
    """
    yaw_lists = []
    for step in q_trajectory:
        ys = [item[1] for item in step]
        ys = sorted(ys)   # 关键：排序后更适合跨步连线
        if use_degree:
            ys = [np.rad2deg(y) for y in ys]
        yaw_lists.append(ys)

    if len(yaw_lists) == 0:
        print("[INFO] q_trajectory is empty, skip plotting.")
        return

    max_num = max((len(ys) for ys in yaw_lists), default=0)
    if max_num == 0:
        print("[INFO] No feasible yaw found in q_trajectory, skip plotting.")
        return

    xs = np.arange(len(yaw_lists))

    plt.figure(figsize=(10, 6))

    # 先画散点：每一步所有 feasible yaw
    for i, ys in enumerate(yaw_lists):
        if len(ys) > 0:
            plt.scatter([i] * len(ys), ys, s=18, alpha=0.8)

    # 再画折线：连接每一步排序后的第 k 个 yaw
    for k in range(max_num):
        line_y = []
        for ys in yaw_lists:
            if k < len(ys):
                line_y.append(ys[k])
            else:
                line_y.append(np.nan)  # 没有第 k 个可行 yaw 时断开
        plt.plot(xs, line_y, linewidth=1.2, alpha=0.75)

    plt.xlabel("i")
    plt.ylabel("feasible yaw (deg)" if use_degree else "feasible yaw (rad)")
    plt.title("Evolution of feasible yaw distribution over degree_tuple iterations")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved yaw evolution plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
