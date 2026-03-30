import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D


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
            if best_dist < 1:
                matches.append((ia, best_ib, best_dist))
        return matches

    # 版本2：一对一贪心匹配
    candidates = []
    for ia, (qa, _) in enumerate(step_a):
        for ib, (qb, _) in enumerate(step_b):
            dist = joint_config_distance(qa, qb, angular_indices=angular_indices)
            if dist < 1:
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


def plot_threshold_3d_with_init_layer(
    q_trajectory,
    q_source_trajectory,
    planned_dict,
    distance_threshold=0.9,
    save_path="threshold_3d_with_init_layer.png",
    show=True,
    use_degree=True,
    z_mode="refine_iter",   # "refine_iter" or "reset_idx"
    draw_failed_exec_edges=False,
    annotate_exec_dist=False,
    base_edge_alpha=0.18,
    exec_edge_alpha=0.95,
    elev=24,
    azim=-62,
):
    """
    x = step index
    y = yaw
    z = refinement layer

    画法：
    1) z=0 (init layer)：
       - 画所有 init 候选点
       - 相邻 step 间，所有 q-distance < threshold 的 pair 都连线

    2) z>0：
       - 只画最终 executed path 的点
       - executed 相邻点间，若 q-distance < threshold，则连线
       - 颜色按“边出生在哪一层”来区分
    """

    if planned_dict is None:
        print("[INFO] planned_dict is None, skip plotting.")
        return
    if len(q_trajectory) == 0:
        print("[INFO] q_trajectory is empty, skip plotting.")
        return

    assert len(q_trajectory) == len(q_source_trajectory)

    def _source_to_z(src):
        if src is None:
            return 0
        tag = src.get("source_tag", {}) or {}

        if z_mode == "refine_iter":
            if tag.get("kind") == "refine":
                return int(tag.get("iter", 0)) + 1
            return 0
        elif z_mode == "reset_idx":
            return int(src.get("reset_idx", 0))
        else:
            raise ValueError(f"Unknown z_mode: {z_mode}")

    def _yaw_vis(yaw):
        return np.rad2deg(yaw) if use_degree else yaw

    def _qdist(q1, q2):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        return float(np.linalg.norm(q2 - q1))

    path = planned_dict["path"]
    indices = planned_dict["indices"]

    # ----------------------------
    # 1) 收集 init layer 全部点
    # ----------------------------
    init_points_by_step = []
    for step_idx, (step, step_sources) in enumerate(zip(q_trajectory, q_source_trajectory)):
        curr = []
        for cand_idx, ((q, yaw), src) in enumerate(zip(step, step_sources)):
            z = _source_to_z(src)
            if z == 0:
                curr.append({
                    "step": step_idx,
                    "cand_idx": cand_idx,
                    "q": np.asarray(q, dtype=float),
                    "yaw": float(yaw),
                    "z": 0,
                    "src": src,
                })
        init_points_by_step.append(curr)

    # ----------------------------
    # 2) 收集 executed path 点
    # ----------------------------
    executed_points = []
    for step_idx, cand_idx in enumerate(indices):
        q, yaw = q_trajectory[step_idx][cand_idx]
        src = q_source_trajectory[step_idx][cand_idx]
        z = _source_to_z(src)
        executed_points.append({
            "step": step_idx,
            "cand_idx": cand_idx,
            "q": np.asarray(q, dtype=float),
            "yaw": float(yaw),
            "z": z,
            "src": src,
        })

    unique_z = sorted(set([0] + [p["z"] for p in executed_points]))
    cmap = plt.get_cmap("tab10")
    z_to_color = {z: cmap(i % 10) for i, z in enumerate(unique_z)}

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # ----------------------------
    # 3) 先画 init layer 所有点
    # ----------------------------
    first_init_label_used = False
    for step_idx, pts in enumerate(init_points_by_step):
        if len(pts) == 0:
            continue

        xs = [p["step"] for p in pts]
        ys = [_yaw_vis(p["yaw"]) for p in pts]
        zs = [0] * len(pts)

        ax.scatter(
            xs,
            ys,
            zs,
            s=20,
            color="0.65",
            alpha=0.85,
            depthshade=False,
            label="all init candidates" if not first_init_label_used else None,
        )
        first_init_label_used = True

    # ----------------------------
    # 4) 再画 init layer 的 threshold edges
    # ----------------------------
    base_edge_count = 0
    for i in range(len(init_points_by_step) - 1):
        A = init_points_by_step[i]
        B = init_points_by_step[i + 1]

        for pa in A:
            for pb in B:
                dist = _qdist(pa["q"], pb["q"])
                if dist < distance_threshold:
                    ax.plot(
                        [pa["step"], pb["step"]],
                        [_yaw_vis(pa["yaw"]), _yaw_vis(pb["yaw"])],
                        [0, 0],
                        color="royalblue",
                        linewidth=1.0,
                        alpha=base_edge_alpha,
                        zorder=1,
                    )
                    base_edge_count += 1

    # ----------------------------
    # 5) 画 executed path 的点
    # ----------------------------
    used_xs, used_ys, used_zs = [], [], []
    for i, p in enumerate(executed_points):
        x = p["step"]
        y = _yaw_vis(p["yaw"])
        z = p["z"]

        used_xs.append(x)
        used_ys.append(y)
        used_zs.append(z)

        ax.scatter(
            x,
            y,
            z,
            s=170,
            color=z_to_color[z],
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            depthshade=False,
            zorder=5,
            label="executed points" if i == 0 else None,
        )
        ax.text(x, y, z, f"{i}", fontsize=9)

        # 帮助看层级：给 executed 点补一条竖线
        if z > 0:
            ax.plot(
                [x, x],
                [y, y],
                [0, z],
                color="0.75",
                linewidth=1.0,
                alpha=0.6,
                zorder=0,
            )

    # ----------------------------
    # 6) 画 executed path 相邻点之间的 threshold edges
    # ----------------------------
    for i in range(len(executed_points) - 1):
        p0 = executed_points[i]
        p1 = executed_points[i + 1]
        dist = _qdist(p0["q"], p1["q"])

        x0, y0, z0 = p0["step"], _yaw_vis(p0["yaw"]), p0["z"]
        x1, y1, z1 = p1["step"], _yaw_vis(p1["yaw"]), p1["z"]

        edge_birth_z = max(z0, z1)
        edge_color = z_to_color[edge_birth_z]

        if dist < distance_threshold:
            ax.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color=edge_color,
                linewidth=3.0,
                alpha=exec_edge_alpha,
                zorder=4,
            )
        elif draw_failed_exec_edges:
            ax.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color="gray",
                linestyle="--",
                linewidth=1.3,
                alpha=0.25,
                zorder=2,
            )

        if annotate_exec_dist:
            ax.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (z0 + z1),
                f"{dist:.2f}",
                fontsize=8,
            )

    # ----------------------------
    # 7) 轴和图例
    # ----------------------------
    ax.set_xlabel("step i")
    ax.set_ylabel("yaw (deg)" if use_degree else "yaw (rad)")
    ax.set_zlabel("refinement iteration" if z_mode == "refine_iter" else "q_reset idx")
    ax.set_title(
        f"3D threshold graph with full init layer (threshold={distance_threshold}, "
        f"init_edges={base_edge_count})"
    )
    ax.set_xticks(range(len(q_trajectory)))

    if len(unique_z) <= 20:
        ax.set_zticks(unique_z)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.65',
               markersize=7, label='all init candidates'),
        Line2D([0], [0], color='royalblue', lw=2, label='init layer threshold edges'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=11, label='executed points'),
    ]
    legend_handles += [
        Line2D([0], [0], color=z_to_color[z], lw=3, label=f"executed edge born at z={z}")
        for z in unique_z
    ]
    ax.legend(handles=legend_handles, loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=220)
        print(f"[INFO] Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_threshold_3d_with_layer_views(
    q_trajectory,
    q_source_trajectory,
    planned_dict,
    distance_threshold=0.9,
    focus_refine_iter="last",   # "last" or int (0-based refine iter)
    save_path="threshold_3d_with_layer_views.png",
    show=True,
    use_degree=True,
    draw_failed_exec_edges=False,
    annotate_exec_dist=False,
    elev=24,
    azim=-62,
):
    """
    左: 3D 总览
        - init/home 层: 全部点 + threshold 边
        - focus refinement 层: 全部点 + threshold 边
        - executed path: 红色星号 + 粗线

    右上: init/home 层 2D
    右下: focus refinement 层 2D

    这里默认 z 的定义是:
        init -> z = 0
        refine 第 j 次 -> z = j + 1
    """

    if planned_dict is None:
        print("[INFO] planned_dict is None, skip plotting.")
        return
    if len(q_trajectory) == 0:
        print("[INFO] q_trajectory is empty, skip plotting.")
        return

    assert len(q_trajectory) == len(q_source_trajectory), \
        "q_source_trajectory must have the same outer length as q_trajectory"

    for i, (step_q, step_src) in enumerate(zip(q_trajectory, q_source_trajectory)):
        assert len(step_q) == len(step_src), \
            f"step {i}: len(q_trajectory[i]) != len(q_source_trajectory[i])"

    def _source_to_z(src):
        if src is None:
            return 0
        tag = src.get("source_tag", {}) or {}
        if tag.get("kind") == "refine":
            return int(tag.get("iter", 0)) + 1
        return 0

    def _yaw_vis(yaw):
        return np.rad2deg(yaw) if use_degree else yaw

    def _qdist(q1, q2):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        return float(np.linalg.norm(q2 - q1))

    def _collect_layer_points(layer_z):
        by_step = []
        for step_idx, (step, step_sources) in enumerate(zip(q_trajectory, q_source_trajectory)):
            curr = []
            for cand_idx, ((q, yaw), src) in enumerate(zip(step, step_sources)):
                z = _source_to_z(src)
                if z == layer_z:
                    curr.append({
                        "step": step_idx,
                        "cand_idx": cand_idx,
                        "q": np.asarray(q, dtype=float),
                        "yaw": float(yaw),
                        "z": z,
                        "src": src,
                    })
            by_step.append(curr)
        return by_step

    def _compute_threshold_edges(points_by_step):
        edges = []
        for i in range(len(points_by_step) - 1):
            A = points_by_step[i]
            B = points_by_step[i + 1]
            for pa in A:
                for pb in B:
                    dist = _qdist(pa["q"], pb["q"])
                    if dist < distance_threshold:
                        edges.append((pa, pb, dist))
        return edges

    # ---------- resolve focus refinement layer ----------
    all_z = sorted(set(
        _source_to_z(src)
        for step_sources in q_source_trajectory
        for src in step_sources
    ))
    refine_zs = [z for z in all_z if z > 0]

    if focus_refine_iter == "last":
        focus_z = refine_zs[-1] if len(refine_zs) > 0 else None
    else:
        focus_z = int(focus_refine_iter) + 1
        if focus_z not in all_z:
            print(f"[WARN] requested focus refine iter={focus_refine_iter}, "
                  f"but z={focus_z} not found. Available refine z: {refine_zs}")
            focus_z = None

    # ---------- collect layers ----------
    init_points_by_step = _collect_layer_points(0)
    init_edges = _compute_threshold_edges(init_points_by_step)

    if focus_z is not None:
        focus_points_by_step = _collect_layer_points(focus_z)
        focus_edges = _compute_threshold_edges(focus_points_by_step)
    else:
        focus_points_by_step = None
        focus_edges = []

    # ---------- executed path ----------
    indices = planned_dict["indices"]
    path = planned_dict["path"]
    path_costs = planned_dict.get("path_costs", None)

    executed_points = []
    for step_idx, cand_idx in enumerate(indices):
        q, yaw = q_trajectory[step_idx][cand_idx]
        src = q_source_trajectory[step_idx][cand_idx]
        z = _source_to_z(src)
        executed_points.append({
            "step": step_idx,
            "cand_idx": cand_idx,
            "q": np.asarray(q, dtype=float),
            "yaw": float(yaw),
            "z": z,
            "src": src,
        })

    # ---------- colors ----------
    init_point_color = "0.65"
    init_edge_color = "royalblue"

    focus_point_color = "#F7B267"
    focus_edge_color = "darkorange"

    exec_point_color = "crimson"
    exec_edge_color = "crimson"

    # ---------- figure layout ----------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.15, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.22,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_init = fig.add_subplot(gs[0, 1])
    ax_focus = fig.add_subplot(gs[1, 1])

    ax3d.view_init(elev=elev, azim=azim)

    # =========================================================
    # 3D: init/home layer (all points + threshold edges)
    # =========================================================
    for pts in init_points_by_step:
        if len(pts) == 0:
            continue
        xs = [p["step"] for p in pts]
        ys = [_yaw_vis(p["yaw"]) for p in pts]
        zs = [0] * len(pts)
        ax3d.scatter(
            xs, ys, zs,
            s=18,
            color=init_point_color,
            alpha=0.88,
            depthshade=False,
        )

    for pa, pb, dist in init_edges:
        ax3d.plot(
            [pa["step"], pb["step"]],
            [_yaw_vis(pa["yaw"]), _yaw_vis(pb["yaw"])],
            [0, 0],
            color=init_edge_color,
            linewidth=1.0,
            alpha=0.18,
        )

    # =========================================================
    # 3D: focus refinement layer (all points + threshold edges)
    # =========================================================
    if focus_points_by_step is not None:
        for pts in focus_points_by_step:
            if len(pts) == 0:
                continue
            xs = [p["step"] for p in pts]
            ys = [_yaw_vis(p["yaw"]) for p in pts]
            zs = [focus_z] * len(pts)
            ax3d.scatter(
                xs, ys, zs,
                s=18,
                color=focus_point_color,
                alpha=0.82,
                depthshade=False,
            )

        for pa, pb, dist in focus_edges:
            ax3d.plot(
                [pa["step"], pb["step"]],
                [_yaw_vis(pa["yaw"]), _yaw_vis(pb["yaw"])],
                [focus_z, focus_z],
                color=focus_edge_color,
                linewidth=1.0,
                alpha=0.24,
            )

    # =========================================================
    # 3D: executed path overlay
    # =========================================================
    used_xs, used_ys, used_zs = [], [], []
    for i, p in enumerate(executed_points):
        x = p["step"]
        y = _yaw_vis(p["yaw"])
        z = p["z"]

        used_xs.append(x)
        used_ys.append(y)
        used_zs.append(z)

        ax3d.scatter(
            x, y, z,
            s=150,
            color=exec_point_color,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            depthshade=False,
        )
        ax3d.text(x, y, z, f"{i}", fontsize=8)

    for i in range(len(executed_points) - 1):
        p0 = executed_points[i]
        p1 = executed_points[i + 1]

        if path_costs is not None and len(path_costs) == len(executed_points) - 1:
            dist = float(path_costs[i])
        else:
            dist = _qdist(p0["q"], p1["q"])

        x0, y0, z0 = p0["step"], _yaw_vis(p0["yaw"]), p0["z"]
        x1, y1, z1 = p1["step"], _yaw_vis(p1["yaw"]), p1["z"]

        if dist < distance_threshold:
            ax3d.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color=exec_edge_color,
                linewidth=2.8,
                alpha=0.95,
            )
        elif draw_failed_exec_edges:
            ax3d.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.25,
            )

        if annotate_exec_dist:
            ax3d.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (z0 + z1),
                f"{dist:.2f}",
                fontsize=7,
            )

    ax3d.set_xlabel("step i")
    ax3d.set_ylabel("yaw (deg)" if use_degree else "yaw (rad)")
    ax3d.set_zlabel("refinement layer z")
    ax3d.set_xticks(range(len(q_trajectory)))

    shown_z = [0] + ([focus_z] if focus_z is not None else [])
    shown_z += sorted(set(p["z"] for p in executed_points))
    shown_z = sorted(set(shown_z))
    if len(shown_z) <= 20:
        ax3d.set_zticks(shown_z)

    focus_str = "none" if focus_z is None else f"z={focus_z}"
    ax3d.set_title(
        f"3D overview: init + focus refinement layer ({focus_str}) + executed path\n"
        f"threshold={distance_threshold}, init_edges={len(init_edges)}, "
        f"focus_edges={len(focus_edges)}"
    )

    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=init_point_color, markersize=7,
               label='init/home layer points'),
        Line2D([0], [0], color=init_edge_color, lw=2,
               label='init/home threshold edges'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=focus_point_color, markersize=7,
               label='focus refinement layer points'),
        Line2D([0], [0], color=focus_edge_color, lw=2,
               label='focus refinement threshold edges'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=exec_point_color, markeredgecolor='black',
               markersize=12, label='executed path points'),
        Line2D([0], [0], color=exec_edge_color, lw=3,
               label='executed path edges'),
    ]
    ax3d.legend(handles=legend_handles, loc="best")

    # =========================================================
    # helper: 2D layer plot
    # =========================================================
    def _plot_layer_2d(ax, points_by_step, edges, layer_z, title,
                       point_color, edge_color):
        if points_by_step is None:
            ax.text(0.5, 0.5, "No such layer", ha="center", va="center")
            ax.set_axis_off()
            return

        for pts in points_by_step:
            if len(pts) == 0:
                continue
            xs = [p["step"] for p in pts]
            ys = [_yaw_vis(p["yaw"]) for p in pts]
            ax.scatter(
                xs, ys,
                s=18,
                color=point_color,
                alpha=0.88,
                zorder=3,
            )

        for pa, pb, dist in edges:
            ax.plot(
                [pa["step"], pb["step"]],
                [_yaw_vis(pa["yaw"]), _yaw_vis(pb["yaw"])],
                color=edge_color,
                linewidth=1.0,
                alpha=0.22,
                zorder=2,
            )

        # overlay executed points that lie on this layer
        layer_exec = [p for p in executed_points if p["z"] == layer_z]
        if len(layer_exec) > 0:
            ex = [p["step"] for p in layer_exec]
            ey = [_yaw_vis(p["yaw"]) for p in layer_exec]
            ax.scatter(
                ex, ey,
                s=120,
                color=exec_point_color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            for p in layer_exec:
                ax.text(
                    p["step"], _yaw_vis(p["yaw"]),
                    f"{p['step']}",
                    fontsize=8,
                    zorder=6,
                )

        # overlay executed edges that stay inside this layer
        for i in range(len(executed_points) - 1):
            p0 = executed_points[i]
            p1 = executed_points[i + 1]
            if not (p0["z"] == layer_z and p1["z"] == layer_z):
                continue

            if path_costs is not None and len(path_costs) == len(executed_points) - 1:
                dist = float(path_costs[i])
            else:
                dist = _qdist(p0["q"], p1["q"])

            if dist < distance_threshold:
                ax.plot(
                    [p0["step"], p1["step"]],
                    [_yaw_vis(p0["yaw"]), _yaw_vis(p1["yaw"])],
                    color=exec_edge_color,
                    linewidth=2.4,
                    alpha=0.95,
                    zorder=4,
                )
            elif draw_failed_exec_edges:
                ax.plot(
                    [p0["step"], p1["step"]],
                    [_yaw_vis(p0["yaw"]), _yaw_vis(p1["yaw"])],
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.25,
                    zorder=1,
                )

        ax.set_title(f"{title} (z={layer_z}, edges={len(edges)})")
        ax.set_xlabel("step i")
        ax.set_ylabel("yaw (deg)" if use_degree else "yaw (rad)")
        ax.set_xticks(range(len(q_trajectory)))
        ax.grid(True, linestyle="--", alpha=0.30)

    # 2D: init layer
    _plot_layer_2d(
        ax=ax_init,
        points_by_step=init_points_by_step,
        edges=init_edges,
        layer_z=0,
        title="Init / home layer",
        point_color=init_point_color,
        edge_color=init_edge_color,
    )

    # 2D: focus refinement layer
    if focus_z is None:
        ax_focus.text(
            0.5, 0.5,
            "No refinement layer present",
            ha="center", va="center",
        )
        ax_focus.set_axis_off()
    else:
        _plot_layer_2d(
            ax=ax_focus,
            points_by_step=focus_points_by_step,
            edges=focus_edges,
            layer_z=focus_z,
            title=f"Refinement layer (iter={focus_z - 1})",
            point_color=focus_point_color,
            edge_color=focus_edge_color,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=220)
        print(f"[INFO] Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)