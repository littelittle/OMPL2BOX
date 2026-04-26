import numpy as np
import matplotlib.pyplot as plt


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def double_hinge_grasp_pose(
    x1, y1, z1,
    theta0,   # box base yaw
    theta1,   # base-lid joint angle
    l1,       # lid length
    theta2,   # lid-flap joint angle
    l2,       # flap/grasp length
    degrees=False,
):
    """
    返回:
        p1 : base-lid hinge point
        p2 : lid-flap hinge point
        pg : grasp point
        Rg : grasp frame rotation matrix (columns = x', y', z')
    """

    if degrees:
        theta0 = np.deg2rad(theta0)
        theta1 = np.deg2rad(theta1)
        theta2 = np.deg2rad(theta2)

    p1 = np.array([x1, y1, z1], dtype=float)

    # 盒子主方向 / hinge axis
    h = np.array([
        np.cos(theta0),
        np.sin(theta0),
        0.0
    ], dtype=float)

    # 开合所在竖直平面中的水平法向
    n = np.array([
        np.sin(theta0),
        -np.cos(theta0),
        0.0
    ], dtype=float)

    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    def link_dir(alpha):
        # alpha=0 -> 朝上
        return np.sin(alpha) * n + np.cos(alpha) * ez

    # lid 方向
    u1 = link_dir(theta1)
    # flap 方向
    phi = theta1 + theta2
    u2 = link_dir(phi)

    # 关键点
    p2 = p1 + l1 * u1
    pg = p2 + l2 * u2

    # grasp frame:
    # x' = hinge axis
    x_axis = _normalize(h)

    # z' = flap 法向 / 夹爪 approach 方向
    z_axis = _normalize(-np.cos(phi) * n + np.sin(phi) * ez)

    # y' 保证右手系
    y_axis = _normalize(np.cross(z_axis, x_axis))

    # 再正交化一遍，避免数值误差
    z_axis = _normalize(np.cross(x_axis, y_axis))

    Rg = np.column_stack([x_axis, y_axis, z_axis])

    return p1, p2, pg, Rg


def set_axes_equal(ax):
    """
    让 3D 图 x/y/z 比例一致，不然看起来会变形
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_double_hinge_box(
    x1, y1, z1,
    theta0, theta1, l1, theta2, l2,
    degrees=False,
    ax=None,
    grasp_axis_len=None,
    show_base_frame=True,
    title=None,
):
    """
    画出:
      - p1 -> p2 : lid
      - p2 -> pg : flap
      - pg 处的 grasp frame
    """

    p1, p2, pg, Rg = double_hinge_grasp_pose(
        x1, y1, z1,
        theta0, theta1, l1, theta2, l2,
        degrees=degrees
    )

    if grasp_axis_len is None:
        grasp_axis_len = 0.25 * max(l1, l2, 1e-6)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True

    # ---- 画 lid 和 flap ----
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        linewidth=3, marker='o', label='lid'
    )
    ax.plot(
        [p2[0], pg[0]], [p2[1], pg[1]], [p2[2], pg[2]],
        linewidth=3, marker='o', label='flap'
    )

    # ---- 画关键点 ----
    ax.scatter([p1[0]], [p1[1]], [p1[2]], s=60, label='p1 (base-lid joint)')
    ax.scatter([p2[0]], [p2[1]], [p2[2]], s=60, label='p2 (lid-flap joint)')
    ax.scatter([pg[0]], [pg[1]], [pg[2]], s=80, label='pg (grasp point)')

    # ---- 画 grasp frame ----
    x_axis = Rg[:, 0]
    y_axis = Rg[:, 1]
    z_axis = Rg[:, 2]

    ax.quiver(pg[0], pg[1], pg[2],
              x_axis[0], x_axis[1], x_axis[2],
              length=grasp_axis_len, normalize=True, linewidth=2)
    ax.quiver(pg[0], pg[1], pg[2],
              y_axis[0], y_axis[1], y_axis[2],
              length=grasp_axis_len, normalize=True, linewidth=2)
    ax.quiver(pg[0], pg[1], pg[2],
              z_axis[0], z_axis[1], z_axis[2],
              length=grasp_axis_len, normalize=True, linewidth=2)

    ax.text(*(pg + grasp_axis_len * x_axis), "x'")
    ax.text(*(pg + grasp_axis_len * y_axis), "y'")
    ax.text(*(pg + grasp_axis_len * z_axis), "z'")

    # ---- 可选：画世界坐标轴在 p1 处 ----
    if show_base_frame:
        world_axis_len = grasp_axis_len * 1.2
        ax.quiver(p1[0], p1[1], p1[2], 1, 0, 0,
                  length=world_axis_len, normalize=True, linestyle='dashed')
        ax.quiver(p1[0], p1[1], p1[2], 0, 1, 0,
                  length=world_axis_len, normalize=True, linestyle='dashed')
        ax.quiver(p1[0], p1[1], p1[2], 0, 0, 1,
                  length=world_axis_len, normalize=True, linestyle='dashed')
        ax.text(*(p1 + np.array([world_axis_len, 0, 0])), "X")
        ax.text(*(p1 + np.array([0, world_axis_len, 0])), "Y")
        ax.text(*(p1 + np.array([0, 0, world_axis_len])), "Z")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if title is None:
        title = (
            f"theta0={theta0}, theta1={theta1}, theta2={theta2}, "
            f"l1={l1}, l2={l2}"
            + (" (deg)" if degrees else " (rad)")
        )
    ax.set_title(title)

    ax.legend(loc='best')
    set_axes_equal(ax)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax

from matplotlib.widgets import Slider


def interactive_double_hinge_viewer(
    x1=0.5, y1=0.0, z1=0.2,
    theta0=0.0, theta1=30.0, l1=0.2, theta2=0.0, l2=0.1,
):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, bottom=0.32)

    # 初始图
    plot_double_hinge_box(
        x1, y1, z1,
        theta0, theta1, l1, theta2, l2,
        degrees=True,
        ax=ax
    )

    # slider 区域
    ax_theta0 = plt.axes([0.15, 0.23, 0.7, 0.03])
    ax_theta1 = plt.axes([0.15, 0.19, 0.7, 0.03])
    ax_theta2 = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_l1     = plt.axes([0.15, 0.11, 0.7, 0.03])
    ax_l2     = plt.axes([0.15, 0.07, 0.7, 0.03])

    s_theta0 = Slider(ax_theta0, 'theta0', -180.0, 180.0, valinit=theta0)
    s_theta1 = Slider(ax_theta1, 'theta1', -90.0,  90.0,  valinit=theta1)
    s_theta2 = Slider(ax_theta2, 'theta2', -90.0,  90.0,  valinit=theta2)
    s_l1     = Slider(ax_l1,     'l1',      0.01,   0.50, valinit=l1)
    s_l2     = Slider(ax_l2,     'l2',      0.01,   0.50, valinit=l2)

    def update(val):
        ax.cla()
        plot_double_hinge_box(
            x1, y1, z1,
            s_theta0.val,
            s_theta1.val,
            s_l1.val,
            s_theta2.val,
            s_l2.val,
            degrees=True,
            ax=ax
        )
        fig.canvas.draw_idle()

    s_theta0.on_changed(update)
    s_theta1.on_changed(update)
    s_theta2.on_changed(update)
    s_l1.on_changed(update)
    s_l2.on_changed(update)

    plt.show()

interactive_double_hinge_viewer()