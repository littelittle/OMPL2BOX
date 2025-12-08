from typing import List
import pybullet as p

def interpolate_joint_line(q_from: List[float], q_to: List[float], steps: int):
    """Joint-space linear interpolation including endpoints."""
    path = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        path.append([qf + alpha * (qt - qf) for qf, qt in zip(q_from, q_to)])
    return path

def draw_point(pos, color=[1, 0, 0], size=0.01, life_time=0):
    """在 PyBullet GUI 中画一个小点（通过短线段模拟）"""
    p.addUserDebugLine(
        [pos[0] - size, pos[1], pos[2]],
        [pos[0] + size, pos[1], pos[2]],
        lineColorRGB=color,
        lineWidth=3,
        lifeTime=life_time
    )
    p.addUserDebugLine(
        [pos[0], pos[1] - size, pos[2]],
        [pos[0], pos[1] + size, pos[2]],
        lineColorRGB=color,
        lineWidth=3,
        lifeTime=life_time
    )