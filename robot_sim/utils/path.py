from typing import List

def interpolate_joint_line(q_from: List[float], q_to: List[float], steps: int):
    """Joint-space linear interpolation including endpoints."""
    path = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        path.append([qf + alpha * (qt - qf) for qf, qt in zip(q_from, q_to)])
    return path