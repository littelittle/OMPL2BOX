import numpy as np
import pybullet as p


def _make_frankik_pose(pos, orn):
    """
    把 PyBullet 的 world pose 转成 frankik.inverse() 所需的 pose 格式。
    这里先假设 frankik 接受 4x4 齐次矩阵；如果它接受别的格式，再按实际 API 改。
    """
    R = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T