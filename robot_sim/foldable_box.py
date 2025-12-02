import math
from pathlib import Path
from typing import List, Optional, Tuple

import pybullet as p

from robot_sim.utils.vector import _cross, _dot, _normalize

def _rotate_axis_angle(v, axis, angle: float):
    """Rodrigues 公式，在局部系中绕单位轴 axis 旋转向量 v。"""
    axis = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    axv = _cross(axis, v)
    ax_dot_v = _dot(axis, v)
    return [
        v[i] * c + axv[i] * s + axis[i] * ax_dot_v * (1.0 - c)
        for i in range(3)
    ]

class FoldableBox:
    """A simple foldable box with four top flaps driven by hinge joints (URDF-based)."""

    def __init__(self, base_pos, cid):
        self.cid = cid
        self.base_pos = base_pos
        self.base_half_extents = [0.15, 0.12, 0.1]
        self.flap_len = 0.12
        self.flap_width = 0.16
        self.thickness = 0.01
        self.open_angle = -1.35
        self.body_id = self._load_urdf()
        self.flap_joint_indices = list(range(4))

    # ----------------------------- model build -----------------------------
    def _asset_path(self) -> str:
        return str(Path(__file__).resolve().parent / "assets" / "foldable_box.urdf")

    def _load_urdf(self):
        body_id = p.loadURDF(
            fileName=self._asset_path(),
            basePosition=self.base_pos,
            # flip 90 degrees to have flaps point upwards initially
            # baseOrientation=[0, 0, math.sin(-math.pi / 4), math.cos(-math.pi / 4)],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self.cid,
        )
        for j in range(p.getNumJoints(body_id, physicsClientId=self.cid)):
            p.resetJointState(body_id, j, targetValue=0.0, physicsClientId=self.cid)
            # p.setJointMotorControl2(
            #     bodyIndex=body_id,
            #     jointIndex=j,
            #     controlMode=p.POSITION_CONTROL,
            #     targetPosition=0.0,
            #     force=0.0,
            #     positionGain=0.4,
            #     velocityGain=1.0,
            #     physicsClientId=self.cid,
            # )
            # p.setJointMotorControl2(
            #     bodyIndex=body_id,
            #     jointIndex=j,
            #     controlMode=p.VELOCITY_CONTROL,
            #     targetVelocity=0.0,
            #     force=0.0,
            #     physicsClientId=self.cid,
            # )
        return body_id

    # ----------------------------- control utils -----------------------------
    def set_flap_angle(self, flap_id: int, angle: float):
        # p.setJointMotorControl2(
        #     bodyIndex=self.body_id,
        #     jointIndex=int(flap_id),
        #     controlMode=p.VELOCITY_CONTROL,
        #     targetPosition=angle,
        #     force=8.0,
        #     positionGain=0.4,
        #     velocityGain=1.0,
        #     physicsClientId=self.cid,
        # )
        p.resetJointState(
            self.body_id,
            int(flap_id),
            targetValue=angle,
            targetVelocity=0.0,
            physicsClientId=self.cid,
        )
        # p.setJointMotorControl2(
        #     bodyIndex=self.body_id,
        #     jointIndex=int(flap_id),
        #     controlMode=p.VELOCITY_CONTROL,
        #     targetVelocity=0.0,
        #     force=0.0,
        #     physicsClientId=self.cid,
        # )

    def open_all(self, angle: Optional[float] = None):
        ang = self.open_angle if angle is None else float(angle)
        for i in self.flap_joint_indices:
            self.set_flap_angle(i, ang)

    def get_flap_keypoint_pose(
        self,
        flap_id: int,
        angle: float,
        edge_ratio: float = 0.9,
        ) -> Tuple[List[float], List[float], List[float]]:
        """
        在给定 flap 角度下，返回 flap 外侧关键点的世界坐标、法向和铰链轴。
        约定：
        - box 局部坐标系：原点在箱体中心，+z 向上，x/y 对应箱体长宽方向；
        - angle = 0.0 ：flap 与箱体顶面共平面（完全“平”在箱口上）；
        - angle > 0 ：绕铰链轴按右手定则旋转，使 flap 朝“侧面”方向竖起（大约 90° 时竖直）。
        
        参数：
        - flap_id: 0 → +x 侧 flap
                    1 → -x 侧 flap
                    2 → +y 侧 flap
                    3 → -y 侧 flap
        - angle:  flap 绕铰链转动的角度（弧度），从“平放在顶面”的姿态开始计。
        - edge_ratio: 关键点沿 flap 长度方向距离铰链的比例（0~1），接近 1 表示靠近自由边。
        """
        assert 0 <= flap_id < 4

        # 当前 box 的基座位姿（注意：不要再只用 self.base_pos，pick-place 会修改 base pose）
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.cid
        )

        hx, hy, hz = self.base_half_extents
        key_dist = float(self.flap_len) * float(edge_ratio)

        # flap 关闭时（贴在箱口上）的法向：朝上
        n_closed = [0.0, 0.0, 1.0]

        # 在 box 局部坐标系下定义各 flap 的铰链位置和轴
        if flap_id == 0:      # +x 边
            hinge_local = [hx, 0.0, hz]
            axis_local = [0.0, 1.0, 0.0]   # 绕 +y 转
        elif flap_id == 1:    # -x 边
            hinge_local = [-hx, 0.0, hz]
            axis_local = [0.0, -1.0, 0.0]  # 绕 -y 转
        elif flap_id == 2:    # +y 边
            hinge_local = [0.0, hy, hz]
            axis_local = [-1.0, 0.0, 0.0]  # 绕 -x 转
        else:                 # flap_id == 3, -y 边
            hinge_local = [0.0, -hy, hz]
            axis_local = [1.0, 0.0, 0.0]   # 绕 +x 转

        axis_local = _normalize(axis_local)

        # flap 关闭时，铰链到关键点的向量：
        #   - 方向指向箱体内部（而不是外面），否则打开时会把关键点旋到箱体里面去
        t_inward = _cross(axis_local, n_closed)
        t_inward = [x for x in t_inward]          # 取“向内”的方向
        t_inward = _normalize(t_inward)

        offset_closed = [t_inward[i] * key_dist for i in range(3)]

        # 在 flap 平放时，关键点在 box 局部系的位置
        key_local_closed = [
            hinge_local[i] + offset_closed[i] for i in range(3)
        ]

        # 把 offset 和法向都绕铰链轴旋转 angle，得到目标角度下的 offset / normal
        offset_rot = _rotate_axis_angle(offset_closed, axis_local, angle)
        normal_rot = _rotate_axis_angle(n_closed, axis_local, angle)

        key_local = [hinge_local[i] + offset_rot[i] for i in range(3)]

        # 从 box 局部系变到世界系
        key_world, _ = p.multiplyTransforms(
            base_pos,
            base_orn,
            key_local,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )
        # import ipdb; ipdb.set_trace()

        # 法向和铰链轴只需要旋转，不带平移
        normal_world = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            base_orn,
            normal_rot,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )[0]

        axis_world = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            base_orn,
            axis_local,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )[0]

        return key_world, normal_world, axis_world


    def get_flap_target_pose(
        self, flap_id: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Return an outer-face contact point (near the free edge), orientation, and outward direction.
        """
        hx, hy, hz = self.base_half_extents
        base_x, base_y, base_z = self.base_pos
        edge_bias = 0.3 * self.flap_len
        height_bias = 0.02

        if flap_id == 0:  # +x
            pt = [base_x + hx + edge_bias, base_y, base_z + hz + height_bias]
            outward = [1, 0, 0]
        elif flap_id == 1:  # -x
            pt = [base_x - hx - edge_bias, base_y, base_z + hz + height_bias]
            outward = [-1, 0, 0]
        elif flap_id == 2:  # +y
            pt = [base_x, base_y + hy + edge_bias, base_z + hz + height_bias]
            outward = [0, 1, 0]
        else:  # -y
            pt = [base_x, base_y - hy - edge_bias, base_z + hz + height_bias]
            outward = [0, -1, 0]

        orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        return pt, orn, outward
