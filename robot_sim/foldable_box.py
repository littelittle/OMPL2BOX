import math
from pathlib import Path
from typing import List, Optional, Tuple

import pybullet as p


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
            baseOrientation=[0, 0, math.sin(-math.pi / 4), math.cos(-math.pi / 4)],
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
