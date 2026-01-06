import math
import time
from typing import Iterable, List, Optional

import pybullet as p


class GenericPlanner:
    """
    Robot-agnostic planning helpers that rely on provided joint/robot metadata.
    This class is intentionally minimal and does not load any specific robot.
    """

    def __init__(
        self,
        *,
        cid: int,
        robot_id: int,
        joint_indices: List[int],
        lower_limits: List[float],
        upper_limits: List[float],
        ee_link_index: int,
        collision_link_indices: Optional[List[int]] = None,
        plane_id: Optional[int] = None,
        box_id: Optional[int] = None,
        box_attached: Optional[int] = None,
        control_dt: float = 1.0 / 240.0,
        segment_duration: float = 0.05,
        max_torque: float = 87.0,
        position_gain: float = 0.2,
        velocity_gain: float = 1.0,
        plane_distance: float = 0.002,
        box_distance: float = 0.008,
    ):
        self.cid = cid
        self.robot_id = robot_id
        self.joint_indices = list(joint_indices)
        self.lower_limits = list(lower_limits)
        self.upper_limits = list(upper_limits)
        self.ee_link_index = ee_link_index
        self.collision_link_indices = list(collision_link_indices or [])
        self.plane_id = plane_id
        self.box_id = box_id
        self.box_attached = box_attached

        self.control_dt = control_dt
        self.segment_duration = segment_duration
        self.max_torque = max_torque
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.plane_distance = plane_distance
        self.box_distance = box_distance
        self.ndof = len(self.joint_indices)

    # @property
    # def ndof(self) -> int:
    #     return len(self.joint_indices)

    def set_robot_config(self, q: List[float]):
        assert len(q) == self.ndof
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=float(q[i]),
                targetVelocity=0.0,
                physicsClientId=self.cid,
            )

    def get_current_config(self) -> List[float]:
        states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.cid
        )
        return [s[0] for s in states]

    def is_state_valid(self, state) -> bool:
        q = [float(state[i]) for i in range(self.ndof)]
        backup = self.get_current_config()
        for i in range(self.ndof):
            if q[i] < self.lower_limits[i] - 1e-5 or q[i] > self.upper_limits[i] + 1e-5:
                return False

        self.set_robot_config(q)

        check_list = self.collision_link_indices or self.joint_indices

        for link_index in check_list:
            pts1 = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                distance=0.002,
                linkIndexA=link_index,
                linkIndexB=-1,
                physicsClientId=self.cid,
            )
            if len(pts1) > 0:
                self.set_robot_config(backup)
                return False

        if not self.box_attached==-2:
            for link_index in check_list:
                # import ipdb; ipdb.set_trace()
                for box_link in range(-1, 4):
                    if box_link == self.box_attached:
                        # print("skipping collision check for attached flap ", box_link)
                        continue
                    pts2 = p.getClosestPoints(
                        bodyA=self.robot_id,
                        bodyB=self.box_id,
                        distance=0.008,
                        linkIndexA=link_index,
                        linkIndexB=box_link,
                        physicsClientId=self.cid,
                    )
                    # print("hh")
                    if len(pts2) > 0:
                        # import ipdb; ipdb.set_trace()
                        self.set_robot_config(backup)
                        return False
        self.set_robot_config(backup)
        return True


    def _set_joint_targets_position_control(self, q_target: List[float]):
        assert len(q_target) == self.ndof
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(q_target[i]),
                force=self.max_torque,
                positionGain=self.position_gain,
                velocityGain=self.velocity_gain,
                physicsClientId=self.cid,
            )

    def execute_joint_trajectory_real(
        self,
        qs: List[List[float]],
        dt: Optional[float] = None,
        segment_duration: Optional[float] = None,
        interpolate: bool = True,
    ):
        if dt is None:
            dt = self.control_dt
        if segment_duration is None:
            segment_duration = self.segment_duration

        steps_per_segment = max(1, int(segment_duration / dt))

        if not qs:
            return

        q_curr = self.get_current_config()

        for q_next in qs:
            assert len(q_next) == self.ndof

            for k in range(steps_per_segment):
                if interpolate:
                    alpha = float(k + 1) / float(steps_per_segment)
                    q_cmd = [
                        q_curr[d] + alpha * (q_next[d] - q_curr[d])
                        for d in range(self.ndof)
                    ]
                else:
                    q_cmd = q_next

                self._set_joint_targets_position_control(q_cmd)
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)

            q_curr = list(q_next)

    def wrap_into_limits(self, q: List[float], q_ref: Optional[List[float]] = None) -> List[float]:
        qn = list(q)
        if q_ref is None:
            q_ref = self.get_current_config()
        period = 2.0 * math.pi

        for i in range(self.ndof):
            lb = float(self.lower_limits[i])
            ub = float(self.upper_limits[i])

            jidx = self.joint_indices[i]
            jtype = p.getJointInfo(self.robot_id, jidx, physicsClientId=self.cid)[2]

            if jtype != p.JOINT_REVOLUTE:
                qn[i] = min(max(qn[i], lb), ub)
                continue

            k_min = math.ceil((lb - qn[i]) / period)
            k_max = math.floor((ub - qn[i]) / period)

            if k_min <= k_max:
                best = None
                best_err = 1e18
                for k in range(k_min, k_max + 1):
                    cand = qn[i] + period * k
                    err = abs(cand - float(q_ref[i]))
                    if err < best_err:
                        best_err = err
                        best = cand
                qn[i] = best

        return qn
