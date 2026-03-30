import math
import time
import random
from typing import Iterable, List, Optional

import pybullet as p
import numpy as np

from utils.vector import _normalize, _mat_to_quat, quat_from_normal_and_axis, _cross

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

        self.active_ids = []
        for ji in range(p.getNumJoints(self.robot_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot_id, ji, physicsClientId=self.cid)
            if info[2] in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                self.active_ids.append(ji)

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

    def is_state_valid(self, state, debug=False) -> bool:
        q = [float(state[i]) for i in range(self.ndof)]
        backup = self.get_current_config()
        for i in range(self.ndof):
            if q[i] < self.lower_limits[i] - 1e-5 or q[i] > self.upper_limits[i] + 1e-5:
                if debug:
                    import ipdb; ipdb.set_trace()
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
                if debug:
                    import ipdb; ipdb.set_trace()
                self.set_robot_config(backup)
                return False

        # import ipdb; ipdb.set_trace()
        if not self.box_attached==-2:
            for link_index in check_list:
                # import ipdb; ipdb.set_trace()
                for box_link in range(-1, 4):
                    if box_link == self.box_attached:
                        print("skipping collision check for attached flap ", box_link)
                        continue
                    pts2 = p.getClosestPoints(
                        bodyA=self.robot_id,
                        bodyB=self.box_id,
                        distance=0.008,
                        linkIndexA=link_index,
                        linkIndexB=box_link,
                        physicsClientId=self.cid,
                    )
                    if len(pts2) > 0:
                        if debug:
                            import ipdb; ipdb.set_trace()
                        self.set_robot_config(backup)
                        return False
        self.set_robot_config(backup)
        return True

    def _set_joint_targets_position_control(self, q_target: List[float]):
        assert len(q_target) == self.ndof
        for i, joint_index in enumerate(self.joint_indices):
            if joint_index in self.gripper_joint_indices:
                print(f"[WARNING] {i} is in self.gripeer_joint_indices!")
                continue
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
        N_ref = 100
    ):
        if dt is None:
            dt = self.control_dt
        if segment_duration is None:
            segment_duration = self.segment_duration * (N_ref / len(qs))

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

    def execute_joint_trajectory(self, qs: List[List[float]], dt: float = 1.0 / 240.0):
        for q in qs:
            self.set_robot_config(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(dt)

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

    def _quat_from_normal_and_yaw(
    self,
    normal_world,
    yaw: float,
    horizontal, 
    finger_axis_is_plus_y: bool = True,  # True: +Y 对齐 normal；False: -Y 对齐 normal
    ):
        n = _normalize(normal_world)

        # 让 EE 的 y 轴对齐 normal（或反向）
        y = [n[0], n[1], n[2]] if finger_axis_is_plus_y else [-n[0], -n[1], -n[2]]

        # 选一个不与 y 平行的参考向量，构造正交基
        # tmp = [1.0, 0.0, 0.0] if abs(y[0]) < 0.9 else [0.0, 0.0, 1.0]
        x0 = _normalize(horizontal)
        z0 = _normalize(_cross(x0, y))  # 保证右手系：x × y = z

        # 绕 y（也就是 normal）旋转 yaw：在 x-z 平面里转
        c, s = math.cos(yaw), math.sin(yaw)
        x = [x0[i] * c + z0[i] * s for i in range(3)]
        z = [-x0[i] * s + z0[i] * c for i in range(3)]

        # 列向量 [x, y, z]
        R = [
            [x[0], y[0], z[0]],
            [x[1], y[1], z[1]],
            [x[2], y[2], z[2]],
        ]
        return _mat_to_quat(R)

    def solve_ik_collision_aware(self, pos, orn, collision=True, max_trials=20, reset=True, q_reset=None):
        base_rest = q_reset if q_reset is not None else self.rest_pose[:] 
        if reset:
            q_backup = self.get_current_config()
        self.set_robot_config(base_rest)

        for t in range(max_trials):
            if t == 1:
                rest = base_rest
            else:
                # add some noise to the rest pose 
                rest = [r + random.uniform(-0.1, 0.1) for r in base_rest]

            ik = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                pos,
                orn,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=[u - l for l, u in zip(self.lower_limits, self.upper_limits)],
                restPoses=rest,
                physicsClientId=self.cid,
                maxNumIterations=1000,
                residualThreshold=1e-4,
            )
            q_candidate = list(ik[: self.ndof])
            q_candidate = self._wrap_into_limits(q_candidate, self.home_config)

            # check in range
            bad = False
            for i in range(self.ndof):
                if q_candidate[i] < self.lower_limits[i] - 1e-5 or q_candidate[i] > self.upper_limits[i] + 1e-5:
                    # print(f"[IK] joint {i} out of bounds: {q_candidate[i]:.5f} not in [{self.lower_limits[i]:.5f}, {self.upper_limits[i]:.5f}]")
                    bad = True
                    break
            if bad:
                continue

            if not collision or self.is_state_valid(q_candidate) :
                if reset:
                    self.set_robot_config(q_backup)
                return q_candidate
        # print("[IK] failed to find collision-free IK solution after", max_trials, "trials")
        if reset:
            self.set_robot_config(q_backup)
        return None 

    def sample_redundant(self, index, q_trajectory, q_reset_list, yaws, normal, horizontal, pos, current_config, q_source_trajectory=None, source_tag=None, finger_axis_is_plus_y=False):
        q_goal_list = []
        q_source_list = []
        current_q_reset_list = q_reset_list.copy()
        if index > 0:
            former_q_list = [q[0] for q in q_trajectory[index-1]]
            # import ipdb; ipdb.set_trace()
            mean_former_q = np.mean(former_q_list, axis=0).tolist()
            q_reset = mean_former_q
            current_q_reset_list.append(mean_former_q)

        for reset_idx, q_reset in enumerate(current_q_reset_list):
            
            for yaw in yaws:
                orn = self._quat_from_normal_and_yaw(normal, yaw, horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)
                self.set_robot_config(current_config)
                q_goal = self.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, reset=False, q_reset=q_reset)
                # q_goal2 = self.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, reset=False, q_reset=q_trajectory[index-1][random.randint(0, len(q_trajectory[index-1])-1)][0] if index>0 else q_reset)
                self.set_robot_config(current_config)
                if q_goal is not None:
                    q_goal_list.append((q_goal, yaw))
                    if q_source_trajectory is not None:
                        q_source_list.append({
                            "source_tag": source_tag,     # 这次调用属于 init 还是 refine
                            "reset_idx": reset_idx,       # 是 q_reset_list 里的第几个
                            "q_reset": np.asarray(q_reset, dtype=float).tolist(),
                            "yaw": float(yaw),
                        })
                # if q_goal2 is not None:
                #     q_goal_list.append((q_goal2, yaw))
        q_trajectory[index] += q_goal_list
        if q_source_trajectory is not None:
            q_source_trajectory[index] += q_source_list

    def get_Jacobian(self, ):
        q_full = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.cid)[0] for j in self.active_ids], float)
        zero = [0.0] * len(self.active_ids)
        q = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.cid)[0] for j in self.joint_indices], dtype=float)
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.cid)
        ee_pos = ls[0]
        ee_orn = ls[1]
        Jlin, Jang = p.calculateJacobian(
            self.robot_id, self.ee_link_index,
            localPosition=[0,0,0],
            objPositions=list(q_full),
            objVelocities=zero,
            objAccelerations=zero,
            physicsClientId=self.cid,
        )
        J = np.vstack([np.array(Jlin), np.array(Jang)])  # 6 x n
        J = J[:, :-2]
        return J

    def qs_refinement(self, q1, q2):
        q_backup = self.get_current_config()

        # get N1
        self.set_robot_config(q1)
        J = self.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N1 = np.eye(len(self.joint_indices)) - J_pinv @ J  # nullspace projector

        # get N2
        self.set_robot_config(q2)
        J = self.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N2 = np.eye(len(self.joint_indices)) - J_pinv @ J  # nullspace projector

        # target vector
        q_delta = np.asarray(q2)-np.asarray(q1)

        q1_delta = N1@q_delta
        q2_delta = N2@q_delta

        print(np.linalg.norm(q_delta), np.linalg.norm(q1_delta), np.linalg.norm(q2_delta))
        self.set_robot_config(q_backup)

        return (np.asarray(q1)+1*q1_delta).tolist(), (np.asarray(q2)-1*q2_delta).tolist() 

