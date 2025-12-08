import math
import time
from typing import List, Optional

import pybullet as p
import pybullet_data
from ompl import base as ob
from ompl import geometric as og

from .foldable_box import FoldableBox
from .utils.vector import quat_from_normal_and_axis
from .utils.path import interpolate_joint_line
from .suck_planner import KukaOmplPlanner

class PandaGripperPlanner(KukaOmplPlanner):
    """
    Franka Panda-based planner with an actuated parallel gripper.
    The core OMPL + PyBullet flow mirrors the KUKA planner but adds
    gripper control so flaps can be pinched before folding.
    """

    def __init__(self, use_gui: bool = True, box_base_pos=None):
        self.cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        box_base_pos = box_base_pos or [0.6, 0.0, 0.1]

        # Environment: plane + foldable box
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.foldable_box = FoldableBox(base_pos=box_base_pos, cid=self.cid)
        self.box_id = self.foldable_box.body_id

        # Robot: Franka Panda with gripper
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.cid,
        )

        self.joint_indices: List[int] = []
        self.collision_link_indices: List[int] = []
        self.gripper_joint_indices: List[int] = []
        self.lower_limits: List[float] = []
        self.upper_limits: List[float] = []
        self.rest_pose: List[float] = []
        self.box_attached: int = -2
        self.ee_link_index: int = -1
        self.box_constraint_id: Optional[int] = None
        self.gripper_open_width: float = 0.08
        self.gripper_close_width: float = 0.0

        self._extract_active_joints()

        self.ndof = len(self.joint_indices)
        # Slightly tucked home pose to keep gripper over the table.
        self.home_config = [0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.8]
        if len(self.home_config) != self.ndof:
            self.home_config = [0.0] * self.ndof
        self.rest_pose = list(self.home_config)

        self.control_dt = 1.0 / 240.0
        self.segment_duration = 0.01
        self.max_torque = 500.0
        self.position_gain = 0.5
        self.velocity_gain = 1.0

        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            physicsClientId=self.cid,
        )

        # ---------- OMPL setup ----------
        self.space = ob.RealVectorStateSpace(self.ndof)

        bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            bounds.setLow(i, float(self.lower_limits[i]))
            bounds.setHigh(i, float(self.upper_limits[i]))
        self.space.setBounds(bounds)

        self.si = ob.SpaceInformation(self.space)
        planner_self = self

        class BulletValidityChecker(ob.StateValidityChecker):
            def __init__(self, si_):
                super().__init__(si_)
                self._planner = planner_self

            def isValid(self, state):
                return self._planner.is_state_valid(state)

        self.si.setStateValidityChecker(BulletValidityChecker(self.si))
        self.si.setStateValidityCheckingResolution(0.01)
        self.si.setup()

    # ---------- PyBullet helpers ----------
    def _extract_active_joints(self):
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        for j in range(num_joints):
            ji = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            joint_type = ji[2]
            joint_name = ji[1].decode("utf-8")
            link_name = ji[12].decode("utf-8")
            is_finger = "finger" in joint_name or "finger" in link_name

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if is_finger:
                    self.gripper_joint_indices.append(j)
                    self.collision_link_indices.append(j)
                    continue

                self.joint_indices.append(j)
                ll = ji[8]
                ul = ji[9]
                if ul < ll or (ll == 0 and ul == -1):
                    ll, ul = -3.14, 3.14
                self.lower_limits.append(ll)
                self.upper_limits.append(ul)
                self.rest_pose.append(0.0)
                self.ee_link_index = j
                self.collision_link_indices.append(j)
            elif is_finger:
                self.collision_link_indices.append(j)

            if link_name == "panda_hand":
                self.ee_link_index = j

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
        self.set_robot_config(q)

        check_links = self.collision_link_indices or self.joint_indices
        for link_index in check_links:
            pts1 = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                distance=0.002,
                linkIndexA=link_index,
                linkIndexB=-1,
                physicsClientId=self.cid,
            )
            if len(pts1) > 0:
                return False

            if not self.box_attached == -2:
                for box_link in range(-1, 4):
                    if box_link == self.box_attached:
                        continue
                    pts2 = p.getClosestPoints(
                        bodyA=self.robot_id,
                        bodyB=self.box_id,
                        distance=0.002,
                        linkIndexA=link_index,
                        linkIndexB=-1,
                        physicsClientId=self.cid,
                    )
                    if len(pts2) > 0:
                        return False
        return True

    def is_config_valid(self, q) -> bool:
        assert len(q) == self.ndof
        self.set_robot_config(q)

        check_links = self.collision_link_indices or self.joint_indices
        for link_index in check_links:
            pts1 = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                distance=0.02,
                linkIndexA=link_index,
                linkIndexB=-1,
                physicsClientId=self.cid,
            )
            if len(pts1) > 0:
                return False

            if self.box_attached != -2:
                for box_link in range(-1, 4):
                    if box_link == self.box_attached:
                        continue
                    pts2 = p.getClosestPoints(
                        bodyA=self.robot_id,
                        bodyB=self.box_id,
                        distance=0.02,
                        linkIndexA=link_index,
                        linkIndexB=box_link,
                        physicsClientId=self.cid,
                    )
                    if len(pts2) > 0:
                        return False
        return True

    # ---------- gripper control ----------
    def command_gripper_width(self, width: float, force: float = 40.0):
        target = max(0.0, min(self.gripper_open_width, width)) * 0.5
        for j in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=force,
                positionGain=0.6,
                velocityGain=1.0,
                physicsClientId=self.cid,
            )

    def open_gripper(self):
        self.command_gripper_width(self.gripper_open_width)

    def close_gripper(self, squeeze: float = 0.0):
        self.command_gripper_width(max(self.gripper_close_width, squeeze))

    # ---------- tasks ----------
    def open_flap_with_ompl(
        self,
        flap_id: int,
        target_angle_deg: float = 90.0,
        approach_dist: float = 0.05,
        timeout: float = 4.0,
    ):
        box = self.foldable_box
        angle_rad = math.radians(target_angle_deg)

        old_box_attached = self.box_attached
        self.box_attached = flap_id

        key_start, normal_start, axis_start = box.get_flap_keypoint_pose(
            flap_id, angle=0.0
        )
        approach_pos = [
            key_start[i] + normal_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(normal_start, axis_start)
        # contact_orn = [-i for i in contact_orn]

        self.open_gripper()
        path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=False)
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id}")
            return

        q_contact = self.solve_ik(key_start, contact_orn)
        interp = interpolate_joint_line(self.get_current_config(), q_contact, 60)
        self.execute_joint_trajectory_real(interp)

        self.close_gripper()
        flap_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.box_id,
            childLinkIndex=flap_id,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )
        print(f"[Flap] gripped flap {flap_id} with fixed constraint")

        key_goal, normal_goal, axis_goal = box.get_flap_keypoint_pose(
            flap_id, angle=-angle_rad
        )
        goal_orn = quat_from_normal_and_axis(normal_goal, axis_goal)

        if flap_id == 0:
            center_dir_local = [-1.0, 0.0, 0.0]
        elif flap_id == 1:
            center_dir_local = [1.0, 0.0, 0.0]
        elif flap_id == 2:
            center_dir_local = [0.0, -1.0, 0.0]
        else:
            center_dir_local = [0.0, 1.0, 0.0]

        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.cid
        )
        center_dir_world = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            base_orn,
            center_dir_local,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )[0]
        norm = (center_dir_world[0] ** 2 + center_dir_world[1] ** 2 + center_dir_world[2] ** 2) ** 0.5
        center_dir_world = [c / norm for c in center_dir_world]

        center_pull_dist = 0.1
        key_goal_pulled = [
            key_goal[i] + center_dir_world[i] * center_pull_dist
            for i in range(3)
        ]

        q_start = self.get_current_config()
        self.box_attached = -2
        q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn)

        path_open = self.plan(q_start, q_goal, timeout=timeout)
        if path_open is None:
            print(f"[Flap] failed to plan opening motion for flap {flap_id}")
            p.removeConstraint(flap_constraint, physicsClientId=self.cid)
            self.open_gripper()
            return
        self.execute_path_real(path_open)

        p.removeConstraint(flap_constraint, physicsClientId=self.cid)
        self.open_gripper()

        retreat_pos = [
            key_goal[i] + normal_goal[i] * approach_dist for i in range(3)
        ]
        q_retreat = self.solve_ik(retreat_pos, goal_orn)
        retreat_traj = interpolate_joint_line(
            self.get_current_config(), q_retreat, 45
        )
        self.execute_joint_trajectory(retreat_traj)

        self.box_attached = old_box_attached
