import math
import time
from typing import List, Optional

import pybullet as p
import pybullet_data
from ompl import base as ob
from ompl import geometric as og

from .foldable_box import FoldableBox
from .utils.vector import quat_from_normal_and_axis
from .utils.path import interpolate_joint_line, draw_point
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

        # Disable gripper joint motors for direct control
        # for j in self.gripper_joint_indices:
        #     p.setJointMotorControl2(
        #         bodyUniqueId=self.robot_id,
        #         jointIndex=j,
        #         controlMode=p.VELOCITY_CONTROL,
        #         targetVelocity=0.0,
        #         force=0.0,             
        #         physicsClientId=self.cid,
        #     )

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
                if True or ul < ll or (ll == 0 and ul == -1): # for simplicty, ignore limits from URDF
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

    # ---------- gripper control ----------
    def command_gripper_width(self, width: float, force: float = 40.0, wait: float = 1.0):
        target = max(0.0, min(self.gripper_open_width, width)) * 0.5
        # for j in self.gripper_joint_indices:
        #     p.setJointMotorControl2(
        #         bodyUniqueId=self.robot_id,
        #         jointIndex=j,
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=target,
        #         force=force,
        #         positionGain=0.6,
        #         velocityGain=1.0,
        #         physicsClientId=self.cid,
        #     )
        # steps = int(wait / self.control_dt) if hasattr(self, "control_dt") else 60
        # for _ in range(steps):
        #     p.stepSimulation(physicsClientId=self.cid)
        #     time.sleep(self.control_dt)
        # # for simplicity, just reset the joint state
        for j in self.gripper_joint_indices:
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=target,
                targetVelocity=0.0,
                physicsClientId=self.cid,
            )
        print("[Gripper] moving to width:", width)

    def open_gripper(self):
        # Disable gripper joint motors for direct control
        for j in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=40.0,             
                physicsClientId=self.cid,
            )
        self.command_gripper_width(self.gripper_open_width)

    def close_gripper(self, squeeze: float = 0.0, force: float = 200.0, wait: float = 1.0):
        # self.command_gripper_width(max(self.gripper_close_width, squeeze))
        for j in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force=force,
                positionGain=0.6,
                velocityGain=1,
                physicsClientId=self.cid,
            )
        steps = int(wait / self.control_dt) if hasattr(self, "control_dt") else 60
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(self.control_dt)
        # for j in self.gripper_joint_indices:
        #     p.setJointMotorControl2(
        #         bodyUniqueId=self.robot_id,
        #         jointIndex=j,
        #         controlMode=p.VELOCITY_CONTROL,
        #         targetVelocity=0.0,
        #         force=40.0,             
        #         physicsClientId=self.cid,
        #     )

    # ---------- tasks ----------
    def open_flap_with_ompl(
        self,
        flap_id: int,
        target_angle_deg: float = 90.0,
        approach_dist: float = 0.12,
        timeout: float = 4.0,
    ):
        box = self.foldable_box

        # old_box_attached = self.box_attached
        self.box_attached = 4 # all flaps attached to avoid collision during motion planning

        key_start, normal_start, axis_start, extended_start = box.get_flap_keypoint_pose(
            flap_id, angle=0.0
        )
        approach_pos = [
            key_start[i] + extended_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(extended_start, axis_start)
        # contact_orn = [-i for i in contact_orn]

        self.open_gripper()

        # draw_point(approach_pos, [1, 0, 0], size=0.2, life_time=500)        
        # self.set_robot_config(self.solve_ik_collision_aware(approach_pos, contact_orn, collision=True))
        path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=True, num_waypoints=1000, optimal=False)
        # import ipdb; ipdb.set_trace()
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id}")
            return
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)
        # approach_pos = [
        #     key_start[i] + extended_start[i] * 0.9* approach_dist for i in range(3)
        # ]
        # path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=False, num_waypoints=500)
        # if path is None:
        #     print(f"[Flap] failed to approach flap {flap_id}")
        #     return
        
        self.box_attached = flap_id

        self.close_gripper()

        motion_planning = False
        if motion_planning:
            for delta_angle in range(10, int(target_angle_deg)+1, 10):
                key_goal, normal_goal, axis_goal, extended_goal = box.get_flap_keypoint_pose(
                    flap_id, angle=-math.radians(delta_angle)
                )
                goal_orn = quat_from_normal_and_axis(extended_goal, axis_goal)

                center_pull_dist = 0.12
                key_goal_pulled = [
                    key_goal[i] + extended_goal[i] * center_pull_dist
                    for i in range(3)
                ]

                q_start = self.get_current_config()
                self.box_attached = -2
                q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn, collision=False)
                # self.set_robot_config(q_start)

                path_open = self.plan(q_start, q_goal, timeout=timeout, num_waypoints=200, optimal=False)
                # if path_open is None:
                #     print(f"[Flap] failed to plan opening motion for flap {flap_id}")
                #     p.removeConstraint(flap_constraint, physicsClientId=self.cid)
                #     self.open_gripper()
                #     return
                self.execute_path_real(path_open, DRAW_DEBUG_LINES=True)

                # p.removeConstraint(flap_constraint, physicsClientId=self.cid)
        else: # using IK interpolation
            for delta_angle in range(5, int(target_angle_deg)+1, 5):
                key_goal, normal_goal, axis_goal, extended_goal = box.get_flap_keypoint_pose(
                    flap_id, angle=-math.radians(delta_angle)
                )
                goal_orn = quat_from_normal_and_axis(extended_goal, axis_goal)

                center_pull_dist = 0.12
                key_goal_pulled = [
                    key_goal[i] + extended_goal[i] * center_pull_dist
                    for i in range(3)
                ]

                q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn, collision=True)
                q_start = self.get_current_config()
                traj = interpolate_joint_line(q_start, q_goal, 90)
                self.execute_joint_trajectory_real(traj)
                # self.close_gripper()
            while True:
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(1.0 / 240.0)
                break
        self.close_gripper()
        self.open_gripper()

        retreat_pos = [
            key_goal[i] + extended_goal[i] * approach_dist for i in range(3)
        ]
        q_retreat = self.solve_ik_collision_aware(retreat_pos, goal_orn, collision=True)
        retreat_traj = interpolate_joint_line(
            self.get_current_config(), q_retreat, 45
        )
        self.execute_joint_trajectory(retreat_traj)

        self.box_attached = 4
