import sys
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import pybullet as p
import pybullet_data
from ompl import base as ob
from ompl import geometric as og

import vamp
# from vamp import pybullet_interface as vpb

import numpy as np

from utils.vector import _normalize, quat_from_normal_and_yaw
from utils.pointcloud import pts2obj
from .generic_planner import GenericPlanner

class PandaGripperPlanner(GenericPlanner):
    """
    Franka Panda-based planner with an actuated parallel gripper.
    The core OMPL + PyBullet flow mirrors the KUKA planner but adds
    gripper control so flaps can be pinched before folding.
    """

    def __init__(self, oracle_function=None, cid: Optional[int] = None, box_id: Optional[int] = None, plane_id: Optional[int] = None):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Robot: Franka Panda with gripper
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=cid,
        )
        self.oracle_function = oracle_function
        if box_id is None:
            print("[Warning] box_id is not provided to PandaGripperPlanner, collision checking with the box may not work properly.")

        self.box_constraint_id: Optional[int] = None
        self.gripper_open_width: float = 0.08
        self.gripper_close_width: float = 0.0
        self.gripper_joint_indices: List[int] = []
        self.left_finger_link_index: Optional[int] = None
        self.right_finger_link_index: Optional[int] = None
        self.tip2body=[0.1, 0.0, 0.0]

        robot_model = self._extract_active_joints(robot_id, cid)
        super().__init__(
            cid=cid,
            robot_id=robot_id,
            joint_indices=robot_model["joint_indices"],
            lower_limits=robot_model["lower_limits"],
            upper_limits=robot_model["upper_limits"],
            ee_link_index=robot_model["ee_link_index"],
            collision_link_indices=robot_model["collision_link_indices"],
            plane_id=plane_id,
            box_id=box_id,
            box_attached=-1,
            control_dt=1.0 / 240.0,
            segment_duration=0.01,
            max_torque=500.0,
            position_gain=0.5,
            velocity_gain=1.0,
        )

        self.gripper_joint_indices = robot_model["gripper_joint_indices"]
        self.left_finger_link_index = robot_model["left_finger_link_index"]
        self.right_finger_link_index = robot_model["right_finger_link_index"]
        self.active_ids = self.collision_link_indices.copy()

        # Slightly tucked home pose to keep gripper over the table.
        self.home_config = [0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.8]
        if len(self.home_config) != self.ndof:
            raise RuntimeError("the size of home_config is different from the joint_indices")
        self.rest_pose = list(self.home_config)
        
        self.set_robot_config(self.home_config)

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
    def _extract_active_joints(self, robot_id: int, cid: int) -> Dict[str, object]:
        joint_indices: List[int] = []
        collision_link_indices: List[int] = []
        gripper_joint_indices: List[int] = []
        lower_limits: List[float] = []
        upper_limits: List[float] = []
        ee_link_index = -1
        left_finger_link_index: Optional[int] = None
        right_finger_link_index: Optional[int] = None

        num_joints = p.getNumJoints(robot_id, physicsClientId=cid)
        for j in range(num_joints):
            ji = p.getJointInfo(robot_id, j, physicsClientId=cid)
            joint_type = ji[2]
            joint_name = ji[1].decode("utf-8")
            link_name = ji[12].decode("utf-8")
            is_finger = "finger" in joint_name or "finger" in link_name

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if is_finger:
                    gripper_joint_indices.append(j)
                    collision_link_indices.append(j)

                    # Try to identify left/right finger links for grasp checking.
                    lname = link_name.lower()
                    jname = joint_name.lower()
                    if ("left" in lname) or ("left" in jname):
                        left_finger_link_index = j
                    elif ("right" in lname) or ("right" in jname):
                        right_finger_link_index = j
                else:
                    joint_indices.append(j)
                    ll = ji[8]
                    ul = ji[9]
                    if ul < ll or (ll == 0 and ul == -1): # for simplicty, ignore limits from URDF
                        ll, ul = -3.14, 3.14
                    lower_limits.append(ll)
                    upper_limits.append(ul)
                    ee_link_index = j
                    collision_link_indices.append(j)
            elif is_finger:
                collision_link_indices.append(j)

            if link_name == "panda_grasptarget":
                print("panda_grasptarget_hand found!")
                ee_link_index = j

        # Fallback: if we didn't find explicit left/right labels, just take the first two finger joints.
        if left_finger_link_index is None or right_finger_link_index is None:
            if len(gripper_joint_indices) >= 2:
                left_finger_link_index = gripper_joint_indices[0]
                right_finger_link_index = gripper_joint_indices[1]

        return {
            "joint_indices": joint_indices,
            "collision_link_indices": collision_link_indices,
            "gripper_joint_indices": gripper_joint_indices,
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "ee_link_index": ee_link_index,
            "left_finger_link_index": left_finger_link_index,
            "right_finger_link_index": right_finger_link_index,
        }

    # --------- Pybullet + VAMP helpers -----------
    def pybullet_depth_to_pointcloud(
        self, p, width=320, height=240,
        cam_pos=(1.0, 0.0, 0.8),
        target=(0.0, 0.0, 0.2),
        up=(0,0,1),
        fov=60, near=0.01, far=3.0,
        *, 
        exclude_body_links: Optional[List[Tuple[int, int]]] = None,   # [(bodyUniqueId, linkIndex), ...]
        exclude_bodies: Optional[List[int]] = None,                   # [bodyUniqueId, ...]
    ):
        view = p.computeViewMatrix(cam_pos, target, up)
        proj = p.computeProjectionMatrixFOV(fov, width/height, near, far)
        flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

        _, _, _, depth_buf, seg = p.getCameraImage(width, height, view, proj, flags=flags)
        depth_buf = np.asarray(depth_buf).reshape(height, width)
        seg = np.asarray(seg).reshape(height, width)

        # valid = (seg != self.plane_id) & (seg > 0) # & (seg != self.robot_id) 
        # 先屏蔽背景（环境 seg 是 -1）
        base_valid = (seg >= 0)

        # 解码 objectUniqueId / linkIndex
        obj_uid = np.full_like(seg, -1, dtype=np.int32)
        link_idx = np.full_like(seg, -1, dtype=np.int32)
        seg_v = seg[base_valid]
        obj_uid[base_valid] = seg_v & ((1 << 24) - 1)
        link_idx[base_valid] = (seg_v >> 24) - 1

        # 基础过滤：去掉 plane / 背景
        valid = base_valid & (obj_uid != self.plane_id) & (obj_uid >= 0)

        # 可选：去掉整个人/机器人（你后面还有 filter_self_from_pointcloud，这里去掉能省点）
        if exclude_bodies:
            for bid in exclude_bodies:
                valid &= (obj_uid != int(bid))

        # 核心：去掉被抓 flap link 的像素
        if exclude_body_links:
            for bid, lid in exclude_body_links:
                valid &= ~((obj_uid == int(bid)) & (link_idx == int(lid)))


        # OpenGL: pixel -> NDC
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        x_ndc = (2.0 * (u + 0.5) / width) - 1.0
        y_ndc = 1.0 - (2.0 * (v + 0.5) / height)      # 注意这里把图像y翻到OpenGL
        z_ndc = 2.0 * depth_buf - 1.0                 # depth in [0,1] -> z in [-1,1]

        ones = np.ones_like(z_ndc)
        pts_clip = np.stack([x_ndc, y_ndc, z_ndc, ones], axis=-1)[valid].reshape(-1, 4)

        # inv(P*V): clip -> world
        V = np.array(view).reshape(4, 4).T
        P = np.array(proj).reshape(4, 4).T
        invPV = np.linalg.inv(P @ V)

        pts_world_h = (invPV @ pts_clip.T).T
        pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]

        m = np.isfinite(pts_world).all(axis=1)
        return pts_world[m]

    def build_vamp_env_from_pybullet(self, pts_world: np.ndarray, q):
        # 1) 取出 VAMP 机器人模块（panda）
        vamp_module = vamp.panda

        # 2) 获取机器人最小/最大球半径（VAMP 内部需要）
        r_min, r_max = vamp_module.min_max_radii()

        # 3) 转换为 list，调用 VAMP 的 CAPT 构建
        env = vamp.Environment()

        # 4) filter the point cloud to remove the panda itself
        filtered_pc = vamp_module.filter_self_from_pointcloud(pts_world.tolist(), vamp.POINT_RADIUS*10, q, env)

        build_time = env.add_pointcloud(filtered_pc, r_min, r_max, vamp.POINT_RADIUS*10)
        # pts2obj(filtered_pc, "vamp_env_pointcloud.obj")

        return env, build_time

    # ---------- OMPL Planning ----------
    def plan_ompl(self, q_start: List[float], q_goal: List[float], timeout: float = 3.0, num_waypoints=1000, optimal: bool = False):
        assert len(q_start) == self.ndof and len(q_goal) == self.ndof

        def clamp_and_warn(name, q):
            q_clamped = list(q)
            for i in range(self.ndof):
                lb = float(self.lower_limits[i])
                ub = float(self.upper_limits[i])
                if q[i] < lb or q[i] > ub:
                    print(f"[WARN] {name} joint {i} out of bounds: {q[i]:.5f} not in [{lb:.5f}, {ub:.5f}]")
                    # 简单粗暴：夹到边界
                    if lb == -3.14 and ub == 3.14:
                        q_clamped[i] = q[i] - ((q[i]+3.14)//6.28) * 6.28
                    else:
                        q_clamped[i] = min(max(q[i], lb), ub)
            return q_clamped
        
        q_start = clamp_and_warn("q_start", q_start)
        q_goal = clamp_and_warn("q_goal", q_goal)

        start = ob.State(self.space)
        goal = ob.State(self.space)
        for i in range(self.ndof):
            start[i] = float(q_start[i])
            goal[i] = float(q_goal[i])

        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start, goal)

        if optimal:
            planner = og.InformedRRTstar(self.si)
        else:
            planner = og.RRTConnect(self.si)
        planner.setRange(0.2)
        planner.setProblemDefinition(pdef)
        planner.setup()

        print("[OMPL] solving ...")
        solved = planner.solve(timeout)

        if not solved:
            print("[OMPL] no solution found")
            return None

        path = pdef.getSolutionPath()
        path.interpolate(num_waypoints)
        print("[OMPL] path length (in joint space):", path.length())
        return path

    # ---------- VAMP Planning-----------
    def plan_vamp(self, q_start: List[float], q_goal: List[float], *, rebuild_env: bool = True, env=None, use_ee_in_nn_metric: bool=True):
        q_start = np.array(q_start, dtype=np.float64)
        q_goal  = np.array(q_goal,  dtype=np.float64)


        if env is None or rebuild_env:
            exclude_links = []
            if self.box_id is not None and isinstance(self.box_attached, int) and (0 <= self.box_attached < 4):
                exclude_links.append((self.box_id, self.box_attached))
                print(f"box_id {self.box_id}, box_attached {self.box_attached} has been excluded!")
            pts = self.pybullet_depth_to_pointcloud(p, cam_pos=(0.8, -0.8, 0.8), exclude_body_links=exclude_links, exclude_bodies=[self.robot_id])
            env, _build_time = self.build_vamp_env_from_pybullet(pts, q_start)
            
        
        if not vamp.panda.validate(q_start, env):
            print('[VAMP] start state invalid!')
            return None
        # if not vamp.panda.validate(q_goal, env):
        #     print('[VAMP] goal state invalid!')
        #     return None
        
        W = vamp.panda.DistanceWeights()
        # W.joint = [0.1] + [0.1] * (vamp.panda.dimension() - 1)
        # W.ee_rpy = [0, 0, 1]
        # W.ee_pos = [0, 0, 1]
        W.ee_rpy = [1,0,0]
        W.ee_pos = [1,0,0]

        settings = vamp.RRTCSettings()
        rng = vamp.panda.xorshift(); rng.reset()

        res = vamp.panda.rrtc(q_start, q_goal, env, settings, rng, W, use_ee_in_nn_metric=use_ee_in_nn_metric)
        if res is None or res.path is None:
            print('[VAMP] planning failed!')
            return None
        
        simple = vamp.panda.simplify(res.path, env, vamp.SimplifySettings(), rng)
        simple.path.interpolate_to_resolution(vamp.panda.resolution())
        q_traj = []
        for i in range(simple.path.__len__()):
            q = [float(x) for x in simple.path[i]]
            if not self.is_state_valid(q):
                print(f"[VAMP] Warning: trajectory waypoint {i} is in collision!")
                # return None
            q_traj.append(q)

        return q_traj, env

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

    def close_gripper_to_width(self, target_width: float, force: float = 100.0, wait: float = 10.0):
        # import ipdb; ipdb.set_trace()
        for j in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=-abs(0.02), 
                velocityGain=1.0,
                force=force,
                physicsClientId=self.cid,
            )
        steps = int(wait / self.control_dt) if hasattr(self, "control_dt") else 60
        target = max(0.0, min(self.gripper_open_width, target_width)) * 0.5
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.cid)
            # time.sleep(self.control_dt)
            js = p.getJointStates(self.robot_id, self.gripper_joint_indices, physicsClientId=self.cid)
            jpos = js[0][0]
            if jpos <= target + 0.001:
                break

    # ---------- utils --------------
    def move_to_pose_unified(
        self,
        pos, orn,
        *,
        planner: Literal['OMPL', 'VAMP'] = "OMPL",          # "ompl" or "vamp"
        timeout: float = 4.0,
        ik_collision: bool = True,
        execute: bool = True,
        vamp_env=None,
        rebuild_vamp_env: bool = True,
    ):
        q_start = self.get_current_config()
        q_goal = self.solve_ik_collision_aware(pos, orn, collision=ik_collision)
        if q_goal is None:
            # import ipdb; ipdb.set_trace()
            return None, vamp_env

        # if not self.is_state_valid(q_start, debug=True):
        #     print(f"[ERROR] q_start({q_start}) is invalid!")
        #     return None, vamp_env
        if ik_collision and not self.is_state_valid(q_goal, debug=True):
            print(f"[ERROR] q_goal({q_goal}) is invalid!")
            return None, vamp_env

        if planner == "OMPL":
            path = self.plan_ompl(q_start, q_goal, timeout=timeout, num_waypoints=200, optimal=False)
            if path is None:
                return None, vamp_env
            # OMPL path -> q_traj
            q_traj = []
            for i in range(path.getStateCount()):
                s = path.getState(i)
                q_traj.append([float(s[j]) for j in range(self.ndof)])

        elif planner == "VAMP":
            # import ipdb; ipdb.set_trace()
            out = self.plan_vamp(q_start, q_goal, rebuild_env=rebuild_vamp_env, env=vamp_env)
            if out is None:
                return None, vamp_env
            q_traj, vamp_env = out

        else:
            raise ValueError(f"unknown planner: {planner}")

        if execute:
            self.execute_joint_trajectory_real(q_traj, dt=0.05, interpolate=False)
        else:
            self.execute_joint_trajectory(q_traj, dt=0.05)

        return q_traj, vamp_env

    def move_to_pose_with_free_yaw(
        self,
        pos,
        normal_world,
        yaw: float=None,
        *,
        yaw_samples: int = 10,
        approach_flip: bool = False,   # True: tool -Z 对齐 normal
        planner: str = "OMPL",
        timeout: float = 4.0,
        ik_collision: bool = True,
        execute: bool = True,
        vamp_env=None,
        rebuild_vamp_env: bool = True,
    ):
        """
        给 pos + normal_world，扫描多个 yaw，找到一个可行的 orn，再调用 move_to_pose_unified。
        返回: (traj, chosen_orn, vamp_env)；若失败，traj=None。
        """
        # TODO: sample from a tighter bound instead of [0, 2pi)
        lower_bound, upper_bound = 2.0 * math.pi * 0.4, 2.0 * math.pi * 0.6
        yaws = [lower_bound+k*(upper_bound-lower_bound)/float(max(1, yaw_samples)) for k in range(max(1, yaw_samples))]
        # yaws = [2.0 * math.pi * k / float(max(1, yaw_samples)) for k in range(4, max(1, yaw_samples))]
        
        # yaw provided, try it first, then fall back to sampling
        if yaw is not None:
            yaws = [yaw] + yaws

        for i, yaw in enumerate(yaws):
            orn = quat_from_normal_and_yaw(normal_world, yaw, finger_axis_is_plus_y=approach_flip)

            # Select an end-effector pose that is reachable within the configuration space and collision-free.
            q_goal = self.solve_ik_collision_aware(pos, orn, collision=ik_collision)
            if q_goal is None:
                continue
            elif ik_collision and (not self.is_state_valid(q_goal)):
                continue
            else:
                print(f"[IK] IK solution found for the {i}th candidate!")
                # import ipdb; ipdb.set_trace()

            # Only for selected configuration(yaw) debugging
            # current_q = self.get_current_config()
            # self.set_robot_config(q_goal)
            # input()
            # self.set_robot_config(current_q)
            
            # Execute
            traj, vamp_env = self.move_to_pose_unified(
                pos, orn,
                planner=planner,
                timeout=timeout,
                ik_collision=ik_collision,
                execute=execute,
                vamp_env=vamp_env,
                rebuild_vamp_env=rebuild_vamp_env,
            )
            if traj is not None:
                return traj, orn, vamp_env, yaw

        return None, None, vamp_env, yaw
