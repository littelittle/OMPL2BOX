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

from utils.vector import _normalize, _mat_to_quat, quat_from_normal_and_axis, _cross
from utils.path import interpolate_joint_line, draw_point, omplpath2traj
from utils.pointcloud import pts2obj
from utils.contactframe import ContactFrame
from .generic_planner import GenericPlanner

class PandaGripperPlanner(GenericPlanner):
    """
    Franka Panda-based planner with an actuated parallel gripper.
    The core OMPL + PyBullet flow mirrors the KUKA planner but adds
    gripper control so flaps can be pinched before folding.
    """

    def __init__(self, oracle_function=None, cid: Optional[int] = None, box_id: Optional[int] = None, plane_id: Optional[int] = None):
        self.cid = cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        # Environment: plane + foldable box
        self.plane_id = plane_id
        # import ipdb; ipdb.set_trace()
        # self.foldable_box = FoldableBox(base_pos=box_base_pos, cid=self.cid)
        # self.box_id = self.foldable_box.body_id

        # Robot: Franka Panda with gripper
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.cid,
        )
        self.oracle_function = oracle_function
        if box_id is None:
            print("[Warning] box_id is not provided to PandaGripperPlanner, collision checking with the box may not work properly.")
        self.box_id = box_id

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
        self.left_finger_link_index: Optional[int] = None
        self.right_finger_link_index: Optional[int] = None
        self.tip2body=[0.1, 0.0, 0.0]

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
            numSolverIterations=100,
            physicsClientId=self.cid,
        )
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

                    # Try to identify left/right finger links for grasp checking.
                    lname = link_name.lower()
                    jname = joint_name.lower()
                    if ("left" in lname) or ("left" in jname):
                        self.left_finger_link_index = j
                    elif ("right" in lname) or ("right" in jname):
                        self.right_finger_link_index = j
                    continue

                self.joint_indices.append(j)
                ll = ji[8]
                ul = ji[9]
                # some manual adjustments
                # if j==5: # the wrist joint has weird limits
                #     ul = 4.4 # 4.8
                # if j==1: # the second joint is better limited
                #     ll, ul = -2.3, 2.3
                if j==6: # the last joint is better limited
                    ll, ul = -3.14, 3.14
                if ul < ll or (ll == 0 and ul == -1): # for simplicty, ignore limits from URDF
                    ll, ul = -3.14, 3.14
                self.lower_limits.append(ll)
                self.upper_limits.append(ul)
                self.rest_pose.append(0.0)
                self.ee_link_index = j
                self.collision_link_indices.append(j)
            elif is_finger:
                self.collision_link_indices.append(j)

            if link_name == "panda_grasptarget":
                print("panda_grasptarget_hand found!")
                # import ipdb; ipdb.set_trace()
                self.ee_link_index = j

        # Fallback: if we didn't find explicit left/right labels, just take the first two finger joints.
        if self.left_finger_link_index is None or self.right_finger_link_index is None:
            if len(self.gripper_joint_indices) >= 2:
                self.left_finger_link_index = self.gripper_joint_indices[0]
                self.right_finger_link_index = self.gripper_joint_indices[1]

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
        # 先屏蔽背景（有些环境 seg 会是 -1）
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

    def close_gripper(self, force: float = 100.0, wait: float = 10.0, flap_id=None, min_normal_force=20.0): 
        # TODO: figure out how the coefficient of friction affect the behaviour of the contact
        # p.changeDynamics(self.robot_id, self.gripper_joint_indices[0],  lateralFriction=1.0, spinningFriction=0.05, rollingFriction=0.05)
        # p.changeDynamics(self.robot_id, self.gripper_joint_indices[1], lateralFriction=2.0, spinningFriction=0.05, rollingFriction=0.05)
        # if flap_id:
        #     p.changeDynamics(self.box_id, flap_id, lateralFriction=1.5, spinningFriction=0.02, rollingFriction=0.02)

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
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.cid)
            # time.sleep(self.control_dt)
            # print(p.getJointInfo(self.robot_id, self.gripper_joint_indices[0], physicsClientId=self.cid))
            # sys.exit()
            if flap_id:
                ok, _ = self.check_grasping_flap(
                    flap_id,
                    require_both_fingers=True,
                    min_normal_force=min_normal_force,
                    return_info=True,
                )
                if ok:
                    print(f"[DEBUG] THE TARGET FORCE {min_normal_force}N HAS BEEN REACHED EARLY STOPPING IN CLOSING LOOP")
                    import ipdb; ipdb.set_trace()
                    break
        ok = self.check_grasping_flap(
            flap_id,
            require_both_fingers=True,
            min_normal_force=0,
            return_info=True,
        )
        # import ipdb; ipdb.set_trace()
        print(ok)

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

    # ---------- grasp checking ----------
    def check_grasping_flap(
        self,
        flap_id: int,
        require_both_fingers: bool = True,
        require_contact: bool = True,
        close_tol: float = 0.002,
        keypoint_tol: Optional[float] = None,
        min_normal_force: float = 0.0,
        debug_draw: bool = False,
        return_info: bool = True,
    ):
        """
        判断夹爪是否“真的夹住了”某个 flap。

        判据（可组合）：
        1) 接触判据（默认开启）：左右 finger link 与 flap link 有 contactPoints；
           且（可选）normalForce >= min_normal_force。
        2) 近距离判据（require_contact=False 时启用）：若没有接触，允许用 getClosestPoints
           判断 finger 到 flap 的最小距离 <= close_tol。
        3) keypoint 判据（keypoint_tol 不为 None 时启用）：
           finger 与 flap 的接触点（positionOnB，落在 flap 上的点）离 oracle keypoint 的距离 <= keypoint_tol。

        返回：
            - return_info=False: bool
            - return_info=True : (bool, info_dict)
        """
        if self.box_id is None:
            info = {"ok": False, "reason": "box_id is None"}
            return (False, info) if return_info else False

        lf = getattr(self, "left_finger_link_index", None)
        rf = getattr(self, "right_finger_link_index", None)
        if lf is None or rf is None:
            info = {"ok": False, "reason": "finger link indices not found", "lf": lf, "rf": rf}
            return (False, info) if return_info else False

        def _dist3(a, b) -> float:
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

        # optional keypoint from oracle
        key_world = None
        if keypoint_tol is not None and self.oracle_function is not None:
            try:
                res = self.oracle_function(flap_id)
                key_world = res[0]
            except Exception:
                key_world = None

        def _analyze_finger(finger_link: int) -> Dict:
            out = {
                "finger_link": finger_link,
                "num_contacts": 0,
                "num_close": 0,
                "max_normal_force": 0.0,
                "min_contact_distance": None,
                "min_keypoint_distance": None,
                "best_point_on_flap": None,
                "engaged": False,
                "keypoint_ok": True,
            }

            # 1) strict contact
            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.box_id,
                linkIndexA=finger_link,
                linkIndexB=flap_id,
                physicsClientId=self.cid,
            )
            out["num_contacts"] = len(contacts)

            # parse contacts
            if contacts:
                # contact tuple indices (PyBullet):
                # 6: positionOnB (world), 8: contactDistance, 9: normalForce
                best_force = -1.0
                best_ptB = None
                min_dist = None
                min_kdist = None

                for c in contacts:
                    # robust guards
                    ptB = c[6] if len(c) > 6 else None
                    cdist = c[8] if len(c) > 8 else None
                    nforce = float(c[9]) if len(c) > 9 else 0.0

                    if nforce > best_force and ptB is not None:
                        best_force = nforce
                        best_ptB = ptB

                    if cdist is not None:
                        min_dist = cdist if (min_dist is None or cdist < min_dist) else min_dist

                    if key_world is not None and ptB is not None:
                        kd = _dist3(ptB, key_world)
                        min_kdist = kd if (min_kdist is None or kd < min_kdist) else min_kdist

                out["max_normal_force"] = max(0.0, best_force)
                # import ipdb; ipdb.set_trace()
                out["best_point_on_flap"] = best_ptB
                out["min_contact_distance"] = min_dist
                out["min_keypoint_distance"] = min_kdist

            # 2) near-contact if allowed
            if (not require_contact) and (out["num_contacts"] == 0):
                close_pts = p.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=self.box_id,
                    distance=close_tol,
                    linkIndexA=finger_link,
                    linkIndexB=flap_id,
                    physicsClientId=self.cid,
                )
                out["num_close"] = len(close_pts)

                if close_pts:
                    # closestPoints tuple usually has:
                    # 6: positionOnB, 8: contactDistance
                    best_ptB = None
                    min_dist = None
                    min_kdist = None
                    for c in close_pts:
                        ptB = c[6] if len(c) > 6 else None
                        cdist = c[8] if len(c) > 8 else None
                        if ptB is not None and best_ptB is None:
                            best_ptB = ptB
                        if cdist is not None:
                            min_dist = cdist if (min_dist is None or cdist < min_dist) else min_dist
                        if key_world is not None and ptB is not None:
                            kd = _dist3(ptB, key_world)
                            min_kdist = kd if (min_kdist is None or kd < min_kdist) else min_kdist

                    out["best_point_on_flap"] = out["best_point_on_flap"] or best_ptB
                    out["min_contact_distance"] = out["min_contact_distance"] if out["min_contact_distance"] is not None else min_dist
                    out["min_keypoint_distance"] = out["min_keypoint_distance"] if out["min_keypoint_distance"] is not None else min_kdist

            # 3) decide engaged
            engaged = False
            if out["num_contacts"] > 0:
                engaged = (out["max_normal_force"] >= float(min_normal_force))
            elif not require_contact:
                # near-contact
                if out["min_contact_distance"] is not None and out["min_contact_distance"] <= float(close_tol):
                    engaged = True

            out["engaged"] = engaged

            # 4) keypoint constraint
            if keypoint_tol is not None:
                if out["min_keypoint_distance"] is None:
                    out["keypoint_ok"] = False
                else:
                    out["keypoint_ok"] = (out["min_keypoint_distance"] <= float(keypoint_tol))

            return out

        left_info = _analyze_finger(lf)
        right_info = _analyze_finger(rf)

        def _finger_ok(fi: Dict) -> bool:
            if not fi["engaged"]:
                return False
            if keypoint_tol is not None and not fi["keypoint_ok"]:
                return False
            return True

        left_ok = _finger_ok(left_info)
        right_ok = _finger_ok(right_info)

        ok = (left_ok and right_ok) if require_both_fingers else (left_ok or right_ok)

        if debug_draw:
            # keypoint green, left contact red, right contact blue
            if key_world is not None:
                draw_point(key_world, color=[0, 1, 0], size=0.02, life_time=0)
            if left_info["best_point_on_flap"] is not None:
                draw_point(left_info["best_point_on_flap"], color=[1, 0, 0], size=0.02, life_time=0)
            if right_info["best_point_on_flap"] is not None:
                draw_point(right_info["best_point_on_flap"], color=[0, 0, 1], size=0.02, life_time=0)

        info = {
            "ok": ok,
            "flap_id": flap_id,
            "require_both_fingers": require_both_fingers,
            "require_contact": require_contact,
            "close_tol": close_tol,
            "keypoint_tol": keypoint_tol,
            "min_normal_force": min_normal_force,
            "keypoint": key_world,
            "left": left_info,
            "right": right_info,
        }
        return (ok, info) if return_info else ok

    # ---------- utils --------------
    def _wrap_into_limits(self, q, q_ref):
        qn = list(q)
        period = 2.0 * math.pi

        for i in range(self.ndof):
            lb = float(self.lower_limits[i])
            ub = float(self.upper_limits[i])

            jidx = self.joint_indices[i]
            jtype = p.getJointInfo(self.robot_id, jidx, physicsClientId=self.cid)[2]

            # prismatic: 直接 clamp
            if jtype != p.JOINT_REVOLUTE:
                qn[i] = min(max(qn[i], lb), ub)
                continue

            # 找所有 q + 2πk 落在 [lb,ub] 的整数 k
            k_min = math.ceil((lb - qn[i]) / period)
            k_max = math.floor((ub - qn[i]) / period)

            if k_min <= k_max:
                # 选一个最接近参考角（当前关节角）的解，避免跳变
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

    def solve_ik_collision_aware(self, pos, orn, collision=True, max_trials=20, reset=True, q_reset=None):
        base_rest = q_reset if q_reset else self.rest_pose[:] 
        if reset:
            q_backup = self.get_current_config()
            self.set_robot_config(base_rest)

        for t in range(max_trials):
            if t == 1:
                rest = base_rest
            else:
                # add some noise to the rest pose 
                rest = [
                    r + random.uniform(-0.1, 0.1) for r in base_rest
                ]

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

            bad = False

            # check in range
            for i in range(self.ndof):
                if q_candidate[i] < self.lower_limits[i] - 1e-5 or q_candidate[i] > self.upper_limits[i] + 1e-5:
                    print(f"[IK] joint {i} out of bounds: {q_candidate[i]:.5f} not in [{self.lower_limits[i]:.5f}, {self.upper_limits[i]:.5f}]")
                    bad = True
                    break
            if bad:
                continue

            # import ipdb; ipdb.set_trace()
            if not collision or self.is_state_valid(q_candidate) :
                if reset:
                    self.set_robot_config(q_backup)
                return q_candidate
        print("[IK] failed to find collision-free IK solution after", max_trials, "trials")
        if reset:
            self.set_robot_config(q_backup)
        return None 

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

    def _quat_from_normal_and_yaw(
        self,
        normal_world,
        yaw: float,
        finger_axis_is_plus_y: bool = True,  # True: +Y 对齐 normal；False: -Y 对齐 normal
    ):
        n = _normalize(normal_world)

        # 让 EE 的 y 轴对齐 normal（或反向）
        y = [n[0], n[1], n[2]] if finger_axis_is_plus_y else [-n[0], -n[1], -n[2]]

        # 选一个不与 y 平行的参考向量，构造正交基
        tmp = [1.0, 0.0, 0.0] if abs(y[0]) < 0.9 else [0.0, 0.0, 1.0]
        x0 = _normalize(_cross(tmp, y))
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
            orn = self._quat_from_normal_and_yaw(normal_world, yaw, finger_axis_is_plus_y=approach_flip)

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
    
    # ---------- frame helpers ------
    def _oracle_frame(self, flap_id: int, *, angle: Optional[float] = None) -> ContactFrame:
        """
        统一封装 oracle 输出为 ContactFrame：
        - 兼容返回 (key, normal, axis, extended) 或 (key, normal, axis, extended, angle)
        - 兼容 oracle 是否支持 angle=... 关键字
        """
        if self.oracle_function is None:
            raise RuntimeError("oracle_function is None. Please pass FoldableBox.get_flap_keypoint_pose (or your own oracle).")

        # 1) 调 oracle：优先用 angle=...，不支持就降级
        try:
            res = self.oracle_function(flap_id, angle=angle) if angle is not None else self.oracle_function(flap_id)
        except TypeError:
            # 有些人写 oracle 用位置参数，不支持关键字
            res = self.oracle_function(flap_id, angle) if angle is not None else self.oracle_function(flap_id)

        if not isinstance(res, (tuple, list)) or len(res) < 4:
            raise RuntimeError(f"oracle_function returned unexpected value: {res}")

        key, normal, axis, extended = res[:4]
        # import ipdb; ipdb.set_trace()
        # 2) angle：优先取 oracle 的第 5 项，否则用输入 angle，否则读 bullet joint
        ang = None
        if len(res) >= 5:
            ang = res[4]
        if ang is None and angle:
            ang = float(angle)
        if ang is None:
            raise RuntimeError("[ERROR] WHERE IS YOUR ANGLE?")

        return ContactFrame(
            key=list(map(float, key)),
            normal=list(map(float, normal)),
            axis=list(map(float, axis)),
            extended=list(map(float, extended)),
            angle=float(ang),
        )

    # ---------- primitives ----------
    def prim_pregrasp_pinch(
        self,
        flap_id: int,
        *,
        approach_dist: float,
        PL: Literal["OMPL", "VAMP"] = "OMPL",
        timeout: float = 4.0,
        vamp_env=None,
        rebuild_vamp_env: bool = True,
        debug_draw: bool = True,
    ):
        """
        Primitive: 走到“可夹取”的预抓位（沿 extended 方向退 approach_dist）
        返回: (ok, frame, (pos, orn), vamp_env)
        """
        frame = self._oracle_frame(flap_id)
        pos = [frame.key[i] + frame.extended[i] * float(approach_dist) for i in range(3)]
        orn = quat_from_normal_and_axis(frame.extended, frame.axis)

        if debug_draw:
            draw_point(pos, [1, 0, 0], size=0.2, life_time=500)

        traj, vamp_env = self.move_to_pose_unified(
            pos,
            orn,
            planner=PL,
            timeout=timeout,
            ik_collision=True,
            execute=True,
            vamp_env=vamp_env,
            rebuild_vamp_env=rebuild_vamp_env,
        )
        ok = traj is not None
        return ok, frame, (pos, orn), vamp_env

    def prim_acquire_pinch(
        self,
        flap_id: int,
        *,
        approach_dist: float,
        PL: Literal["OMPL", "VAMP"] = "OMPL",
        timeout: float = 4.0,
        close_wait: float = 5.0,
        min_normal_force: float = 20.0,
        max_attempts: int = 6,
        debug_draw: bool = True,
    ):
        """
        Primitive: Acquire（预抓 + 合爪 + grasp 验证）
        - 内部自带 max_attempts 防止无限 while 卡死
        返回: (ok, grasp_frame, info_dict)
        """
        self.open_gripper()
        last_info = None
        grasp_frame = None

        for k in range(int(max_attempts)):
            ok, frame, _pose, _ = self.prim_pregrasp_pinch(
                flap_id,
                approach_dist=approach_dist,
                PL=PL,
                timeout=timeout,
                vamp_env=None,
                rebuild_vamp_env=True,
                debug_draw=debug_draw,
            )
            if not ok:
                last_info = {"ok": False, "reason": "pregrasp_failed", "attempt": k}
                print(last_info)
                continue

            # 合爪，直到达到一定法向力就提前停（你现在 close_gripper 已支持）
            self.close_gripper(wait=close_wait, flap_id=flap_id, min_normal_force=min_normal_force)
            # import ipdb; ipdb.set_trace()
            g_ok, info = self.check_grasping_flap(
                flap_id,
                require_both_fingers=True,
                min_normal_force=0,
                debug_draw=debug_draw,
                return_info=True,
            )
            last_info = info
            grasp_frame = frame

            if g_ok:
                return True, grasp_frame, info

        return False, grasp_frame, (last_info or {"ok": False, "reason": "unknown"})

    def prim_follow_hinge_open_loop(
        self,
        flap_id: int,
        *,
        target_angle_deg: float,
        step_deg: float = 20.0,
        pull_dist: float = 0.12,
        PL: Literal["OMPL", "VAMP"] = "VAMP",
        timeout: float = 4.0,
        vamp_env=None,
        rebuild_vamp_env: bool = False,
        min_normal_force: float = 20.0,
        reacquire_on_drop: bool = True,
        reacquire_max: int = 3,
        approach_dist_for_reacquire: float = 0.12,
        debug_draw: bool = True,
    ):
        """
        Primitive: 沿铰链“逐步走角度”（每次把 angle 往 |angle| 增大的方向推进 step_deg）
        - 每步都 move_to_pose_unified(goal_pos, goal_orn)
        - VAMP 可复用 vamp_env（第一步 rebuild，之后不 rebuild）---不要rebuild!
        返回: (ok, last_frame, (last_pos, last_orn), vamp_env)
        """
        # 复用 env：第一步按参数决定 rebuild，之后都不 rebuild
        rebuild = bool(rebuild_vamp_env)
        rebuild = True
        reacq_count = 0

        last_frame = None
        last_pose = None

        # 用 bullet 读当前角度更可靠
        current_angle = self.oracle_function(flap_id)[-1]

        # 防止死循环：最多走这么多步
        max_steps = int(max(1, abs(float(target_angle_deg)) / max(float(step_deg), 1e-6))) + 100

        for _ in range(max_steps):
            current_angle = self.oracle_function(flap_id)[-1]
            current_deg = math.degrees(current_angle)
            print(f"[DEG]: {current_deg}")

            if abs(current_deg) >= float(target_angle_deg) - 1e-3:
                return True, last_frame, last_pose, vamp_env

            # 1) 夹持检查
            start_time = time.time()
            g_ok, _info = self.check_grasping_flap(
                flap_id,
                require_both_fingers=True,
                # min_normal_force=min_normal_force,
                debug_draw=debug_draw,
                return_info=True,
            )
            if not g_ok:
                if not reacquire_on_drop or reacq_count >= int(reacquire_max):
                    return False, last_frame, last_pose, vamp_env

                # 掉了就重新 acquire
                self.box_attached = 4
                ok, _gf, _gi = self.prim_acquire_pinch(
                    flap_id,
                    approach_dist=approach_dist_for_reacquire,
                    PL=PL,
                    timeout=timeout,
                    close_wait=5.0,
                    min_normal_force=min_normal_force,
                    max_attempts=4,
                    debug_draw=debug_draw,
                )
                # if not ok:
                #     return False, last_frame, last_pose, vamp_env

                self.box_attached = flap_id
                reacq_count += 1
                rebuild = True  # 重新抓取后建议 rebuild 一次 env
                continue

            # 2) 推进下一步角度：往 |angle| 增大方向走
            sign = -1.0 # if current_angle <= 0.0 else 1.0
            next_angle = float(current_angle) + sign * math.radians(float(step_deg))

            frame = self._oracle_frame(flap_id, angle=next_angle)

            goal_pos = [frame.key[i] + frame.extended[i] * float(pull_dist) for i in range(3)]
            goal_orn = quat_from_normal_and_axis(frame.extended, frame.axis)

            if debug_draw:
                draw_point(goal_pos, [1, 1, 0], size=0.12, life_time=200)
            checked_time = time.time()
            traj, vamp_env = self.move_to_pose_unified(
                goal_pos,
                goal_orn,
                planner=PL,
                timeout=timeout,
                ik_collision=True,
                execute=True,
                vamp_env=vamp_env,
                rebuild_vamp_env=rebuild,
            )
            if traj is None:
                return False, frame, (goal_pos, goal_orn), vamp_env

            last_frame = frame
            last_pose = (goal_pos, goal_orn)
            # rebuild = False  # 后续复用 env

        return False, last_frame, last_pose, vamp_env    

    def prim_retreat_linear_ik(
        self,
        pos,
        orn,
        *,
        direction: List[float],
        distance: float,
        steps: int = 45,
        collision: bool = True,
        segment_duration: float = 0.05,
    ):
        """
        Primitive: 用 IK + 关节直线插值撤退（你原来 Phase2/Phase3 后的 retreat 就是这个）
        返回: (ok, (retreat_pos, retreat_orn))
        """
        retreat_pos = [pos[i] + float(direction[i]) * float(distance) for i in range(3)]
        q_goal = self.solve_ik_collision_aware(retreat_pos, orn, collision=collision)
        if q_goal is None:
            return False, (retreat_pos, orn)

        q_start = self.get_current_config()
        traj = interpolate_joint_line(q_start, q_goal, int(steps))
        self.execute_joint_trajectory_real(traj, segment_duration=segment_duration)
        return True, (retreat_pos, orn)

    def prim_press_stab_sequence(
        self,
        flap_id: int,
        *,
        start_deg: int,
        end_deg: int = 180,
        step_deg: int = 5,
        press_dist: float = 0.12,
        interp_steps: int = 90,
        segment_duration: float = 0.05,
        debug_draw: bool = True,
    ):
        """
        Primitive: “stabbing/press”序列（你原 Phase3）
        - 每个角度：算 key/normal/axis -> 生成一个从外侧 press 的 pose
        - 用 IK + 关节直线插值执行（不走全局 planner）
        返回: (last_frame, (last_pos, last_orn))
        """
        last_frame = None
        last_pose = None

        for deg in range(int(start_deg), int(end_deg), int(step_deg)):
            frame = self._oracle_frame(flap_id, angle=-math.radians(float(deg)))

            # 从外侧沿 normal “顶进去”：pos = key - normal * press_dist
            goal_pos = [frame.key[i] - frame.normal[i] * float(press_dist) for i in range(3)]
            goal_orn = quat_from_normal_and_axis([-x for x in frame.normal], frame.axis)

            if debug_draw:
                draw_point(frame.key, [0, 1, 0], size=0.2, life_time=120)

            q_goal = self.solve_ik_collision_aware(goal_pos, goal_orn, collision=False)
            if q_goal is None:
                continue

            q_start = self.get_current_config()
            traj = interpolate_joint_line(q_start, q_goal, int(interp_steps))
            self.execute_joint_trajectory_real(traj, segment_duration=segment_duration)

            # 这里你原来是 close_gripper(wait=0.0) 当作“施压/就位动作”
            self.close_gripper(wait=0.0, flap_id=flap_id)

            last_frame = frame
            last_pose = (goal_pos, goal_orn)

        return last_frame, last_pose

    def reach_flap(
        self,
        flap_id: int,
        approach_dist: float = 0.10,
        timeout: float = 4.0,
        PL: Literal['OMPL', 'VAMP'] = 'OMPL',
    ):
        self.open_gripper()
        key_start, normal_start, axis_start, extended_start, angle = self.oracle_function(flap_id)
        approach_pos = [
            key_start[i] + extended_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(extended_start, axis_start)
        draw_point(approach_pos, [1, 0, 0], size=0.2, life_time=500)  

        path, _ = self.move_to_pose_unified(approach_pos, contact_orn, planner=PL)
        
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id} using {PL}")
            return
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

    def close_flap(
        self,
        flap_id: int,
        target_angle_deg: float = 120.0,
        approach_dist: float = 0.12,
        timeout: float = 4.0,
        motion_planning = True,
        PL: Literal['OMPL', 'VAMP'] = 'OMPL',
    ):
        
        # Phase 1: reach out to the flap and attach it 
        self.box_attached = 4 # 4 means all flaps should be taken into account for collision checking
        self.reach_flap(flap_id, approach_dist=approach_dist, timeout=timeout, PL=PL)
        self.close_gripper(wait=5, flap_id=flap_id)
        while not self.check_grasping_flap(flap_id, debug_draw=True)[0]:
            self.reach_flap(flap_id, approach_dist=approach_dist, timeout=timeout, PL=PL)
            self.close_gripper(wait=5, flap_id=flap_id)

        # Phase 2: lift the flap by opening the gripper while planning motion 
        self.box_attached = flap_id # the attached flap id is ignored for collision checking
        if motion_planning:
            stride=10
            while True:
                ok = self.check_grasping_flap(flap_id, debug_draw=True)
                if not ok[0]:
                    self.box_attached = 4
                    print(f"[Flap] lost grasping on flap {flap_id} at angle {current_degree} deg")
                    print("retrying...")
                    self.reach_flap(flap_id, approach_dist=approach_dist, timeout=timeout)
                    self.close_gripper(wait=5)
                    self.box_attached = flap_id
                _, _, _, _, current_angle = self.oracle_function(flap_id)
                current_degree = math.degrees(current_angle)
                if abs(current_degree) > target_angle_deg:
                    break
                key_goal, normal_goal, axis_goal, extended_goal, _ = self.oracle_function(
                    flap_id, angle=math.radians(current_degree-10)
                )
                print(f"current degree is {current_degree}")
                goal_orn = quat_from_normal_and_axis(extended_goal, axis_goal)
                center_pull_dist = 0.12
                key_goal_pulled = [
                    key_goal[i] + extended_goal[i] * center_pull_dist
                    for i in range(3)
                ]

                q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn, collision=False)
                # self.close_gripper(wait=0.0)

                q_start = self.get_current_config()
                # import ipdb; ipdb.set_trace()
                if PL == 'OMPL':
                    path = self.plan_ompl(q_start, q_goal, num_waypoints=50)
                    # OMPL path -> q_traj
                    q_traj = []
                    for i in range(path.getStateCount()):
                        s = path.getState(i)
                        q_traj.append([float(s[j]) for j in range(self.ndof)])
                    path = q_traj
                elif PL == 'VAMP':
                    path, _ = self.plan_vamp(q_start, q_goal)
                print("the lenght of the path is: ", len(path))
                # traj = omplpath2traj(path_open)
                self.execute_joint_trajectory_real(path, segment_duration=0.1)
        else: # using IK interpolation
            for delta_angle in range(-int(math.degrees(self.oracle_function(flap_id)[-1])), int(target_angle_deg)+1, 5):
                key_goal, normal_goal, axis_goal, extended_goal, _ = self.oracle_function(
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
                ok = self.check_grasping_flap(flap_id, debug_draw=True)
                if not ok[0]:
                    print(f"[Flap] lost grasping on flap {flap_id} at angle {delta_angle} deg")
                    print("retrying...")
                    self.reach_flap(flap_id, approach_dist=approach_dist, timeout=timeout)
                    self.close_gripper(wait=0.5)
                self.close_gripper(wait=0.0)
        
        self.open_gripper()

        retreat_pos = [key_goal_pulled[i] + extended_goal[i] * approach_dist * 0.7 for i in range(3)]
        q_retreat = self.solve_ik_collision_aware(retreat_pos, goal_orn, collision=True)
        retreat_traj = interpolate_joint_line(self.get_current_config(), q_retreat, 45)
        self.execute_joint_trajectory_real(retreat_traj)

        # Phase3: fully close the flap to the target angle using stabbing motion
        for delta_angle in range(-int(math.degrees(self.oracle_function(flap_id)[-1])), 180, 5):
            key_goal, normal_goal, axis_goal, _, _ = self.oracle_function(
                flap_id, angle=-math.radians(delta_angle)
            )
            goal_orn = quat_from_normal_and_axis([-i for i in normal_goal], axis_goal)
            draw_point(key_goal, [0, 1, 0], size=0.2, life_time=500)
            center_pull_dist = 0.12
            key_goal_pulled = [key_goal[i] - normal_goal[i] * center_pull_dist for i in range(3)]

            q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn, collision=False)
            # self.set_robot_config(q_goal)
            # import ipdb; ipdb.set_trace()
            q_start = self.get_current_config()
            traj = interpolate_joint_line(q_start, q_goal, 90)
            self.execute_joint_trajectory_real(traj)
            self.close_gripper(wait=0.0)
        # import ipdb; ipdb.set_trace()
        
        retreat_pos = [
            key_goal_pulled[i] - normal_goal[i] * approach_dist  for i in range(3)
        ]
        q_retreat = self.solve_ik_collision_aware(retreat_pos, goal_orn, collision=True)
        retreat_traj = interpolate_joint_line(
            self.get_current_config(), q_retreat, 90
        )
        self.execute_joint_trajectory_real(retreat_traj)
        # import ipdb; ipdb.set_trace()
        self.box_attached = 4

    def close_flap(
        self,
        flap_id: int,
        target_angle_deg: float = 100.0,
        approach_dist: float = 0.00,
        timeout: float = 4.0,
        motion_planning: bool = True,
        PL: Literal["OMPL", "VAMP"] = "VAMP",
    ):
        """
        close_flap = orchestration (Level-2)
        依赖的 Level-1 primitives:
          - prim_acquire_pinch
          - prim_follow_hinge_open_loop
          - prim_retreat_linear_ik
          - prim_press_stab_sequence
        """

        # -------------------------
        # Phase 1: acquire pinch
        # -------------------------
        self.box_attached = 4  # 抓取前：所有 flap 都算碰撞
        ok, grasp_frame, grasp_info = self.prim_acquire_pinch(
            flap_id,
            approach_dist=approach_dist,
            PL=PL,
            timeout=timeout,
            close_wait=5.0,
            min_normal_force=20.0,
            max_attempts=8,
            debug_draw=True,
        )
        if not ok:
            print(f"[Flap] acquire_pinch failed on flap {flap_id}. info={grasp_info}")
            return False

        # -------------------------
        # Phase 2: follow hinge (planning)
        # -------------------------
        self.box_attached = flap_id  # 抓住后：忽略该 flap 的碰撞 + VAMP 点云里可排除该 link

        vamp_env = None
        if motion_planning:
            ok, last_frame, last_pose, vamp_env = self.prim_follow_hinge_open_loop(
                flap_id,
                target_angle_deg=target_angle_deg,
                step_deg=25.0,
                pull_dist=approach_dist,
                PL=PL,
                timeout=timeout,
                vamp_env=vamp_env,
                rebuild_vamp_env=True,
                min_normal_force=20.0,
                reacquire_on_drop=True,
                reacquire_max=3,
                approach_dist_for_reacquire=approach_dist,
                debug_draw=True,
            )
            if not ok:
                print(f"[Flap] follow_hinge failed on flap {flap_id}.")
                return False
        else:
            print("[Flap] motion_planning=False not implemented in refactor (use motion_planning=True).")
            return False

        # -------------------------
        # Phase 2.5: release + retreat a bit
        # -------------------------
        self.open_gripper()
        # import ipdb; ipdb.set_trace()
        # if last_frame is not None and last_pose is not None:
        #     last_pos, last_orn = last_pose
        #     # 沿 extended 方向退一点（原逻辑：approach_dist*0.7）
        #     self.prim_retreat_linear_ik(
        #         last_pos,
        #         last_orn,
        #         direction=last_frame.extended,
        #         distance=approach_dist * 0.7,
        #         steps=10,
        #         collision=True,
        #         segment_duration=0.05,
        #     )
        # else:
        #     raise RuntimeWarning("last_frame or last_pose is None!")

        # -------------------------
        # Phase 3: press/stab to "seat" / fully close
        # -------------------------
        current_deg = abs(math.degrees(self.oracle_function(flap_id)[-1]))
        start_deg = int(current_deg)

        last_frame2, last_pose2 = self.prim_press_stab_sequence(
            flap_id,
            start_deg=start_deg,
            end_deg=180,
            step_deg=5,
            press_dist=approach_dist,
            interp_steps=10,
            segment_duration=0.05,
            debug_draw=True,
        )

        # if last_frame2 is not None and last_pose2 is not None:
        #     last_pos, last_orn = last_pose
        #     # 沿 extended 方向退一点（原逻辑：approach_dist*0.7）
        #     self.prim_retreat_linear_ik(
        #         last_pos,
        #         last_orn,
        #         direction=last_frame.extended,
        #         distance=approach_dist * 0.7,
        #         steps=10,
        #         collision=True,
        #         segment_duration=0.05,
        #     )

        self.box_attached = 4
        return True

    def back_home(
        self,
        *,
        PL: Literal['OMPL', 'VAMP'] = "OMPL",
        timeout: float = 4.0,
        ik_collision: bool = True,
        execute: bool = True,
        vamp_env=None,
        rebuild_vamp_env: bool = True,
    ):
        q_backup = self.get_current_config()
        self.set_robot_config(self.home_config)
        link_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.cid)
        home_pos, home_orn = link_state[0], link_state[1]
        self.set_robot_config(q_backup)
        return self.move_to_pose_unified(
            home_pos,
            home_orn,
            planner=PL,
            timeout=timeout,
            ik_collision=ik_collision,
            execute=execute,
            vamp_env=vamp_env,
            rebuild_vamp_env=rebuild_vamp_env,
        )

    # --------- tasks -----------
    def close_double_flap(self,):
        print("[Demo] Closing a foldable box with 2 flaps ...")
        for i in range(3, 1, -1):
            self.close_flap(i, PL='VAMP')
            self.back_home(PL='VAMP')
            print(f"Flap {i} opened.")