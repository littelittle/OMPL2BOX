import math
import time
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, oracle_function=None, cid: Optional[int] = None, box_id: Optional[int] = None):
        self.cid = cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        # Environment: plane + foldable box
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
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
        self.set_robot_config(self.home_config)

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
                # some hand adjustments
                if j==5: # the wrist joint has weird limits
                    ul = 4.8
                if j==1: # the second joint is better limited
                    ll, ul = -2.0, 2.0
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

            if link_name == "panda_hand":
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

    def close_gripper(self, force: float = 100.0, wait: float = 1.0):
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

    # ---------- tasks ----------
    def reach_flap_with_ompl(
        self,
        flap_id: int,
        approach_dist: float = 0.12,
        timeout: float = 4.0,
    ):
        self.open_gripper()

        # key_start, normal_start, axis_start, extended_start = box.get_flap_keypoint_pose(flap_id)
        key_start, normal_start, axis_start, extended_start, angle = self.oracle_function(flap_id)

        approach_pos = [
            key_start[i] + extended_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(extended_start, axis_start)
        # contact_orn = [-i for i in contact_orn]

        draw_point(approach_pos, [1, 0, 0], size=0.2, life_time=500)        
        # self.set_robot_config(self.solve_ik_collision_aware(approach_pos, contact_orn, collision=True))
        # import ipdb; ipdb.set_trace()
        path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=True, num_waypoints=500, optimal=False, debug=False)
        # import ipdb; ipdb.set_trace()
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id}")
            return
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

    def open_flap_with_ompl(
        self,
        flap_id: int,
        target_angle_deg: float = 120.0,
        approach_dist: float = 0.12,
        timeout: float = 4.0,
        motion_planning = False
    ):
        
        # Phase 1: reach out to the flap and attach it 
        self.box_attached = 4 # 4 means all flaps should be taken into account for collision checking

        self.reach_flap_with_ompl(flap_id, approach_dist=approach_dist, timeout=timeout)
        self.close_gripper(wait=0.5)
        while not self.check_grasping_flap(flap_id, debug_draw=True)[0]:
            self.reach_flap_with_ompl(flap_id, approach_dist=approach_dist, timeout=timeout)
            self.close_gripper(wait=0.5)

        # Phase 2: lift the flap by opening the gripper while planning motion 
        self.box_attached = flap_id # the attached flap id is ignored for collision checking

        if motion_planning:
            for delta_angle in range(10, int(target_angle_deg)+1, 10):
                # key_goal, normal_goal, axis_goal, extended_goal = box.get_flap_keypoint_pose(
                #     flap_id, angle=-math.radians(delta_angle)
                # )
                key_goal, normal_goal, axis_goal, extended_goal, _ = self.oracle_function(
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
                    self.reach_flap_with_ompl(flap_id, approach_dist=approach_dist, timeout=timeout)
                    self.close_gripper(wait=0.5)

                # print("[GraspCheck]", ok)
                # input()
                self.close_gripper(wait=0.0)
            # while True:
            #     p.stepSimulation(physicsClientId=self.cid)
            #     time.sleep(1.0 / 240.0)
            #     break
        self.open_gripper()

        retreat_pos = [
            key_goal_pulled[i] + extended_goal[i] * approach_dist * 0.7 for i in range(3)
        ]
        q_retreat = self.solve_ik_collision_aware(retreat_pos, goal_orn, collision=True)
        retreat_traj = interpolate_joint_line(
            self.get_current_config(), q_retreat, 45
        )
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
