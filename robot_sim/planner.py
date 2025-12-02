import math
import time
from typing import List, Optional

import pybullet as p
import pybullet_data
from ompl import base as ob
from ompl import geometric as og

from .foldable_box import FoldableBox
from .utils.vector import quat_from_normal_and_axis

def interpolate_joint_line(q_from: List[float], q_to: List[float], steps: int):
    """Joint-space linear interpolation including endpoints."""
    path = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        path.append([qf + alpha * (qt - qf) for qf, qt in zip(q_from, q_to)])
    return path


class KukaOmplPlanner:
    """
    OMPL-based joint-space motion planner for the KUKA iiwa in PyBullet.
    Includes a foldable box with actuated flaps and unpacking routine.
    """

    def __init__(self, use_gui: bool = True, box_base_pos=None):
        self.cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        box_base_pos = box_base_pos or [0.7, 0.0, 0.1]

        # Environment: plane + foldable box
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.foldable_box = FoldableBox(base_pos=box_base_pos, cid=self.cid)
        self.box_id = self.foldable_box.body_id

        # Robot: KUKA iiwa
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.cid,
        )

        self.joint_indices: List[int] = []
        self.lower_limits: List[float] = []
        self.upper_limits: List[float] = []
        self.rest_pose: List[float] = []
        self.box_attached: int = -2 # i.e. -1 is the base, 0~3 are flaps, -2 means none
        self.ee_link_index: int = -1
        self.box_constraint_id: Optional[int] = None

        self._extract_active_joints()

        self.ndof = len(self.joint_indices)
        self.home_config = [0.0] * self.ndof

        # 控制相关参数（你可以按需要调整）
        self.control_dt = 1.0 / 240.0  # 单步仿真时间
        self.segment_duration = 0.01    # 每个 OMPL waypoint 用多少秒走完
        self.max_torque = 500.0         # 每个关节的最大力矩（牛米左右，可根据机器人调大/调小）
        self.position_gain = 0.5       # 位置环增益
        self.velocity_gain = 1.0       # 速度环增益

        # 物理引擎迭代次数可以调高一点，让约束更稳定
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
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_indices.append(j)
                ll = ji[8]
                ul = ji[9]
                if ul < ll or (ll == 0 and ul == -1):
                    ll, ul = -3.14, 3.14
                self.lower_limits.append(ll)
                self.upper_limits.append(ul)
                self.rest_pose.append(0.0)
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

        for link_index in self.joint_indices:
            pts1 = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                distance=0.0,
                linkIndexA=link_index,
                linkIndexB=-1,
                physicsClientId=self.cid,
            )
            if len(pts1) > 0:
                return False

        # if not self.box_attached:
            for box_link in range(-1, 4):
                if box_link == self.box_attached:
                    continue
                pts2 = p.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=self.box_id,
                    distance=0.0,
                    linkIndexA=link_index,
                    linkIndexB=-1,
                    physicsClientId=self.cid,
                )
                if len(pts2) > 0:
                    return False

        return True

    def _set_joint_targets_position_control(self, q_target: List[float]):
        """给所有关节设置 position control 目标，不做 stepSimulation。"""
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

    # ---------- OMPL planning ----------
    def plan(self, q_start: List[float], q_goal: List[float], timeout: float = 3.0):
        assert len(q_start) == self.ndof and len(q_goal) == self.ndof

        def clamp_and_warn(name, q):
            q_clamped = list(q)
            for i in range(self.ndof):
                lb = float(self.lower_limits[i])
                ub = float(self.upper_limits[i])
                if q[i] < lb or q[i] > ub:
                    print(f"[WARN] {name} joint {i} out of bounds: {q[i]:.5f} not in [{lb:.5f}, {ub:.5f}]")
                    # 简单粗暴：夹到边界
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
        path.interpolate(500)
        print("[OMPL] path length (in joint space):", path.length())
        return path

    def execute_path(self, path, dt: float = 1.0 / 240.0, substeps=100):
        if path is None:
            return
        print("[PyBullet] executing trajectory with", path.getStateCount(), "waypoints")
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = [float(state[j]) for j in range(self.ndof)]
            self.set_robot_config(q)
            # for _ in range(substeps):
            #     p.stepSimulation(physicsClientId=self.cid)
            time.sleep(dt)

    def execute_path_real(
        self,
        path,
        dt: float = None,
        segment_duration: float = None,
        interpolate: bool = True,
    ):
        """用 position control 执行 OMPL 规划出来的关节轨迹。

        - path: OMPL 的 PathGeometric
        - dt:   每次 stepSimulation 的时间间隔（默认用 self.control_dt）
        - segment_duration: 每两个 waypoint 之间走多久（秒）
        - interpolate: 是否在两个 waypoint 之间做线性插值
        """
        if path is None:
            return

        if dt is None:
            dt = self.control_dt
        if segment_duration is None:
            segment_duration = self.segment_duration

        steps_per_segment = max(1, int(segment_duration / dt))

        print(
            "[PyBullet] executing trajectory with",
            path.getStateCount(),
            "waypoints,",
            steps_per_segment,
            "steps per segment",
        )

        # 当前关节配置（从仿真里读，而不是假定从0开始）
        q_curr = self.get_current_config()

        # 对 OMPL 路径的每个 waypoint 依次执行
        for i in range(path.getStateCount()):
            print("[PyBullet] moving to waypoint", i + 1, "/", path.getStateCount())
            state = path.getState(i)
            q_next = [float(state[j]) for j in range(self.ndof)]

            # 在 q_curr -> q_next 之间走 steps_per_segment 步
            for k in range(steps_per_segment):
                if interpolate:
                    alpha = float(k + 1) / float(steps_per_segment)
                    q_cmd = [
                        q_curr[d] + alpha * (q_next[d] - q_curr[d])
                        for d in range(self.ndof)
                    ]
                else:
                    # 不插值的话，直接对 q_next 做 position control
                    q_cmd = q_next

                # 把这一时刻的目标位置发给电机
                self._set_joint_targets_position_control(q_cmd)

                # 让物理引擎滚动一步（这一步里会解约束、算接触等）
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)

            # 段结束，更新当前 q
            q_curr = q_next

    def execute_joint_trajectory(self, qs: List[List[float]], dt: float = 1.0 / 240.0):
        for q in qs:
            self.set_robot_config(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(dt)

    def execute_joint_trajectory_real(
        self,
        qs: List[List[float]],
        dt: float = None,
        segment_duration: float = None,
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


    # ---------- IK / helpers ----------
    def solve_ik(self, pos, orn):
        ik = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            pos,
            orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=[u - l for l, u in zip(self.lower_limits, self.upper_limits)],
            restPoses=self.rest_pose,
            physicsClientId=self.cid,
        )
        return list(ik[: self.ndof])

    def move_to_pose(self, pos, orn, timeout: float = 4.0, real=True):
        q_start = self.get_current_config()
        q_goal = self.solve_ik(pos, orn)
        path = self.plan(q_start, q_goal, timeout=timeout)
        if path is not None:
            if real:
                self.execute_path_real(path)
            else:
                self.execute_path(path)
        return path

    # ---------- grasp helpers ----------
    def attach_box(self):
        if self.box_constraint_id is not None:
            return
        self.box_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.box_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )
        self.box_attached = True

    def detach_box(self):
        if self.box_constraint_id is not None:
            p.removeConstraint(self.box_constraint_id, physicsClientId=self.cid)
        self.box_constraint_id = None
        self.box_attached = False

    # ---------- tasks ----------
    def pick_and_place(self, box_pos, place_pos):
        box_z = box_pos[2]
        place_z = place_pos[2]
        approach_h = 0.25

        p.resetBasePositionAndOrientation(
            self.box_id,
            box_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.cid,
        )

        self.set_robot_config(self.home_config)
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

        grasp_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        q_start = self.get_current_config()
        q_pick_above = self.solve_ik(
            [box_pos[0], box_pos[1], box_z + approach_h], grasp_orn
        )
        path1 = self.plan(q_start, q_pick_above, timeout=5.0)
        if path1 is None:
            print("[Demo] failed to find path to pick position")
            return
        self.execute_path(path1)

        q_grasp = self.solve_ik([box_pos[0], box_pos[1], box_z + 0.05], grasp_orn)
        interp_to_grasp = interpolate_joint_line(self.get_current_config(), q_grasp, 60)
        self.execute_joint_trajectory(interp_to_grasp)
        self.attach_box()

        q_place_above = self.solve_ik(
            [place_pos[0], place_pos[1], place_z + approach_h], grasp_orn
        )
        path2 = self.plan(self.get_current_config(), q_place_above, timeout=5.0)
        if path2 is None:
            print("[Demo] failed to find path to place position")
            return
        self.execute_path(path2)

        q_place = self.solve_ik([place_pos[0], place_pos[1], place_z + 0.05], grasp_orn)
        interp_to_place = interpolate_joint_line(self.get_current_config(), q_place, 60)
        self.execute_joint_trajectory(interp_to_place)

        self.detach_box()
        p.resetBasePositionAndOrientation(
            self.box_id,
            [place_pos[0], place_pos[1], place_z],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.cid,
        )
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

    def open_flap_with_ompl(
        self,
        flap_id: int,
        target_angle_deg: float = 90.0,
        approach_dist: float = 0.05,
        timeout: float = 4.0,
    ):
        """
        示例：利用几何 + OMPL 打开某个 flap 到指定角度。
        流程：
          1) 通过几何接口算出 flap 在 angle=0 和 angle=target_angle 时关键点 & 法向；
          2) 先从当前姿态规划到“接近关键点”的姿态；
          3) 直接插值下压/贴合关键点（模拟吸附）；
          4) （可选）创建 P2P 约束把 flap 铰接到末端；
          5) 将 flap 视作已 attach，在关节空间规划到目标关键点对应的末端 pose；
          6) 执行轨迹，最后撤离一点距离。
        注意：
          - 这里只用到我们自己定义的“虚拟几何 flap”，并不依赖 URDF 里 flap 当前的真实角度。
        """

        # print("using open_flap_with_ompl")
        box = self.foldable_box
        angle_rad = math.radians(target_angle_deg)

        old_box_attached = self.box_attached
        self.box_attached = flap_id

        # 1) 算出 angle = 0（平放）时的关键点和法向：用作“接触点”
        key_start, normal_start, axis_start = box.get_flap_keypoint_pose(
            flap_id, angle=0.0
        )

        # 接近点：沿法向外侧退一点
        approach_pos = [
            key_start[i] + normal_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(normal_start, axis_start)


        # 2) 规划到接近点
        path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=False)
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id}")
            return
        
        # import ipdb; ipdb.set_trace()

        # 3) 插值下压到关键点（比再跑一次 OMPL 稳定）
        # q_contact = self.solve_ik(key_start, contact_orn)
        # interp = interpolate_joint_line(self.get_current_config(), q_contact, 60)
        # self.execute_joint_trajectory_real(interp)

        # import ipdb; ipdb.set_trace()

        # 4) （可选）创建约束，模拟末端吸附 flap
        flap_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.box_id,
            childLinkIndex=flap_id,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )
        print(f"[Flap] grasped flap {flap_id} with point-to-point constraint")

        # import ipdb; ipdb.set_trace()

        # while True:
        #     p.stepSimulation(physicsClientId=self.cid)

        # 5) 准备目标姿态：目标角度下的关键点 & 法向
        key_goal, normal_goal, axis_goal = box.get_flap_keypoint_pose(
            flap_id, angle=-angle_rad
        )
        goal_orn = quat_from_normal_and_axis(normal_goal, axis_goal)

        q_start = self.get_current_config()
        q_goal = self.solve_ik(key_goal, goal_orn)

        # 在规划阶段暂时“忽略 box 碰撞”（相当于把 flap 视为机器人一部分）
        
        path_open = self.plan(q_start, q_goal, timeout=timeout)
        self.box_attached = old_box_attached

        if path_open is None:
            print(f"[Flap] failed to plan opening motion for flap {flap_id}")
            p.removeConstraint(flap_constraint, physicsClientId=self.cid)
            return

        self.execute_path_real(path_open)

        # import ipdb; ipdb.set_trace()

        # 6) 松开 flap，做一个小撤离
        p.removeConstraint(flap_constraint, physicsClientId=self.cid)

        retreat_pos = [
            key_goal[i] + normal_goal[i] * approach_dist for i in range(3)
        ]
        q_retreat = self.solve_ik(retreat_pos, goal_orn)
        retreat_traj = interpolate_joint_line(
            self.get_current_config(), q_retreat, 45
        )
        self.execute_joint_trajectory(retreat_traj)

        # import ipdb; ipdb.set_trace()

    def unpack_box(self):
        box = self.foldable_box
        approach_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        approach_h = 0.14
        settle_steps = 120

        for flap_id in box.flap_joint_indices:
            print("----------Processing flap", flap_id, "----------")
            target_pos, _, outward = box.get_flap_target_pose(flap_id)
            approach_pos = [target_pos[0], target_pos[1], target_pos[2] + approach_h]
            print(f"[Demo] approaching flap {flap_id} at", approach_pos)
            path = self.move_to_pose(approach_pos, approach_orn, timeout=5.0)
            if path is None:
                print(f"[Demo] failed to reach flap {flap_id}")
                continue

            tap_pos = [target_pos[0]-0.01, target_pos[1], target_pos[2]-0.01]
            # q_tap = self.solve_ik(tap_pos, approach_orn)
            # import ipdb; ipdb.set_trace()
            # self.execute_joint_trajectory(
            #     interpolate_joint_line(self.get_current_config(), q_tap, 100)
            # )
            path = self.move_to_pose(tap_pos, approach_orn, timeout=3.0)
            if path is None:
                print(f"[Demo] failed to reach flap {flap_id}")
                continue

            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.box_id,
                linkIndexA=self.ee_link_index,
                linkIndexB=flap_id,
                physicsClientId=self.cid,
            )
            if len(contacts) == 0:
                print(f"[Demo] no contact with flap {flap_id}, skip grasp")
                # continue

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
            print(f"[Demo] grasped flap {flap_id}")
            
            pull_pos = [
                target_pos[0] + 0.08 * outward[0],
                target_pos[1] + 0.08 * outward[1],
                target_pos[2] + 0.08,
            ]

            # import ipdb; ipdb.set_trace()

            # q_pull = self.solve_ik(pull_pos, approach_orn)
            # self.execute_joint_trajectory(
            #     interpolate_joint_line(self.get_current_config(), q_pull, 600)
            # )
            self.move_to_pose(pull_pos, approach_orn, timeout=5.0)

            # import ipdb; ipdb.set_trace()

            p.removeConstraint(flap_constraint, physicsClientId=self.cid)

            retreat_pos = [
                target_pos[0] + 0.03 * outward[0],
                target_pos[1] + 0.03 * outward[1],
                target_pos[2] + approach_h + 0.03,
            ]
            q_retreat = self.solve_ik(retreat_pos, approach_orn)
            self.execute_joint_trajectory(
                interpolate_joint_line(self.get_current_config(), q_retreat, 45)
            )

            # import ipdb; ipdb.set_trace()

        # box.open_all()
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
