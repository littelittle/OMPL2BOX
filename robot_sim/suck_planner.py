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
        self.box_attached: int = -2 # explain: -1 is the base, 0~3 are flaps, -2 means none
        self.ee_link_index: int = -1
        self.box_constraint_id: Optional[int] = None
        self.ee_contact_offset = [0.0, 0.0, 0.05] # the offset from end-effector link frame to suction cup tip

        self._extract_active_joints()

        self.ndof = len(self.joint_indices)
        self.home_config = [0.0] * self.ndof

        # ---------- Controller setup ----------
        self.control_dt = 1.0 / 400.0  
        self.segment_duration = 0.01    #  the time duration to move between two waypoints
        self.max_torque = 700.0        
        self.position_gain = 0.2       # Using PD position control
        self.velocity_gain = 1.0       

        # ---------- PyBullet setup ----------
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
                self.ee_link_index = 6 # the last active joint is the end-effector link

    def set_robot_config(self, q: List[float]):
        '''
        set the robot joint configuration directly to q in the simulation
        '''
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
                distance=0.002,
                linkIndexA=link_index,
                linkIndexB=-1,
                physicsClientId=self.cid,
            )
            if len(pts1) > 0:
                return False

        if not self.box_attached==-2:
            for link_index in self.joint_indices:
                for box_link in range(-1, 4):
                    if box_link == self.box_attached:
                        continue
                    pts2 = p.getClosestPoints(
                        bodyA=self.robot_id,
                        bodyB=self.box_id,
                        distance=0.00,
                        linkIndexA=link_index,
                        linkIndexB=-1,
                        physicsClientId=self.cid,
                    )
                    # print("hh")
                    if len(pts2) > 0:
                        # import ipdb; ipdb.set_trace()
                        return False

        return True

    # ---------- Controller ----------
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
    def plan(self, q_start: List[float], q_goal: List[float], timeout: float = 3.0, num_waypoints=1000):
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
        path.interpolate(num_waypoints)
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
        interpolate: bool = False,
        DRAW_DEBUG_LINES: bool = False,
    ):
        """用 position control 执行 OMPL 规划出来的关节轨迹。

        - path: OMPL 的 PathGeometric
        - dt:   每次 stepSimulation 的时间间隔（默认用 self.control_dt）
        - segment_duration: 每两个 waypoint 之间走多久（秒）
        - interpolate: 是否在两个 waypoint 之间做线性插值
        """
        # import time
        # start_time = time.time()
        if path is None:
            return

        if dt is None:
            dt = self.control_dt
        if segment_duration is None:
            segment_duration = self.segment_duration

        steps_per_segment = max(1, int(segment_duration / dt))
        num_states = path.getStateCount()

        print(
            "[PyBullet] executing trajectory with",
            path.getStateCount(),
            "waypoints,",
            steps_per_segment,
            "steps per segment",
        )

        if DRAW_DEBUG_LINES:
            q_backup = self.get_current_config()
            ee_positions = []

            for i in range(num_states):
                state = path.getState(i)
                q = [float(state[j]) for j in range(self.ndof)]

                self.set_robot_config(q)

                link_state = p.getLinkState(
                    self.robot_id,
                    self.ee_link_index,
                    physicsClientId=self.cid,
                )
                ee_pos = link_state[0]  # the world position of end-effector link
                ee_positions.append(ee_pos)

            self.set_robot_config(q_backup)

            for i in range(len(ee_positions) - 1):
                p.addUserDebugLine(
                    ee_positions[i],
                    ee_positions[i + 1],
                    [1, 0, 0],      # 红色
                    lineWidth=5,
                    lifeTime=100,
                    physicsClientId=self.cid,
                )

        q_curr = self.get_current_config()

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
        # time_elapsed = time.time() - start_time
        # print(f"[PyBullet] trajectory execution finished in {time_elapsed:.3f} seconds")
        # import ipdb; ipdb.set_trace()

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
    def _contact_to_ee_pose(self, contact_pos, contact_orn):
        """给定“接触点”的世界系位姿，算出末端连杆原点应该放哪。

        contact_pos, contact_orn : 接触点在世界系下的位姿
        self.ee_contact_offset   : 接触点在末端连杆坐标系下的位置（常量向量）
        """
        offset = getattr(self, "ee_contact_offset", None)
        if offset is None:
            return contact_pos, contact_orn

        # 偏移几乎为 0 就直接用原值，省一次运算
        if max(abs(o) for o in offset) < 1e-6:
            return contact_pos, contact_orn

        # 把末端坐标系中的 offset 旋到世界系
        offset_world, _ = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            contact_orn,
            offset,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )
        # 末端原点 = 接触点 - R * offset
        ee_pos = [contact_pos[i] - offset_world[i] for i in range(3)]
        return ee_pos, contact_orn

    def solve_ik_collision_aware(self, pos, orn, collision=True, max_trials=2000):
        import random
        base_rest = self.rest_pose[:]

        for t in range(max_trials):
            if t == 0:
                rest = base_rest
            else:
                # add some noise to the rest pose 
                rest = [
                    r + random.uniform(-1, 1) for r in base_rest
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
            )
            q_candidate = list(ik[: self.ndof])

            if self.is_state_valid(q_candidate) or not collision:
                if not collision:
                    print("[IK] found collision-free IK solution in trial", t)
                else:
                    print("[IK] found IK(not sure if collision free!) solution in trial", t)
                # import ipdb; ipdb.set_trace()
                return q_candidate
        print("[IK] failed to find collision-free IK solution after", max_trials, "trials")
        import ipdb; ipdb.set_trace()
        return None 

    def move_to_pose(self, pos, orn, timeout: float = 4.0, real=True, debug=False, num_waypoints=1000):
        q_start = self.get_current_config()
        q_goal = self.solve_ik_collision_aware(pos, orn)

        start_state = ob.State(self.space)
        goal_state = ob.State(self.space)

        for i in range(self.ndof):
            start_state[i] = float(q_start[i])
            goal_state[i] = float(q_goal[i])

        start_valid = self.is_state_valid(start_state)
        goal_valid = self.is_state_valid(goal_state)
        self.set_robot_config(q_start)

        if not start_valid:
            print("[IK] start configuration is in collision, abort plan")
            print("q_start:", q_start)
            self.set_robot_config(q_start)
            while True:
                p.stepSimulation()
            return None

        if not goal_valid:
            print("[IK] goal configuration collides with environment, abort plan")
            print("q_goal:", q_goal)
            self.set_robot_config(q_goal)
            while True:
                p.stepSimulation()
            return None

        path = self.plan(q_start, q_goal, timeout=timeout, num_waypoints=num_waypoints) 
        if path is not None:
            if debug:
                for i in range(path.getStateCount()):
                    state = path.getState(i)
                    q = [float(state[j]) for j in range(self.ndof)]
                    self.set_robot_config(q)
                    for _ in range(5):
                        p.stepSimulation(physicsClientId=self.cid)
                    # time.sleep(0.5)
            elif real:
                self.execute_path_real(path)
            else:
                self.execute_path(path)
        return path

    def get_ee_contact_offset(self, flap_id):
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.box_id,
            linkIndexA=self.ee_link_index,
            linkIndexB=flap_id,
            physicsClientId=self.cid,
        )
        if contacts:
            # positionOnA/worldspace
            if len(contacts) > 1:
                print(f"[WARN] multiple contact points found between ee link and flap {flap_id}, using the first one")
            contact_world = contacts[0][5]
        else:
            print(f"[WARN] no contact point found between ee link and flap {flap_id}, using default offset")
            return self.ee_contact_offset
        
        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            physicsClientId=self.cid,
        )
        ee_pos, ee_orn = ee_state[4], ee_state[5]

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)

        parentFramePosition, _ = p.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            contact_world,
            [0, 0, 0, 1],
        )

        return parentFramePosition

    # ---------- tasks ----------
    def open_flap_with_ompl(
        self,
        flap_id: int,
        target_angle_deg: float = 90.0,
        approach_dist: float = 0.00,
        timeout: float = 4.0,
    ):
        """
        利用几何 + OMPL 打开某个 flap 到指定角度。
          1) 通过几何接口算出 flap 在 angle=0 和 angle=target_angle 时关键点 & 法向；
          2) 先从当前姿态规划到“接近关键点”的姿态；
          3) 直接插值下压/贴合关键点（模拟吸附）；
          4) （可选）创建 P2P 约束把 flap 铰接到末端；
          5) 将 flap 视作已 attach, 在关节空间规划到目标关键点对应的末端 pose
          6) 执行轨迹，最后撤离一点距离。
        """

        # print("using open_flap_with_ompl")
        # while True:
        #     p.stepSimulation(physicsClientId=self.cid)
        box = self.foldable_box
        angle_rad = math.radians(target_angle_deg)

        # old_box_attached = self.box_attached
        # NOTE: when considering collision, we treat the flap as part of the robot after grasping(i.e. ignore the collision)!
        self.box_attached = flap_id

        # 1) 算出此时指定flap的关键点和法向：用作“接触点”
        key_start, normal_start, axis_start = box.get_flap_keypoint_pose(
            flap_id, p.getJointState(box.body_id, flap_id)[0], edge_ratio=0.9
        )

        # ---FOR DEBUGGING THE STARTING POINT---
        # for temp_flap in range(4):
        #     key_start, normal_start, axis_start = box.get_flap_keypoint_pose(
        #         temp_flap, -90, edge_ratio=0.9
        #     )
        #     draw_point(key_start, [temp_flap%2, temp_flap%3, temp_flap%4], size=0.1, life_time=500)
        # key_start, normal_start, axis_start = box.get_flap_keypoint_pose(
        #     flap_id, -90, edge_ratio=0.9)

        # 接近点：沿法向外侧退一点
        DOWNWARD = False  # Flip the end-effector 
        if DOWNWARD:
            approach_pos = [
                key_start[i] - normal_start[i] * approach_dist for i in range(3)
            ]
        else:
            approach_pos = [
                key_start[i] + normal_start[i] * approach_dist for i in range(3)
            ]
        contact_orn = quat_from_normal_and_axis(normal_start, axis_start, downward=DOWNWARD)

        # ---FOR DEBUGGING THE DESIRED POSITION---

        # approach_pos = [0.65, 0.29, 0.6]

        draw_point(approach_pos, [1, 0, 0], size=0.2, life_time=500)
        self.set_robot_config(self.solve_ik_collision_aware(approach_pos, contact_orn, collision=False))
        # import ipdb; ipdb.set_trace()
        # while True:
        #     p.stepSimulation(physicsClientId=self.cid)

        # 2) 规划到接近点
        path = self.move_to_pose(approach_pos, contact_orn, timeout=timeout, real=False, debug=True)
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id}")
            return
        
        # import ipdb; ipdb.set_trace()

        # 3) 插值下压到关键点（比再跑一次 OMPL 稳定）
        # q_contact = self.solve_ik_collision_aware(key_start, contact_orn, collision=False)
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

        # get the contact point offset to the end-effector link frame
        ee_contact_offset = self.get_ee_contact_offset(flap_id)
        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            physicsClientId=self.cid,
        )
        ee_pos, ee_orn = ee_state[4], ee_state[5]
        contact_pos = p.multiplyTransforms(
            ee_pos,
            ee_orn,
            ee_contact_offset,
            [0, 0, 0, 1],
        )[0]
        draw_point(contact_pos, [0, 1, 0], size=0.6, life_time=500)


        # import ipdb; ipdb.set_trace()

        # while True:
        #     p.stepSimulation(physicsClientId=self.cid)

        # 5) 准备目标姿态：目标角度下的关键点 & 法向
        key_goal, normal_goal, axis_goal = box.get_flap_keypoint_pose(
            flap_id, angle=-angle_rad
        )
        goal_orn = quat_from_normal_and_axis(normal_goal, axis_goal)

        if flap_id == 0:      # +x 边的 flap，中心在 -x 方向
            center_dir_local = [-1.0, 0.0, 0.0]
        elif flap_id == 1:    # -x 边的 flap，中心在 +x 方向
            center_dir_local = [1.0, 0.0, 0.0]
        elif flap_id == 2:    # +y 边
            center_dir_local = [0.0, -1.0, 0.0]
        else:                 # flap_id == 3, -y 边
            center_dir_local = [0.0, 1.0, 0.0]

        # 把这个方向变到世界系
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

        # Normalize
        norm = (center_dir_world[0]**2 + center_dir_world[1]**2 + center_dir_world[2]**2) ** 0.5
        center_dir_world = [c / norm for c in center_dir_world]

        # Let the end align with the box edge more
        center_pull_dist = 0.0

        key_goal_pulled = [
            key_goal[i] + center_dir_world[i] * center_pull_dist
            for i in range(3)
        ]

        # ---FOR DEBUG---
        # self.set_robot_config(self.solve_ik_collision_aware(key_goal_pulled, goal_orn, collision=True))
        # import ipdb; ipdb.set_trace()

        q_start = self.get_current_config()
        self.box_attached = -2

        q_goal = self.solve_ik_collision_aware(key_goal_pulled, goal_orn)

        path_open = self.plan(q_start, q_goal, timeout=timeout)
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
        # q_retreat = self.solve_ik_collision_aware(retreat_pos, goal_orn)
        # retreat_traj = interpolate_joint_line(
        #     self.get_current_config(), q_retreat, 45
        # )
        self.move_to_pose(retreat_pos, goal_orn, timeout=timeout, real=True, num_waypoints=200)
        # self.execute_joint_trajectory(retreat_traj)

        # import ipdb; ipdb.set_trace()

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass