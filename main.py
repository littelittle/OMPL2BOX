import math
import random
import time
from typing import List, Optional

import pybullet as p
import pybullet_data
from ompl import base as ob
from ompl import geometric as og


def interpolate_joint_line(q_from: List[float], q_to: List[float], steps: int):
    """关节空间直线插补，包含起点和终点。"""
    path = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        path.append([qf + alpha * (qt - qf) for qf, qt in zip(q_from, q_to)])
    return path


class KukaOmplPlanner:
    """
    用 OMPL 对 PyBullet 里的 KUKA 机械臂做关节空间 motion planning 的简单 demo。
    规划空间：R^n（各个关节角）
    碰撞检测：调用 PyBullet 的碰撞检测接口。
    """

    def __init__(self, use_gui: bool = True):
        self.cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        # 环境：地面 + 一个障碍物盒子
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.box_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.7, 0.0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=2.0,
            physicsClientId=self.cid,
        )

        # 机械臂：KUKA iiwa
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
        self.box_attached: bool = False
        self.ee_link_index: int = -1
        self.box_constraint_id: Optional[int] = None

        self._extract_active_joints()

        self.ndof = len(self.joint_indices)
        self.home_config = [0.0] * self.ndof

        # ---------- OMPL 部分 ----------
        # 关节空间 R^ndof
        self.space = ob.RealVectorStateSpace(self.ndof)

        # 关节上下界 -> RealVectorBounds
        bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            bounds.setLow(i, float(self.lower_limits[i]))
            bounds.setHigh(i, float(self.upper_limits[i]))
        self.space.setBounds(bounds)

        # SpaceInformation
        self.si = ob.SpaceInformation(self.space)

        # 定义 StateValidityChecker，调用 PyBullet 做碰撞检测
        planner_self = self

        class BulletValidityChecker(ob.StateValidityChecker):
            def __init__(self, si_):
                super().__init__(si_)
                self._planner = planner_self

            def isValid(self, state):
                return self._planner.is_state_valid(state)

        self.si.setStateValidityChecker(BulletValidityChecker(self.si))
        # 碰撞检测分辨率（越小越精细，计算越慢）
        self.si.setStateValidityCheckingResolution(0.01)
        self.si.setup()

    # ---------- PyBullet 辅助函数 ----------
    def _extract_active_joints(self):
        """从 URDF 中提取可动关节及其上下限。"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        for j in range(num_joints):
            ji = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            joint_type = ji[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_indices.append(j)
                ll = ji[8]
                ul = ji[9]
                # 有些 URDF 会给连续关节 [0, -1] 之类的奇怪范围，这里做个容错
                if ul < ll or (ll == 0 and ul == -1):
                    ll, ul = -3.14, 3.14
                self.lower_limits.append(ll)
                self.upper_limits.append(ul)
                self.rest_pose.append(0.0)
                self.ee_link_index = j

    def set_robot_config(self, q: List[float]):
        """把关节向量 q 写入 PyBullet 机器人。"""
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
        """读取当前关节角。"""
        states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.cid
        )
        return [s[0] for s in states]

    def is_state_valid(self, state) -> bool:
        """
        OMPL 的状态有效性检查函数：
        1. 把 state 映射成关节角
        2. 写入 PyBullet
        3. 查看是否与地面 / 障碍物盒子碰撞
        """
        q = [float(state[i]) for i in range(self.ndof)]
        self.set_robot_config(q)

        # 检查与环境的碰撞（地面 + 障碍物）
        # 这里只看 link vs plane / box 的接触，简化处理
        for link_index in self.joint_indices:
            # robot vs plane
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

            # robot vs box
            if not self.box_attached:
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

    # ---------- OMPL 规划 ----------
    def plan(self, q_start: List[float], q_goal: List[float], timeout: float = 3.0):
        """在关节空间中，从 q_start 规划到 q_goal。"""
        assert len(q_start) == self.ndof and len(q_goal) == self.ndof

        start = ob.State(self.space)
        goal = ob.State(self.space)
        for i in range(self.ndof):
            start[i] = float(q_start[i])
            goal[i] = float(q_goal[i])

        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start, goal)

        # 使用 RRTConnect 做一个快速 demo
        planner = og.RRTConnect(self.si)
        planner.setRange(0.2)  # 步长可以根据关节尺度调
        planner.setProblemDefinition(pdef)
        planner.setup()

        print("[OMPL] solving ...")
        solved = planner.solve(timeout)

        if not solved:
            print("[OMPL] no solution found")
            return None

        path = pdef.getSolutionPath()
        # 为了可视化更平滑一点
        path.interpolate(100)
        print("[OMPL] path length (in joint space):", path.length())
        return path

    def execute_path(self, path, dt: float = 1.0 / 240.0):
        """把规划出来的路径在 PyBullet 中播放出来。"""
        if path is None:
            return
        print("[PyBullet] executing trajectory with", path.getStateCount(), "waypoints")
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = [float(state[j]) for j in range(self.ndof)]
            self.set_robot_config(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(dt)

    def execute_joint_trajectory(self, qs: List[List[float]], dt: float = 1.0 / 240.0):
        """直接播放给定的关节轨迹。"""
        for q in qs:
            self.set_robot_config(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(dt)

    def solve_ik(self, pos, orn):
        """求解末端到达给定位姿的逆解。"""
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

    def attach_box(self):
        """用约束把盒子挂到末端，模拟抓取。"""
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

    def pick_and_place(self, box_pos, place_pos):
        """完整的抓取-搬运-放置流程。"""
        box_z = box_pos[2]
        place_z = place_pos[2]
        approach_h = 0.25

        # 更新盒子位置
        p.resetBasePositionAndOrientation(
            self.box_id,
            box_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.cid,
        )

        # 回到 home
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

        # 释放盒子并让它落到目标位置
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

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass


def main():
    planner = KukaOmplPlanner(use_gui=True)

    # 随机放一个盒子，然后把它搬到用户指定的地方
    box_xy_range = (0.3, 0.8)
    box_pos = [
        random.uniform(*box_xy_range),
        random.uniform(-0.35, 0.35),
        0.1,
    ]
    place_pos = [0.55, -0.4, 0.1]  # 可以根据需要改成任意可达点

    print("[Demo] box starts at", box_pos)
    print("[Demo] placing box to", place_pos)
    planner.pick_and_place(box_pos, place_pos)

    # print("Press Ctrl+C to quit the GUI window.")
    while True:
        p.stepSimulation(physicsClientId=planner.cid)
        time.sleep(1.0 / 20.0)


if __name__ == "__main__":
    main()
