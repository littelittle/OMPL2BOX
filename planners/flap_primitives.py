import math
import time
import pybullet as p
from typing import Optional, Literal, List, Dict

from utils.path import draw_point, interpolate_joint_line
from utils.contactframe import ContactFrame
from utils.vector import quat_from_normal_and_axis
from .grip_planner import PandaGripperPlanner


class FlapManipulationPrimitives:

    def __init__(self, robot: PandaGripperPlanner, oracle_function, box_id):
        self.robot = robot
        self.oracle_function = oracle_function
        self.box_id = box_id

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

        lf = getattr(self.robot, "left_finger_link_index", None)
        rf = getattr(self.robot, "right_finger_link_index", None)
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
                bodyA=self.robot.robot_id,
                bodyB=self.box_id,
                linkIndexA=finger_link,
                linkIndexB=flap_id,
                physicsClientId=self.robot.cid,
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
                    bodyA=self.robot.robot_id,
                    bodyB=self.box_id,
                    distance=close_tol,
                    linkIndexA=finger_link,
                    linkIndexB=flap_id,
                    physicsClientId=self.robot.cid,
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

    def close_gripper(self, force: float = 100.0, wait: float = 10.0, flap_id=None, min_normal_force=20.0): 
        # TODO: figure out how the coefficient of friction affect the behaviour of the contact
        # p.changeDynamics(self.robot_id, self.gripper_joint_indices[0],  lateralFriction=1.0, spinningFriction=0.05, rollingFriction=0.05)
        # p.changeDynamics(self.robot_id, self.gripper_joint_indices[1], lateralFriction=2.0, spinningFriction=0.05, rollingFriction=0.05)
        # if flap_id:
        #     p.changeDynamics(self.box_id, flap_id, lateralFriction=1.5, spinningFriction=0.02, rollingFriction=0.02)

        for j in self.robot.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=-abs(0.02), 
                velocityGain=1.0,
                force=force,
                physicsClientId=self.robot.cid,
            )
        steps = int(wait / self.robot.control_dt) if hasattr(self.robot, "control_dt") else 60
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.robot.cid)
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
                    # import ipdb; ipdb.set_trace()
                    break
        ok = self.check_grasping_flap(
            flap_id,
            require_both_fingers=True,
            min_normal_force=0,
            return_info=True,
        )
        # import ipdb; ipdb.set_trace()
        print("close gripper")
        print(ok)

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

        traj, vamp_env = self.robot.move_to_pose_unified(
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
        self.robot.open_gripper()
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
                self.robot.box_attached = 4
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

                self.robot.box_attached = flap_id
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
            traj, vamp_env = self.robot.move_to_pose_unified(
                goal_pos,
                goal_orn,
                planner=PL,
                timeout=timeout,
                ik_collision=True,
                execute=True,
                vamp_env=vamp_env,
                rebuild_vamp_env=False,
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
        q_goal = self.robot.solve_ik_collision_aware(retreat_pos, orn, collision=collision)
        if q_goal is None:
            return False, (retreat_pos, orn)

        q_start = self.robot.get_current_config()
        traj = interpolate_joint_line(q_start, q_goal, int(steps))
        self.robot.execute_joint_trajectory_real(traj, segment_duration=segment_duration)
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

            q_goal = self.robot.solve_ik_collision_aware(goal_pos, goal_orn, collision=False)
            if q_goal is None:
                continue

            q_start = self.robot.get_current_config()
            traj = interpolate_joint_line(q_start, q_goal, int(interp_steps))
            self.robot.execute_joint_trajectory_real(traj, segment_duration=segment_duration)

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
        self.robot.open_gripper()
        key_start, normal_start, axis_start, extended_start, angle = self.oracle_function(flap_id)
        approach_pos = [
            key_start[i] + extended_start[i] * approach_dist for i in range(3)
        ]
        contact_orn = quat_from_normal_and_axis(extended_start, axis_start)
        draw_point(approach_pos, [1, 0, 0], size=0.2, life_time=500)  

        path, _ = self.robot.move_to_pose_unified(approach_pos, contact_orn, planner=PL)
        
        if path is None:
            print(f"[Flap] failed to approach flap {flap_id} using {PL}")
            return
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.robot.cid)
            time.sleep(1.0 / 240.0)

    def close_flap(
        self,
        flap_id: int,
        target_angle_deg: float = 100.0,
        approach_dist: float = 0.0,
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
        self.robot.box_attached = 4  # 抓取前所有 flap 都算碰撞
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
        self.robot.box_attached = flap_id  # 抓住后忽略该 flap 的碰撞

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
        self.robot.open_gripper()
        self.robot.box_attached = 4 # 所有 flap 都算碰撞

        # if last_frame is not None and last_pose is not None:
        #     last_pos, last_orn = last_pose
        #     # 沿 extended 方向退一点
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
        frame = self._oracle_frame(flap_id)
        goal_pos = [frame.key[i] - frame.normal[i] * float(0.14) for i in range(3)]
        goal_orn = quat_from_normal_and_axis([-x for x in frame.normal], frame.axis)
        self.robot.move_to_pose_unified(goal_pos, goal_orn, planner="VAMP")

        current_deg = abs(math.degrees(self.oracle_function(flap_id)[-1]))
        start_deg = int(current_deg)

        last_frame2, last_pose2 = self.prim_press_stab_sequence(
            flap_id,
            start_deg=start_deg,
            end_deg=170,
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

        self.robot.box_attached = 4
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
        q_backup = self.robot.get_current_config()
        self.robot.set_robot_config(self.robot.home_config)
        link_state = p.getLinkState(self.robot.robot_id, self.robot.ee_link_index, physicsClientId=self.robot.cid)
        home_pos, home_orn = link_state[0], link_state[1]
        self.robot.set_robot_config(q_backup)
        return self.robot.move_to_pose_unified(
            home_pos,
            home_orn,
            planner=PL,
            timeout=timeout,
            ik_collision=ik_collision,
            execute=execute,
            vamp_env=vamp_env,
            rebuild_vamp_env=rebuild_vamp_env,
        )