import numpy as np

from planners.grip_planner import PandaGripperPlanner
from utils.vector import quat_from_normal_and_yaw

class TaskConstraintPlanner:
    def __init__(self, robot_planner: PandaGripperPlanner):
        self.robot = robot_planner


    def sample_redundant(self, index, q_trajectory, q_reset_list, yaws, normal, horizontal, pos, current_config, q_source_trajectory=None, source_tag=None, finger_axis_is_plus_y=False):
        q_goal_list = []
        q_source_list = []
        current_q_reset_list = q_reset_list.copy()
        for reset_idx, q_reset in enumerate(current_q_reset_list):
            for yaw in yaws:
                orn = quat_from_normal_and_yaw(normal, yaw, horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)
                self.robot.set_robot_config(current_config)
                q_goal = self.robot.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, reset=False, q_reset=q_reset)
                self.robot.set_robot_config(current_config)
                if q_goal is not None:
                    q_goal_list.append((q_goal, yaw))
                    if q_source_trajectory is not None:
                        q_source_list.append({
                            "source_tag": source_tag,     # init or refine
                            "reset_idx": reset_idx,       # which one is it in q_reset_list
                            "q_reset": np.asarray(q_reset, dtype=float).tolist(),
                            "yaw": float(yaw),
                        })
        q_trajectory[index] += q_goal_list
        if q_source_trajectory is not None:
            q_source_trajectory[index] += q_source_list


    def qs_refinement(self, q1, q2, step_size:float=0.01):
        q_backup = self.get_current_config()

        # get N1
        self.robot.set_robot_config(q1)
        J = self.robot.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N1 = np.eye(len(self.robot.joint_indices)) - J_pinv @ J  # nullspace projector

        # get N2
        self.robot.set_robot_config(q2)
        J = self.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N2 = np.eye(len(self.robot.joint_indices)) - J_pinv @ J  # nullspace projector

        # target vector
        q_delta = np.asarray(q2)-np.asarray(q1)

        q1_delta = N1@q_delta
        q2_delta = N2@q_delta

        print(np.linalg.norm(q_delta), np.linalg.norm(q1_delta), np.linalg.norm(q2_delta))
        self.set_robot_config(q_backup)

        return (np.asarray(q1)+step_size*q1_delta).tolist(), (np.asarray(q2)-step_size*q2_delta).tolist()