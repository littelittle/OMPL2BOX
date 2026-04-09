import time
import numpy as np
from typing import List, Literal 

from planners.grip_planner import PandaGripperPlanner
from planners.constraint_sequence_rrt_planner import ConstraintSequenceRRTPlanner
from utils.vector import quat_from_normal_and_yaw, WaypointConstraint
from utils.yaw_dp import dp_plan_yaw_path, Q_RESET_SEEDS, uniform_q_sampling


class TaskConstraintPlanner:
    def __init__(self, robot_planner: PandaGripperPlanner):
        self.robot = robot_planner


    def _sample_redundant(self, constraint:WaypointConstraint, source_tag, finger_axis_is_plus_y=False):
        q_goal_list = []
        q_source_list = []
        for reset_idx, q_reset in enumerate(self.q_reset_list):
            for yaw in self.yaws:
                orn = quat_from_normal_and_yaw(constraint.normal, yaw, constraint.horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)  
                self.robot.set_robot_config(self.current_config)
                q_goal = self.robot.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=False, q_reset=q_reset)
                self.robot.set_robot_config(self.current_config)
                if q_goal is not None:
                    q_goal_list.append((q_goal, yaw))
                    q_source_list.append({
                        "source_tag": source_tag,     # init or refine
                        "reset_idx": reset_idx,       # which one is it in q_reset_list
                        "q_reset": np.asarray(q_reset, dtype=float).tolist(),
                        "yaw": float(yaw),
                    })
        self.q_trajectory[constraint.task_step] += q_goal_list
        self.q_source_trajectory[constraint.task_step] += q_source_list


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
        self.robot.set_robot_config(q_backup)

        return (np.asarray(q1)+step_size*q1_delta).tolist(), (np.asarray(q2)-step_size*q2_delta).tolist()


    def get_yaw_candidates(self, mid_degree=90, num_steps:int=5, max_offset=np.deg2rad(60)):

        step = max_offset / float(max(1, num_steps))
        yaws = [np.deg2rad(mid_degree)]
        for k in range(1, num_steps + 1):
            offset = k * step
            yaws.append(yaws[0] + offset)
            yaws.append(yaws[0] - offset)
        return yaws


    def solve_constraint_path(self, constraints: List[WaypointConstraint], method: Literal["Sampling", "Iteration", "RRT"], ):
        if method == "RRT":
            planner = ConstraintSequenceRRTPlanner(
                self.robot,
                joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]),
            )

            start_time = time.time()
            solution = planner.solve(constraints)
            total_time = time.time()-start_time
            print(f"Total time: {total_time}")
            print(f"Max edge cost is {solution['max_edge_cost']}, Total cost is {solution['total_cost']}")
            return solution

        # First, open the gripper
        self.robot.open_gripper()
        
        # Initialize the q_trajectory
        self.q_trajectory = [[] for _ in range(len(constraints))]
        self.q_source_trajectory = [[] for _ in range(len(constraints))]

        # Initialize the yaw candidates list
        self.yaws = self.get_yaw_candidates()

        # Initialize the q_reset fed into the IK solver
        if method == "Iteration": 
            self.q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                Q_RESET_SEEDS["left_elbow_out"],
                Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            MaxIteration = 50

        elif method == "Sampling":
            self.q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                Q_RESET_SEEDS["left_elbow_out"],
                Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            self.q_reset_list += uniform_q_sampling(20)
            MaxIteration = 0
 
        start_time = time.time()
        self.current_config = self.robot.get_current_config()
        for constraint in constraints:
            self._sample_redundant(constraint, source_tag={"kind":"init","step":constraint.task_step})
            if len(self.q_trajectory[constraint.task_step]) == 0:
                q_goal = None
                raise RuntimeError(f"constraint{constraint.task_step} has no feasible q at all!")
        planned_dict = dp_plan_yaw_path(feasible_by_step=self.q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))
        if planned_dict is None:
            return {
                "success": False,
                "total_cost": None,
                "max_edge_cost": None,
                "planned_dict": None,
                "path": None,
            }
        
        # Evaluation&Refinemnt!
        sorted_idx = 0
        for j in range(MaxIteration):
            path = planned_dict['path']
            selected_index, selected_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[sorted_idx]
            max_index, max_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[0]
            
            # Add some randomness ...
            sorted_idx += 1
            if sorted_idx > 4:
                sorted_idx = 0

            new_q_rest_list = [path[selected_index][0].tolist(), path[selected_index+1][0].tolist()]
            self.q_reset_list = new_q_rest_list
            print(f"Iter times: {j}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}, selected_index: {selected_index}")
            if max_edge_cost < 0.6: # set this threashold accordingly
                break

            # Refining...
            refine_yaw_offset = np.deg2rad(30)

            for i, constraint in enumerate(constraints):
                if abs(i-selected_index) < 15:
                    old_yaw = path[i][1]
                    low_bound = max(np.deg2rad(30), old_yaw - refine_yaw_offset)
                    high_bound = min(np.deg2rad(150), old_yaw + refine_yaw_offset)
                    self.yaws = np.linspace(low_bound, high_bound, 20).tolist()
                    self._sample_redundant(constraint, source_tag={"kind":"refine", "iter":j+1, "from_edge":max_index})

            planned_dict = dp_plan_yaw_path(feasible_by_step=self.q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))

        total_time = time.time()-start_time
        print(f"Total time: {total_time}")
        path = planned_dict['path'] 
        path_sources = [
            self.q_source_trajectory[step][cand_idx]
            for step, cand_idx in enumerate(planned_dict["indices"])
        ]
        max_index, max_edge_cost = max(enumerate(planned_dict["path_costs"]), key=lambda x: x[1])
        print(f"Iter times: {MaxIteration}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}")
        return {
            "success": True,
            "total_cost": float(planned_dict["total_cost"]),
            "max_edge_cost": float(max_edge_cost),
            "planned_dict": planned_dict,
            "path": path,
        }

            
            



        
 
