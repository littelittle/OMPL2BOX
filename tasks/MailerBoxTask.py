import csv
import numpy as np
from functools import partial
import pybullet as p
import time
import sys
from pathlib import Path

from tasks import Task
from models import MailerBox
from planners import PandaGripperPlanner, TaskConstraintPlanner, TaskConstraintPlannerNew
from test_env import is_feasible
from planners.lid_flap_planner import search_traj, search_traj_cache
from scene import create_pedestal
from utils.path import interpolate_joint_line
from utils.vector import WaypointConstraint, quat_from_normal_and_yaw
from utils.loader import load_path
from perception.data_generator import get_estimation



class MailerBoxTask(Task):
    def __init__(self, config, sim):
        super().__init__(config, sim)
        self.box_closed = self.config.get("box_closed", False)
        self.box_scaling = self.config.get("box_scaling", 1)
        self.method = self.config.get("method", "Iteration")

    def setup_scene(self, load_panda:bool=True):
        # Load mailerbox pose from self.config
        mailerbox_pos = list(self.config.get("box_pos", [0.6, 0.1, 0.4]))
        mailerbox_yaw = self.config.get('box_yaw', 0.0)

        # Create the pedestal
        self.pedestal_id = create_pedestal(self.sim.cid, mailerbox_pos[:2], size_xy=(0.2*self.box_scaling, 0.2*self.box_scaling), height=mailerbox_pos[2]-0.05) # NOTE: -0.05 is only empirical

        # Set up the mailerbox
        file_path = self.config.get("box_file_path","assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf")
        self.mailerbox = MailerBox(self.sim.cid, file_path=file_path, scaling=self.box_scaling, pos=mailerbox_pos, yaw=mailerbox_yaw, closed=self.box_closed)
        box_id = self.mailerbox.body_id

        label = get_estimation(p, self)
        self.mailerbox._save_estimation(label)

        # Set up the robot
        if load_panda: 
            self.planner = PandaGripperPlanner(oracle_function=self.mailerbox.get_flap_keypoint_pose, cid=self.sim.cid, box_id=box_id, plane_id=self.sim.plane_id)
            tc_planner_impl = self.config.get("task_constraint_planner", "new")
            if tc_planner_impl == "old":
                self.tc_planner = TaskConstraintPlanner(robot_planner=self.planner)
            else:
                self.tc_planner = TaskConstraintPlannerNew(robot_planner=self.planner)
            # planner.box_attached = 10 # # TODO: This is a remnant of ompl. Investigate whether this tag actually has any potential effect.

    def search_task_path(self, resolution:int=10):
        # Configure the order of the array
        if self.box_closed:
            start_angle_tuple = (90, 90)
            goal_angle_tuple = (-90, -90)
        else:
            start_angle_tuple = (-90, -90)
            goal_angle_tuple = (90, 90)
        start_yaw = np.deg2rad(90)

        # Searching for potential feasible task space!
        # TODO: move is_feasible and search_traj to this file(maybe class)
        is_feasible_bound = partial(is_feasible, mailerbox=self.mailerbox, planner=self.planner, former_yaw=np.deg2rad(90), closed=self.box_closed) 
        # degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)
        # print(f"degree_tuple_list: {degree_tuple_list}")
        degree_tuple_list, q_list = search_traj_cache(start_angle_tuple, goal_angle_tuple, is_feasible_bound, resolution=resolution)
        print(f"degree_tuple_list: {degree_tuple_list}")
        # import ipdb; ipdb.set_trace()
        degree_tuple_list = [start_angle_tuple] + degree_tuple_list
        # gen_2D_map((-90, -90), (90, 90), is_feasible_bound)
        return degree_tuple_list
    
    def build_constraint_sequence(self, degree_tuple_list):
        constraints = []
        for i, degree_tuple in enumerate(degree_tuple_list):
            # Searching for feasible configuration!
            pos, normal, horizontal = self.mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
            constraints.append(
                WaypointConstraint(pos, normal, horizontal, i)
            )
        
        return constraints

    def _compute_plan_quality(self, coarse2fine_ratio:int=5):
        degree_tuple_list = self.search_task_path(resolution=10)
        if degree_tuple_list is None:
            return {
                "success": True,
                "total_cost": None,
                "max_edge_cost": None,
                "planned_dict": None,
                "path": None,
            }

        constraints = self.build_constraint_sequence(degree_tuple_list)
        metric = self.tc_planner.solve_constraint_path(constraints, self.method)
        _, max_edge_cost = self.get_traj_coarse2fine(metric['path'], degree_tuple_list, coarse2fine_ratio)
        metric['max_edge_cost'] = max_edge_cost

        return metric

    def execute_plan(self, path, interpolate_ratio:int=45):
        candidate_yaw_trajectory = []

        # grasp_q_goal = path[0][0]
        # ompl_path = self.planner.plan_ompl(
        #     self.planner.get_current_config(),
        #     grasp_q_goal,
        #     num_waypoints=200,
        #     optimal=False,
        # )
        # if ompl_path is None:
        #     raise RuntimeError("OMPL failded to plan a feasible trajectory to the grasp pose!")
        self.planner.open_gripper()
        self.planner.set_robot_config(path[0][0])
        self.planner.close_gripper_to_width(0)

        # q_traj = []
        # for i in range(ompl_path.getStateCount()):
        #     s = ompl_path.getState(i)
        #     q_traj.append([float(s[j]) for j in range(self.planner.ndof)])

        # self.planner.execute_joint_trajectory_real(q_traj, dt=0.05, interpolate=False)
        # self.planner.set_robot_config(grasp_q_goal)
        # self.planner.close_gripper_to_width(target_width=0.0, force=1000)

        for q_goal, yaw in path:
            candidate_yaw_trajectory.append(yaw)
            q_start = self.planner.get_current_config()
            traj = interpolate_joint_line(q_start, q_goal, interpolate_ratio)
            self.planner.execute_joint_trajectory_real(traj, N_ref=75)
            self.planner.close_gripper_to_width(target_width=0, force=1000, wait=0.5)

        return candidate_yaw_trajectory

    def execute_plan_coarse2fine_old(self, path, degree_tuple_list, fine_ratio:int=5):
        for i, (q_goal, yaw_goal) in enumerate(path[1:]):
            q_start = self.planner.get_current_config()
            yaw_start = path[i][1]
            q_seed_list = interpolate_joint_line(q_start, q_goal, fine_ratio)
            yaw_list = [yaw_start + alpha/(fine_ratio)*(yaw_goal-yaw_start) for alpha in range(1, fine_ratio+1)]
            degree_tuple_start, degree_tuple_goal = degree_tuple_list[i], degree_tuple_list[i+1]
            degree_tuple_slices = interpolate_joint_line(degree_tuple_start, degree_tuple_goal, fine_ratio)
            constraints_slices = self.build_constraint_sequence(degree_tuple_slices)
            for j, constraint in enumerate(constraints_slices):
                orn = quat_from_normal_and_yaw(constraint.normal, yaw_list[j], constraint.horizontal, finger_axis_is_plus_y=False)
                self.planner.set_robot_config(q_start)
                q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_seed_list[j])
                # q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_start)
                if q_goal is None:
                    # import ipdb; ipdb.set_trace()
                    q_goal = q_seed_list[j]
                self.planner.set_robot_config(q_start)
                traj = interpolate_joint_line(q_start, q_goal, 9)
                self.planner.execute_joint_trajectory_real(traj, N_ref=75)
                self.planner.close_gripper_to_width(target_width=0, force=1000, wait=0.1)
                q_start = self.planner.get_current_config()

    def execute_plan_coarse2fine(self, path, degree_tuple_list, fine_ratio:int=5):
        constraints_slices = []
        q_seed_list = []
        yaw_list = []
        full_traj = []
        for i, (q_goal, yaw_goal) in enumerate(path[1:]):
            # q_start = self.planner.get_current_config()
            q_start = path[i][0]
            yaw_start = path[i][1]
            q_seed_list += interpolate_joint_line(q_start, q_goal, fine_ratio)
            yaw_list += [yaw_start + alpha/(fine_ratio)*(yaw_goal-yaw_start) for alpha in range(0, fine_ratio)]
            degree_tuple_start, degree_tuple_goal = degree_tuple_list[i], degree_tuple_list[i+1]
            degree_tuple_slices = interpolate_joint_line(degree_tuple_start, degree_tuple_goal, fine_ratio)
            if i == len(path)-2: # Which means this is the end of the constraint list
                q_seed_list.append(q_goal)
                yaw_list.append(yaw_goal)
                degree_tuple_slices.append(degree_tuple_goal)
            constraints_slices += self.build_constraint_sequence(degree_tuple_slices)
        # import ipdb; ipdb.set_trace()
        q_start = self.planner.get_current_config()
        for j, constraint in enumerate(constraints_slices):
            orn = quat_from_normal_and_yaw(constraint.normal, yaw_list[j], constraint.horizontal, finger_axis_is_plus_y=False)
            self.planner.set_robot_config(q_start)
            q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_seed_list[j])
            if q_goal is None:
                # import ipdb; ipdb.set_trace()
                q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_start)
                if q_goal is None:
                    q_goal = q_seed_list[j+1]
            self.planner.set_robot_config(q_start)
            if j != 0:
                # if j == 40:
                #     import ipdb; ipdb.set_trace()
                traj = interpolate_joint_line(q_start, q_goal, 9)
                self.planner.execute_joint_trajectory_real(traj, N_ref=100//fine_ratio)
                full_traj += traj
                # self.planner.set_robot_config(q_goal)
                # for i in range(5):
                #     p.stepSimulation()
                #     time.sleep(0.1)
                # self.planner.close_gripper_to_width(target_width=0, force=1000, wait=0.1)
                # print(j)
            else: # First moving to the starting point using motion planning!
                grasp_q_goal = q_goal
                # ompl_path = self.planner.plan_ompl(
                #     self.planner.get_current_config(),
                #     grasp_q_goal,
                #     num_waypoints=200,
                #     optimal=False,
                # )
                # if ompl_path is None:
                #     raise RuntimeError("OMPL failded to plan a feasible trajectory to the grasp pose!")

                # q_traj = []
                # for i in range(ompl_path.getStateCount()):
                #     s = ompl_path.getState(i)
                #     q_traj.append([float(s[j]) for j in range(self.planner.ndof)])

                # self.planner.execute_joint_trajectory_real(q_traj, dt=0.05, interpolate=False)
                self.planner.set_robot_config(grasp_q_goal)
                self.planner.close_gripper_to_width(target_width=0.0, force=1000)
            q_start = self.planner.get_current_config()
    
    def get_traj_coarse2fine(self, path, degree_tuple_list, fine_ratio:int =5, output_path:str=None, dt:float=1.0/2, WRITE:bool=False):
        constraints_slices = []
        q_seed_list = []
        yaw_list = []
        full_traj = []
        for i, (q_goal, yaw_goal) in enumerate(path[1:]):
            # q_start = self.planner.get_current_config()
            q_start = path[i][0]
            yaw_start = path[i][1]
            q_seed_list += interpolate_joint_line(q_start, q_goal, fine_ratio)
            yaw_list += [yaw_start + alpha/(fine_ratio)*(yaw_goal-yaw_start) for alpha in range(0, fine_ratio)]
            degree_tuple_start, degree_tuple_goal = degree_tuple_list[i], degree_tuple_list[i+1]
            degree_tuple_slices = interpolate_joint_line(degree_tuple_start, degree_tuple_goal, fine_ratio)
            if i == len(path)-2: # Which means this is the end of the constraint list
                q_seed_list.append(q_goal)
                yaw_list.append(yaw_goal)
                degree_tuple_slices.append(degree_tuple_goal)
            constraints_slices += self.build_constraint_sequence(degree_tuple_slices)
        # import ipdb; ipdb.set_trace()
        q_start = None
        max_q_diff = 0
        for j, constraint in enumerate(constraints_slices):
            orn = quat_from_normal_and_yaw(constraint.normal, yaw_list[j], constraint.horizontal, finger_axis_is_plus_y=False)
            q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_seed_list[j])
            if q_goal is None:
                q_goal = self.planner.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_start)
                if q_goal is None:
                    q_goal = q_seed_list[j]
            if q_start is not None:
                max_q_diff = max(max_q_diff, np.linalg.norm(np.asarray(q_start)-np.asarray(q_goal)))
            q_start = q_goal
            # Save the ee pose as [x, y, z, qx, qy, qz, qw].
            full_traj.append(list(constraint.pos) + list(orn))

        print(f"max_q_diff is {max_q_diff}")

        if WRITE:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "time",
                "target-POS_X",
                "target-POS_Y",
                "target-POS_Z",
                "target-ROT_X",
                "target-ROT_Y",
                "target-ROT_Z",
                "target-ROT_W",
            ]
            rows = []
            for i, pose in enumerate(full_traj):
                rows.append(
                    {
                        "time": i * dt,
                        "target-POS_X": pose[0],
                        "target-POS_Y": pose[1],
                        "target-POS_Z": pose[2],
                        "target-ROT_X": pose[3],
                        "target-ROT_Y": pose[4],
                        "target-ROT_Z": pose[5],
                        "target-ROT_W": pose[6],
                    }
                )

            with output_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"Saved full trajectory CSV to: {output_path}")

        return output_path, max_q_diff

    def run(self, execute=True):
        # First, find the feasible degree_tuple_list
        degree_tuple_list = self.search_task_path(10)
        # Second, generate the constraint
        constraints = self.build_constraint_sequence(degree_tuple_list)
        # Third, open the panda ee gripper
        self.planner.open_gripper()
        # Fourth, plan the feasible q trajectory 
        plan_metrics = self.tc_planner.solve_constraint_path(constraints, self.method)
        if not plan_metrics["success"]:
            print("Debug Info:")
            print(plan_metrics)
            raise RuntimeError("Failed to compute a feasible MailerBoxTask plan.")

        path = plan_metrics["path"]
        # TEST_ANYTIME = True
        # if TEST_ANYTIME:
        #     path = load_path('/home/littletree/playground/IKLink/output_motions/panda_ee_pose4+iklink_anytime4+764.csv')
        candidate_yaw_trajectory = []

        if execute:
            candidate_yaw_trajectory = self.execute_plan_coarse2fine(path, degree_tuple_list, fine_ratio=5)
            # candidate_yaw_trajectory = self.execute_plan(path)
            # self.get_traj_coarse2fine(path, degree_tuple_list, fine_ratio=5, output_path=f"data/waypoints/{self.box_closed}closed_{self.box_scaling}_coarse2fine_traj.csv", WRITE=True)
 


        # Visualization for C-bundles...
        # plot_feasible_yaw_evolution_greedy(
        #     q_trajectory,
        #     chosen_yaw_trajectory=candidate_yaw_trajectory,
        #     save_path=f"exp/03_28/{self.box_closed}closed_{self.scaling}_iter_mix_rand.png",
        #     show=True,
        #     use_degree=True,
        #     angular_indices=range(7),   # Panda arm joints
        #     one_to_one=False,           
        # )
        
        # plot_threshold_3d_with_init_layer(
        #     q_trajectory=q_trajectory,
        #     q_source_trajectory=q_source_trajectory,
        #     planned_dict=planned_dict,
        #     distance_threshold=0.9,
        #     save_path=f"exp/NEW/{self.box_closed}closed_{self.scaling}_threshold_3d_with_init.png",
        #     show=True,
        #     use_degree=True,
        #     z_mode="refine_iter",
        #     draw_failed_exec_edges=False,
        #     annotate_exec_dist=False,
        # )

        # plot_threshold_3d_with_layer_views(
        #     q_trajectory=q_trajectory,
        #     q_source_trajectory=q_source_trajectory,
        #     planned_dict=planned_dict,
        #     distance_threshold=1.11,
        #     focus_refine_iter="last",   # 或者 0, 1, 2 ...
        #     save_path=f"exp/paper/{self.box_closed}closed_{self.scaling}_3d_with_layer_views_max_classic.png",
        #     show=True,
        #     use_degree=True,
        #     draw_failed_exec_edges=False,
        #     annotate_exec_dist=False,
        # )

        # Ending...
        if self.box_closed==False:
            print(f"[INFO] The box has been closed!")
        else:
            print(f"[INFO] The box has been opened!")
     
    def compute_quality_metrics(self):
        return self._compute_plan_quality()
