import numpy as np
from functools import partial

from tasks import Task
from models import MailerBox
from planners import PandaGripperPlanner, TaskConstraintPlanner
from test_env import is_feasible, search_traj
from scene import create_pedestal
from utils.path import interpolate_joint_line
from utils.vector import WaypointConstraint



class MailerBoxTask(Task):
    def __init__(self, config, sim):
        super().__init__(config, sim)
        self.box_closed = self.config.get("box_closed", False)
        self.box_scaling = self.config.get("box_scaling", 1)
        self.method = self.config.get("method", "Iteration")


    def setup_scene(self, ):
        # Load mailerbox pose from self.config
        mailerbox_pos = list(self.config.get("box_pos", [0.6, 0.1, 0.4]))
        mailerbox_yaw = self.config.get('box_yaw', 0.0)

        # Create the pedestal
        create_pedestal(self.sim.cid, mailerbox_pos[:2], size_xy=(0.2, 0.2), height=mailerbox_pos[2]-0.05) # NOTE: -0.05 is only empirical

        # Set up the mailerbox
        file_path = self.config.get("box_file_path","assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf")
        self.mailerbox = MailerBox(self.sim.cid, file_path=file_path, scaling=self.box_scaling, pos=mailerbox_pos, yaw=mailerbox_yaw, closed=self.box_closed)
        box_id = self.mailerbox.body_id

        # Set up the robot 
        self.planner = PandaGripperPlanner(oracle_function=self.mailerbox.get_flap_keypoint_pose, cid=self.sim.cid, box_id=box_id, plane_id=self.sim.plane_id)
        self.tc_planner = TaskConstraintPlanner(robot_planner=self.planner)
        # planner.box_attached = 10 # # TODO: This is a remnant of ompl. Investigate whether this tag actually has any potential effect.


    def search_task_path(self):
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
        is_feasible_bound = partial(is_feasible, mailerbox=self.mailerbox, planner=self.planner, former_yaw=start_yaw, closed=self.box_closed) 
        degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)
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
        

    def _compute_plan_quality(self):
        degree_tuple_list = self.search_task_path()
        if degree_tuple_list is None:
            return {
                "success": False,
                "total_cost": None,
                "max_edge_cost": None,
                "planned_dict": None,
                "path": None,
            }

        constraints = self.build_constraint_sequence(degree_tuple_list)
        return self.tc_planner.solve_constraint_path(constraints, self.method)


    def execute_plan(self, path):
        candidate_yaw_trajectory = []

        grasp_q_goal = path[0][0]
        ompl_path = self.planner.plan_ompl(
            self.planner.get_current_config(),
            grasp_q_goal,
            num_waypoints=200,
            optimal=False,
        )
        if ompl_path is None:
            raise RuntimeError("OMPL failded to plan a feasible trajectory to the grasp pose!")

        q_traj = []
        for i in range(ompl_path.getStateCount()):
            s = ompl_path.getState(i)
            q_traj.append([float(s[j]) for j in range(self.planner.ndof)])

        self.planner.execute_joint_trajectory_real(q_traj, dt=0.05, interpolate=False)
        self.planner.set_robot_config(grasp_q_goal)
        self.planner.close_gripper_to_width(target_width=0.0, force=1000)

        for q_goal, yaw in path:
            candidate_yaw_trajectory.append(yaw)
            q_start = self.planner.get_current_config()
            traj = interpolate_joint_line(q_start, q_goal, 45)
            self.planner.execute_joint_trajectory_real(traj, N_ref=75)
            self.planner.close_gripper_to_width(target_width=0, force=1000, wait=0.5)

        return candidate_yaw_trajectory


    def run(self, execute=True):
        degree_tuple_list = self.search_task_path()
        constraints = self.build_constraint_sequence(degree_tuple_list)
        plan_metrics = self.tc_planner.solve_constraint_path(constraints, self.method)
        if not plan_metrics["success"]:
            print("Debug Info:")
            print(plan_metrics)
            raise RuntimeError("Failed to compute a feasible MailerBoxTask plan.")

        path = plan_metrics["path"]
        candidate_yaw_trajectory = []

        if execute:
            candidate_yaw_trajectory = self.execute_plan(path)


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
