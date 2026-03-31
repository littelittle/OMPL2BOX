import math
import random
import numpy as np
import pybullet as p
from functools import partial

from tasks import Task
from models import MailerBox
from planners import PandaGripperPlanner
from test_env import is_feasible, search_traj
from scene import create_pedestal
from utils.yaw_dp import dp_plan_yaw_path, Q_RESET_SEEDS, uniform_q_sampling
from utils.path import interpolate_joint_line
from utils.drawer import plot_feasible_yaw_evolution_greedy, plot_threshold_3d_with_init_layer, plot_threshold_3d_with_layer_views



class MailerBoxTask(Task):
    def setup_scene(self, ):
        mailerbox_pos = np.array(self.config.get("box_pos", [0.6, 0.1, 0.4]), dtype=float)
        mailerbox_pos = mailerbox_pos.tolist() 
        mailerbox_yaw = self.config.get('box_yaw', 0.0)
        self.closed = self.config.get("closed", False)
        self.scaling = self.config.get("scaling", 1)
        self.method = self.config.get("method", "Iteration")

        # create the pedestal
        create_pedestal(self.sim.cid, mailerbox_pos[:2], size_xy=(0.2, 0.2), height=mailerbox_pos[2]-0.05)

        # set up the mailerbox
        file_path = self.config.get("file_path","assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf")
        self.mailerbox = MailerBox(self.sim.cid, file_path=file_path, scaling=self.scaling, pos=mailerbox_pos, yaw=mailerbox_yaw, closed=self.closed)
        box_id = self.mailerbox.body_id

        # set up the robot 
        self.planner = PandaGripperPlanner(oracle_function=self.mailerbox.get_flap_keypoint_pose, cid=self.sim.cid, box_id=box_id, plane_id=self.sim.plane_id)
        # planner.box_attached = 10 # # TODO: This is a remnant of ompl. Investigate whether this tag actually has any effect.

    def run(self, ):
        planner = self.planner
        mailerbox = self.mailerbox
        candidate_yaw_trajectory = []

        # Reaching out to the flap grasp point!
        # NOTE: We'll skip this process for now and select the optimal grasping pose after the planning is complete.
        planner.open_gripper()

        # Configure the order of the array
        if self.closed:
            start_angle_tuple = (90, 90)
            goal_angle_tuple = (-90, -90)
        else:
            start_angle_tuple = (-90, -90)
            goal_angle_tuple = (90, 90)

        # Searching for potential fesible task space!
        start_yaw = np.deg2rad(90)
        is_feasible_bound = partial(is_feasible, mailerbox=mailerbox, planner=planner, former_yaw=start_yaw, closed=self.closed) 
        degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)

        # TODO: Optimize the way q_trajectory and q_source_trajectory is recorded
        q_trajectory = [[] for _ in range(len(degree_tuple_list))]
        q_source_trajectory = [[] for _ in range(len(degree_tuple_list))]

        # Initialize the yaw candidates list
        num_steps = 10
        max_offset = np.deg2rad(60)
        step = max_offset / float(max(1, num_steps))
        yaws = [np.deg2rad(90)] 
        for k in range(1, num_steps + 1):
            offset = k * step
            yaws.append(yaws[0] + offset)
            yaws.append(yaws[0] - offset)

        # Initialize the q_reset fed in to the IK solver
        if self.method == "Iteration": 
            q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                Q_RESET_SEEDS["left_elbow_out"],
                Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            MaxIteration = 5                           # tweak this param by mode

        elif self.method == "Sampling":
            q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                Q_RESET_SEEDS["left_elbow_out"],
                Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            q_reset_list += uniform_q_sampling(5)
            MaxIteration = 0


        # Following the task space constraint to execute!
        for i, degree_tuple in enumerate(degree_tuple_list):

            # Searching for feasible configuration!
            pos, normal, horizontal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
            current_config = planner.get_current_config()

            # TODO: Optimize the params passing to planner.sample_redundant()
            planner.sample_redundant(i, q_trajectory, q_reset_list, yaws, normal, horizontal, pos, current_config, q_source_trajectory=q_source_trajectory, source_tag={"kind":"init","step":i})

            if len(q_trajectory[i]) == 0:
                q_goal = None
                break
            else:
                q_current = np.asarray(current_config, dtype=float)
                q_goal, candidate_yaw = min(q_trajectory[i], key=lambda candidate: np.sum((np.asarray(candidate[0], dtype=float)-q_current)**2))
                # candidate_yaw_trajectory.append(candidate_yaw)
            if q_goal is None:
                #break for checking
                import ipdb; ipdb.set_trace()
                break
            
                print("failed to find ik for flap angle ", degree_tuple)

                print("try null space search....")
                for yaw in yaws:
                    orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=False)
                    q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, reset=True)
                    if q_goal:
                        while True:
                            current_config = planner.get_current_config()
                            z = np.array(q_goal) - np.array(current_config)
                            z = z/np.linalg.norm(z)
                            nullspace_slide_step(planner.robot_id, planner.joint_indices, planner.ee_link_index, cid, z, step=0.01)
                            err = np.linalg.norm(np.array(current_config)-np.array(q_goal))
                            if err < 0.7:
                                break
                            print(err)
                            p.stepSimulation()
                        break
        planned_dict = dp_plan_yaw_path(feasible_by_step=q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))


        # Evaluate the feasible q_goal!
        sorted_idx = 0
        for j in range(MaxIteration):
            path = planned_dict['path']
            selected_index, selected_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[sorted_idx]
            max_index, max_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[0]
            
            # Add some randomness ...
            sorted_idx += 1
            if sorted_idx > 2:
                sorted_idx = 0
            
            # print("the leaf local structure is:")
            # print(selected_index, selected_index+1)

            # NOTE: Below is legacy, but worth another try
            # q1_refined, q2_refined = planner.qs_refinement(path[max_index][0], path[max_index+1][0])

            # NOTE: Not good at all! 
            # q1_list, q2_list = [q[0] for q in q_trajectory[selected_index]], [q[0] for q in q_trajectory[selected_index+1]]
            # q1_array = np.asarray(q1_list, dtype=float)
            # q2_array = np.asarray(q2_list, dtype=float)
            # pairwise_dist_sq = np.sum((q1_array[:, None, :] - q2_array[None, :, :]) ** 2, axis=2)
            # q1_idx, q2_idx = np.unravel_index(np.argmin(pairwise_dist_sq), pairwise_dist_sq.shape)
            # new_q_rest_list = [q1_array[q1_idx].tolist(), q2_array[q2_idx].tolist()]

            new_q_rest_list = [path[selected_index][0].tolist(), path[selected_index+1][0].tolist()]
            print(f"Iter times: {j}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}, selected_index: {selected_index}")
            if max_edge_cost < 0: # set this threashold accordingly
                break
            # Refining...
            for i, degree_tuple in enumerate(degree_tuple_list):
                # new_q_rest = [new_q_rest_list[1]] if i <= selected_index else [new_q_rest_list[0]]
                if abs(i-selected_index) < 5:
                    temp_q_rest_list = new_q_rest_list.copy()
                else:
                    continue
                # if i > 0
                #     temp_q_rest_list.append(path[i-1][0].tolist())
                # if i < len(degree_tuple)-1:
                #     temp_q_:rest_list.append(path[i+1][0].tolist())
                pos, normal, horizontal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
                planner.sample_redundant(i, q_trajectory, temp_q_rest_list, yaws, normal, horizontal, pos, current_config, q_source_trajectory=q_source_trajectory, source_tag={"kind":"refine", "iter":j+1, "from_edge":max_index})
            planned_dict = dp_plan_yaw_path(feasible_by_step=q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))
            

        path = planned_dict['path'] 
        path_sources = [
            q_source_trajectory[step][cand_idx]
            for step, cand_idx in enumerate(planned_dict["indices"])
        ]
        max_index, max_edge_cost = max(enumerate(planned_dict["path_costs"]), key=lambda x: x[1])
        print(f"Iter times: {MaxIteration}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}")
        # for step, ((q_goal, yaw), src) in enumerate(zip(path, path_sources)):
        #     print(f"[path step {step}] yaw={yaw:.4f} q_goal={q_goal}")
        #     print(f"  source = {src}")

        # NOTE: temp, simply for saving the trajectory
        # q_list = [i[0] for i in path]
        # import pickle
        # with open('data/q6.pkl', 'wb') as f:
        #     pickle.dump(q_list, f)
        # return 

        # Start excuting...
        Excute = True
        if Excute:
            # move to grasp...
            grasp_q_goal = path[0][0]
            ompl_path = planner.plan_ompl(planner.get_current_config(), grasp_q_goal, num_waypoints=200, optimal=True)
            if ompl_path is None:
                raise RuntimeError("path is none!")
            q_traj = []
            for i in range(ompl_path.getStateCount()):
                s = ompl_path.getState(i)
                q_traj.append([float(s[j]) for j in range(planner.ndof)])
            planner.execute_joint_trajectory_real(q_traj, dt=0.05, interpolate=False)
            
            planner.close_gripper_to_width(target_width=0.0, force=1000)

            # open the flap/lid...
            for i, (q_goal, yaw) in enumerate(path):
                candidate_yaw_trajectory.append(yaw)
                q_start = planner.get_current_config()
                traj = interpolate_joint_line(q_start, q_goal, 45)
                planner.execute_joint_trajectory_real(traj, N_ref=75)
                planner.close_gripper_to_width(target_width=0, force=1000, wait=0.5)


        # Visualization for C-bundles...
        # plot_feasible_yaw_evolution_greedy(
        #     q_trajectory,
        #     chosen_yaw_trajectory=candidate_yaw_trajectory,
        #     save_path=f"exp/03_28/{self.closed}closed_{self.scaling}_iter_mix_rand.png",
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
        #     save_path=f"exp/NEW/{self.closed}closed_{self.scaling}_threshold_3d_with_init.png",
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
        #     save_path=f"exp/paper/{self.closed}closed_{self.scaling}_3d_with_layer_views_max_classic.png",
        #     show=True,
        #     use_degree=True,
        #     draw_failed_exec_edges=False,
        #     annotate_exec_dist=False,
        # )



        # Ending...
        if self.closed==False:
            print(f"[INFO] The box has been closed!")
        else:
            print(f"[INFO] The box has been opened!")



