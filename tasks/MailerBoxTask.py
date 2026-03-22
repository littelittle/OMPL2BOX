import math
import numpy as np
import pybullet as p
from functools import partial

from tasks import Task
from models import MailerBox
from planners import PandaGripperPlanner
from test_env import is_feasible, search_traj
from scene import create_pedestal
from utils.yaw_dp import dp_plan_yaw_path, Q_RESET_SEEDS
from utils.path import interpolate_joint_line
from utils.drawer import plot_feasible_yaw_evolution_greedy


class MailerBoxTask(Task):
    def setup_scene(self, ):
        mailerbox_pos = self.config.get("mailerbox_pos", [0.6, 0.1, 0.4])
        self.closed = self.config.get("closed", False)
        self.scaling = self.config.get("scaling", 1)
        file_path = self.config.get("file_path","assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf")

        # create the pedestal
        create_pedestal(self.sim.cid, mailerbox_pos[:2], size_xy=(0.2, 0.2), height=0.35)

        # set up the mailerbox
        self.mailerbox = MailerBox(self.sim.cid, file_path=file_path, scaling=self.scaling, pos=mailerbox_pos, closed=self.closed)
        box_id = self.mailerbox.body_id

        # set up the robot 
        self.planner = PandaGripperPlanner(oracle_function=self.mailerbox.get_flap_keypoint_pose, cid=self.sim.cid, box_id=box_id, plane_id=self.sim.plane_id)
        # planner.box_attached = 10 # # TODO: This is a remnant of ompl. Investigate whether this tag actually has any effect.

    def run(self, ):
        planner = self.planner
        mailerbox = self.mailerbox
        candidate_yaw_trajectory = []

        # reaching out to the flap grasp point! 
        key_point, normal = mailerbox.get_flap_keypoint_pose()
        planner.open_gripper()
        _, _, _, start_yaw = planner.move_to_pose_with_free_yaw(key_point, normal, planner='VAMP', execute=True, ik_collision=False, approach_flip=False, yaw=np.deg2rad(180))
        planner.close_gripper_to_width(target_width=0.0, force=1000)

        # searching for potential fesible task space!
        if self.closed:
            start_angle_tuple = (90, 90)
            goal_angle_tuple = (-90, -90)
        else:
            start_angle_tuple = (-90, -90)
            goal_angle_tuple = (90, 90)
        is_feasible_bound = partial(is_feasible, mailerbox=mailerbox, planner=planner, former_yaw=start_yaw, closed=self.closed) 
        degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)
        q_trajectory = [[] for _ in range(len(degree_tuple_list))]

        # following the task space to execute!
        for i, degree_tuple in enumerate(degree_tuple_list):

            # Update yaw list every IK tracking step
            num_steps = 20
            max_offset = 0.1 * 2.0 * math.pi 
            step = max_offset / float(max(1, num_steps))
            yaws = [start_yaw] # if i==0 else [q_goal_list[0][1]] 
            for k in range(1, num_steps + 1):
                offset = k * step
                yaws.append(yaws[0] + offset)
                yaws.append(yaws[0] - offset)


            # Searching for feasible configuration!
            pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
            q_goal = None
            q_goal_list = []
            current_config = planner.get_current_config()
            q_reset1 = [planner.rest_pose[i] for i in range(len(planner.get_current_config()))]
            q_reset2 = [planner.get_current_config()[i] for i in range(len(planner.get_current_config()))]
            q_reset3 = [(current_config[i]+planner.rest_pose[i])/2 for i in range(len(current_config))]
            q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                # Q_RESET_SEEDS["left_elbow_out"],
                # Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]

            planner.sample_redundant(i, q_trajectory, q_reset_list, yaws, normal, pos, current_config)

            if len(q_trajectory[i]) == 0:
                q_goal = None
                break
            else:
                # q_goal = q_goal_list[0][0]
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
        if planned_dict:
            path = planned_dict['path']
            max_index, max_edge_cost = max(enumerate(planned_dict["path_costs"]), key=lambda x: x[1])
            new_q_rest = path[max_index+1][0].tolist()
            # import ipdb; ipdb.set_trace()

            # Refining...
            # NOTE: later, yaws = [np.deg2rad(130+10*i) for i in range(11)]
            for i, degree_tuple in enumerate(degree_tuple_list):
                pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
                planner.sample_redundant(i, q_trajectory, [new_q_rest], yaws, normal, pos, current_config)

            planned_dict = dp_plan_yaw_path(feasible_by_step=q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))
            path = planned_dict['path']

            for i, (q_goal, yaw) in enumerate(path):
                candidate_yaw_trajectory.append(yaw)
                q_start = planner.get_current_config()
                traj = interpolate_joint_line(q_start, q_goal, 45)
                planner.execute_joint_trajectory_real(traj, N_ref=75)
                planner.close_gripper_to_width(target_width=0, force=1000, wait=0.5)


        # Visualization for C-bundles!
        # plot_feasible_yaw_evolution_greedy(
        #     q_trajectory,
        #     chosen_yaw_trajectory=candidate_yaw_trajectory,
        #     save_path=f"exp/merge_bundles/{self.closed}closed_{self.scaling}_baesline_new.png",
        #     show=True,
        #     use_degree=True,
        #     angular_indices=range(7),   # Panda arm joints
        #     one_to_one=False,           
        # )

        # Ending...
        if self.closed==False:
            print(f"[INFO] The box has been closed!")
        else:
            print(f"[INFO] The box has been opened!")




