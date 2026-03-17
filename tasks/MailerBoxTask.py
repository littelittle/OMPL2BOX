import math
import numpy as np
import pybullet as p

from tasks import Task
from models import MailerBox
from planners import PandaGripperPlanner
from test_env import is_feasible, search_traj
from functools import partial
from utils.path import interpolate_joint_line


class MailerBoxTask(Task):
    def setup_scene(self, ):
        mailerbox_pos = [0.6, 0.1, 0.4]
        self.closed = False
        scaling = 1
        file_path = "assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf"

        # set up the mailerbox
        self.mailerbox = MailerBox(self.sim.cid, file_path=file_path, scaling=scaling, pos=mailerbox_pos, closed=self.closed)
        box_id = self.mailerbox.body_id

        # set up the robot 
        self.planner = PandaGripperPlanner(oracle_function=self.mailerbox.get_flap_keypoint_pose, cid=self.sim.cid, box_id=box_id, plane_id=self.sim.plane_id)
        # planner.box_attached = 10 # # TODO: This is a remnant of ompl. Investigate whether this tag actually has any effect.

    def run(self, ):
        planner = self.planner
        mailerbox = self.mailerbox

        key_point, normal = mailerbox.get_flap_keypoint_pose()
        planner.open_gripper()
        _, _, _, yaw = planner.move_to_pose_with_free_yaw(key_point, normal, planner='VAMP', execute=True, ik_collision=False, approach_flip=False)
        planner.close_gripper_to_width(target_width=0.0, force=1000)

        if self.closed:
            start_angle_tuple = (90, 90)
            goal_angle_tuple = (-90, -90)
        else:
            start_angle_tuple = (-90, -90)
            goal_angle_tuple = (90, 90)

        is_feasible_bound = partial(is_feasible, mailerbox=mailerbox, planner=planner, former_yaw=yaw, closed=self.closed) 
        degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)

        for degree_tuple in degree_tuple_list:
            # input("press enter to continue")
            # Update yaw every IK tracking step
            num_steps = 50
            max_offset = 0.1 * 2.0 * math.pi 
            step = max_offset / float(max(1, num_steps))

            yaws = [yaw]
            for k in range(1, num_steps + 1):
                offset = k * step
                yaws.append(yaw + offset)
                yaws.append(yaw - offset)

            pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
            q_goal = None
            for yaw in yaws:
                orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=False)
                q_reset = [(planner.get_current_config()[i]+planner.rest_pose[i])/2 for i in range(len(planner.get_current_config()))]
                q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_reset)
                if q_goal is not None:
                    q_curr = planner.get_current_config()
                    if q_curr is not None and len(q_curr) == len(q_goal):
                        delta = q_goal[-1] - q_curr[-1]
                        q_goal[-1] = q_curr[-1] + ((delta + math.pi) % (2.0 * math.pi) - math.pi)
                    break
            if q_goal is None:
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

            # Execute
            q_start = planner.get_current_config()
            if q_start is None or q_goal is None:
                import ipdb; ipdb.set_trace()
            traj = interpolate_joint_line(q_start, q_goal, 45)
            planner.execute_joint_trajectory_real(traj, N_ref=75)

        if self.closed==False:
            print(f"[INFO] The box has been closed!")
        else:
            print(f"[INFO] The box has been opened!")




