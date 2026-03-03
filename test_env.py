import argparse
import json
import time
from pathlib import Path
import random

import pybullet as p
import numpy as np
import math

from functools import partial

from robot_sim import (
    PandaGripperPlanner,
    make_sim,
    physics_from_config,
    MailerBox,
    create_pedestal,
    interpolate_joint_line,
)

def ee_axes_in_world(body_id, ee_link, cid):
    ls = p.getLinkState(body_id, ee_link, computeForwardKinematics=True, physicsClientId=cid)
    pos_w, orn_w = ls[4], ls[5]  # link frame
    x_w = p.multiplyTransforms([0,0,0], orn_w, [1,0,0], [0,0,0,1], physicsClientId=cid)[0]
    y_w = p.multiplyTransforms([0,0,0], orn_w, [0,1,0], [0,0,0,1], physicsClientId=cid)[0]
    z_w = p.multiplyTransforms([0,0,0], orn_w, [0,0,1], [0,0,0,1], physicsClientId=cid)[0]
    return list(x_w), list(y_w), list(z_w)

def is_feasible(lid_flap_tuple: tuple, mailerbox, planner, former_yaw=None, num_samples=10):
    pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(lid_flap_tuple[1]), lid_angle=np.deg2rad(lid_flap_tuple[0]))

    # TODO: get the orn from normal and yaw, how should I determine the yaw or yaw list?
    if former_yaw is not None:
        yaws = [former_yaw] # + [former_yaw+0.1*2.0 * math.pi * k / float(max(1, num_samples)) for k in range(max(1, num_samples))]
        print("hhh")
    else:
        # in this case, do uniform sampling from [0, 2pi)
        yaws = [2.0 * math.pi * k / float(max(1, num_samples)) for k in range(max(1, num_samples))]
    
    for yaw in yaws:
        orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=False)
        q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=10)
        if q_goal is not None:
            return True
    
    return False

def search_traj(left_angle_tuple: tuple, right_angle_tuple: tuple, is_feasible=None, num_sample:int=10, verbose=True):
    """
    find the feasible lid/flap angle trajectory 
    *_angle_tuple: (lid_angle, flap_angle)
    """
    middle_flap_angle = (left_angle_tuple[1]+right_angle_tuple[1])/2 
    step = (right_angle_tuple[1]-left_angle_tuple[1])/(num_sample-1)
    samples = [left_angle_tuple[1]+i*step for i in range(num_sample)]
    mid = (num_sample-1)/2.0
    ordered = sorted(range(num_sample), key=lambda i:abs(i-mid))
    flap_angle_candidate = [middle_flap_angle] + [samples[i] for i in ordered]
    # flap_angle_candidate = samples
    middle_angle_tuple_list = [((left_angle_tuple[0]+right_angle_tuple[0])/2, i) for i in flap_angle_candidate]
    for i, candidate_tuple in enumerate(middle_angle_tuple_list):
        if verbose:
            print(i)
        if is_feasible(candidate_tuple):
            if (right_angle_tuple[0]-left_angle_tuple[0]) > 10:
                left_traj = search_traj(left_angle_tuple, candidate_tuple, is_feasible, num_sample, verbose=False)
                if left_traj is None:
                    continue
                right_traj = search_traj(candidate_tuple, right_angle_tuple, is_feasible, num_sample, verbose=False)
                if right_traj is None:
                    continue
                return left_traj + right_traj
            else:
                return [candidate_tuple, right_angle_tuple]
    return None

def find_finger_slide_axes(robot_id, cid, name_keyword="finger"):
    axes = []
    for j in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        ji = p.getJointInfo(robot_id, j, physicsClientId=cid)
        joint_name = ji[1].decode("utf-8")
        joint_type = ji[2]  # 1: prismatic, 0: revolute, etc.
        if (name_keyword in joint_name) and (joint_type == p.JOINT_PRISMATIC):
            joint_axis = ji[13]
            parent_frame_pos = ji[14]
            parent_frame_orn = ji[15]
            parent_index = ji[16]

            # parent link world pose
            if parent_index == -1:
                parent_pos_w, parent_orn_w = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
            else:
                ls = p.getLinkState(robot_id, parent_index, computeForwardKinematics=True, physicsClientId=cid)
                parent_pos_w, parent_orn_w = ls[4], ls[5]

            # joint frame world orn
            _, joint_orn_w = p.multiplyTransforms(
                parent_pos_w, parent_orn_w,
                parent_frame_pos, parent_frame_orn,
                physicsClientId=cid
            )

            # axis in world (rotate only)
            axis_w = p.multiplyTransforms(
                [0,0,0], joint_orn_w,
                joint_axis, [0,0,0,1],
                physicsClientId=cid
            )[0]

            axes.append((j, joint_name, joint_axis, axis_w))
    return axes


def main():
    sim = make_sim(gui=True, load_ground_plane=True)
    cid = sim.cid

    # TODO:figure out the graspable region of the mailerbox_pos
    mailerbox_pos = [0.6, 0.1, 0.4]

    mailerbox = MailerBox(cid, file_path="robot_sim/assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf", scaling=0.3, pos=mailerbox_pos, )
    box_id = mailerbox.body_id
    # pedestal_id = create_pedestal(cid, center_xy=[mailerbox_pos[0], mailerbox_pos[1]], height=mailerbox_pos[2])

    planner = PandaGripperPlanner(oracle_function=mailerbox.get_flap_keypoint_pose, cid=cid, box_id=box_id, plane_id=sim.plane_id)
    planner.box_attached = 10

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    key_point, normal = mailerbox.get_flap_keypoint_pose()
    # while True:
    #     p.stepSimulation()
    # import ipdb; ipdb.set_trace()

    planner.open_gripper()
    rid = planner.robot_id

    _, _, _, yaw = planner.move_to_pose_with_free_yaw(key_point, normal, planner='VAMP', execute=True)

    time.sleep(0.5)

    # planner.close_gripper_to_width(target_width=0.0, force=1000)

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    # while True:
    #     p.stepSimulation()

    planner.box_attached = 1

    yaws = [yaw] + [yaw+0.1*2.0 * math.pi * k / float(max(1, 10)) for k in range(max(1, 10))]

    # planning!!!

    left_angle_tuple = (-90, -90)
    right_angle_tuple = (90, 90)
    is_feasible_bound = partial(is_feasible, mailerbox=mailerbox, planner=planner, former_yaw=yaw) 
    degree_tuple_list = search_traj(left_angle_tuple, right_angle_tuple, is_feasible_bound, num_sample=10)

    print(degree_tuple_list)
    # import ipdb; ipdb.set_trace()

    # for degree in range(0, 180, 5):
    #     # rad1 = np.deg2rad(180-degree)
    #     # rad2 = np.deg2rad(max(180-5*degree, 0))
    #     rad1 = np.deg2rad(min(-90+1.5*degree, 90))
    #     rad2 = np.deg2rad(min(-90+1*degree, 90))
    for degree_tuple in degree_tuple_list:
        pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(degree_tuple[1]), lid_angle=np.deg2rad(degree_tuple[0]))
        # planner.move_to_pose_with_free_yaw(pos, normal, yaw=yaw, planner='VAMP')
        q_goal = None
        for yaw in yaws:
            orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=False)
            q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1)
            if q_goal is not None:
                break
        if q_goal is None:
            print("failed to find ik for flap angle ", degree_tuple)
            import ipdb; ipdb.set_trace()
            continue

        # Update yaw every IK tracking step
        yaws = [yaw] + [yaw+0.1*2.0 * math.pi * k / float(max(1, 10)) for k in range(max(1, 10))]

        # Execute
        q_start = planner.get_current_config()
        traj = interpolate_joint_line(q_start, q_goal, 90)
        planner.execute_joint_trajectory_real(traj)
        time.sleep(0.3)
    import ipdb; ipdb.set_trace()
    while True:
        p.stepSimulation()

if __name__ == "__main__":
    main()