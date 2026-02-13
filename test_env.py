import argparse
import json
import time
from pathlib import Path
import random

import pybullet as p
import numpy as np
import math

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
    mailerbox_pos = [0.7, 0.1, 0.2]

    mailerbox = MailerBox(cid, file_path="robot_sim/assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf", scaling=0.3, pos=mailerbox_pos, )
    box_id = mailerbox.body_id
    pedestal_id = create_pedestal(cid, center_xy=[mailerbox_pos[0], mailerbox_pos[1]], height=mailerbox_pos[2])

    planner = PandaGripperPlanner(oracle_function=mailerbox.get_flap_keypoint_pose, cid=cid, box_id=box_id, plane_id=sim.plane_id)
    planner.box_attached = 10

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    key_point, normal = mailerbox.get_flap_keypoint_pose()
    # while True:
    #     p.stepSimulation()
    # import ipdb; ipdb.set_trace()

    planner.open_gripper()

    _, _, _, yaw = planner.move_to_pose_with_free_yaw(key_point, normal, planner='VAMP')

    time.sleep(0.5)

    planner.close_gripper_to_width(target_width=0.0)

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    while True:
        p.stepSimulation()

    planner.box_attached = 1

    yaws = [yaw] + [yaw+0.1*2.0 * math.pi * k / float(max(1, 10)) for k in range(max(1, 10))]

    for degree in range(0, 175, 5):
        # rad1 = np.deg2rad(180-degree)
        # rad2 = np.deg2rad(max(180-5*degree, 0))
        rad1 = np.deg2rad(-90+degree)
        rad2 = np.deg2rad(-90+degree)
        pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=rad2, lid_angle=rad1)
        # planner.move_to_pose_with_free_yaw(pos, normal, yaw=yaw, planner='VAMP')
        q_goal = None
        for yaw in yaws:
            orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=True)
            q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False)
            if q_goal is not None:
                break
        if q_goal is None:
            print("failed to find ik for flap angle ", rad2)
            continue

        # update yaw every IK tracking step
        yaws = [yaw] + [yaw+0.1*2.0 * math.pi * k / float(max(1, 10)) for k in range(max(1, 10))]

        q_start = planner.get_current_config()
        traj = interpolate_joint_line(q_start, q_goal, 90)
        planner.execute_joint_trajectory_real(traj)
        time.sleep(0.1)
    import ipdb; ipdb.set_trace()
    while True:
        p.stepSimulation()

if __name__ == "__main__":
    main()