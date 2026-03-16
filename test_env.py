import argparse
import json
import time
from pathlib import Path
import random

import pybullet as p
import numpy as np
import math

from functools import partial

from scene import (
    make_sim,
    physics_from_config,
    create_pedestal,
)

from utils.path import interpolate_joint_line
from models import MailerBox

from planners import PandaGripperPlanner

def ee_axes_in_world(body_id, ee_link, cid):
    ls = p.getLinkState(body_id, ee_link, computeForwardKinematics=True, physicsClientId=cid)
    pos_w, orn_w = ls[4], ls[5]  # link frame
    x_w = p.multiplyTransforms([0,0,0], orn_w, [1,0,0], [0,0,0,1], physicsClientId=cid)[0]
    y_w = p.multiplyTransforms([0,0,0], orn_w, [0,1,0], [0,0,0,1], physicsClientId=cid)[0]
    z_w = p.multiplyTransforms([0,0,0], orn_w, [0,0,1], [0,0,0,1], physicsClientId=cid)[0]
    return list(x_w), list(y_w), list(z_w)

def is_feasible(lid_flap_tuple: tuple, mailerbox, planner, closed, former_yaw=None, num_samples=5, q_reset=None):
    pos, normal = mailerbox.get_flap_keypoint_pose(flap_angle=np.deg2rad(lid_flap_tuple[1]), lid_angle=np.deg2rad(lid_flap_tuple[0]))

    # TODO: get the orn from normal and yaw, how should I determine the yaw or yaw list?
    if former_yaw is not None:
        yaws = [former_yaw]
        for i in range(1, num_samples+1):
            yaws.append(former_yaw+0.07*2.0*math.pi*i/float(max(1, num_samples)))
            yaws.append(former_yaw-0.07*2.0*math.pi*i/float(max(1, num_samples)))
        print("former_yawwwww")
    else:
        # in this case, do uniform sampling from [0, 2pi)
        yaws = [2.0 * math.pi * k / float(max(1, num_samples)) for k in range(max(1, num_samples))]
    
    for i, yaw in enumerate(yaws):
        orn = planner._quat_from_normal_and_yaw(normal, yaw, finger_axis_is_plus_y=False)
        q_goal = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1, q_reset=q_reset)
        if q_goal is not None:
            print(yaw, i)
            return q_goal
    
    return None

def gen_2D_map(left_angle_tuple: tuple, right_angle_tuple: tuple, is_feasible=None, step=10):
    """
    x: lid angle e.g [-90, 90]
    y: flap angle() e.g [-90, 90]
    """
    lid_angle_range = [left_angle_tuple[0], right_angle_tuple[0]]
    flap_angle_range = [left_angle_tuple[1], right_angle_tuple[1]]
    lid_angles = np.arange(lid_angle_range[0], lid_angle_range[1], step, dtype=np.int32)
    flap_angles = np.arange(flap_angle_range[0], flap_angle_range[1], step, dtype=np.int32)

    feasible_map = np.zeros((lid_angles.size, flap_angles.size), dtype=np.uint8)
    for i, lid_angle in enumerate(lid_angles):
        row = feasible_map[i]
        for j, flap_angle in enumerate(flap_angles):
            if is_feasible((lid_angle, flap_angle)) is not None:
                row[j] = 1
            else:
                row[j] = 0

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(
        feasible_map.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap="viridis",
        extent=[lid_angles[0], lid_angles[-1] + step, flap_angles[0], flap_angles[-1] + step],
    )
    ax.set_xlabel("lid angle (deg)")
    ax.set_ylabel("flap angle (deg)")
    ax.set_title("Feasible map")
    fig.colorbar(im, ax=ax, label="feasible")
    plt.tight_layout()
    plt.show()
    return feasible_map
    
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
        q_candidate = is_feasible(candidate_tuple, q_reset=is_feasible(right_angle_tuple)) # q_reset=is_feasible(right_angle_tuple)
        if q_candidate is not None:
            if abs(right_angle_tuple[0]-left_angle_tuple[0]) > 20:
                left_traj, left_q = search_traj(left_angle_tuple, candidate_tuple, is_feasible, num_sample, verbose=False)
                if left_traj is None:
                    continue
                right_traj, right_q = search_traj(candidate_tuple, right_angle_tuple, is_feasible, num_sample, verbose=False)
                if right_traj is None:
                    continue
                return left_traj + right_traj, left_q + right_q
            else:
                return [candidate_tuple, right_angle_tuple], [q_candidate, is_feasible(right_angle_tuple)]
    return None, None

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

def nullspace_slide_step(robot_id, joint_indices, ee_link, cid,
                         z, step=0.5, damping=1e-4):
    """
    z: (ndof,) 在关节空间的“推动方向/梯度”
    step: 积分步长
    """
    active_ids = []
    for ji in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, ji, physicsClientId=cid)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            active_ids.append(ji)

    # 2) 用全 active dof 的 q 去算 Jacobian（满足 pybullet API 要求）
    q_full = np.array([p.getJointState(robot_id, j, physicsClientId=cid)[0] for j in active_ids], float)
    zero = [0.0] * len(active_ids)

    # 当前 q
    q = np.array([p.getJointState(robot_id, j, physicsClientId=cid)[0] for j in joint_indices], dtype=float)

    # 取当前 ee link state
    ls = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True, physicsClientId=cid)
    ee_pos = ls[0]
    ee_orn = ls[1]

    # import ipdb; ipdb.set_trace()
    Jlin, Jang = p.calculateJacobian(
        robot_id, ee_link,
        localPosition=[0,0,0],
        objPositions=list(q_full),
        objVelocities=zero,
        objAccelerations=zero,
        physicsClientId=cid,
    )
    J = np.vstack([np.array(Jlin), np.array(Jang)])  # 6 x n
    J = J[:, :-2] 
    # damped pseudoinverse: J^T (J J^T + λI)^-1
    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + damping * np.eye(6))

    N = np.eye(len(joint_indices)) - J_pinv @ J  # nullspace projector

    qdot = N @ np.array(z, dtype=float)
    qdot = qdot
    q_next = q + step * qdot

    # 用 position control 跟踪（也可以 resetJointState 做纯运动学）
    for i, j in enumerate(joint_indices):
        p.setJointMotorControl2(
            robot_id, j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(q_next[i]),
            force=200,
            positionGain=0.2,
            velocityGain=1.0,
            physicsClientId=cid,
        )
    return q_next

def build_tracking_traj(planner, q_start, q_goal, cid, min_steps=10, max_joint_step=0.08, far_jump_norm=0.8):
    q_start_arr = np.asarray(q_start, dtype=np.float64)
    q_goal_arr = np.asarray(q_goal, dtype=np.float64)
    dq = q_goal_arr - q_start_arr
    base_steps = max(min_steps, int(np.max(np.abs(dq)) / max_joint_step) + 1)

    if float(np.linalg.norm(dq)) <= far_jump_norm:
        return interpolate_joint_line(q_start, q_goal, base_steps)

    start_state = p.getLinkState(
        planner.robot_id,
        planner.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=planner.cid,
    )
    start_pos = np.asarray(start_state[4], dtype=np.float64) # [ 0.47002286, -0.01110263,  0.67446291]
    start_orn = start_state[5] # (0.6365908980369568, -0.037826232612133026, 0.7696377635002136, -0.0312880277633667)

    planner.set_robot_config(q_goal)
    goal_state = p.getLinkState(
        planner.robot_id,
        planner.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=planner.cid,
    )
    goal_pos = np.asarray(goal_state[4], dtype=np.float64) # [0.4699856 , 0.00246661, 0.67502356]
    goal_orn = goal_state[5] # (0.6845521926879883, 1.8573606212157756e-05, 0.7289639115333557, -1.8069353245664388e-05)
    planner.set_robot_config(q_start) 

    # q_start [-0.3446061745174167, -0.5868990680473989, 0.16621974609398724, -2.58913825752405, -0.2712792283555172, 3.804286415506545, 1.227880626025723]
    # q_goal [-0.07881589390675175, -0.6450530660477215, 0.09764490061850846, -2.135587948377031, 2.9513882223897387, 3.157086654713307, -2.1074344897436097]

    while True:
        current_config = planner.get_current_config()
        z = np.array(q_goal) - np.array(current_config)
        z = z/np.linalg.norm(z)
        nullspace_slide_step(planner.robot_id, planner.joint_indices, planner.ee_link_index, cid, z, step=0.01)
        err = np.linalg.norm(np.array(current_config)-np.array(q_goal))
        if err < far_jump_norm*0.5:
            break
        print(err)
        p.stepSimulation()
    return interpolate_joint_line(planner.get_current_config(), q_goal, base_steps*2)

    # ee_dist = float(np.linalg.norm(goal_pos - start_pos))
    # steps = max(base_steps, int(ee_dist / 0.01) + 1)

    # traj = [list(q_start)]
    # q_ref = list(q_start)
    # for i in range(1, steps):
    #     alpha = i / float(steps - 1)
    #     pos = ((1.0 - alpha) * start_pos + alpha * goal_pos).tolist()
    #     orn = p.getQuaternionSlerp(start_orn, goal_orn, alpha)
    #     q_wp = planner.solve_ik_collision_aware(pos, orn, collision=False, max_trials=1)
    #     if q_wp is None:
    #         return interpolate_joint_line(q_start, q_goal, steps)
    #     q_wp = planner.wrap_into_limits(q_wp, q_ref)
    #     traj.append(q_wp)
    #     q_ref = q_wp

    # traj[-1] = list(q_goal)
    # return traj

def main(closed=False):
    sim = make_sim(gui=True, load_ground_plane=True)
    cid = sim.cid

    # TODO:figure out the graspable region of the mailerbox_pos
    mailerbox_pos = [0.6, 0.1, 0.4]

    mailerbox = MailerBox(cid, file_path="assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf", scaling=1, pos=mailerbox_pos, closed=closed)
    box_id = mailerbox.body_id
    # pedestal_id = create_pedestal(cid, center_xy=[mailerbox_pos[0], mailerbox_pos[1]], height=mailerbox_pos[2])

    planner = PandaGripperPlanner(oracle_function=mailerbox.get_flap_keypoint_pose, cid=cid, box_id=box_id, plane_id=sim.plane_id)
    planner.box_attached = 10

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    key_point, normal = mailerbox.get_flap_keypoint_pose()
    planner.open_gripper()
    rid = planner.robot_id
    _, _, _, yaw = planner.move_to_pose_with_free_yaw(key_point, normal, planner='VAMP', execute=True, ik_collision=False, approach_flip=False)
    # import ipdb;ipdb.set_trace()
    time.sleep(0.5)

    planner.close_gripper_to_width(target_width=0.0, force=1000)

    # print(ee_axes_in_world(planner.robot_id, planner.ee_link_index, cid))
    # print(find_finger_slide_axes(planner.robot_id, cid))

    # while True:
    #     p.stepSimulation()

    planner.box_attached = 1

    # planning!!!

    if closed:
        start_angle_tuple = (90, 90)
        goal_angle_tuple = (-90, -90)
    else:
        start_angle_tuple = (-90, -90)
        goal_angle_tuple = (90, 90)
    # right_angle_tuple = (-90, -90)
    # left_angle_tuple = (90, 90)

    is_feasible_bound = partial(is_feasible, mailerbox=mailerbox, planner=planner, former_yaw=yaw, closed=closed) 
    # gen_2D_map(start_angle_tuple, goal_angle_tuple, is_feasible_bound, step=10)
    degree_tuple_list, q_list = search_traj(start_angle_tuple, goal_angle_tuple, is_feasible_bound, num_sample=10)


    # print(degree_tuple_list)
    # print(is_feasible_bound((75.9375, 75.9375))) 
    # import ipdb; ipdb.set_trace()

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

    if closed==False:
        print(f"[INFO] The box has been closed!")
    else:
        print(f"[INFO] The box has been opened!")

    while True:
        p.stepSimulation()

if __name__ == "__main__":
    main(closed=False)
