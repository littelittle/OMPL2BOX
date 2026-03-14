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

def get_active_dof(robot_id, cid):
    dof_joint_ids = []
    q_full = []
    for ji in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, ji, physicsClientId=cid)
        jtype = info[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            dof_joint_ids.append(ji)
            q_full.append(p.getJointState(robot_id, ji, physicsClientId=cid)[0])
    return np.array(q_full, float), dof_joint_ids

def nullspace_slide_step(robot_id, joint_indices, ee_link, cid,
                         z, step=0.01, damping=1e-4):
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

def main():
    sim = make_sim(gui=True, load_ground_plane=True)
    cid = sim.cid

    planner = PandaGripperPlanner(cid=cid, plane_id=sim.plane_id)

    # get the current pose
    link_state = p.getLinkState(planner.robot_id, planner.ee_link_index)
    pos = link_state[0]
    orn = link_state[1]

    print("EE position:", pos)
    print("EE orientation:", orn)    
    print(get_active_dof(planner.robot_id, cid))

    start_pos = [ 0.47002286, -0.01110263,  0.67446291]
    start_orn = [0.6365908980369568, -0.037826232612133026, 0.7696377635002136, -0.0312880277633667]
    start_config = planner.solve_ik_collision_aware(start_pos, start_orn)
    start_config = [-0.3446061745174167, -0.5868990680473989, 0.16621974609398724, -2.58913825752405, -0.2712792283555172, 3.804286415506545, 1.227880626025723]
    planner.set_robot_config(start_config)

    goal_pos = [0.4699856 , 0.00246661, 0.67502356]
    goal_orn = [0.6845521926879883, 1.8573606212157756e-05, 0.7289639115333557, -1.8069353245664388e-05]
    goal_config = planner.solve_ik_collision_aware(goal_pos, goal_orn)
    goal_config = [-0.07881589390675175, -0.6450530660477215, 0.09764490061850846, -2.135587948377031, 2.9513882223897387, 3.157086654713307, -2.1074344897436097]

    print(np.linalg.norm(np.array(start_config)-np.array(goal_config)))


    while True:
        current_config = planner.get_current_config()
        z = np.array(goal_config) - np.array(current_config)
        z = z/np.linalg.norm(z)
        nullspace_slide_step(planner.robot_id, planner.joint_indices, planner.ee_link_index, cid, z, step=0.01)
        err = np.linalg.norm(np.array(current_config)-np.array(goal_config))
        if err < 0.01:
            break
        print(err)
        p.stepSimulation()

    while True:
        p.stepSimulation()

if __name__ == "__main__":
    main()