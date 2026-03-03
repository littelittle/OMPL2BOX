#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone PyBullet experiment:
- Load Franka Panda with gripper
- Put a cube between two fingers
- Close gripper with velocity control and motor force limit
- Measure contact normal forces (grip force proxy)
- Apply increasing lateral external force until slip
- Report F_slip and mu_eff ~ F_slip / N_total

Run:
  pip install pybullet
  python panda_grip_test.py

Tips:
- Toggle GUI = True/False
- Adjust cube size/mass/friction, motor_force, solver iterations, time step, etc.
"""

import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pybullet as p
import pybullet_data


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    GUI: bool = True

    # physics
    time_step: float = 1.0 / 240.0
    gravity: float = -9.81
    num_solver_iterations: int = 100

    # cube
    cube_half_extent: float = 0.02  # 4cm cube edge length
    cube_mass: float = 0.05         # kg
    cube_lateral_friction: float = 1.0
    cube_spinning_friction: float = 0.0
    cube_rolling_friction: float = 0.0

    # finger friction
    finger_lateral_friction: float = 1.0
    finger_spinning_friction: float = 0.0
    finger_rolling_friction: float = 0.0

    # placement
    base_z: float = 0.0
    table_z: float = 0.0
    panda_base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda_base_orn_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # gripper close control
    motor_force: float = 80.0     # "max actuator force" for finger joints
    close_velocity: float = -0.03 # rad/s (for prismatic-like mimic joints it's in m/s-ish units; still works)
    settle_steps: int = 480       # steps to settle after closing

    # slip test
    slip_direction_world: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # push along +X
    F_start: float = 0.0
    F_step: float = 1.0          # N increment per stage
    F_max: float = 200.0         # N max trial force
    stage_steps: int = 120       # steps per force stage
    slip_speed_threshold: float = 0.02  # m/s considered "slipping"
    slip_drift_threshold: float = 0.002 # m drift during a stage considered "slipping"

    # optional: reposition cube slightly to ensure contacts
    cube_preload_offset_x: float = 0.01  # can be small like 0.002 if needed


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------

def step_sim(steps: int, sleep: bool = False):
    for _ in range(steps):
        p.stepSimulation()
        if sleep and CFG.GUI:
            time.sleep(CFG.time_step)


def find_joint_and_link_indices(body_id: int):
    """
    Identify Panda gripper finger joints & finger links by name.
    For the default pybullet_data panda URDF, finger joints are typically:
      - panda_finger_joint1
      - panda_finger_joint2
    Finger links are typically:
      - panda_leftfinger
      - panda_rightfinger
    But we don't assume; we search names robustly.
    """
    nj = p.getNumJoints(body_id)

    joint_name_to_index = {}
    link_name_to_index = {}

    for ji in range(nj):
        info = p.getJointInfo(body_id, ji)
        joint_index = info[0]
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        joint_name_to_index[joint_name] = joint_index
        link_name_to_index[link_name] = joint_index

    # Try canonical names first
    finger_joint_candidates = ["panda_finger_joint1", "panda_finger_joint2"]
    left_link_candidates = ["panda_leftfinger", "panda_leftfinger_tip", "leftfinger", "left_finger"]
    right_link_candidates = ["panda_rightfinger", "panda_rightfinger_tip", "rightfinger", "right_finger"]

    finger_joints = []
    for n in finger_joint_candidates:
        if n in joint_name_to_index:
            finger_joints.append(joint_name_to_index[n])

    # Fallback: search for "finger_joint"
    if len(finger_joints) < 2:
        finger_joints = [idx for name, idx in joint_name_to_index.items() if "finger_joint" in name]
        finger_joints = sorted(finger_joints)

    def pick_link(cands: List[str]) -> Optional[int]:
        for n in cands:
            if n in link_name_to_index:
                return link_name_to_index[n]
        return None

    left_link = pick_link(left_link_candidates)
    right_link = pick_link(right_link_candidates)

    # Fallback: search by substring
    if left_link is None:
        for lname, idx in link_name_to_index.items():
            if "leftfinger" in lname or ("left" in lname and "finger" in lname):
                left_link = idx
                break
    if right_link is None:
        for lname, idx in link_name_to_index.items():
            if "rightfinger" in lname or ("right" in lname and "finger" in lname):
                right_link = idx
                break

    if len(finger_joints) < 2 or left_link is None or right_link is None:
        print("---- Joint/Link dump ----")
        for ji in range(nj):
            info = p.getJointInfo(body_id, ji)
            print(
                f"jointIndex={info[0]:2d} jointName={info[1].decode('utf-8'):24s} "
                f"linkName={info[12].decode('utf-8'):24s} jointType={info[2]}"
            )
        raise RuntimeError("Failed to auto-detect finger joints/links. See dump above and adjust name lists.")

    return finger_joints[:2], left_link, right_link


def set_dynamics_for_fingers_and_cube(panda_id: int, left_link: int, right_link: int, cube_id: int):
    # fingers
    p.changeDynamics(
        panda_id, left_link,
        lateralFriction=CFG.finger_lateral_friction,
        spinningFriction=CFG.finger_spinning_friction,
        rollingFriction=CFG.finger_rolling_friction,
    )
    p.changeDynamics(
        panda_id, right_link,
        lateralFriction=CFG.finger_lateral_friction,
        spinningFriction=CFG.finger_spinning_friction,
        rollingFriction=CFG.finger_rolling_friction,
    )
    # cube
    p.changeDynamics(
        cube_id, -1,
        lateralFriction=CFG.cube_lateral_friction,
        spinningFriction=CFG.cube_spinning_friction,
        rollingFriction=CFG.cube_rolling_friction,
    )


def sum_normal_force(bodyA: int, linkA: int, bodyB: int, linkB: int = -1) -> float:
    cps = p.getContactPoints(bodyA=bodyA, bodyB=bodyB, linkIndexA=linkA, linkIndexB=linkB)
    # contact tuple: normalForce is index 9 in PyBullet getContactPoints()
    return sum(float(c[9]) for c in cps)


def open_gripper(panda_id: int, finger_joints: List[int], target_open: float = 0.04):
    # Panda gripper opening per finger is around 0.04 in default URDF.
    for j in finger_joints:
        p.setJointMotorControl2(
            panda_id, j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_open,
            force=CFG.motor_force,
        )
    step_sim(240, sleep=False)


def close_gripper_velocity(panda_id: int, finger_joints: List[int]):
    for j in finger_joints:
        p.setJointMotorControl2(
            panda_id, j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=CFG.close_velocity,
            force=CFG.motor_force,
        )


def stop_gripper(panda_id: int, finger_joints: List[int]):
    for j in finger_joints:
        p.setJointMotorControl2(
            panda_id, j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=CFG.motor_force,
        )


def create_cube(pos_xyz: Tuple[float, float, float], half_extent: float, mass: float) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_extent]*3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_extent]*3, rgbaColor=[0.9, 0.2, 0.2, 1.0])
    cube_id = p.createMultiBody(
        baseMass=0.0, # mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos_xyz,
        baseOrientation=[0, 0, 0, 1],
    )
    return cube_id


def apply_lateral_force(cube_id: int, force_mag: float, direction: Tuple[float, float, float]):
    fx, fy, fz = (force_mag * direction[0], force_mag * direction[1], force_mag * direction[2])
    p.applyExternalForce(
        objectUniqueId=cube_id,
        linkIndex=-1,
        forceObj=[fx, fy, fz],
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME,
    )


def measure_slip_threshold(cube_id: int) -> Optional[float]:
    # Reset cube velocity so detection is cleaner.
    p.resetBaseVelocity(cube_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    last_pos, _ = p.getBasePositionAndOrientation(cube_id)

    F = CFG.F_start
    while F <= CFG.F_max:
        F += CFG.F_step

        # Run one stage at constant force
        stage_start_pos, _ = p.getBasePositionAndOrientation(cube_id)
        for _ in range(CFG.stage_steps):
            apply_lateral_force(cube_id, F, CFG.slip_direction_world)
            p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(cube_id)
        lin_vel, _ = p.getBaseVelocity(cube_id)

        speed = math.sqrt(lin_vel[0]**2 + lin_vel[1]**2 + lin_vel[2]**2)
        drift_stage = math.sqrt(
            (pos[0] - stage_start_pos[0])**2 +
            (pos[1] - stage_start_pos[1])**2 +
            (pos[2] - stage_start_pos[2])**2
        )
        drift_from_last = math.sqrt(
            (pos[0] - last_pos[0])**2 +
            (pos[1] - last_pos[1])**2 +
            (pos[2] - last_pos[2])**2
        )

        # Simple slip criteria: noticeable drift in stage or high speed
        if speed > CFG.slip_speed_threshold or drift_stage > CFG.slip_drift_threshold:
            return F

        last_pos = pos

    return None


# -----------------------------
# Main
# -----------------------------

def main():
    cid = p.connect(p.GUI if CFG.GUI else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, CFG.gravity)
    p.setTimeStep(CFG.time_step)
    p.setPhysicsEngineParameter(numSolverIterations=CFG.num_solver_iterations)

    if CFG.GUI:
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=35, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.25])

    # Plane
    plane_id = p.loadURDF("plane.urdf", [0, 0, CFG.table_z])

    # Panda
    panda_orn = p.getQuaternionFromEuler(CFG.panda_base_orn_euler)
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=CFG.panda_base_pos,
        baseOrientation=panda_orn,
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    finger_joints, left_finger_link, right_finger_link = find_joint_and_link_indices(panda_id)
    print(f"[Detect] finger_joints={finger_joints}, left_finger_link={left_finger_link}, right_finger_link={right_finger_link}")

    # Put arm in a reasonable pose so gripper is accessible.
    # We'll set a standard "ready" configuration for 7 arm joints.
    arm_joint_targets = [0.0, -0.5, 0.0, -2.2, 0.0, 2.0, 0.8]
    # Identify first 7 revolute joints of panda arm by type
    arm_joints = []
    for ji in range(p.getNumJoints(panda_id)):
        info = p.getJointInfo(panda_id, ji)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            arm_joints.append(ji)
        if len(arm_joints) == 7:
            break

    for j, q in zip(arm_joints, arm_joint_targets):
        p.resetJointState(panda_id, j, q)

    open_gripper(panda_id, finger_joints, target_open=0.04)


    # Create cube near gripper tips.
    # We'll query a point roughly between finger links as initial placement:
    left_state = p.getLinkState(panda_id, left_finger_link)
    right_state = p.getLinkState(panda_id, right_finger_link)
    left_pos = left_state[0]
    right_pos = right_state[0]
    mid = [(left_pos[i] + right_pos[i]) * 0.5 for i in range(3)]
    mid[0] += CFG.cube_preload_offset_x
    # Lift a bit to avoid initial interpenetration weirdness
    # import ipdb; ipdb.set_trace()
    mid[2] = max(mid[2], 0.18)
    mid[2] -= 0.02

    cube_id = create_cube(tuple(mid), CFG.cube_half_extent, CFG.cube_mass)
    # import ipdb; ipdb.set_trace()
    set_dynamics_for_fingers_and_cube(panda_id, left_finger_link, right_finger_link, cube_id)

    # Let cube settle slightly
    step_sim(120, sleep=False)

    # Close gripper
    print("[Action] Closing gripper...")
    close_gripper_velocity(panda_id, finger_joints)
    step_sim(CFG.settle_steps, sleep=False)
    stop_gripper(panda_id, finger_joints)
    step_sim(120, sleep=False)

    # Measure grip normal forces
    N_left = sum_normal_force(panda_id, left_finger_link, cube_id, -1)
    N_right = sum_normal_force(panda_id, right_finger_link, cube_id, -1)
    N_total = N_left + N_right
    print(f"[Grip] N_left={N_left:.3f} N, N_right={N_right:.3f} N, N_total={N_total:.3f} N")

    # Slip test
    print("[Test] Sweeping lateral force until slip...")
    F_slip = measure_slip_threshold(cube_id)
    if F_slip is None:
        print(f"[Result] No slip up to F_max={CFG.F_max:.1f} N")
        print("[Result] mu_eff is at least:", (CFG.F_max / max(N_total, 1e-9)))
    else:
        mu_eff = F_slip / max(N_total, 1e-9)
        print(f"[Result] F_slip ≈ {F_slip:.3f} N")
        print(f"[Result] mu_eff ≈ {mu_eff:.4f}  (≈ F_slip / N_total)")

    print("\n--- Notes ---")
    print("1) motor_force is actuator limit, not directly the contact force.")
    print("2) Contact forces depend on solver/timeStep/penetration/friction settings.")
    print("3) For better repeatability: increase numSolverIterations, keep timeStep small, avoid initial interpenetration.")
    print("4) Try scanning CFG.motor_force and friction to map F_slip vs N_total.\n")

    if CFG.GUI:
        print("GUI is on. Close the window to exit.")
        while p.isConnected():
            time.sleep(0.05)
    p.disconnect()


if __name__ == "__main__":
    main()