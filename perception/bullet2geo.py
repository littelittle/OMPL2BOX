import numpy as np
import pybullet as p


def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _cid_kwargs(cid):
    return {} if cid is None else {"physicsClientId": cid}


def get_joint_world_pose(body_id, joint_id, cid=None):
    """
    Return the world pose of a joint frame in PyBullet.

    This uses getJointInfo:
        parentFramePos = joint origin position in parent link frame
        parentFrameOrn = joint origin orientation in parent link frame
    """
    kw = _cid_kwargs(cid)

    info = p.getJointInfo(body_id, joint_id, **kw)

    parent_index = info[16]
    parent_frame_pos = info[14]
    parent_frame_orn = info[15]

    if parent_index == -1:
        parent_world_pos, parent_world_orn = p.getBasePositionAndOrientation(body_id, **kw)
    else:
        ls = p.getLinkState(
            body_id,
            parent_index,
            computeForwardKinematics=True,
            **kw,
        )
        parent_world_pos = ls[4]
        parent_world_orn = ls[5]

    joint_world_pos, joint_world_orn = p.multiplyTransforms(
        parent_world_pos,
        parent_world_orn,
        parent_frame_pos,
        parent_frame_orn,
        **kw,
    )

    return np.asarray(joint_world_pos, dtype=float), np.asarray(joint_world_orn, dtype=float)


def get_gt_box_geometry_from_pybullet(
    body_id,
    lid_id,
    flap_id,
    cid=None,
    lid_angle=None,
    flap_angle=None,
    restore=True,
):
    """
    Read clean geometric parameters directly from PyBullet.

    Returns:
        x1, y1, z1:
            world position of the base-lid joint

        lid_length:
            distance between base-lid joint and lid-flap joint

        hinge_axis_world:
            base-lid hinge axis in world frame

        theta0:
            yaw angle of hinge axis, computed from PyBullet GT
    """
    kw = _cid_kwargs(cid)

    original_states = None
    if lid_angle is not None or flap_angle is not None:
        original_states = p.getJointStates(body_id, [lid_id, flap_id], **kw)

        if lid_angle is not None:
            p.resetJointState(body_id, lid_id, targetValue=lid_angle, physicsClientId=cid)
        if flap_angle is not None:
            p.resetJointState(body_id, flap_id, targetValue=flap_angle, physicsClientId=cid)
        

    try:
        # base-lid joint world pose
        p_base_lid, q_base_lid = get_joint_world_pose(body_id, lid_id, cid=cid)

        # lid-flap joint world pose
        p_lid_flap, q_lid_flap = get_joint_world_pose(body_id, flap_id, cid=cid)

        lid_vec_world = p_lid_flap - p_base_lid
        lid_length = float(np.linalg.norm(lid_vec_world))
        lid_dir_world = _normalize(lid_vec_world)

        # read hinge axis from PyBullet joint info
        lid_joint_info = p.getJointInfo(body_id, lid_id, **kw)
        hinge_axis_local = np.asarray(lid_joint_info[13], dtype=float)

        R_base_lid = np.asarray(
            p.getMatrixFromQuaternion(q_base_lid),
            dtype=float,
        ).reshape(3, 3)

        hinge_axis_world = _normalize(R_base_lid @ hinge_axis_local)

        # yaw of hinge axis
        theta0 = float(np.arctan2(hinge_axis_world[1], hinge_axis_world[0]))

        return {
            "p_base_lid_joint": p_base_lid,
            "p_lid_flap_joint": p_lid_flap,
            "x1": float(p_base_lid[0]),
            "y1": float(p_base_lid[1]),
            "z1": float(p_base_lid[2]),
            "lid_length": lid_length,
            "lid_vec_world": lid_vec_world,
            "lid_dir_world": lid_dir_world,
            "hinge_axis_world": hinge_axis_world,
            "theta0": theta0,
        }

    finally:
        if restore and original_states is not None:
            p.resetJointState(body_id, lid_id, targetValue=original_states[0][0], **kw)
            p.resetJointState(body_id, flap_id, targetValue=original_states[1][0], **kw)