import numpy as np


def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _horizontal_frame(horizontal, world_up=(0.0, 0.0, 1.0)):
    z_axis = _normalize(world_up)
    horizontal = np.asarray(horizontal, dtype=float)
    horizontal = horizontal - np.dot(horizontal, z_axis) * z_axis
    horizontal = _normalize(horizontal)
    plane_dir = _normalize(np.cross(horizontal, z_axis))
    return horizontal, plane_dir, z_axis


def _link_direction(angle, plane_dir, z_axis):
    return _normalize(np.sin(angle) * plane_dir + np.cos(angle) * z_axis)


def _flap_normal_from_direction(flap_dir, horizontal):
    return _normalize(-np.cross(flap_dir, horizontal))


def _maybe_degrees(degrees, *angles):
    if degrees:
        return tuple(np.deg2rad(a) for a in angles)
    return angles


def analytic_flap_keypoint_pose(
    x1,
    y1,
    z1,
    box_base_yaw,
    lid_angle,
    flap_angle,
    lid_length,
    scaling=1.0,
    key_local=(0.12, 0.0, 0.05),
    degrees=False,
    local_z_sign=1.0,
):
    """
    Analytic version of MailerBox.get_flap_keypoint_pose(...).

    x1, y1, z1 are the world position of the base-lid joint center.
    lid_length is the distance from the base-lid joint to the lid-flap joint.
    """
    if degrees:
        box_base_yaw, lid_angle, flap_angle = _maybe_degrees(
            True, box_base_yaw, lid_angle, flap_angle
        )

    p1 = np.array([x1, y1, z1], dtype=float)

    horizontal_world = np.array(
        [np.cos(box_base_yaw), np.sin(box_base_yaw), 0.0],
        dtype=float,
    )
    horizontal_world, plane_dir, world_z = _horizontal_frame(horizontal_world)

    lid_dir = _link_direction(lid_angle, plane_dir, world_z)
    flap_origin_world = p1 + lid_length * lid_dir

    flap_dir = _link_direction(lid_angle + flap_angle, plane_dir, world_z)
    local_z_world = _normalize(local_z_sign * flap_dir)
    local_y_world = _normalize(np.cross(local_z_world, horizontal_world))
    normal_world = _normalize(-local_y_world)

    key_local = np.asarray(key_local, dtype=float) * scaling
    key_world = (
        flap_origin_world
        + key_local[0] * horizontal_world
        + key_local[1] * local_y_world
        + key_local[2] * local_z_world
    )

    return key_world, normal_world, horizontal_world


def base_pose_from_keypoint_pose(
    key_pb,
    normal_pb,
    horizontal_pb,
    l1,
    l2,
    lid_angle,
    flap_angle,
    *,
    degrees=False,
    world_up=(0.0, 0.0, 1.0),
    align_horizontal_sign=False,
):
    """
    Recover the invariant base point pose from the current keypoint pose.

    Geometry convention:
    - horizontal_pb is the hinge direction and is kept as the base pose x axis.
    - world_up is the base pose z axis.
    - l1 is the in-plane distance from the lid/flap joint slice to key_pb.
    - l2 is the in-plane distance from the base/lid joint slice to the
      lid/flap joint slice.

    The returned base point is the base/lid point in the same opening plane as
    key_pb, not necessarily the center of the physical base/lid joint line.
    """
    lid_angle, flap_angle = _maybe_degrees(degrees, lid_angle, flap_angle)

    key_pb = np.asarray(key_pb, dtype=float)
    observed_normal = _normalize(normal_pb)
    horizontal, plane_dir, z_axis = _horizontal_frame(horizontal_pb, world_up)

    if align_horizontal_sign:
        total_angle = lid_angle + flap_angle
        expected_normal = _flap_normal_from_direction(
            _link_direction(total_angle, plane_dir, z_axis),
            horizontal,
        )
        flipped_horizontal, flipped_plane_dir, _ = _horizontal_frame(
            -horizontal, z_axis
        )
        flipped_normal = _flap_normal_from_direction(
            _link_direction(total_angle, flipped_plane_dir, z_axis),
            flipped_horizontal,
        )
        if np.dot(flipped_normal, observed_normal) > np.dot(
            expected_normal, observed_normal
        ):
            horizontal = flipped_horizontal
            plane_dir = flipped_plane_dir

    lid_dir = _link_direction(lid_angle, plane_dir, z_axis)
    flap_dir = _link_direction(lid_angle + flap_angle, plane_dir, z_axis)

    base_position = key_pb - float(l1) * flap_dir - float(l2) * lid_dir
    base_normal = plane_dir

    return {
        "position": base_position,
        "normal": base_normal,
        "horizontal": horizontal,
        "z_axis": z_axis,
    }


def _unpack_base_pose(base_pose, world_up):
    if isinstance(base_pose, dict):
        position = base_pose.get("position", base_pose.get("pos"))
        horizontal = base_pose.get("horizontal")
        z_axis = base_pose.get("z_axis", world_up)
        normal = base_pose.get("normal")
    else:
        if len(base_pose) == 2:
            position, horizontal = base_pose
            normal = None
            z_axis = world_up
        elif len(base_pose) == 3:
            position, normal, horizontal = base_pose
            z_axis = world_up
        else:
            raise ValueError(
                "base_pose must be a dict, (position, horizontal), or "
                "(position, normal, horizontal)."
            )

    if position is None:
        raise ValueError("base_pose is missing position.")

    z_axis = _normalize(z_axis)
    if horizontal is None:
        if normal is None:
            raise ValueError("base_pose needs horizontal, or normal to infer it.")
        horizontal = np.cross(z_axis, _normalize(normal))

    return np.asarray(position, dtype=float), horizontal, z_axis


def keypoint_pose_from_base_pose(
    base_pose,
    l1,
    l2,
    lid_angle,
    flap_angle,
    *,
    degrees=False,
    world_up=(0.0, 0.0, 1.0),
):
    """
    Compute key_pb and normal_pb from an invariant base point pose.

    This is the forward counterpart of base_pose_from_keypoint_pose() under the
    same in-plane l1/l2 convention.
    """
    lid_angle, flap_angle = _maybe_degrees(degrees, lid_angle, flap_angle)

    base_position, horizontal, z_axis = _unpack_base_pose(base_pose, world_up)
    horizontal, plane_dir, z_axis = _horizontal_frame(horizontal, z_axis)

    lid_dir = _link_direction(lid_angle, plane_dir, z_axis)
    flap_dir = _link_direction(lid_angle + flap_angle, plane_dir, z_axis)

    key_pb = base_position + float(l2) * lid_dir + float(l1) * flap_dir
    normal_pb = _flap_normal_from_direction(flap_dir, horizontal)

    return key_pb, normal_pb, horizontal
