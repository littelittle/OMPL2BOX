"""
MailerBoxTask cfg:
{
'mode': 'MailerBoxTask', 
'robot': 'panda', 'gui': True, 
'box_pos': [0.6, 0.1, 0.35], 
'box_yaw': 0.0, 
'box_closed': False, 
'box_scaling': 1.0,
'method': 'Iteration',
'box_file_path': 'assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf'
}
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene import make_sim, physics_from_config
from utils.pointcloud import pts2obj
from utils.path import draw_point
from perception.bullet2geo import get_gt_box_geometry_from_pybullet
from perception.model import TinyPointNetRegressor, decode_labels
from perception.evaluate_model import load_checkpoint

def load_config(path: str | Path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)

def pybullet_depth_to_pointcloud(
    p, width=160, height=120,
    cam_pos=(0.0, -0.4, 1),
    target=(1.0, 0.0, 0.2),
    up=(0,0,1),
    fov=60, near=0.01, far=3.0,
    *, 
    exclude_body_links: Optional[List[Tuple[int, int]]] = None,   # [(bodyUniqueId, linkIndex), ...]
    exclude_bodies: Optional[List[int]] = None,                   # [bodyUniqueId, ...]
):
    view = p.computeViewMatrix(cam_pos, target, up)
    proj = p.computeProjectionMatrixFOV(fov, width/height, near, far)
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

    _, _, _, depth_buf, seg = p.getCameraImage(width, height, view, proj, flags=flags)
    depth_buf = np.asarray(depth_buf).reshape(height, width)
    seg = np.asarray(seg).reshape(height, width)

    # 先屏蔽背景（环境 seg 是 -1）
    base_valid = (seg >= 0)

    # 解码 objectUniqueId / linkIndex
    obj_uid = np.full_like(seg, -1, dtype=np.int32)
    link_idx = np.full_like(seg, -1, dtype=np.int32)
    seg_v = seg[base_valid]
    obj_uid[base_valid] = seg_v & ((1 << 24) - 1)
    link_idx[base_valid] = (seg_v >> 24) - 1

    # 基础过滤：去掉 plane / 背景
    valid = base_valid & (obj_uid >= 0)

    if exclude_bodies:
        for bid in exclude_bodies:
            valid &= (obj_uid != int(bid))

    # 核心：去掉被抓 flap link 的像素
    if exclude_body_links:
        for bid, lid in exclude_body_links:
            valid &= ~((obj_uid == int(bid)) & (link_idx == int(lid)))


    # OpenGL: pixel -> NDC
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_ndc = (2.0 * (u + 0.5) / width) - 1.0
    y_ndc = 1.0 - (2.0 * (v + 0.5) / height)      # 注意这里把图像y翻到OpenGL
    z_ndc = 2.0 * depth_buf - 1.0                 # depth in [0,1] -> z in [-1,1]

    ones = np.ones_like(z_ndc)
    pts_clip = np.stack([x_ndc, y_ndc, z_ndc, ones], axis=-1)[valid].reshape(-1, 4)

    # inv(P*V): clip -> world
    V = np.array(view).reshape(4, 4).T
    P = np.array(proj).reshape(4, 4).T
    invPV = np.linalg.inv(P @ V)

    pts_world_h = (invPV @ pts_clip.T).T
    pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]

    m = np.isfinite(pts_world).all(axis=1)
    return pts_world[m]

def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def analytic_flap_keypoint_pose(
    x1, y1, z1,
    box_base_yaw,
    lid_angle,
    flap_angle,
    lid_length,
    scaling=1.0,
    key_local=(0.125, 0.0, 0.05),
    degrees=False,
    local_z_sign=1.0,
):
    """
    Analytic version of:

        key_world, normal_world, horizontal_world = get_flap_keypoint_pose(...)

    Coordinate convention:
    - flap local x axis  -> horizontal_world
    - flap local -y axis -> normal_world
    - flap local z axis  -> local_z_sign * geometric flap direction

    Parameters
    ----------
    x1, y1, z1:
        World position of the base-lid hinge point.

    box_base_yaw:
        Box yaw angle theta_0.

    lid_angle:
        Base-lid joint angle theta_1.

    flap_angle:
        Lid-flap joint angle theta_2.

    lid_length:
        Distance from base-lid joint to lid-flap joint, i.e. l1.

    scaling:
        Same scaling factor as self.scaling in your PyBullet code.

    key_local:
        Local keypoint in flap frame before scaling.
        This corresponds to [0.13, 0.0, 0.05] in your code.

    degrees:
        Whether input angles are in degrees.

    local_z_sign:
        +1 means flap local +z axis is along geometric flap direction.
        -1 means flap local +z axis is opposite to geometric flap direction.

        If analytic result is flipped compared with PyBullet, try changing this.
    """

    if degrees:
        box_base_yaw = np.deg2rad(box_base_yaw)
        lid_angle = np.deg2rad(lid_angle)
        flap_angle = np.deg2rad(flap_angle)

    p1 = np.array([x1, y1, z1], dtype=float)

    theta0 = box_base_yaw
    theta1 = lid_angle
    theta2 = flap_angle

    # flap local x axis / hinge direction
    horizontal_world = np.array([
        np.cos(theta0),
        np.sin(theta0),
        0.0,
    ], dtype=float)
    horizontal_world = _normalize(horizontal_world)

    # direction inside the vertical opening plane
    plane_dir = np.array([
        np.sin(theta0),
        -np.cos(theta0),
        0.0,
    ], dtype=float)

    world_z = np.array([0.0, 0.0, 1.0], dtype=float)

    def link_direction(alpha):
        """
        Direction of a link whose angle alpha is measured from vertical.
        alpha = 0 means pointing upward.
        """
        return np.sin(alpha) * plane_dir + np.cos(alpha) * world_z

    # lid direction
    lid_dir = link_direction(theta1)

    # lid-flap joint position
    flap_origin_world = p1 + lid_length * lid_dir

    # geometric flap direction
    flap_dir = link_direction(theta1 + theta2)

    # PyBullet flap local +z axis in world frame
    local_z_world = local_z_sign * flap_dir
    local_z_world = _normalize(local_z_world)

    # In a right-handed local frame:
    # x cross y = z
    # therefore y = z cross x
    local_y_world = np.cross(local_z_world, horizontal_world)
    local_y_world = _normalize(local_y_world)

    # Your PyBullet normal_local = [0, -1, 0]
    normal_world = -local_y_world
    normal_world = _normalize(normal_world)

    # key_local is in flap local frame
    key_local = np.asarray(key_local, dtype=float) * scaling
    key_world = (
        flap_origin_world
        + key_local[0] * horizontal_world
        + key_local[1] * local_y_world
        + key_local[2] * local_z_world
    )

    return key_world, normal_world, horizontal_world

def get_flap_keypoint_pose_from_model(model, points, ckpt, device):
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected a point cloud with shape [N, 3], got {tuple(points.shape)}")

    point_mean = torch.as_tensor(ckpt["point_mean"], dtype=torch.float32, device=device)
    point_std = torch.as_tensor(ckpt["point_std"], dtype=torch.float32, device=device)
    label_mean = torch.as_tensor(ckpt["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.as_tensor(ckpt["label_std"], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        points_norm = ((points - point_mean) / point_std).unsqueeze(0)
        pred_norm = model(points_norm)
        pred_out = pred_norm * label_std + label_mean
        output_label_names = ckpt.get("output_label_names", ckpt.get("label_names"))
        pred, _ = decode_labels(pred_out.squeeze(0), output_label_names)
        return pred.cpu().numpy()

def get_estimation(p, task, checkpoint:str="perception/data/pointnet_10k_lr3e-4_sincos.pt", device:str="cuda"):
    pts = pybullet_depth_to_pointcloud(p, exclude_bodies=[task.sim.plane_id, task.pedestal_id])
    rng = np.random.default_rng(0)
    if len(pts) > 768:
        selected_idx = rng.choice(len(pts), size=768, replace=False)
    else:
        selected_idx = rng.choice(len(pts), size=768, replace=True)
    pts = pts[selected_idx]

    device = device
    ckpt = load_checkpoint(checkpoint)
    width = int(ckpt.get("width"))
    model = TinyPointNetRegressor(out_dim=len(ckpt["label_mean"]), width=width).to(device)
    model.load_state_dict(ckpt["model_state"])
    label = get_flap_keypoint_pose_from_model(model, pts, ckpt, device)

    pred_x1, pred_y1, pred_z1 = label[0:3]
    pred_yaw, pred_lid_angle, pred_flap_angle = [np.rad2deg(i) for i in label[3:6]]
    pred_lid_length = label[6]

    return [pred_x1, pred_y1, pred_z1, pred_yaw, pred_lid_angle, pred_flap_angle, pred_lid_length]


if __name__ == "__main__":
    from tasks import MailerBoxTask
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="perception/data/tiny_pointnet.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config("config/MailerBoxTask.json")
    gui = cfg.get("gui", True)

    # Temp:
    cfg['box_closed'] = False
    x0, y0, z0 = cfg['box_pos']
    box_scaling = cfg.get('box_scaling', 1.0)
    box_yaw = cfg.get('box_yaw', 0.0)

    # x1 = x0 - box_scaling * 0.13
    # y1 = y0 - box_scaling * 0.1
    # z1 = z0 + box_scaling * 0.05
       
    sim = make_sim(gui=True, physics=physics_from_config(cfg), load_ground_plane=True)
    task = MailerBoxTask(cfg, sim)
    task.setup_scene(load_panda=False)

    gt = get_gt_box_geometry_from_pybullet(
        body_id=task.mailerbox.body_id,
        lid_id=task.mailerbox.lid_id,
        flap_id=task.mailerbox.flap_id,
        cid=task.mailerbox.cid,
        lid_angle=np.deg2rad(50),
        flap_angle=np.deg2rad(50),
        restore=False
    )

    x1 = gt["x1"]
    y1 = gt["y1"]
    z1 = gt["z1"]
    l1 = gt["lid_length"]
    l1 = 0.18*box_scaling   # override with measured lid length, since GT from PyBullet is not accurate
    theta0 = gt["theta0"]

    print("GT base-lid joint:", gt["p_base_lid_joint"])
    print("GT lid-flap joint:", gt["p_lid_flap_joint"])
    print("GT lid length:", l1)
    print("GT hinge axis:", gt["hinge_axis_world"])
    print("GT theta0:", np.rad2deg(theta0))


    draw_point((x1, y1, z1), size=0.1) 
    pts = pybullet_depth_to_pointcloud(p, exclude_bodies=[sim.plane_id, task.pedestal_id])
    # random select 768 points fed into model
    rng = np.random.default_rng(0)
    if len(pts) > 768:
        selected_idx = rng.choice(len(pts), size=768, replace=False)
    else:
        selected_idx = rng.choice(len(pts), size=768, replace=True)
    pts = pts[selected_idx]

    # load model
    device = args.device
    ckpt = load_checkpoint(args.checkpoint)
    width = int(ckpt.get("width"))
    model = TinyPointNetRegressor(out_dim=len(ckpt["label_mean"]), width=width).to(device)
    model.load_state_dict(ckpt["model_state"])

    label = get_flap_keypoint_pose_from_model(model, pts, ckpt, device)

    # pts2obj(pts, "perception/pointcloud.obj")
    # key_pb, normal_pb, horizontal_pb = task.mailerbox.get_flap_keypoint_pose(np.deg2rad(90), np.deg2rad(90))
    # print("PyBullet key point:", key_pb)
    # draw_point(key_pb, size=0.1, color=(0,0,1))

    pred_x1, pred_y1, pred_z1 = label[0:3]
    pred_yaw, pred_lid_angle, pred_flap_angle = [np.rad2deg(i) for i in label[3:6]]
    pred_lid_length = label[6]

    print(f"Predicted base-lid joint: ({pred_x1:.3f}, {pred_y1:.3f}, {pred_z1:.3f})")
    print(f"Predicted yaw: {pred_yaw:.3f}°")
    print(f"Predicted lid angle: {pred_lid_angle:.3f}°")
    print(f"Predicted flap angle: {pred_flap_angle:.3f}°")
    print(f"Predicted lid length: {pred_lid_length:.3f}")

    mean_error = 0.0
    for i in range(-90, 91, 10):
        key_an, normal_an, horizontal_an = analytic_flap_keypoint_pose(x1=pred_x1, y1=pred_y1, z1=pred_z1, box_base_yaw=np.deg2rad(pred_yaw), lid_angle=np.deg2rad(i), lid_length=pred_lid_length, flap_angle=np.deg2rad(i))
        # print(f"Analytic key point (flap angle {i}):", key_an)
        draw_point(key_an, size=0.1, color=(0,1,0))
        key_pb, normal_pb, horizontal_pb = task.mailerbox.get_flap_keypoint_pose(np.deg2rad(i), np.deg2rad(i))
        # print(f"PyBullet key point (flap angle {i}):", key_pb)
        draw_point(key_pb, size=0.1, color=(0,0,1))

        # print("key error:", np.linalg.norm(np.array(key_pb) - key_an))
        # print("normal dot:", np.dot(_normalize(normal_pb), _normalize(normal_an)))
        # print("horizontal dot:", np.dot(_normalize(horizontal_pb), _normalize(horizontal_an)))
        mean_error += np.linalg.norm(np.array(key_pb) - key_an)
    mean_error /= 19
    print("Mean key error across angles:", mean_error)


    while True:
        p.stepSimulation()
        time.sleep(5)
