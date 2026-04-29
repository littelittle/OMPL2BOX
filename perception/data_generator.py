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
from utils.pointcloud import pts2obj, pybullet_depth_to_pointcloud
from utils.path import draw_point
from perception.bullet2geo import get_gt_box_geometry_from_pybullet
from perception.mailerbox_geometry import _normalize, analytic_flap_keypoint_pose, base_pose_from_keypoint_pose, keypoint_pose_from_base_pose
from perception.model import (
    build_model,
    decode_labels,
    denormalize_label_coordinates,
    normalize_points_and_labels,
)
from perception.evaluate_model import load_checkpoint

def load_config(path: str | Path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)

def get_flap_keypoint_pose_from_model(model, points, ckpt, device):
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected a point cloud with shape [N, 3], got {tuple(points.shape)}")

    label_mean = torch.as_tensor(ckpt["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.as_tensor(ckpt["label_std"], dtype=torch.float32, device=device)
    output_label_names = ckpt.get("output_label_names", ckpt.get("label_names"))
    point_normalization = ckpt.get("point_normalization", "global_mean_std")
    if point_normalization == "global_mean_std":
        point_mean = torch.as_tensor(ckpt["point_mean"], dtype=torch.float32, device=device)
        point_std = torch.as_tensor(ckpt["point_std"], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        if point_normalization == "per_sample_center_scale":
            points_norm, _, center, scale = normalize_points_and_labels(points)
            points_norm = points_norm.unsqueeze(0)
        else:
            points_norm = ((points - point_mean) / point_std).unsqueeze(0)
            center = scale = None
        pred_norm = model(points_norm)
        pred_out = pred_norm * label_std + label_mean
        if point_normalization == "per_sample_center_scale":
            pred_out = denormalize_label_coordinates(pred_out.squeeze(0), output_label_names, center, scale)
        else:
            pred_out = pred_out.squeeze(0)
        pred, _ = decode_labels(pred_out, output_label_names)
        return pred.cpu().numpy()

def get_estimation(pts, checkpoint:str="perception/data/keypointNet_002.pt", device:str="cuda"):
    rng = np.random.default_rng(0)
    if len(pts) > 768:
        selected_idx = rng.choice(len(pts), size=768, replace=False)
    else:
        selected_idx = rng.choice(len(pts), size=768, replace=True)
    pts = pts[selected_idx]

    device = device
    ckpt = load_checkpoint(checkpoint)
    width = int(ckpt.get("width", 64))
    model_name = ckpt.get("model", "tiny")
    output_label_names = ckpt.get("output_label_names", ckpt.get("label_names"))
    model = build_model(
        model_name,
        out_dim=len(ckpt["label_mean"]),
        width=width,
        label_names=output_label_names,
        label_mean=ckpt["label_mean"],
        label_std=ckpt["label_std"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    label = get_flap_keypoint_pose_from_model(model, pts, ckpt, device)

    pred_x1, pred_y1, pred_z1 = label[0:3]
    pred_yaw, pred_lid_angle, pred_flap_angle = [np.rad2deg(i) for i in label[3:6]]
    pred_lid_length = label[6]

    return [
        pred_x1,
        pred_y1,
        pred_z1,
        pred_yaw,
        pred_lid_angle,
        pred_flap_angle,
        pred_lid_length,
        *label[7:].tolist(),
    ]


if __name__ == "__main__":
    from tasks import MailerBoxTask
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="perception/data/pointnetplus_10k_lr1e-4_150.pt")
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
    width = int(ckpt.get("width", 64))
    model_name = ckpt.get("model", "tiny")
    output_label_names = ckpt.get("output_label_names", ckpt.get("label_names"))
    model = build_model(
        model_name,
        out_dim=len(ckpt["label_mean"]),
        width=width,
        label_names=output_label_names,
        label_mean=ckpt["label_mean"],
        label_std=ckpt["label_std"],
    ).to(device)
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
        key_an, normal_an, horizontal_an = analytic_flap_keypoint_pose(x1=pred_x1, y1=pred_y1, z1=pred_z1, box_base_yaw=np.deg2rad(pred_yaw), lid_angle=np.deg2rad(i), lid_length=pred_lid_length, flap_angle=np.deg2rad(i), scaling=box_scaling)
        # print(f"Analytic key point (flap angle {i}):", key_an)
        draw_point(key_an, size=0.1, color=(0,1,0))
        key_pb, normal_pb, horizontal_pb = task.mailerbox.get_flap_keypoint_pose(np.deg2rad(i), np.deg2rad(i), estimate=False)
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
