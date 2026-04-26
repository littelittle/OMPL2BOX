import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.bullet2geo import get_gt_box_geometry_from_pybullet
from scene import make_sim, physics_from_config
from scene.sim_context import configure_physics, load_plane
from tasks import MailerBoxTask


LABEL_NAMES = np.array(
    [
        "x1",
        "y1",
        "z1",
        "box_base_yaw",
        "lid_angle",
        "flap_angle",
        "lid_length",
    ]
)


def load_config(path):
    with Path(path).open("r") as f:
        return json.load(f)


def fixed_size_points(points, num_points, rng):
    if len(points) == 0:
        raise RuntimeError("Camera produced an empty point cloud.")
    replace = len(points) < num_points
    idx = rng.choice(len(points), size=num_points, replace=replace)
    return points[idx].astype(np.float32)


def depth_to_pointcloud(
    width=160,
    height=120,
    cam_pos=(0.0, -0.4, 1.0),
    target=(1.0, 0.0, 0.2),
    up=(0, 0, 1),
    fov=60,
    near=0.01,
    far=3.0,
    exclude_bodies=None,
):
    view = p.computeViewMatrix(cam_pos, target, up)
    proj = p.computeProjectionMatrixFOV(fov, width / height, near, far)
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    _, _, _, depth, seg = p.getCameraImage(
        width, height, view, proj, flags=flags, renderer=p.ER_TINY_RENDERER
    )

    depth = np.asarray(depth).reshape(height, width)
    seg = np.asarray(seg).reshape(height, width)
    valid = seg >= 0

    if exclude_bodies:
        obj_uid = np.full_like(seg, -1, dtype=np.int32)
        obj_uid[valid] = seg[valid] & ((1 << 24) - 1)
        for body_id in exclude_bodies:
            valid &= obj_uid != int(body_id)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_ndc = 2.0 * (u + 0.5) / width - 1.0
    y_ndc = 1.0 - 2.0 * (v + 0.5) / height
    z_ndc = 2.0 * depth - 1.0
    pts_clip = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1)[valid]

    view_mat = np.array(view).reshape(4, 4).T
    proj_mat = np.array(proj).reshape(4, 4).T
    pts_world_h = (np.linalg.inv(proj_mat @ view_mat) @ pts_clip.T).T
    pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]
    return pts_world[np.isfinite(pts_world).all(axis=1)]


def sample_cfg(base_cfg, rng):
    cfg = dict(base_cfg)
    cfg["gui"] = False
    cfg["box_closed"] = False
    cfg["box_scaling"] = float(rng.uniform(0.9, 1.1))
    cfg["box_yaw"] = float(rng.uniform(-45.0, 45.0))
    lid_angle = float(np.deg2rad(rng.uniform(-90.0, 90.0)))
    flap_angle = float(np.deg2rad(rng.uniform(-90.0, 90.0)))
    return cfg, lid_angle, flap_angle


def generate_sample(base_cfg, sim, rng, num_points):
    cfg, lid_angle, flap_angle = sample_cfg(base_cfg, rng)

    p.resetSimulation(physicsClientId=sim.cid)
    configure_physics(sim.cid, physics_from_config(cfg))
    sim.plane_id = load_plane(sim.cid)

    task = MailerBoxTask(cfg, sim)
    task.setup_scene(load_panda=False)

    box = task.mailerbox
    p.resetJointState(box.body_id, box.lid_id, lid_angle, physicsClientId=box.cid)
    p.resetJointState(box.body_id, box.flap_id, flap_angle, physicsClientId=box.cid)

    gt = get_gt_box_geometry_from_pybullet(
        body_id=box.body_id,
        lid_id=box.lid_id,
        flap_id=box.flap_id,
        cid=box.cid,
    )

    points = depth_to_pointcloud(exclude_bodies=[sim.plane_id, task.pedestal_id])
    points = fixed_size_points(points, num_points, rng)

    label = np.array(
        [
            gt["x1"],
            gt["y1"],
            gt["z1"],
            gt["theta0"],
            lid_angle,
            flap_angle,
            0.18 * cfg["box_scaling"],
        ],
        dtype=np.float32,
    )
    return points, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/MailerBoxTask.json")
    parser.add_argument("--output", default="perception/data/mailerbox_poc.npz")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_points", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base_cfg = load_config(args.config)

    points = np.empty((args.num_samples, args.num_points, 3), dtype=np.float32)
    labels = np.empty((args.num_samples, len(LABEL_NAMES)), dtype=np.float32)

    sim = make_sim(gui=False, physics=physics_from_config(base_cfg), load_ground_plane=True)
    for i in range(args.num_samples):
        points[i], labels[i] = generate_sample(base_cfg, sim, rng, args.num_points)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"generated {i + 1}/{args.num_samples}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=points, labels=labels, label_names=LABEL_NAMES)
    print(f"saved {output}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
