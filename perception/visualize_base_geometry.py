import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.bullet2geo import get_gt_box_geometry_from_pybullet
from perception.mailerbox_geometry import (
    base_pose_from_keypoint_pose,
    keypoint_pose_from_base_pose,
)
from scene import make_sim, physics_from_config
from tasks import MailerBoxTask
from utils.path import draw_point


def load_config(path):
    with Path(path).open("r") as f:
        return json.load(f)


def draw_vector(origin, vec, color, scale=0.08, life_time=0):
    origin = np.asarray(origin, dtype=float)
    vec = np.asarray(vec, dtype=float)
    end = origin + scale * vec
    p.addUserDebugLine(
        origin.tolist(),
        end.tolist(),
        lineColorRGB=color,
        lineWidth=3,
        lifeTime=life_time,
    )


def draw_text(text, pos, color=(1.0, 1.0, 1.0), life_time=0):
    p.addUserDebugText(
        text,
        np.asarray(pos, dtype=float).tolist(),
        textColorRGB=color,
        textSize=1.1,
        lifeTime=life_time,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/MailerBoxTask.json")
    parser.add_argument("--lid_angle", type=float, default=30.0)
    parser.add_argument("--flap_angle", type=float, default=30.0)
    parser.add_argument("--angle_min", type=float, default=-90.0)
    parser.add_argument("--angle_max", type=float, default=90.0)
    parser.add_argument("--angle_step", type=float, default=15.0)
    parser.add_argument("--l1", type=float, default=None)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--key_local_z", type=float, default=0.05)
    parser.add_argument("--sleep", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["gui"] = True

    sim = make_sim(gui=True, physics=physics_from_config(cfg), load_ground_plane=True)
    task = MailerBoxTask(cfg, sim)
    task.setup_scene(load_panda=False)

    box = task.mailerbox
    lid_angle = np.deg2rad(args.lid_angle)
    flap_angle = np.deg2rad(args.flap_angle)

    gt = get_gt_box_geometry_from_pybullet(
        body_id=box.body_id,
        lid_id=box.lid_id,
        flap_id=box.flap_id,
        cid=box.cid,
        lid_angle=lid_angle,
        flap_angle=flap_angle,
    )

    l1 = args.l1
    if l1 is None:
        l1 = float(args.key_local_z) * float(box.scaling)

    l2 = args.l2
    if l2 is None:
        l2 = gt['lid_length']

    key_pb, normal_pb, horizontal_pb = box.get_flap_keypoint_pose(
        lid_angle=lid_angle,
        flap_angle=flap_angle,
        estimate=False,
    )

    base_pose = base_pose_from_keypoint_pose(
        key_pb=key_pb,
        normal_pb=normal_pb,
        horizontal_pb=horizontal_pb,
        l1=l1,
        l2=l2,
        lid_angle=lid_angle,
        flap_angle=flap_angle,
    )

    base_position = base_pose["position"]
    draw_point(base_position, color=[1.0, 0.0, 1.0], size=0.05)
    draw_text("base", base_position + np.array([0.0, 0.0, 0.035]), color=(1.0, 0.0, 1.0))
    draw_vector(base_position, base_pose["horizontal"], color=[1.0, 0.0, 1.0])
    draw_vector(base_position, base_pose["normal"], color=[0.0, 1.0, 1.0])

    draw_point(key_pb, color=[0.0, 0.0, 1.0], size=0.035)
    draw_text("source key", np.asarray(key_pb) + np.array([0.0, 0.0, 0.035]), color=(0.0, 0.0, 1.0))
    draw_vector(key_pb, normal_pb, color=[0.0, 0.0, 1.0])

    angles = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step)
    last_key = None
    for angle_deg in angles:
        angle = np.deg2rad(angle_deg)
        key, normal, horizontal = keypoint_pose_from_base_pose(
            base_pose=base_pose,
            l1=l1,
            l2=l2,
            lid_angle=angle,
            flap_angle=angle,
        )
        draw_point(key, color=[0.0, 1.0, 0.0], size=0.025)
        draw_vector(key, normal, color=[0.0, 0.7, 0.0], scale=0.045)
        if last_key is not None:
            p.addUserDebugLine(
                np.asarray(last_key).tolist(),
                np.asarray(key).tolist(),
                lineColorRGB=[0.0, 0.6, 0.0],
                lineWidth=2,
                lifeTime=0,
            )
        last_key = key
        if abs(angle_deg % 30.0) < 1e-6:
            draw_text(f"{angle_deg:.0f}", key + np.array([0.0, 0.0, 0.02]), color=(0.0, 1.0, 0.0))
        time.sleep(args.sleep)

    print("source lid/flap angle deg:", args.lid_angle, args.flap_angle)
    print("l1:", l1)
    print("l2:", l2)
    print("source key_pb:", np.asarray(key_pb))
    print("base_position:", base_position)
    print("base_horizontal:", base_pose["horizontal"])
    print("base_normal:", base_pose["normal"])
    print("green points: reconstructed keypoints for lid_angle=flap_angle sweep")
    print("magenta point: recovered base position")

    while True:
        p.stepSimulation(physicsClientId=sim.cid)
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
