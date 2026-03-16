import argparse
import json
import time
from pathlib import Path
import random

import pybullet as p

from robot_sim import (
    KukaOmplPlanner,
    PandaGripperPlanner,
    FoldableBox,
    make_sim,
    physics_from_config
)

def load_config(path: str | Path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)

def run_pick_place(planner: KukaOmplPlanner, cfg: dict):
    print("[Demo] Pick-and-place demo ...")
    pick_cfg = cfg.get("pick_place", {})
    box_pos = pick_cfg.get("box_pos", [0.6, 0.0, 0.1])
    place_pos = pick_cfg.get("place_pos", [0.5, -0.35, 0.1])
    planner.pick_and_place(box_pos, place_pos)

def create_pedestal(cid, center_xy, size_xy=(0.40, 0.34), height=0.10, rgba=(0.6, 0.6, 0.6, 1.0)):
    """创建一个静态台子（mass=0），顶面高度=height，底面贴地 z=0。"""
    hx, hy, hz = size_xy[0] * 0.5, size_xy[1] * 0.5, height * 0.5
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba, physicsClientId=cid)
    pedestal_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[center_xy[0], center_xy[1], hz],  # z = height/2
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=cid,
    )
    return pedestal_id

def main():
    parser = argparse.ArgumentParser(description="KUKA + foldable box demos")
    parser.add_argument(
        "--config",
        type=str,
        default="config/defaults.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--mode",
        choices=["unpack", "pick-place"],
        default=None,
        help="Which demo to run (overrides config if set)",
    )
    parser.add_argument(
        "--gui", action="store_true", default=None, help="Enable PyBullet GUI"
    )
    parser.add_argument(
        "--nogui", dest="gui", action="store_false", help="Run in DIRECT mode"
    )
    parser.add_argument(
        "--robot",
        choices=["kuka", "panda"],
        default="panda",
        help="Robot model to use",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "unpack")
    gui = cfg.get("gui", True) if args.gui is None else args.gui
    robot_name = args.robot or cfg.get("robot", "kuka")
    box_base_pos = cfg.get("foldable_box_pos", [0.7, 0.0, 0.1])
    box_base_orn = cfg.get("foldable_box_orn", p.getQuaternionFromEuler([0, 0, random.uniform(-0.3, 0.3)])) # 

    sim = make_sim(gui=gui, physics=physics_from_config(cfg), load_ground_plane=True)
    cid = sim.cid

    pedestal_h = 0.20
    pedestal_id = create_pedestal(cid, center_xy=[box_base_pos[0], box_base_pos[1]], height=pedestal_h)

    box_half_h = 0.07  # box base_half_extents[2] 就是 0.1 :contentReference[oaicite:3]{index=3}
    box_base_pos = [box_base_pos[0], box_base_pos[1], pedestal_h + box_half_h]

    # create the task(box)
    foldable_box = FoldableBox(base_pos=box_base_pos, base_orn=box_base_orn, cid=cid)
    box_id = foldable_box.body_id       # for collision detection

    if robot_name == "panda":
        planner = PandaGripperPlanner(oracle_function=foldable_box.get_flap_keypoint_pose, cid=cid, box_id=box_id, plane_id=sim.plane_id)
    else:
        raise NotImplementedError("ONLY SUPPORT PANDA FRANKA NOW")

    if mode == "unpack":
        planner.close_double_flap()
    else:
        raise NotImplementedError("ONLY SUPPORT CLOSE DOUBLE FLAP NOW")

    print("Press Ctrl+C to quit the GUI window.")
    while True:
        p.stepSimulation(physicsClientId=planner.cid)
        time.sleep(1.0 / 20.0)

if __name__ == "__main__":
    main()
