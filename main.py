import argparse
import json
import time
from pathlib import Path

import pybullet as p

from robot_sim.planner import KukaOmplPlanner


def load_config(path: str | Path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)

def run_unpack(planner: KukaOmplPlanner):
    print("[Demo] Unpacking a foldable box with 4 flaps ...")
    planner.unpack_box()


def run_pick_place(planner: KukaOmplPlanner, cfg: dict):
    print("[Demo] Pick-and-place demo ...")
    pick_cfg = cfg.get("pick_place", {})
    box_pos = pick_cfg.get("box_pos", [0.6, 0.0, 0.1])
    place_pos = pick_cfg.get("place_pos", [0.5, -0.35, 0.1])
    planner.pick_and_place(box_pos, place_pos)


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "unpack")
    gui = cfg.get("gui", True) if args.gui is None else args.gui
    box_base_pos = cfg.get("foldable_box_pos", [0.7, 0.0, 0.1])

    planner = KukaOmplPlanner(use_gui=gui, box_base_pos=box_base_pos)
    if mode == "unpack":
        run_unpack(planner)
    else:
        run_pick_place(planner, cfg)

    print("Press Ctrl+C to quit the GUI window.")
    while True:
        p.stepSimulation(physicsClientId=planner.cid)
        time.sleep(1.0 / 20.0)


if __name__ == "__main__":
    main()
