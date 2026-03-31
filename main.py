import argparse
import json
import time
from pathlib import Path

import pybullet as p

from scene import make_sim, physics_from_config
from tasks import FlapBoxTask, MailerBoxTask

def load_config(path: str | Path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Panda + foldable box demos")
    parser.add_argument(
        "--config",
        type=str,
        default="config/MailerBoxTask.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--mode",
        choices=["FlapBoxTask", "MailerBoxTask"],
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
        "--method", choices=['Sampling', "Iteration"], default=None, help="refinement method to use",
    )
    parser.add_argument(
        "--scaling", type=float, default=None, help="sacle of the box"
    )
    parser.add_argument(
        "--box_pos", type=list, default=None, help="postion of the box"
    )
    parser.add_argument(
        "--box_yaw", type=float, default=None, help="degree of the box"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "FlapBoxTask")
    gui = args.gui if args.gui is not None else cfg.get("gui", True) 
    cfg["method"] = args.method or cfg.get("method", "Iteration")
    cfg["scaling"] = args.scaling or cfg.get("scaling", 1.0)
    cfg["box_pos"] = args.box_pos or cfg.get("box_pos", [0.6, 0.1, 0.4])
    cfg["box_yaw"] = args.box_yaw or cfg.get("box_yaw", 0.0)

    sim = make_sim(gui=gui, physics=physics_from_config(cfg), load_ground_plane=True)

    task_map = {
        "FlapBoxTask": FlapBoxTask,
        "MailerBoxTask": MailerBoxTask,
    }
    task_cls = task_map.get(mode)
    if task_cls is None:
        raise NotImplementedError(f"{mode} not supported")
    task = task_cls(cfg, sim)
    task.setup_scene()
    task.run()

    # print("Press Ctrl+C to quit the GUI window.")
    # while True:
    #     p.stepSimulation(physicsClientId=sim.cid)
    #     time.sleep(1.0 / 20.0)

if __name__ == "__main__":
    main()
