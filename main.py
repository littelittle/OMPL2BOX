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
        "--method", choices=['Sampling', "Iteration"], default=None, help="refinement method to use"
    )
    parser.add_argument(
        "--box_scaling", type=float, default=None, help="sacle of the box"
    )
    parser.add_argument(
        "--box_pos", type=list, default=None, help="postion of the box"
    )
    parser.add_argument(
        "--box_yaw", type=float, default=None, help="degree of the box"
    )
    parser.add_argument(
        "--box_file_path", type=str, default=None, help="the file path of the loaded box"
    )
    parser.add_argument(
        "--box_closed", action="store_true", help="Set box as closed"
    )
    parser.add_argument(
        "--box_open", dest="box_closed", action="store_false", help="Set box as open"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "FlapBoxTask")
    gui = args.gui if args.gui is not None else cfg.get("gui", True) 
    cfg["method"] = args.method or cfg.get("method", "Iteration")
    cfg["box_scaling"] = args.box_scaling or cfg.get("box_scaling", 1.0)
    cfg["box_pos"] = args.box_pos or cfg.get("box_pos", [0.6, 0.1, 0.4])
    cfg["box_yaw"] = args.box_yaw or cfg.get("box_yaw", 0.0)
    cfg["box_file_path"] = args.box_file_path or cfg.get('box_file_path', "assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf")
    cfg["box_closed"] = args.box_closed if args.box_closed is not None else cfg.get('box_closed', True)    

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
