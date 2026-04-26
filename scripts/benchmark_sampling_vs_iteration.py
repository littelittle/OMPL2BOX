import argparse
import csv
import json
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pybullet as p

from scene import make_sim, physics_from_config
from tasks import MailerBoxTask


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_bool_list(text: str):
    values = []
    for item in text.split(","):
        val = item.strip().lower()
        if val in ("1", "true", "t", "yes", "y", "closed"):
            values.append(True)
        elif val in ("0", "false", "f", "no", "n", "open"):
            values.append(False)
        else:
            raise ValueError(f"Unsupported boolean value: {item}")
    return values


def parse_box_positions(text: str):
    positions = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x.strip()) for x in chunk.split(":") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Each box_pos must have exactly 3 values (x:y:z), got: {chunk}")
        positions.append(parts)
    return positions


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def run_trial(base_cfg, method, scaling, box_pos, box_yaw, box_closed, seed):
    cfg = dict(base_cfg)
    cfg["mode"] = "MailerBoxTask"
    cfg["gui"] = False
    cfg["method"] = method
    cfg["box_scaling"] = scaling
    cfg["box_pos"] = list(box_pos)
    cfg["box_yaw"] = box_yaw
    cfg["box_closed"] = box_closed

    set_seed(seed)

    sim = make_sim(gui=False, physics=physics_from_config(cfg), load_ground_plane=True)
    try:
        task = MailerBoxTask(cfg, sim)
        task.setup_scene()
        metrics = task.compute_quality_metrics()
    except Exception as exc:
        metrics = {
            "success": False,
            "total_cost": None,
            "max_edge_cost": None,
            "error": str(exc),
        }
    finally:
        p.disconnect(physicsClientId=sim.cid)

    return metrics


def safe_mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["method"]].append(row)

    print("\n=== Summary by method ===")
    for method, rows in grouped.items():
        success_rows = [r for r in rows if r["success"]]
        total_costs = [r["total_cost"] for r in success_rows if r["total_cost"] is not None]
        max_edge_costs = [r["max_edge_cost"] for r in success_rows if r["max_edge_cost"] is not None]
        success_rate = len(success_rows) / len(rows) if rows else 0.0
        print(
            f"method={method:<10} trials={len(rows):4d} success_rate={success_rate:.3f} "
            f"mean_total_cost={safe_mean(total_costs):.6f} mean_max_edge_cost={safe_mean(max_edge_costs):.6f}"
        )



def main():
    parser = argparse.ArgumentParser(description="Benchmark MailerBoxTask planning quality across Sampling, Iteration, and optional RRT runs.")
    parser.add_argument("--config", type=str, default="config/MailerBoxTask.json")
    parser.add_argument("--methods", type=str, default="Sampling,Iteration,Greedy", help="Comma-separated methods, e.g. Sampling,Iteration,RRT")
    parser.add_argument("--scalings", type=str, default="1.0, 1.2")
    parser.add_argument("--box-positions", type=str, default="0.6:0.1:0.4;0.65:0.15:0.45")
    parser.add_argument("--box-yaws", type=str, default="0.0, 15,-15, 30,-30")
    parser.add_argument("--closed-states", type=str, default="true,false")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--output", type=str, default="data/benchmark_sampling_vs_iteration_vs_greedy.csv")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    base_cfg = load_json(cfg_path)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    scalings = parse_float_list(args.scalings)
    box_positions = parse_box_positions(args.box_positions)
    box_yaws = parse_float_list(args.box_yaws)
    closed_states = parse_bool_list(args.closed_states)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "setting",
        "scaling",
        "box_pos",
        "box_yaw",
        "box_closed",
        "success",
        "total_cost",
        "total_time",
        "max_edge_cost",
        "seed",
        "error",
    ]

    results = []
    all_trials = list(product(methods, scalings, box_positions, box_yaws, closed_states, seeds))
    print(f"Running {len(all_trials)} trials...")

    for method, scaling, box_pos, box_yaw, box_closed, seed in all_trials:
        metrics = run_trial(
            base_cfg=base_cfg,
            method=method,
            scaling=scaling,
            box_pos=box_pos,
            box_yaw=box_yaw,
            box_closed=box_closed,
            seed=seed,
        )
        setting = {
            "scaling": scaling,
            "box_pos": list(box_pos),
            "box_yaw": box_yaw,
            "box_closed": box_closed,
        }
        row = {
            "method": method,
            "setting": json.dumps(setting, sort_keys=True),
            "scaling": scaling,
            "box_pos": json.dumps(list(box_pos)),
            "box_yaw": box_yaw,
            "box_closed": box_closed,
            "success": bool(metrics.get("success", False)),
            "total_cost": metrics.get("total_cost"),
            "total_time": metrics.get("time"),
            "max_edge_cost": metrics.get("max_edge_cost"),
            "seed": seed,
            "error": metrics.get("error", ""),
        }
        results.append(row)
        print(
            f"method={method:<10} seed={seed:<4d} scaling={scaling:.3f} yaw={box_yaw:.3f} "
            f"closed={box_closed} success={row['success']} total_cost={row['total_cost']} max_edge_cost={row['max_edge_cost']}"
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark CSV to: {output_path}")
    summarize(results)


if __name__ == "__main__":
    main()
