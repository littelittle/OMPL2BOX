import argparse
import csv
import json
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import make_sim, physics_from_config
from tasks import MailerBoxTask


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_bool_list(text):
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


def parse_box_positions(text):
    positions = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x.strip()) for x in chunk.split(":") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Each box position must be x:y:z, got: {chunk}")
        positions.append(parts)
    return positions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def maybe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def maybe_min(values):
    values = [v for v in values if v is not None]
    return float(min(values)) if values else None


def maybe_max(values):
    values = [v for v in values if v is not None]
    return float(max(values)) if values else None


def run_trial(
    base_cfg,
    planner_impl,
    *,
    seed,
    scaling,
    box_pos,
    box_yaw,
    box_closed,
    resolution,
    coarse2fine_ratio,
):
    cfg = dict(base_cfg)
    cfg["mode"] = "MailerBoxTask"
    cfg["gui"] = False
    cfg["method"] = "Iteration"
    cfg["task_constraint_planner"] = planner_impl
    cfg["box_scaling"] = float(scaling)
    cfg["box_pos"] = list(box_pos)
    cfg["box_yaw"] = float(box_yaw)
    cfg["box_closed"] = bool(box_closed)

    set_seed(seed)
    wall_start = time.time()
    sim = make_sim(gui=False, physics=physics_from_config(cfg), load_ground_plane=True)

    row = {
        "planner_impl": planner_impl,
        "seed": seed,
        "scaling": scaling,
        "box_pos": json.dumps(list(box_pos)),
        "box_yaw": box_yaw,
        "box_closed": box_closed,
        "success": False,
        "planner_total_cost": None,
        "planner_max_edge_cost": None,
        "fine_max_edge_cost": None,
        "planner_time": None,
        "wall_time": None,
        "ik_attempts": None,
        "ik_successes": None,
        "candidate_duplicates": None,
        "seed_duplicates": None,
        "candidate_counts": None,
        "error": "",
    }

    try:
        task = MailerBoxTask(cfg, sim)
        task.setup_scene()
        degree_tuple_list = task.search_task_path(resolution=resolution)
        if degree_tuple_list is None:
            raise RuntimeError("search_task_path returned None")

        constraints = task.build_constraint_sequence(degree_tuple_list)
        metrics = task.tc_planner.solve_constraint_path(constraints, "Iteration")
        if not metrics.get("success", False):
            raise RuntimeError("solve_constraint_path returned success=False")

        planned_dict = metrics.get("planned_dict") or {}
        _, fine_max_edge_cost = task.get_traj_coarse2fine(
            metrics["path"],
            degree_tuple_list,
            coarse2fine_ratio,
        )

        debug = planned_dict.get("debug") or {}
        q_trajectory = getattr(task.tc_planner, "q_trajectory", [])

        row.update(
            {
                "success": True,
                "planner_total_cost": metrics.get("total_cost"),
                "planner_max_edge_cost": planned_dict.get(
                    "max_edge_cost",
                    metrics.get("max_edge_cost"),
                ),
                "fine_max_edge_cost": fine_max_edge_cost,
                "planner_time": metrics.get("time"),
                "ik_attempts": debug.get("ik_attempts"),
                "ik_successes": debug.get("ik_successes"),
                "candidate_duplicates": debug.get("candidate_duplicates"),
                "seed_duplicates": debug.get("seed_duplicates"),
                "candidate_counts": json.dumps([len(layer) for layer in q_trajectory]),
            }
        )
    except Exception as exc:
        row["error"] = repr(exc)
    finally:
        row["wall_time"] = time.time() - wall_start
        p.disconnect(physicsClientId=sim.cid)

    return row


def print_summary(rows):
    print("\n=== Summary by planner_impl ===")
    for planner_impl in sorted({row["planner_impl"] for row in rows}):
        impl_rows = [row for row in rows if row["planner_impl"] == planner_impl]
        successes = [row for row in impl_rows if row["success"]]
        print(
            f"planner={planner_impl:<3} trials={len(impl_rows):3d} "
            f"success={len(successes):3d}/{len(impl_rows):3d} "
            f"mean_planner_time={maybe_mean([r['planner_time'] for r in successes])} "
            f"mean_planner_max={maybe_mean([r['planner_max_edge_cost'] for r in successes])} "
            f"mean_fine_max={maybe_mean([r['fine_max_edge_cost'] for r in successes])}"
        )

    print("\n=== Paired new - old deltas by identical setting ===")
    by_key = {}
    for row in rows:
        key = (
            row["seed"],
            row["scaling"],
            row["box_pos"],
            row["box_yaw"],
            row["box_closed"],
        )
        by_key.setdefault(key, {})[row["planner_impl"]] = row

    deltas = []
    for key, pair in by_key.items():
        old = pair.get("old")
        new = pair.get("new")
        if not old or not new or not old["success"] or not new["success"]:
            continue
        deltas.append(
            {
                "planner_time": new["planner_time"] - old["planner_time"],
                "planner_max": new["planner_max_edge_cost"] - old["planner_max_edge_cost"],
                "fine_max": new["fine_max_edge_cost"] - old["fine_max_edge_cost"],
            }
        )

    if not deltas:
        print("No successful paired trials.")
        return

    for field in ("planner_time", "planner_max", "fine_max"):
        values = [delta[field] for delta in deltas]
        print(
            f"{field}: mean_delta={maybe_mean(values):.6f} "
            f"min_delta={maybe_min(values):.6f} max_delta={maybe_max(values):.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark old vs new TaskConstraintPlanner implementations."
    )
    parser.add_argument("--config", default="config/MailerBoxTask.json")
    parser.add_argument("--planner-impls", default="old,new")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--scalings", default="1.0")
    parser.add_argument("--box-positions", default="0.6:0.1:0.35")
    parser.add_argument("--box-yaws", default="0.0")
    parser.add_argument("--closed-states", default="true")
    parser.add_argument("--nogui", action="store_true", help="Accepted for parity with main.py; benchmark always runs DIRECT.")
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--coarse2fine-ratio", type=int, default=5)
    parser.add_argument(
        "--output",
        default="data/benchmark_task_constraint_planner_old_new.csv",
    )
    args = parser.parse_args()

    base_cfg = load_json(args.config)
    planner_impls = [x.strip() for x in args.planner_impls.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scalings = parse_float_list(args.scalings)
    box_positions = parse_box_positions(args.box_positions)
    box_yaws = parse_float_list(args.box_yaws)
    closed_states = parse_bool_list(args.closed_states)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "planner_impl",
        "seed",
        "scaling",
        "box_pos",
        "box_yaw",
        "box_closed",
        "success",
        "planner_total_cost",
        "planner_max_edge_cost",
        "fine_max_edge_cost",
        "planner_time",
        "wall_time",
        "ik_attempts",
        "ik_successes",
        "candidate_duplicates",
        "seed_duplicates",
        "candidate_counts",
        "error",
    ]

    rows = []
    trials = list(
        product(
            planner_impls,
            seeds,
            scalings,
            box_positions,
            box_yaws,
            closed_states,
        )
    )
    print(f"Running {len(trials)} no-GUI trials...")

    for planner_impl, seed, scaling, box_pos, box_yaw, box_closed in trials:
        row = run_trial(
            base_cfg,
            planner_impl,
            seed=seed,
            scaling=scaling,
            box_pos=box_pos,
            box_yaw=box_yaw,
            box_closed=box_closed,
            resolution=args.resolution,
            coarse2fine_ratio=args.coarse2fine_ratio,
        )
        rows.append(row)
        print(
            f"planner={planner_impl:<3} seed={seed:<3d} yaw={box_yaw:<6g} "
            f"closed={box_closed} success={row['success']} "
            f"planner_time={row['planner_time']} planner_max={row['planner_max_edge_cost']} "
            f"fine_max={row['fine_max_edge_cost']} error={row['error']}"
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark CSV to: {output_path}")
    print_summary(rows)


if __name__ == "__main__":
    main()
