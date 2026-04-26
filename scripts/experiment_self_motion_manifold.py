from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


POSE_COLUMNS = (
    "target-POS_X",
    "target-POS_Y",
    "target-POS_Z",
    "target-ROT_X",
    "target-ROT_Y",
    "target-ROT_Z",
    "target-ROT_W",
)

Q_COLUMNS = [f"q{i}" for i in range(1, 8)]
SEED_COLUMNS = [f"seed_q{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    orn: np.ndarray
    time: Optional[float]


@dataclass(frozen=True)
class RobotModel:
    cid: int
    robot_id: int
    joint_indices: list[int]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    ee_link_index: int
    finger_joint_indices: list[int]


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray
    pos_err: float
    orn_err: float


@dataclass(frozen=True)
class PcaBasis:
    mean: np.ndarray
    components: np.ndarray
    explained: np.ndarray
    source: str
    coord_limits: Optional[np.ndarray] = None


def load_ee_pose_csv(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in POSE_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing pose columns: {missing}")

        for row in reader:
            pos = np.array([float(row[col]) for col in POSE_COLUMNS[:3]], dtype=float)
            orn = np.array([float(row[col]) for col in POSE_COLUMNS[3:]], dtype=float)
            norm = float(np.linalg.norm(orn))
            if norm <= 1e-12:
                raise ValueError(f"Invalid zero quaternion in row {len(poses)} of {path}")
            orn = orn / norm
            time_value = float(row["time"]) if "time" in row and row["time"] != "" else None
            poses.append(Pose(pos=pos, orn=orn, time=time_value))

    if not poses:
        raise ValueError(f"{path} does not contain any waypoint rows")
    return poses


def find_default_waypoint_csv(repo_root: Path) -> Path:
    waypoint_dir = repo_root / "data" / "waypoints"
    candidates = sorted(waypoint_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No waypoint CSV files found under {waypoint_dir}")
    return candidates[0]


def connect_panda(gui: bool) -> RobotModel:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
        physicsClientId=cid,
    )

    joint_indices: list[int] = []
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    finger_joint_indices: list[int] = []
    ee_link_index: Optional[int] = None

    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, joint_index, physicsClientId=cid)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        is_finger = "finger" in joint_name or "finger" in link_name

        if link_name == "panda_grasptarget":
            ee_link_index = joint_index

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            if is_finger:
                finger_joint_indices.append(joint_index)
                continue

            joint_indices.append(joint_index)
            lower = float(info[8])
            upper = float(info[9])
            if upper < lower or (lower == 0.0 and upper == -1.0):
                lower, upper = -math.pi, math.pi
            lower_limits.append(lower)
            upper_limits.append(upper)

    if ee_link_index is None:
        raise RuntimeError("Could not find panda_grasptarget link in franka_panda/panda.urdf")
    if len(joint_indices) != 7:
        raise RuntimeError(f"Expected 7 Panda arm joints, got {len(joint_indices)}")

    for finger_joint in finger_joint_indices:
        p.resetJointState(
            robot_id,
            finger_joint,
            targetValue=0.04,
            targetVelocity=0.0,
            physicsClientId=cid,
        )

    return RobotModel(
        cid=cid,
        robot_id=robot_id,
        joint_indices=joint_indices,
        lower_limits=np.array(lower_limits, dtype=float),
        upper_limits=np.array(upper_limits, dtype=float),
        ee_link_index=ee_link_index,
        finger_joint_indices=finger_joint_indices,
    )


def reset_arm_config(robot: RobotModel, q: Iterable[float]) -> None:
    for joint_index, value in zip(robot.joint_indices, q):
        p.resetJointState(
            robot.robot_id,
            joint_index,
            targetValue=float(value),
            targetVelocity=0.0,
            physicsClientId=robot.cid,
        )


def get_ee_pose(robot: RobotModel) -> tuple[np.ndarray, np.ndarray]:
    link_state = p.getLinkState(
        robot.robot_id,
        robot.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=robot.cid,
    )
    return np.array(link_state[4], dtype=float), np.array(link_state[5], dtype=float)


def wrap_to_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def joint_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    return float(np.linalg.norm(wrap_to_pi(np.asarray(q_b) - np.asarray(q_a))))


def wrap_into_limits(
    q: np.ndarray,
    q_ref: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    qn = np.array(q, dtype=float, copy=True)
    for i in range(len(qn)):
        period = 2.0 * math.pi
        low = float(lower_limits[i])
        high = float(upper_limits[i])
        k_min = math.ceil((low - qn[i]) / period)
        k_max = math.floor((high - qn[i]) / period)
        if k_min <= k_max:
            candidates = [qn[i] + period * k for k in range(k_min, k_max + 1)]
            qn[i] = min(candidates, key=lambda value: abs(value - float(q_ref[i])))
    return qn


def quaternion_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_a = q_a / np.linalg.norm(q_a)
    q_b = q_b / np.linalg.norm(q_b)
    dot = float(abs(np.dot(q_a, q_b)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.acos(dot))


def solve_pybullet_ik(
    robot: RobotModel,
    pose: Pose,
    q_seed: np.ndarray,
    *,
    pos_tol: float,
    orn_tol: float,
    max_iterations: int,
    residual_threshold: float,
) -> Optional[IkResult]:
    q_seed = np.asarray(q_seed, dtype=float)
    reset_arm_config(robot, q_seed)

    ik = p.calculateInverseKinematics(
        robot.robot_id,
        robot.ee_link_index,
        pose.pos.tolist(),
        pose.orn.tolist(),
        lowerLimits=robot.lower_limits.tolist(),
        upperLimits=robot.upper_limits.tolist(),
        jointRanges=(robot.upper_limits - robot.lower_limits).tolist(),
        restPoses=q_seed.tolist(),
        physicsClientId=robot.cid,
        maxNumIterations=max_iterations,
        residualThreshold=residual_threshold,
    )
    if len(ik) < len(robot.joint_indices):
        return None

    q_candidate = wrap_into_limits(
        np.array(ik[: len(robot.joint_indices)], dtype=float),
        q_seed,
        robot.lower_limits,
        robot.upper_limits,
    )
    if np.any(q_candidate < robot.lower_limits - 1e-6) or np.any(q_candidate > robot.upper_limits + 1e-6):
        return None

    reset_arm_config(robot, q_candidate)
    ee_pos, ee_orn = get_ee_pose(robot)
    pos_err = float(np.linalg.norm(ee_pos - pose.pos))
    orn_err = quaternion_angle_error(ee_orn, pose.orn)
    if pos_err > pos_tol or orn_err > orn_tol:
        return None

    return IkResult(q=q_candidate, pos_err=pos_err, orn_err=orn_err)


def is_duplicate_q(q: np.ndarray, existing: list[dict], threshold: float) -> bool:
    return any(joint_distance(q, row["q"]) < threshold for row in existing)


def sample_initial_self_motion_targets(
    robot: RobotModel,
    first_pose: Pose,
    *,
    num_seeds: int,
    rng: np.random.Generator,
    dedup_distance: float,
    pos_tol: float,
    orn_tol: float,
    max_iterations: int,
    residual_threshold: float,
    progress_every: int,
) -> list[dict]:
    targets: list[dict] = []
    for seed_index in range(num_seeds):
        q_seed = rng.uniform(robot.lower_limits, robot.upper_limits)
        result = solve_pybullet_ik(
            robot,
            first_pose,
            q_seed,
            pos_tol=pos_tol,
            orn_tol=orn_tol,
            max_iterations=max_iterations,
            residual_threshold=residual_threshold,
        )
        if result is not None and not is_duplicate_q(result.q, targets, dedup_distance):
            targets.append(
                {
                    "target_index": len(targets),
                    "source_seed_index": seed_index,
                    "q_seed": q_seed,
                    "q": result.q,
                    "pos_err": result.pos_err,
                    "orn_err": result.orn_err,
                }
            )

        if progress_every > 0 and (seed_index + 1) % progress_every == 0:
            print(f"sampled {seed_index + 1}/{num_seeds} seeds, unique IK targets={len(targets)}")

    return targets


def padded_limits(coords: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    lows = coords.min(axis=0)
    highs = coords.max(axis=0)
    spans = highs - lows
    fallback = np.maximum(np.abs(highs), 1.0) * pad_ratio
    padding = np.where(spans > 1e-12, spans * pad_ratio, fallback)
    return np.column_stack([lows - padding, highs + padding])


def fit_pca_basis(values: np.ndarray, dims: int = 3, source: str = "self") -> PcaBasis:
    if values.ndim != 2:
        raise ValueError("PCA input must be a 2D array")
    if values.shape[0] == 0:
        raise ValueError("PCA input is empty")

    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:dims]

    if components.shape[0] < dims:
        components = np.pad(components, ((0, dims - components.shape[0]), (0, 0)))

    variance = singular_values**2
    total = float(variance.sum())
    explained = variance[:dims] / total if total > 1e-12 else np.zeros(min(dims, len(variance)))
    if explained.shape[0] < dims:
        explained = np.pad(explained, (0, dims - explained.shape[0]))
    coords = centered @ components[:dims].T
    if coords.shape[1] < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])))
    return PcaBasis(
        mean=mean,
        components=components,
        explained=explained,
        source=source,
        coord_limits=padded_limits(coords),
    )


def project_with_pca(values: np.ndarray, basis: PcaBasis, dims: int = 3) -> np.ndarray:
    coords = (values - basis.mean) @ basis.components[:dims].T
    if coords.shape[1] < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])))
    return coords


def compute_pca(values: np.ndarray, dims: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = fit_pca_basis(values, dims=dims, source="self")
    coords = project_with_pca(values, basis, dims=dims)
    return coords, basis.components, basis.explained


def load_reference_pca_basis(path: Path, dims: int = 3) -> PcaBasis:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    if "target_q" not in data.files:
        raise ValueError(f"{path} does not contain target_q")

    target_q = np.asarray(data["target_q"], dtype=float)
    if "valid" in data.files:
        valid = np.asarray(data["valid"], dtype=bool)
        if valid.shape[0] != target_q.shape[0]:
            raise ValueError(f"{path} has mismatched target_q and valid lengths")
        target_q = target_q[valid]

    if len(target_q) < 2:
        raise ValueError(f"{path} has fewer than 2 valid target_q rows")

    return fit_pca_basis(target_q, dims=dims, source=str(path))


def save_pca_basis(path: Path, basis: PcaBasis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=basis.mean,
        components=basis.components,
        explained=basis.explained,
        source=np.asarray(basis.source),
        coord_limits=basis.coord_limits,
    )


def resolve_default_reference_pca(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "paper" / "ik_seed_manifold_output" / "ik_seed_results.npz",
        repo_root / "paper" / "ik_seed_waypoints_output_10" / "ik_seed_results.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def select_targets_for_tracking(
    targets: list[dict],
    *,
    target_stride: int,
    max_tracks: Optional[int],
) -> list[dict]:
    if not targets:
        return []

    q_values = np.array([row["q"] for row in targets], dtype=float)
    coords, _, _ = compute_pca(q_values, dims=1)
    order = np.argsort(coords[:, 0])
    ordered = [targets[int(i)] for i in order]

    if target_stride > 1:
        ordered = ordered[::target_stride]

    if max_tracks is not None and max_tracks > 0 and len(ordered) > max_tracks:
        keep = np.linspace(0, len(ordered) - 1, num=max_tracks, dtype=int)
        ordered = [ordered[int(i)] for i in keep]

    return ordered


def track_pose_sequence(
    robot: RobotModel,
    poses: list[Pose],
    initial_targets: list[dict],
    *,
    max_step_norm: float,
    pos_tol: float,
    orn_tol: float,
    max_iterations: int,
    residual_threshold: float,
) -> tuple[list[dict], list[dict]]:
    track_rows: list[dict] = []
    summary_rows: list[dict] = []

    for track_id, target in enumerate(initial_targets):
        q_current = np.array(target["q"], dtype=float)
        max_observed_step = 0.0
        stop_reason = "completed"
        stopped_at_waypoint: Optional[int] = None
        accepted = 1

        track_rows.append(
            {
                "track_id": track_id,
                "target_index": target["target_index"],
                "source_seed_index": target["source_seed_index"],
                "waypoint_index": 0,
                "q": q_current.copy(),
                "pos_err": target["pos_err"],
                "orn_err": target["orn_err"],
                "step_norm": 0.0,
            }
        )

        for waypoint_index, pose in enumerate(poses[1:], start=1):
            result = solve_pybullet_ik(
                robot,
                pose,
                q_current,
                pos_tol=pos_tol,
                orn_tol=orn_tol,
                max_iterations=max_iterations,
                residual_threshold=residual_threshold,
            )
            if result is None:
                stop_reason = "ik_fail"
                stopped_at_waypoint = waypoint_index
                break

            step_norm = joint_distance(q_current, result.q)
            if step_norm > max_step_norm:
                stop_reason = "jump"
                stopped_at_waypoint = waypoint_index
                break

            max_observed_step = max(max_observed_step, step_norm)
            q_current = result.q
            accepted += 1
            track_rows.append(
                {
                    "track_id": track_id,
                    "target_index": target["target_index"],
                    "source_seed_index": target["source_seed_index"],
                    "waypoint_index": waypoint_index,
                    "q": q_current.copy(),
                    "pos_err": result.pos_err,
                    "orn_err": result.orn_err,
                    "step_norm": step_norm,
                }
            )

        summary_rows.append(
            {
                "track_id": track_id,
                "target_index": target["target_index"],
                "source_seed_index": target["source_seed_index"],
                "accepted_waypoints": accepted,
                "stopped_at_waypoint": "" if stopped_at_waypoint is None else stopped_at_waypoint,
                "stop_reason": stop_reason,
                "max_step_norm": max_observed_step,
            }
        )

        if (track_id + 1) % 25 == 0 or track_id + 1 == len(initial_targets):
            print(f"tracked {track_id + 1}/{len(initial_targets)} initial targets")

    return track_rows, summary_rows


def write_initial_targets(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["target_index", "source_seed_index", *SEED_COLUMNS, *Q_COLUMNS, "pos_err", "orn_err"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "target_index": row["target_index"],
                "source_seed_index": row["source_seed_index"],
                "pos_err": row["pos_err"],
                "orn_err": row["orn_err"],
            }
            out.update({col: float(value) for col, value in zip(SEED_COLUMNS, row["q_seed"])})
            out.update({col: float(value) for col, value in zip(Q_COLUMNS, row["q"])})
            writer.writerow(out)


def write_track_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "target_index",
        "source_seed_index",
        "waypoint_index",
        *Q_COLUMNS,
        "pc1",
        "pc2",
        "pc3",
        "pos_err",
        "orn_err",
        "step_norm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "track_id": row["track_id"],
                "target_index": row["target_index"],
                "source_seed_index": row["source_seed_index"],
                "waypoint_index": row["waypoint_index"],
                "pc1": row.get("pc", [np.nan, np.nan, np.nan])[0],
                "pc2": row.get("pc", [np.nan, np.nan, np.nan])[1],
                "pc3": row.get("pc", [np.nan, np.nan, np.nan])[2],
                "pos_err": row["pos_err"],
                "orn_err": row["orn_err"],
                "step_norm": row["step_norm"],
            }
            out.update({col: float(value) for col, value in zip(Q_COLUMNS, row["q"])})
            writer.writerow(out)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "target_index",
        "source_seed_index",
        "accepted_waypoints",
        "stopped_at_waypoint",
        "stop_reason",
        "max_step_norm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pca_2d(
    path: Path,
    rows: list[dict],
    explained: np.ndarray,
    title: str,
    coord_limits: Optional[np.ndarray] = None,
) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["track_id"])].append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    for track in grouped.values():
        track = sorted(track, key=lambda row: row["waypoint_index"])
        xy = np.array([row["pc"][:2] for row in track], dtype=float)
        if len(xy) > 1:
            ax.plot(xy[:, 0], xy[:, 1], color="0.65", linewidth=0.6, alpha=0.35, zorder=1)

    coords = np.array([row["pc"][:2] for row in rows], dtype=float)
    steps = np.array([row["waypoint_index"] for row in rows], dtype=float)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=steps, s=10, cmap="viridis", zorder=2)
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}%)")
    ax.set_title(title)
    if coord_limits is not None and coord_limits.shape[0] >= 2:
        ax.set_xlim(float(coord_limits[0, 0]), float(coord_limits[0, 1]))
        ax.set_ylim(float(coord_limits[1, 0]), float(coord_limits[1, 1]))
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="waypoint index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca_3d(
    path: Path,
    rows: list[dict],
    explained: np.ndarray,
    title: str,
    coord_limits: Optional[np.ndarray] = None,
) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["track_id"])].append(row)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for track in grouped.values():
        track = sorted(track, key=lambda row: row["waypoint_index"])
        xyz = np.array([row["pc"][:3] for row in track], dtype=float)
        if len(xyz) > 1:
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.65", linewidth=0.6, alpha=0.35)

    coords = np.array([row["pc"][:3] for row in rows], dtype=float)
    steps = np.array([row["waypoint_index"] for row in rows], dtype=float)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=steps, s=9, cmap="viridis")
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2] * 100.0:.1f}%)")
    ax.set_title(title)
    if coord_limits is not None and coord_limits.shape[0] >= 3:
        ax.set_xlim(float(coord_limits[0, 0]), float(coord_limits[0, 1]))
        ax.set_ylim(float(coord_limits[1, 0]), float(coord_limits[1, 1]))
        ax.set_zlim(float(coord_limits[2, 0]), float(coord_limits[2, 1]))
    fig.colorbar(scatter, ax=ax, label="waypoint index", shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def positive_int_or_none(value: str) -> Optional[int]:
    if value.lower() in ("none", "all", "0"):
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_waypoints = find_default_waypoint_csv(repo_root)

    parser = argparse.ArgumentParser(
        description=(
            "Sample a self-motion manifold for the first EE pose in a waypoint CSV, "
            "track the remaining EE poses with PyBullet IK, and visualize q-space with PCA."
        )
    )
    parser.add_argument("--waypoints", type=str, default=str(default_waypoints))
    parser.add_argument("--output-dir", type=str, default="data/self_motion_manifold")
    parser.add_argument("--num-seeds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--max-tracks", type=positive_int_or_none, default=200)
    parser.add_argument("--max-waypoints", type=positive_int_or_none, default=None)
    parser.add_argument("--dedup-distance", type=float, default=0.04)
    parser.add_argument("--max-step-norm", type=float, default=0.8)
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--orn-tol", type=float, default=0.05)
    parser.add_argument("--ik-iterations", type=int, default=1000)
    parser.add_argument("--residual-threshold", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--reference-pca-npz",
        type=str,
        default=None,
        help=(
            "Optional ik_seed_results.npz from paper/analyze_pybullet_ik_seed_manifold.py. "
            "When set, tracked q values are projected with that waypoint PCA basis instead "
            "of fitting PCA on this experiment's q values."
        ),
    )
    parser.add_argument(
        "--use-default-reference-pca",
        action="store_true",
        help=(
            "Use paper/ik_seed_manifold_output/ik_seed_results.npz if present, otherwise "
            "paper/ik_seed_waypoints_output_10/ik_seed_results.npz."
        ),
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waypoint_path = Path(args.waypoints)
    output_root = Path(args.output_dir) / waypoint_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    poses = load_ee_pose_csv(waypoint_path)
    if args.max_waypoints is not None:
        poses = poses[: args.max_waypoints]
    if len(poses) < 2:
        raise ValueError("Need at least two EE poses to run the tracking experiment")

    rng = np.random.default_rng(args.seed)
    robot = connect_panda(gui=args.gui)
    try:
        print(f"Loaded {len(poses)} EE poses from {waypoint_path}")
        print(f"Sampling {args.num_seeds} q seeds for the first pose")
        initial_targets = sample_initial_self_motion_targets(
            robot,
            poses[0],
            num_seeds=args.num_seeds,
            rng=rng,
            dedup_distance=args.dedup_distance,
            pos_tol=args.pos_tol,
            orn_tol=args.orn_tol,
            max_iterations=args.ik_iterations,
            residual_threshold=args.residual_threshold,
            progress_every=args.progress_every,
        )
        if not initial_targets:
            raise RuntimeError("No valid first-pose IK targets were found")

        selected_targets = select_targets_for_tracking(
            initial_targets,
            target_stride=max(1, args.target_stride),
            max_tracks=args.max_tracks,
        )
        print(f"Unique first-pose IK targets: {len(initial_targets)}")
        print(f"Tracking from {len(selected_targets)} selected initial targets")

        track_rows, summary_rows = track_pose_sequence(
            robot,
            poses,
            selected_targets,
            max_step_norm=args.max_step_norm,
            pos_tol=args.pos_tol,
            orn_tol=args.orn_tol,
            max_iterations=args.ik_iterations,
            residual_threshold=args.residual_threshold,
        )
    finally:
        p.disconnect(physicsClientId=robot.cid)

    q_values = np.array([row["q"] for row in track_rows], dtype=float)
    reference_pca_path = Path(args.reference_pca_npz) if args.reference_pca_npz else None
    if reference_pca_path is None and args.use_default_reference_pca:
        reference_pca_path = resolve_default_reference_pca(Path(__file__).resolve().parents[1])

    if reference_pca_path is not None:
        basis = load_reference_pca_basis(reference_pca_path, dims=3)
        output_prefix = "tracked_q_reference_pca"
        plot_title = "Tracked IK solutions projected onto reference waypoint PCA"
    else:
        basis = fit_pca_basis(q_values, dims=3, source="tracked_q")
        output_prefix = "tracked_q_pca"
        plot_title = "Tracked IK solutions projected by PCA"

    coords = project_with_pca(q_values, basis, dims=3)
    explained = basis.explained
    for row, coord in zip(track_rows, coords):
        row["pc"] = coord

    write_initial_targets(output_root / "initial_self_motion_targets.csv", initial_targets)
    write_track_rows(output_root / f"{output_prefix}.csv", track_rows)
    write_summary(output_root / "track_summary.csv", summary_rows)
    plot_pca_2d(
        output_root / f"{output_prefix}_2d.png",
        track_rows,
        explained,
        plot_title,
        basis.coord_limits,
    )
    plot_pca_3d(
        output_root / f"{output_prefix}_3d.png",
        track_rows,
        explained,
        plot_title,
        basis.coord_limits,
    )
    save_pca_basis(output_root / f"{output_prefix}_basis.npz", basis)

    completed = sum(1 for row in summary_rows if row["stop_reason"] == "completed")
    longest = max(int(row["accepted_waypoints"]) for row in summary_rows)
    print(f"Completed tracks: {completed}/{len(summary_rows)}")
    print(f"Longest accepted waypoint count: {longest}/{len(poses)}")
    print(f"PCA source: {basis.source}")
    print(f"PCA explained variance: {explained[0]:.4f}, {explained[1]:.4f}, {explained[2]:.4f}")
    print(f"Saved experiment outputs to: {output_root}")


if __name__ == "__main__":
    main()
