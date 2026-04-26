from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from planners.grip_planner import PandaGripperPlanner
from utils.vector import WaypointConstraint, quat_from_normal_and_yaw
from utils.yaw_dp import Q_RESET_SEEDS, uniform_q_sampling


QType = Sequence[float]


class ConstraintSequenceGreedyPlanner:
    def __init__(
        self,
        robot_planner: PandaGripperPlanner,
        *,
        joint_weights: Optional[QType] = None,
        yaw_samples: int = 5,
        random_seed_count: int = 10,
    ):
        self.robot = robot_planner
        self.joint_weights = None if joint_weights is None else np.asarray(joint_weights, dtype=float)
        self.yaws = np.linspace(math.radians(30.0), math.radians(150.0), int(yaw_samples)).tolist()
        self.random_seed_count = int(random_seed_count)
        self.base_q_seeds = [
            Q_RESET_SEEDS["home"],
            # Q_RESET_SEEDS["left_relaxed"],
            # Q_RESET_SEEDS["right_relaxed"],
            Q_RESET_SEEDS["left_elbow_out"],
            Q_RESET_SEEDS["right_elbow_out"],
            # Q_RESET_SEEDS["forward_mid"],
        ]

    def solve(self, constraints: List[WaypointConstraint]):
        if not constraints:
            return self._failure(None)

        self.robot.open_gripper()
        start_q = np.asarray(self.robot.get_current_config(), dtype=float)
        current_q = start_q.copy()
        path: List[Tuple[np.ndarray, float]] = []
        debug: Dict[str, object] = {
            "ik_attempts": 0,
            "ik_successes": 0,
            "candidate_counts": [],
        }

        for constraint in constraints:
            candidates = self._sample_candidates(constraint, current_q, debug)
            debug["candidate_counts"].append(len(candidates))
            if not candidates:
                self.robot.set_robot_config(start_q.tolist())
                return self._failure({"debug": debug, "path": path})

            q_next, yaw_next = min(
                candidates,
                key=lambda item: self._weighted_distance(current_q, item[0]),
            )
            path.append((q_next, yaw_next))
            current_q = q_next

        self.robot.set_robot_config(start_q.tolist())

        path_costs = [
            self._weighted_distance(path[i - 1][0], path[i][0])
            for i in range(1, len(path))
        ]
        total_cost = float(sum(path_costs))
        max_edge_cost = float(max(path_costs)) if path_costs else 0.0
        start_edge_cost = self._weighted_distance(start_q, path[0][0])

        planned_dict = {
            "path": path,
            "path_costs": path_costs,
            "indices": None,
            "total_cost": total_cost,
            "max_edge_cost": max_edge_cost,
            "start_edge_cost": float(start_edge_cost),
            "lexicographic_cost": (max_edge_cost, total_cost),
            "edge_trajs": [None for _ in path],
            "debug": debug,
        }
        return {
            "success": True,
            "total_cost": total_cost,
            "max_edge_cost": max_edge_cost,
            "planned_dict": planned_dict,
            "path": path,
        }

    def _sample_candidates(
        self,
        constraint: WaypointConstraint,
        current_q: np.ndarray,
        debug: Dict[str, object],
    ) -> List[Tuple[np.ndarray, float]]:
        candidates: List[Tuple[np.ndarray, float]] = []
        seeds = [current_q.tolist()]
        seeds += self.base_q_seeds
        seeds += uniform_q_sampling(self.random_seed_count)

        for q_seed in seeds:
            for yaw in self.yaws:
                orn = quat_from_normal_and_yaw(
                    constraint.normal,
                    yaw,
                    constraint.horizontal,
                    finger_axis_is_plus_y=False,
                )
                self.robot.set_robot_config(current_q.tolist())
                debug["ik_attempts"] += 1
                q_goal = self.robot.solve_ik_collision_aware(
                    constraint.pos,
                    orn,
                    collision=False,
                    max_trials=1,
                    reset=True,
                    q_reset=q_seed,
                )
                if q_goal is None:
                    continue

                q_goal = np.asarray(
                    self.robot.wrap_into_limits(list(q_goal), current_q.tolist()),
                    dtype=float,
                )
                candidates.append((q_goal, float(yaw)))
                debug["ik_successes"] += 1

        return candidates

    def _weighted_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        diff = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
        if self.joint_weights is not None:
            diff = diff * self.joint_weights
        return float(np.linalg.norm(diff))

    def _failure(self, planned_dict):
        return {
            "success": False,
            "total_cost": None,
            "max_edge_cost": None,
            "planned_dict": planned_dict,
            "path": None,
        }
