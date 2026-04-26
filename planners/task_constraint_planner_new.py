from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from planners.constraint_sequence_greedy_planner import ConstraintSequenceGreedyPlanner
from planners.constraint_sequence_rrt_planner import ConstraintSequenceRRTPlanner
from planners.grip_planner import PandaGripperPlanner
from utils.vector import WaypointConstraint, quat_from_normal_and_yaw
from utils.yaw_dp import Q_RESET_SEEDS, dp_plan_yaw_path, uniform_q_sampling


QType = Sequence[float]
State = Tuple[np.ndarray, float]


@dataclass(frozen=True)
class SeedSpec:
    q: QType
    yaw: float
    kind: str
    from_step: Optional[int] = None


class TaskConstraintPlannerNew:
    """
    Cleaner constraint-sequence planner for Panda mailer-box tasks.

    The iteration method is deliberately small:
      1. build a clustered candidate layer for every constraint;
      2. run DP over those layers;
      3. take the worst DP edge and use each endpoint q as the IK seed for
         the other endpoint's constraint.

    Random seeds are only a rescue path when bad-edge bridge seeds stop adding
    new clustered candidates.
    """

    def __init__(
        self,
        robot_planner: PandaGripperPlanner,
        *,
        max_iterations: int = 3,
        target_max_edge_cost: float = 0.8,
        trace_seed_count: int = 4,
        refinement_window: int = 1,
        random_refine_seed_count: int = 2,
        cluster_q_eps: float = 0.08,
        cluster_yaw_eps: float = math.radians(2.0),
        seed_q_eps: float = 0.04,
        seed_yaw_eps: float = math.radians(1.0),
    ):
        self.robot = robot_planner
        self.path = None

        self.max_iterations = int(max_iterations)
        self.target_max_edge_cost = float(target_max_edge_cost)
        self.trace_seed_count = int(trace_seed_count)
        self.refinement_window = int(refinement_window)
        self.random_refine_seed_count = int(random_refine_seed_count)

        self.cluster_q_eps = float(cluster_q_eps)
        self.cluster_yaw_eps = float(cluster_yaw_eps)
        self.seed_q_eps = float(seed_q_eps)
        self.seed_yaw_eps = float(seed_yaw_eps)

        self.joint_weights = np.ones(7, dtype=float)

    def solve_constraint_path(
        self,
        constraints: List[WaypointConstraint],
        method: Literal["Sampling", "Iteration", "RRT", "Greedy"],
        ik_backend: Literal["frankik", "pybullet"] = "pybullet",
    ):
        if method == "RRT":
            return self._solve_with_rrt(constraints)
        if method == "Greedy":
            return self._solve_with_greedy(constraints)
        if method not in ("Sampling", "Iteration"):
            raise ValueError(f"unsupported planning method: {method}")

        if not constraints:
            return self._failure(None, elapsed=0.0)

        start_time = time.time()
        self._reset_candidate_state(len(constraints))
        self.current_config = self.robot.get_current_config()
        self.yaws = self.get_yaw_candidates(num_steps=2)

        if method == "Sampling":
            self._build_sampling_layers(constraints, ik_backend=ik_backend)
        else:
            self._build_iteration_initial_layers(constraints, ik_backend=ik_backend)

        planned_dict = self._run_dp()
        if planned_dict is None:
            elapsed = time.time() - start_time
            self.robot.set_robot_config(self.current_config)
            return self._failure(None, elapsed=elapsed)

        if method == "Iteration":
            for iter_idx in range(self.max_iterations):
                self.path = planned_dict["path"]
                worst_idx = self._worst_edge_idx(planned_dict)
                worst_cost = planned_dict["path_costs"][worst_idx] if worst_idx is not None else 0.0
                print(
                    f"Iter {iter_idx}, best max_edge={planned_dict['max_edge_cost']:.3f}, "
                    f"worst_idx={worst_idx}, worst_cost={worst_cost:.3f}"
                )

                if planned_dict["max_edge_cost"] < self.target_max_edge_cost:
                    break

                added = self._refine_bad_edge(
                    constraints,
                    planned_dict,
                    iter_idx=iter_idx,
                    ik_backend=ik_backend,
                )
                if added == 0:
                    self._rescue_worst_edge(
                        constraints,
                        planned_dict,
                        iter_idx=iter_idx,
                        ik_backend=ik_backend,
                    )

                next_plan = self._run_dp()
                if next_plan is None:
                    break
                planned_dict = next_plan

        elapsed = time.time() - start_time
        self.robot.set_robot_config(self.current_config)

        path = planned_dict["path"]
        max_edge_cost = float(planned_dict["max_edge_cost"])
        worst_idx = self._worst_edge_idx(planned_dict)
        print(f"Total time: {elapsed}")
        print(
            f"Iter times: {self.max_iterations if method == 'Iteration' else 0}, "
            f"Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, "
            f"worst_idx: {worst_idx}"
        )

        planned_dict["debug"] = self._debug_summary()
        return {
            "success": True,
            "total_cost": float(planned_dict["total_cost"]),
            "max_edge_cost": max_edge_cost,
            "planned_dict": planned_dict,
            "path": path,
            "time": elapsed,
        }

    def _solve_with_rrt(self, constraints: List[WaypointConstraint]):
        planner = ConstraintSequenceRRTPlanner(
            self.robot,
            joint_weights=np.array([1, 1, 1, 1, 1, 1, 1], dtype=float),
        )
        start_time = time.time()
        metric = planner.solve(constraints)
        metric["time"] = time.time() - start_time
        print(f"Total time: {metric['time']}")
        print(f"Max edge cost is {metric['max_edge_cost']}, Total cost is {metric['total_cost']}")
        return metric

    def _solve_with_greedy(self, constraints: List[WaypointConstraint]):
        planner = ConstraintSequenceGreedyPlanner(
            self.robot,
            joint_weights=np.array([1, 1, 1, 1, 1, 1, 1], dtype=float),
        )
        start_time = time.time()
        metric = planner.solve(constraints)
        metric["time"] = time.time() - start_time
        print(f"Total time: {metric['time']}")
        print(f"Max edge cost is {metric['max_edge_cost']}, Total cost is {metric['total_cost']}")
        return metric

    def _reset_candidate_state(self, num_steps: int):
        self.q_trajectory: List[List[State]] = [[] for _ in range(num_steps)]
        self.q_source_trajectory: List[List[Dict[str, object]]] = [[] for _ in range(num_steps)]
        self._tried_seed_trajectory: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(num_steps)]
        self.debug: Dict[str, object] = {
            "ik_attempts": 0,
            "ik_successes": 0,
            "candidate_duplicates": 0,
            "seed_duplicates": 0,
            "candidate_counts_by_round": [],
        }

    def _build_sampling_layers(
        self,
        constraints: List[WaypointConstraint],
        *,
        ik_backend: Literal["frankik", "pybullet"],
    ):
        named_seeds = [
            Q_RESET_SEEDS["home"],
            Q_RESET_SEEDS["left_elbow_out"],
            Q_RESET_SEEDS["right_elbow_out"],
        ]
        q_seeds = named_seeds + uniform_q_sampling(10)

        for constraint in constraints:
            specs = self._seed_specs_from_qs(q_seeds, self.yaws, kind="sampling")
            self._append_candidates_from_seeds(
                constraint,
                specs,
                source_tag={"kind": "sampling", "step": constraint.task_step},
                ik_backend=ik_backend,
            )

        self._record_candidate_counts("sampling")

    def _build_iteration_initial_layers(
        self,
        constraints: List[WaypointConstraint],
        *,
        ik_backend: Literal["frankik", "pybullet"],
    ):
        for step, constraint in enumerate(constraints):
            if step == 0:
                specs = self._seed_specs_from_qs(
                    [Q_RESET_SEEDS["home"]],
                    self.yaws,
                    kind="anchor",
                )
            else:
                specs = [
                    SeedSpec(q=q, yaw=yaw, kind="trace_forward", from_step=step - 1)
                    for q, yaw in self._select_diverse_states(
                        self.q_trajectory[step - 1],
                        max_count=self.trace_seed_count,
                    )
                ]

            added = self._append_candidates_from_seeds(
                constraint,
                specs,
                source_tag={"kind": "init", "step": step},
                ik_backend=ik_backend,
            )

            if len(self.q_trajectory[step]) <= 2 or added == 0:
                rescue_specs = self._seed_specs_from_qs(
                    [
                        Q_RESET_SEEDS["home"],
                        Q_RESET_SEEDS["left_elbow_out"],
                        Q_RESET_SEEDS["right_elbow_out"],
                    ],
                    self.yaws,
                    kind="anchor_rescue",
                )
                self._append_candidates_from_seeds(
                    constraint,
                    rescue_specs,
                    source_tag={"kind": "init_rescue", "step": step},
                    ik_backend=ik_backend,
                )

        self._record_candidate_counts("init")

    def _refine_bad_edge(
        self,
        constraints: List[WaypointConstraint],
        planned_dict: Dict[str, object],
        *,
        iter_idx: int,
        ik_backend: Literal["frankik", "pybullet"],
    ) -> int:
        edge_idx = self._worst_edge_idx(planned_dict)
        if edge_idx is None:
            return 0

        path = planned_dict["path"]
        left = max(0, edge_idx - self.refinement_window)
        right = min(len(constraints) - 1, edge_idx + 1 + self.refinement_window)
        added = 0

        for target_step in range(edge_idx, left - 1, -1):
            neighbor_step = target_step + 1
            if neighbor_step >= len(path):
                continue
            added += self._bridge_from_path_neighbor(
                constraints[target_step],
                path,
                target_step=target_step,
                neighbor_step=neighbor_step,
                source_tag={
                    "kind": "bad_edge_bridge",
                    "iter": iter_idx + 1,
                    "from_edge": edge_idx,
                    "direction": "backward",
                },
                ik_backend=ik_backend,
            )

        for target_step in range(edge_idx + 1, right + 1):
            neighbor_step = target_step - 1
            if neighbor_step < 0:
                continue
            added += self._bridge_from_path_neighbor(
                constraints[target_step],
                path,
                target_step=target_step,
                neighbor_step=neighbor_step,
                source_tag={
                    "kind": "bad_edge_bridge",
                    "iter": iter_idx + 1,
                    "from_edge": edge_idx,
                    "direction": "forward",
                },
                ik_backend=ik_backend,
            )

        self._record_candidate_counts(f"refine_{iter_idx + 1}")
        return added

    def _bridge_from_path_neighbor(
        self,
        constraint: WaypointConstraint,
        path: List[State],
        *,
        target_step: int,
        neighbor_step: int,
        source_tag: Dict[str, object],
        ik_backend: Literal["frankik", "pybullet"],
    ) -> int:
        seed_q, seed_yaw = path[neighbor_step]
        _, target_yaw = path[target_step]
        yaws = self._unique_yaws([seed_yaw, target_yaw])
        specs = [
            SeedSpec(
                q=seed_q,
                yaw=yaw,
                kind="path_neighbor",
                from_step=neighbor_step,
            )
            for yaw in yaws
        ]
        return self._append_candidates_from_seeds(
            constraint,
            specs,
            source_tag=source_tag,
            ik_backend=ik_backend,
        )

    def _rescue_worst_edge(
        self,
        constraints: List[WaypointConstraint],
        planned_dict: Dict[str, object],
        *,
        iter_idx: int,
        ik_backend: Literal["frankik", "pybullet"],
    ) -> int:
        edge_idx = self._worst_edge_idx(planned_dict)
        if edge_idx is None or self.random_refine_seed_count <= 0:
            return 0

        added = 0
        path = planned_dict["path"]
        for step in (edge_idx, edge_idx + 1):
            _, yaw = path[step]
            specs = self._seed_specs_from_qs(
                uniform_q_sampling(self.random_refine_seed_count),
                self._local_yaws(yaw),
                kind="random_rescue",
            )
            added += self._append_candidates_from_seeds(
                constraints[step],
                specs,
                source_tag={
                    "kind": "random_rescue",
                    "iter": iter_idx + 1,
                    "from_edge": edge_idx,
                },
                ik_backend=ik_backend,
            )

        self._record_candidate_counts(f"rescue_{iter_idx + 1}")
        return added

    def _append_candidates_from_seeds(
        self,
        constraint: WaypointConstraint,
        seed_specs: Sequence[SeedSpec],
        *,
        source_tag: Dict[str, object],
        ik_backend: Literal["frankik", "pybullet"],
        finger_axis_is_plus_y: bool = False,
    ) -> int:
        added = 0
        step = int(constraint.task_step)
        for seed_idx, seed in enumerate(self._unique_seed_specs(step, seed_specs)):
            orn = quat_from_normal_and_yaw(
                constraint.normal,
                float(seed.yaw),
                constraint.horizontal,
                finger_axis_is_plus_y=finger_axis_is_plus_y,
            )
            q_seed = np.asarray(seed.q, dtype=float)

            self.robot.set_robot_config(self.current_config)
            self.debug["ik_attempts"] = int(self.debug["ik_attempts"]) + 1
            q_goal = self.robot.solve_ik_collision_aware(
                constraint.pos,
                orn,
                collision=False,
                max_trials=1,
                reset=True,
                q_reset=q_seed.tolist(),
                ik_backend=ik_backend,
            )
            self.robot.set_robot_config(self.current_config)

            if q_goal is None:
                continue

            self.debug["ik_successes"] = int(self.debug["ik_successes"]) + 1
            q_goal = np.asarray(
                self.robot.wrap_into_limits(list(q_goal), q_seed.tolist()),
                dtype=float,
            )
            source = {
                "source_tag": dict(source_tag),
                "seed_idx": seed_idx,
                "seed_kind": seed.kind,
                "seed_from_step": seed.from_step,
                "q_seed": q_seed.tolist(),
                "q": q_goal.tolist(),
                "yaw": float(seed.yaw),
            }
            if self._add_clustered_candidate(step, q_goal, float(seed.yaw), source):
                added += 1

        return added

    def _add_clustered_candidate(
        self,
        step: int,
        q_goal: np.ndarray,
        yaw: float,
        source: Dict[str, object],
    ) -> bool:
        for idx, (q_existing, yaw_existing) in enumerate(self.q_trajectory[step]):
            if (
                self._q_distance(q_goal, q_existing) <= self.cluster_q_eps
                and self._yaw_distance(yaw, yaw_existing) <= self.cluster_yaw_eps
            ):
                src = self.q_source_trajectory[step][idx]
                src["cluster_size"] = int(src.get("cluster_size", 1)) + 1
                src["last_duplicate_source"] = source["source_tag"]
                self.debug["candidate_duplicates"] = int(self.debug["candidate_duplicates"]) + 1
                return False

        self.q_trajectory[step].append((q_goal, float(yaw)))
        source["cluster_size"] = 1
        self.q_source_trajectory[step].append(source)
        return True

    def _unique_seed_specs(self, step: int, seed_specs: Sequence[SeedSpec]) -> List[SeedSpec]:
        unique_specs: List[SeedSpec] = []
        for seed in seed_specs:
            q = np.asarray(seed.q, dtype=float)
            yaw = float(seed.yaw)
            if self._seed_was_tried(step, q, yaw):
                self.debug["seed_duplicates"] = int(self.debug["seed_duplicates"]) + 1
                continue

            self._tried_seed_trajectory[step].append((q, yaw))
            unique_specs.append(seed)

        return unique_specs

    def _seed_was_tried(self, step: int, q: np.ndarray, yaw: float) -> bool:
        for old_q, old_yaw in self._tried_seed_trajectory[step]:
            if (
                self._q_distance(q, old_q) <= self.seed_q_eps
                and self._yaw_distance(yaw, old_yaw) <= self.seed_yaw_eps
            ):
                return True
        return False

    def _run_dp(self):
        return dp_plan_yaw_path(
            feasible_by_step=self.q_trajectory,
            joint_weights=self.joint_weights,
        )

    def _seed_specs_from_qs(
        self,
        q_seeds: Sequence[QType],
        yaws: Sequence[float],
        *,
        kind: str,
    ) -> List[SeedSpec]:
        return [
            SeedSpec(q=q_seed, yaw=float(yaw), kind=kind)
            for q_seed in q_seeds
            for yaw in yaws
        ]

    def _select_diverse_states(
        self,
        states: Sequence[State],
        *,
        max_count: int,
    ) -> List[State]:
        selected: List[State] = []
        for q, yaw in states:
            if all(self._q_distance(q, old_q) > self.cluster_q_eps for old_q, _ in selected):
                selected.append((np.asarray(q, dtype=float), float(yaw)))
            if len(selected) >= max_count:
                break
        return selected

    def _record_candidate_counts(self, label: str):
        self.debug["candidate_counts_by_round"].append(
            {
                "label": label,
                "counts": [len(layer) for layer in self.q_trajectory],
            }
        )

    def _debug_summary(self) -> Dict[str, object]:
        summary = dict(self.debug)
        summary["candidate_counts"] = [len(layer) for layer in self.q_trajectory]
        summary["source_counts"] = [len(layer) for layer in self.q_source_trajectory]
        return summary

    def _failure(self, planned_dict, *, elapsed: float):
        return {
            "success": False,
            "total_cost": None,
            "max_edge_cost": None,
            "planned_dict": planned_dict,
            "path": None,
            "time": elapsed,
        }

    def get_yaw_candidates(
        self,
        mid_degree: float = 90,
        num_steps: int = 5,
        max_offset: float = math.radians(60.0),
    ) -> List[float]:
        step = max_offset / float(max(1, num_steps))
        yaws = [math.radians(mid_degree)]
        for k in range(1, num_steps + 1):
            offset = k * step
            yaws.append(yaws[0] + offset)
            yaws.append(yaws[0] - offset)
        return self._unique_yaws(yaws)

    def _local_yaws(self, center_yaw: float) -> List[float]:
        return self._unique_yaws(
            [
                float(center_yaw),
                float(center_yaw) + math.radians(30.0),
                float(center_yaw) - math.radians(30.0),
            ]
        )

    def _unique_yaws(self, yaws: Sequence[float]) -> List[float]:
        result: List[float] = []
        low = math.radians(30.0)
        high = math.radians(150.0)
        for yaw in yaws:
            yaw = min(max(float(yaw), low), high)
            if all(self._yaw_distance(yaw, old) > self.seed_yaw_eps for old in result):
                result.append(yaw)
        return result

    def _worst_edge_idx(self, planned_dict: Dict[str, object]) -> Optional[int]:
        path_costs = planned_dict.get("path_costs") or []
        if len(path_costs) == 0:
            return None
        return int(np.argmax(path_costs))

    def _q_distance(self, q1: QType, q2: QType) -> float:
        diff = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
        if self.joint_weights is not None:
            diff = diff * self.joint_weights
        return float(np.linalg.norm(diff))

    @staticmethod
    def _yaw_distance(yaw1: float, yaw2: float) -> float:
        diff = (float(yaw1) - float(yaw2) + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff)
