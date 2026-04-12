from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from planners.grip_planner import PandaGripperPlanner
from utils.path import interpolate_joint_line
from utils.vector import WaypointConstraint, quat_from_normal_and_yaw
from utils.yaw_dp import Q_RESET_SEEDS, uniform_q_sampling


QType = Sequence[float]


@dataclass
class _TreeNode:
    step_idx: int
    q: np.ndarray
    yaw: Optional[float]
    parent: Optional["_TreeNode"]
    edge_cost: float
    max_edge_cost: float
    total_cost: float
    edge_traj: Optional[List[List[float]]] = None


class ConstraintSequenceRRTPlanner:
    """
    Layered RRT-style planner for a fixed task-constraint sequence.

    Each tree node is an IK solution that satisfies exactly one waypoint
    constraint in the ordered sequence. The tree only grows from layer i-1
    to layer i, so the search stays aligned with the given task ordering.
    """

    def __init__(
        self,
        robot_planner: PandaGripperPlanner,
        *,
        joint_weights: Optional[QType] = None,
        max_expansions: int = 6000,
        max_stall_expansions: int = 6000,
        max_child_attempts: int = 6,
        edge_cost_limit: float = 1.2,
        goal_max_edge_cost: float = 0.6,
        interp_joint_step: float = 0.05,
        interp_step_cost_limit: float = 0.15,
        parent_neighbor_count: int = 8,
        layer_cap: int = 64,
        target_layer_population: int = 12,
        dedup_q_threshold: float = 0.15,
        dedup_yaw_threshold_deg: float = 8.0,
        local_yaw_probability: float = 0.75,
        local_yaw_sigma_deg: float = 12.0,
        global_yaw_bounds_deg: Tuple[float, float] = (30.0, 150.0),
        random_seed_probability: float = 0.25,
        check_state_valid: bool = False,
    ):
        self.robot = robot_planner
        self.joint_weights = None if joint_weights is None else np.asarray(joint_weights, dtype=float)
        self.max_expansions = int(max_expansions)
        self.max_stall_expansions = int(max_stall_expansions)
        self.max_child_attempts = int(max_child_attempts)
        self.edge_cost_limit = float(edge_cost_limit)
        self.goal_max_edge_cost = float(goal_max_edge_cost)
        self.interp_joint_step = float(interp_joint_step)
        self.interp_step_cost_limit = float(interp_step_cost_limit)
        self.parent_neighbor_count = int(parent_neighbor_count)
        self.layer_cap = int(layer_cap)
        self.target_layer_population = int(target_layer_population)
        self.dedup_q_threshold = float(dedup_q_threshold)
        self.dedup_yaw_threshold = math.radians(float(dedup_yaw_threshold_deg))
        self.local_yaw_probability = float(local_yaw_probability)
        self.local_yaw_sigma = math.radians(float(local_yaw_sigma_deg))
        self.global_yaw_bounds = tuple(math.radians(v) for v in global_yaw_bounds_deg)
        self.random_seed_probability = float(random_seed_probability)
        self.check_state_valid = bool(check_state_valid)
        self.static_q_seeds = [
            np.asarray(Q_RESET_SEEDS["home"], dtype=float),
            np.asarray(Q_RESET_SEEDS["left_relaxed"], dtype=float),
            np.asarray(Q_RESET_SEEDS["right_relaxed"], dtype=float),
            np.asarray(Q_RESET_SEEDS["left_elbow_out"], dtype=float),
            np.asarray(Q_RESET_SEEDS["right_elbow_out"], dtype=float),
            np.asarray(Q_RESET_SEEDS["forward_mid"], dtype=float),
        ]
        self.best_goal_node: Optional[_TreeNode] = None

    def solve(self, constraints: List[WaypointConstraint]):
        if not constraints:
            return {
                "success": False,
                "total_cost": None,
                "max_edge_cost": None,
                "planned_dict": None,
                "path": None,
            }

        self.best_goal_node = None
        self.robot.open_gripper()
        start_q = np.asarray(self.robot.get_current_config(), dtype=float)
        root = _TreeNode(
            step_idx=-1,
            q=start_q,
            yaw=None,
            parent=None,
            edge_cost=0.0,
            max_edge_cost=0.0,
            total_cost=0.0,
            edge_traj=None,
        )
        layer_nodes: List[List[_TreeNode]] = [[] for _ in range(len(constraints))]
        debug: Dict[str, object] = {
            "expansions": 0,
            "ik_attempts": 0,
            "ik_successes": 0,
            "edge_reject_count": 0,
            "duplicate_reject_count": 0,
            "rewire_parent_successes": 0,
            "layer_node_counts": [0 for _ in constraints],
            "target_layer_histogram": [0 for _ in constraints],
        }

        stagnation = 0
        for expansion_idx in range(self.max_expansions):
            debug["expansions"] = expansion_idx + 1
            target_layer = self._choose_target_layer(layer_nodes, len(constraints))
            if target_layer is None:
                break

            debug["target_layer_histogram"][target_layer] += 1

            parent_pool = [root] if target_layer == 0 else layer_nodes[target_layer - 1]
            if not parent_pool:
                stagnation += 1
                if stagnation >= self.max_stall_expansions:
                    break
                continue

            guide_parent = self._choose_guide_parent(parent_pool)
            sample = self._sample_state_for_constraint(constraints[target_layer], guide_parent, debug)
            if sample is None:
                stagnation += 1
                if stagnation >= self.max_stall_expansions:
                    break
                continue

            q_child, yaw_child = sample
            child_node = self._attach_node(layer_nodes, target_layer, q_child, yaw_child, root, debug)
            if child_node is None:
                stagnation += 1
                if stagnation >= self.max_stall_expansions:
                    break
                continue

            debug["layer_node_counts"][target_layer] = len(layer_nodes[target_layer])

            improved = False
            if target_layer == len(constraints) - 1:
                if self.best_goal_node is None or self._is_better_cost(child_node, self.best_goal_node):
                    self.best_goal_node = child_node
                    improved = True

            if improved:
                stagnation = 0
            else:
                stagnation += 1

            if self.best_goal_node is not None and self.best_goal_node.max_edge_cost < self.goal_max_edge_cost:
                break

            if stagnation >= self.max_stall_expansions:
                break

        extracted = self._extract_best_path_from_layers(layer_nodes, root)
        if extracted is None:
            return {
                "success": False,
                "total_cost": None,
                "max_edge_cost": None,
                "planned_dict": {
                    "debug": debug,
                    "layer_node_counts": [len(nodes) for nodes in layer_nodes],
                },
                "path": None,
            }

        path_nodes, path, path_costs, edge_trajs, total_cost, max_edge_cost, start_edge_cost = extracted

        planned_dict = {
            "path": path,
            "path_costs": path_costs,
            "indices": None,
            "total_cost": float(total_cost),
            "max_edge_cost": float(max_edge_cost),
            "start_edge_cost": start_edge_cost,
            "lexicographic_cost": (
                float(max_edge_cost),
                float(total_cost),
            ),
            "edge_trajs": edge_trajs,
            "debug": debug,
            "layer_node_counts": [len(nodes) for nodes in layer_nodes],
        }
        return {
            "success": True,
            "total_cost": float(total_cost),
            "max_edge_cost": float(max_edge_cost),
            "planned_dict": planned_dict,
            "path": path,
        }

    def _choose_target_layer(self, layer_nodes: List[List[_TreeNode]], num_layers: int) -> Optional[int]:
        reachable_layers = []
        for layer_idx in range(num_layers):
            if layer_idx == 0 or layer_nodes[layer_idx - 1]:
                reachable_layers.append(layer_idx)

        if not reachable_layers:
            return None

        for layer_idx in reachable_layers:
            if len(layer_nodes[layer_idx]) < self.target_layer_population:
                return layer_idx

        if self.best_goal_node is not None and random.random() < 0.6:
            worst_node = max(self._backtrack_path(self.best_goal_node), key=lambda node: node.edge_cost)
            return worst_node.step_idx

        return random.choice(reachable_layers)

    def _choose_guide_parent(self, parent_pool: List[_TreeNode]) -> _TreeNode:
        if len(parent_pool) == 1:
            return parent_pool[0]

        if random.random() < 0.7:
            ranked = sorted(parent_pool, key=lambda node: self._cost_key(node))
            top_count = min(8, len(ranked))
            return random.choice(ranked[:top_count])

        return random.choice(parent_pool)

    def _sample_state_for_constraint(
        self,
        constraint: WaypointConstraint,
        guide_parent: _TreeNode,
        debug: Dict[str, object],
    ) -> Optional[Tuple[np.ndarray, float]]:
        yaw_center = guide_parent.yaw if guide_parent.yaw is not None else math.radians(90.0)

        for _ in range(self.max_child_attempts):
            yaw = self._sample_yaw(yaw_center, prefer_local=guide_parent.yaw is not None)
            orn = quat_from_normal_and_yaw(
                constraint.normal,
                yaw,
                constraint.horizontal,
                finger_axis_is_plus_y=False,
            )
            for seed in self._build_seed_candidates(guide_parent.q):
                debug["ik_attempts"] += 1
                q_goal = self.robot.solve_ik_collision_aware(
                    constraint.pos,
                    orn,
                    collision=False,
                    max_trials=1,
                    reset=True,
                    q_reset=seed.tolist(),
                )
                if q_goal is None:
                    continue

                q_wrapped = np.asarray(
                    self.robot.wrap_into_limits(list(q_goal), guide_parent.q.tolist()),
                    dtype=float,
                )
                debug["ik_successes"] += 1
                return q_wrapped, float(yaw)

        return None

    def _build_seed_candidates(self, parent_q: np.ndarray) -> List[np.ndarray]:
        seeds: List[np.ndarray] = [np.asarray(parent_q, dtype=float)]
        seeds.append(0.5 * (parent_q + self.static_q_seeds[0]))

        remaining = self.static_q_seeds[1:].copy()
        random.shuffle(remaining)
        seeds.extend(remaining)

        if random.random() < self.random_seed_probability:
            seeds.extend(np.asarray(sample, dtype=float) for sample in uniform_q_sampling(1))

        unique: List[np.ndarray] = []
        for seed in seeds:
            if any(np.allclose(seed, existing, atol=1e-6) for existing in unique):
                continue
            unique.append(seed)
        return unique

    def _sample_yaw(self, yaw_center: float, *, prefer_local: bool) -> float:
        lower, upper = self.global_yaw_bounds
        if prefer_local and random.random() < self.local_yaw_probability:
            yaw = random.gauss(yaw_center, self.local_yaw_sigma)
        else:
            yaw = random.uniform(lower, upper)
        return float(min(max(yaw, lower), upper))

    def _attach_node(
        self,
        layer_nodes: List[List[_TreeNode]],
        target_layer: int,
        q_child: np.ndarray,
        yaw_child: float,
        root: _TreeNode,
        debug: Dict[str, object],
    ) -> Optional[_TreeNode]:
        parents = [root] if target_layer == 0 else layer_nodes[target_layer - 1]
        if not parents:
            return None

        scored_parents = []
        for parent in parents:
            q_wrapped = np.asarray(
                self.robot.wrap_into_limits(q_child.tolist(), parent.q.tolist()),
                dtype=float,
            )
            heuristic = self._weighted_distance(parent.q, q_wrapped)
            scored_parents.append((heuristic, parent, q_wrapped))

        scored_parents.sort(key=lambda item: item[0])
        nearest_parents = scored_parents[: max(1, self.parent_neighbor_count)]

        best_node = None
        for _, parent, q_wrapped in nearest_parents:
            if parent.step_idx < 0:
                edge_cost = 0.0
                edge_traj = None
            else:
                edge_validation = self._validate_edge(parent.q, q_wrapped)
                if edge_validation is None:
                    debug["edge_reject_count"] += 1
                    continue
                edge_cost, edge_traj = edge_validation
            candidate = _TreeNode(
                step_idx=target_layer,
                q=q_wrapped,
                yaw=float(yaw_child),
                parent=parent,
                edge_cost=float(edge_cost),
                max_edge_cost=parent.max_edge_cost if parent.step_idx < 0 else max(parent.max_edge_cost, float(edge_cost)),
                total_cost=parent.total_cost if parent.step_idx < 0 else parent.total_cost + float(edge_cost),
                edge_traj=edge_traj,
            )
            if best_node is None or self._is_better_cost(candidate, best_node):
                best_node = candidate

        if best_node is None:
            return None

        existing_idx = self._find_duplicate(layer_nodes[target_layer], best_node.q, best_node.yaw)
        if existing_idx is not None:
            existing = layer_nodes[target_layer][existing_idx]
            if self._is_better_cost(best_node, existing):
                layer_nodes[target_layer][existing_idx] = best_node
                debug["rewire_parent_successes"] += 1
                self._prune_layer(layer_nodes[target_layer])
                return best_node
            debug["duplicate_reject_count"] += 1
            return None

        layer_nodes[target_layer].append(best_node)
        self._prune_layer(layer_nodes[target_layer])
        return best_node

    def _validate_edge(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
    ) -> Optional[Tuple[float, List[List[float]]]]:
        edge_cost = self._weighted_distance(q_from, q_to)
        if not math.isfinite(edge_cost) or edge_cost > self.edge_cost_limit:
            return None

        steps = max(2, int(math.ceil(np.max(np.abs(q_to - q_from)) / self.interp_joint_step)) + 1)
        traj = interpolate_joint_line(q_from.tolist(), q_to.tolist(), steps)

        prev = np.asarray(traj[0], dtype=float)
        if self.check_state_valid and (not self.robot.is_state_valid(prev.tolist())):
            return None

        for waypoint in traj[1:]:
            curr = np.asarray(waypoint, dtype=float)
            if self._weighted_distance(prev, curr) > self.interp_step_cost_limit:
                return None
            if self.check_state_valid and (not self.robot.is_state_valid(curr.tolist())):
                return None
            prev = curr

        return float(edge_cost), traj

    def _find_duplicate(self, nodes: List[_TreeNode], q: np.ndarray, yaw: float) -> Optional[int]:
        for idx, node in enumerate(nodes):
            if self._weighted_distance(node.q, q) > self.dedup_q_threshold:
                continue
            if self._yaw_distance(node.yaw, yaw) > self.dedup_yaw_threshold:
                continue
            return idx
        return None

    def _prune_layer(self, nodes: List[_TreeNode]):
        if len(nodes) <= self.layer_cap:
            return

        protected_ids = {id(node) for node in self._backtrack_path(self.best_goal_node)} if self.best_goal_node is not None else set()
        nodes.sort(
            key=lambda node: (
                0 if id(node) in protected_ids else 1,
                node.max_edge_cost,
                node.total_cost,
            )
        )
        del nodes[self.layer_cap :]

    def _backtrack_path(self, goal_node: Optional[_TreeNode]) -> List[_TreeNode]:
        if goal_node is None:
            return []

        nodes: List[_TreeNode] = []
        node = goal_node
        while node is not None and node.step_idx >= 0:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _extract_best_path_from_layers(
        self,
        layer_nodes: List[List[_TreeNode]],
        root: _TreeNode,
    ) -> Optional[Tuple[List[_TreeNode], List[Tuple[np.ndarray, float]], List[float], List[Optional[List[List[float]]]], float, float, Optional[float]]]:
        if any(len(nodes) == 0 for nodes in layer_nodes):
            return None

        n_layers = len(layer_nodes)
        dp_sum: List[List[float]] = [[math.inf] * len(nodes) for nodes in layer_nodes]
        dp_max: List[List[float]] = [[math.inf] * len(nodes) for nodes in layer_nodes]
        prev_choice: List[List[Optional[int]]] = [[None] * len(nodes) for nodes in layer_nodes]
        edge_trajs: List[List[Optional[List[List[float]]]]] = [[None] * len(nodes) for nodes in layer_nodes]

        for j, node in enumerate(layer_nodes[0]):
            dp_sum[0][j] = 0.0
            dp_max[0][j] = 0.0

        for layer_idx in range(1, n_layers):
            for j, node in enumerate(layer_nodes[layer_idx]):
                best_cost = None
                best_prev = None
                best_traj = None
                for i, prev_node in enumerate(layer_nodes[layer_idx - 1]):
                    if not math.isfinite(dp_sum[layer_idx - 1][i]) or not math.isfinite(dp_max[layer_idx - 1][i]):
                        continue
                    edge = self._validate_edge(prev_node.q, node.q)
                    if edge is None:
                        continue
                    edge_cost, edge_traj = edge
                    candidate = (max(dp_max[layer_idx - 1][i], edge_cost), dp_sum[layer_idx - 1][i] + edge_cost)
                    if best_cost is None or candidate < best_cost:
                        best_cost = candidate
                        best_prev = i
                        best_traj = edge_traj

                if best_cost is None:
                    continue

                dp_max[layer_idx][j] = float(best_cost[0])
                dp_sum[layer_idx][j] = float(best_cost[1])
                prev_choice[layer_idx][j] = best_prev
                edge_trajs[layer_idx][j] = best_traj

        best_last_idx = None
        best_last_cost = None
        last_layer_idx = n_layers - 1
        for j in range(len(layer_nodes[last_layer_idx])):
            if not math.isfinite(dp_sum[last_layer_idx][j]) or not math.isfinite(dp_max[last_layer_idx][j]):
                continue
            candidate = (dp_max[last_layer_idx][j], dp_sum[last_layer_idx][j])
            if best_last_cost is None or candidate < best_last_cost:
                best_last_cost = candidate
                best_last_idx = j

        if best_last_idx is None or best_last_cost is None:
            return None

        indices = [best_last_idx]
        for layer_idx in range(last_layer_idx, 0, -1):
            prev_idx = prev_choice[layer_idx][indices[-1]]
            if prev_idx is None:
                return None
            indices.append(prev_idx)
        indices.reverse()

        path_nodes = [layer_nodes[layer_idx][node_idx] for layer_idx, node_idx in enumerate(indices)]
        path = [(node.q.copy(), float(node.yaw)) for node in path_nodes]
        path_costs = [self._weighted_distance(path_nodes[i - 1].q, path_nodes[i].q) for i in range(1, len(path_nodes))]
        path_edge_trajs: List[Optional[List[List[float]]]] = [None]
        for layer_idx, node_idx in enumerate(indices[1:], start=1):
            path_edge_trajs.append(edge_trajs[layer_idx][node_idx])

        start_edge_cost = self._weighted_distance(root.q, path_nodes[0].q) if path_nodes else None
        return (
            path_nodes,
            path,
            [float(cost) for cost in path_costs],
            path_edge_trajs,
            float(best_last_cost[1]),
            float(best_last_cost[0]),
            float(start_edge_cost) if start_edge_cost is not None else None,
        )

    def _cost_key(self, node: _TreeNode) -> Tuple[float, float]:
        return float(node.max_edge_cost), float(node.total_cost)

    def _is_better_cost(self, lhs: _TreeNode, rhs: _TreeNode) -> bool:
        return self._cost_key(lhs) < self._cost_key(rhs)

    def _weighted_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        diff = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
        if self.joint_weights is not None:
            diff = diff * self.joint_weights
        return float(np.linalg.norm(diff))

    def _yaw_distance(self, yaw1: Optional[float], yaw2: Optional[float]) -> float:
        if yaw1 is None or yaw2 is None:
            return math.inf
        diff = abs(float(yaw1) - float(yaw2))
        return min(diff, 2.0 * math.pi - diff)
