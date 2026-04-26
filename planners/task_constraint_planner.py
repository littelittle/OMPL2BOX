import time
import numpy as np
import pickle
import random
from typing import List, Literal 

from planners.grip_planner import PandaGripperPlanner
from planners.constraint_sequence_rrt_planner import ConstraintSequenceRRTPlanner
from planners.constraint_sequence_greedy_planner import ConstraintSequenceGreedyPlanner
from utils.vector import quat_from_normal_and_yaw, WaypointConstraint
from utils.yaw_dp import dp_plan_yaw_path, Q_RESET_SEEDS, uniform_q_sampling, find_diverse_beam, find_counterfactual_alt_for_worst_edge

# with open('temp/path_o0.pkl', 'rb') as f:
#     IK_seed_list = pickle.load(f)


def _find_nearest_q_yaw(q_yaw_list, target_q):
    if len(q_yaw_list) == 0:
        return None

    target_q = np.asarray(target_q, dtype=float)

    best_idx = min(
        range(len(q_yaw_list)),
        key=lambda i: np.linalg.norm(
            np.asarray(q_yaw_list[i][0], dtype=float) - target_q
        )
    )
    best_q, best_yaw = q_yaw_list[best_idx]
    return best_idx, best_q, best_yaw

def _dedup_layer(layer, src_layer, q_eps=0.1, yaw_eps=np.deg2rad(2)):
    keep_states = []
    keep_sources = []

    for (q, yaw), src in zip(layer, src_layer):
        q = np.asarray(q, dtype=float)
        duplicate = False

        for q2, yaw2 in keep_states:
            if np.linalg.norm(q - np.asarray(q2, dtype=float)) < q_eps and abs(float(yaw) - float(yaw2)) < yaw_eps:
                duplicate = True
                break

        if not duplicate:
            keep_states.append((q.tolist(), float(yaw)))
            keep_sources.append(src)

    return keep_states, keep_sources

class TaskConstraintPlanner:
    def __init__(self, robot_planner: PandaGripperPlanner):
        self.robot = robot_planner
        self.path = None


    def _sample_redundant(self, constraint:WaypointConstraint, source_tag, finger_axis_is_plus_y=False, ik_backend:Literal['frankik', 'pybullet']='pybullet'):
        q_goal_list = []
        q_source_list = []
        for reset_idx, q_reset in enumerate(self.q_reset_list):
            for yaw in self.yaws:
                orn = quat_from_normal_and_yaw(constraint.normal, yaw, constraint.horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)  
                self.robot.set_robot_config(self.current_config)
                q_goal = self.robot.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_reset, ik_backend=ik_backend)
                self.robot.set_robot_config(self.current_config)
                if q_goal is not None:
                    q_goal_list.append((q_goal, yaw))
                    q_source_list.append({
                        "source_tag": source_tag,     # init or refine
                        "reset_idx": reset_idx,       # which one is it in q_reset_list
                        "q_reset": np.asarray(q_reset, dtype=float).tolist(),
                        "yaw": float(yaw),
                    })
        self.q_trajectory[constraint.task_step] += q_goal_list
        self.q_source_trajectory[constraint.task_step] += q_source_list

    def _trace_sample_redundant(self, constraint:WaypointConstraint, source_tag, seed_tag:Literal['const', 'former', 'latter'], MAX_SEED=4, finger_axis_is_plus_y=False, reference_path=None, ik_backend:Literal['frankik', 'pybullet']='pybullet'):
        q_goal_list = []
        q_source_list = []
        if seed_tag == 'const':
            # Which means that there is no former step q_goal, so get the yaws from self.yaws
            for reset_idx, q_reset in enumerate(self.q_reset_list):
                for yaw in self.yaws:
                    orn = quat_from_normal_and_yaw(constraint.normal, yaw, constraint.horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)  
                    self.robot.set_robot_config(self.current_config)
                    q_goal = self.robot.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=q_reset, ik_backend=ik_backend)
                    self.robot.set_robot_config(self.current_config)
                    if q_goal is not None:
                        q_goal_list.append((q_goal, yaw))
                        q_source_list.append({
                            "source_tag": source_tag,     # init or refine
                            "reset_idx": reset_idx,       # which one is it in q_reset_list
                            "q": np.asarray(q_goal, dtype=float).tolist(),
                            'q_seed': np.asarray(q_reset, dtype=float).tolist(),
                            "yaw": float(yaw),
                        })
        elif seed_tag == 'former':
            # Select MAX_SEED representative former qs to trace 
            former_seed_list = []
            # if reference_path: # Which means that there is selected q at this current step
            #     # the former step selected (q, yaw) should be added as the seed
            #     temp_best_former_q, temp_best_former_yaw = reference_path[constraint.task_step-1]
            #     former_seed_list.append((temp_best_former_q, temp_best_former_yaw))

            #     # the nearest last step q should also be added as the seed
            #     temp_best_q, _ = reference_path[constraint.task_step]
            #     nearest_result = _find_nearest_q_yaw(self.q_trajectory[constraint.task_step-1], temp_best_q)
            #     if nearest_result is not None:
            #         nearest_idx, nearest_q, nearest_yaw = nearest_result
            #         if not np.allclose(np.asarray(nearest_q), np.asarray(temp_best_former_q)):
            #             former_seed_list.append((nearest_q, nearest_yaw))
            if len(former_seed_list)<MAX_SEED:
                # Add latest former to the seed list
                # former_seed_list += self.q_trajectory[constraint.task_step-1][-(MAX_SEED-len(former_seed_list)):]

                # Or add some random sampled (q, yaw) pair to the seed list?
                selected_yaw = random.choices(self.yaws, k=MAX_SEED-len(former_seed_list))
                selected_q = uniform_q_sampling(MAX_SEED-len(former_seed_list))
                former_seed_list += [(q, yaw) for q, yaw in zip(selected_q, selected_yaw)]
                
            for reset_idx, (former_q, former_yaw) in enumerate(former_seed_list):
                orn = quat_from_normal_and_yaw(constraint.normal, former_yaw, constraint.horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)  
                self.robot.set_robot_config(self.current_config)
                q_goal = self.robot.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=former_q, ik_backend=ik_backend)
                self.robot.set_robot_config(self.current_config)
                if q_goal is not None:
                    # if np.linalg.norm(np.asarray(q_goal)-np.asarray(former_q)) > 1:
                    #     continue
                    q_goal_list.append((q_goal, former_yaw))
                    q_source_list.append({
                        "source_tag": source_tag,     # init or refine
                        "reset_idx": reset_idx,       # which one is it in former_list
                        "q": np.asarray(q_goal, dtype=float).tolist(),
                        "q_seed": np.asarray(former_q, dtype=float).tolist(),
                        "yaw": float(former_yaw),
                    })
        
        elif seed_tag == 'latter':
            # Select MAX_SEED representative former qs to trace 
            latter_seed_list = []
            # if reference_path: # Which means that there is selected q at this current step
            #     # the former step selected (q, yaw) should be added as the seed
            #     temp_best_former_q, temp_best_former_yaw = reference_path[constraint.task_step+1]
            #     latter_seed_list.append((temp_best_former_q, temp_best_former_yaw))

            #     # the nearest last step q should also be added as the seed
            #     temp_best_q, _ = reference_path[constraint.task_step]
            #     nearest_result = _find_nearest_q_yaw(self.q_trajectory[constraint.task_step+1], temp_best_q)
            #     if nearest_result is not None:
            #         nearest_idx, nearest_q, nearest_yaw = nearest_result
            #         if not np.allclose(np.asarray(nearest_q), np.asarray(temp_best_former_q)):
            #             latter_seed_list.append((nearest_q, nearest_yaw))
            if len(latter_seed_list)<MAX_SEED:
                # Add latest former to the seed list
                # latter_seed_list += self.q_trajectory[constraint.task_step+1][-(MAX_SEED-len(latter_seed_list)):] 

                # Or add some random sampled (q, yaw) pair to the seed list?
                selected_yaw = random.choices(self.yaws, k=MAX_SEED-len(latter_seed_list))
                selected_q = uniform_q_sampling(MAX_SEED-len(latter_seed_list))
                latter_seed_list += [(q, yaw) for q, yaw in zip(selected_q, selected_yaw)]
   
            for reset_idx, (letter_q, former_yaw) in enumerate(latter_seed_list):
                orn = quat_from_normal_and_yaw(constraint.normal, former_yaw, constraint.horizontal, finger_axis_is_plus_y=finger_axis_is_plus_y)  
                self.robot.set_robot_config(self.current_config)
                q_goal = self.robot.solve_ik_collision_aware(constraint.pos, orn, collision=False, max_trials=1, reset=True, q_reset=letter_q, ik_backend=ik_backend)
                self.robot.set_robot_config(self.current_config)
                if q_goal is not None:
                    # if np.linalg.norm(np.asarray(q_goal)-np.asarray(former_q)) > 1:
                    #     continue
                    q_goal_list.append((q_goal, former_yaw))
                    q_source_list.append({
                        "source_tag": source_tag,     # init or refine
                        "reset_idx": reset_idx,       # which one is it in former_list
                        "q": np.asarray(q_goal, dtype=float).tolist(),
                        "q_seed": np.asarray(letter_q, dtype=float).tolist(),
                        "yaw": float(former_yaw),
                    })
        
        self.q_trajectory[constraint.task_step] += q_goal_list
        self.q_source_trajectory[constraint.task_step] += q_source_list
        # print(f"|constraint step: {constraint.task_step} |q_goal_list_length: {len(q_goal_list)} |tag: {seed_tag}")


    def qs_refinement(self, q1, q2, step_size:float=0.01):
        q_backup = self.get_current_config()

        # get N1
        self.robot.set_robot_config(q1)
        J = self.robot.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N1 = np.eye(len(self.robot.joint_indices)) - J_pinv @ J  # nullspace projector

        # get N2
        self.robot.set_robot_config(q2)
        J = self.get_Jacobian()
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + 1e-4 * np.eye(6))
        N2 = np.eye(len(self.robot.joint_indices)) - J_pinv @ J  # nullspace projector

        # target vector
        q_delta = np.asarray(q2)-np.asarray(q1)

        q1_delta = N1@q_delta
        q2_delta = N2@q_delta

        print(np.linalg.norm(q_delta), np.linalg.norm(q1_delta), np.linalg.norm(q2_delta))
        self.robot.set_robot_config(q_backup)

        return (np.asarray(q1)+step_size*q1_delta).tolist(), (np.asarray(q2)-step_size*q2_delta).tolist()


    def get_yaw_candidates(self, mid_degree=90, num_steps:int=5, max_offset=np.deg2rad(60)):

        step = max_offset / float(max(1, num_steps))
        yaws = [np.deg2rad(mid_degree)]
        for k in range(1, num_steps + 1):
            offset = k * step
            yaws.append(yaws[0] + offset)
            yaws.append(yaws[0] - offset)
        return yaws

    def local_refine_around_edge(self, constraints, edge_idx, reference_path, iter_id, window=1):
        left = max(0, edge_idx - window)
        right = min(len(constraints) - 1, edge_idx + 1 + window)

        # backward: 用 latter，从 edge 左端往回 trace
        for constraint in constraints[left:edge_idx + 1][::-1]:
            self._trace_sample_redundant(
                constraint,
                source_tag={
                    "kind": "counterfactual_latter_refine",
                    "iter": iter_id,
                    "from_edge": edge_idx,
                },
                seed_tag="latter",
                reference_path=reference_path,
            )

        # forward: 用 former，从 edge 右端往前 trace
        for constraint in constraints[edge_idx + 1:right + 1]:
            self._trace_sample_redundant(
                constraint,
                source_tag={
                    "kind": "counterfactual_former_refine",
                    "iter": iter_id,
                    "from_edge": edge_idx,
                },
                seed_tag="former",
                reference_path=reference_path,
            )

    def solve_constraint_path(self, constraints: List[WaypointConstraint], method: Literal["Sampling", "Iteration", "RRT", "Greedy"], ik_backend:Literal["frankik", "pybullet"]="pybullet"):
        if method == "RRT":
            planner = ConstraintSequenceRRTPlanner(
                self.robot,
                joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]),
            )

            start_time = time.time()
            metric = planner.solve(constraints)
            total_time = time.time()-start_time
            print(f"Total time: {total_time}")
            print(f"Max edge cost is {metric['max_edge_cost']}, Total cost is {metric['total_cost']}")
            metric["time"] = total_time
            return metric
            
        if method == "Greedy":
            planner = ConstraintSequenceGreedyPlanner(
                self.robot,
                joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]),
            )

            start_time = time.time()
            metric = planner.solve(constraints)
            total_time = time.time()-start_time
            print(f"Total time: {total_time}")
            print(f"Max edge cost is {metric['max_edge_cost']}, Total cost is {metric['total_cost']}")
            metric["time"] = total_time
            return metric

        
        # Initialize the q_trajectory
        self.q_trajectory = [[] for _ in range(len(constraints))]
        self.q_source_trajectory = [[] for _ in range(len(constraints))]

        # Initialize the yaw candidates list
        self.yaws = self.get_yaw_candidates(num_steps=2)

        # Initialize the q_reset fed into the IK solver
        if method == "Iteration": 
            self.q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                # Q_RESET_SEEDS["left_elbow_out"],
                # Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            MaxIteration = 3

        elif method == "Sampling":
            self.q_reset_list = [
                Q_RESET_SEEDS["home"],
                # [0.21122026522160325, -0.44400245603577937, -0.23161109603481303, -2.743793599968008, -1.0309511129162083, 3.7166966782496167, -1.110594041641138], # this is the refined q_reset!
                # Q_RESET_SEEDS["left_relaxed"],
                # Q_RESET_SEEDS["right_relaxed"],
                Q_RESET_SEEDS["left_elbow_out"],
                Q_RESET_SEEDS["right_elbow_out"],
                # [(a+b)/2 for a, b in zip(planner.get_current_config(), Q_RESET_SEEDS["home"])],
            ]
            self.q_reset_list += uniform_q_sampling(10)
            MaxIteration = 0
 
        start_time = time.time()
        self.current_config = self.robot.get_current_config()
        for constraint in constraints:
            self._trace_sample_redundant(constraint, source_tag={"kind":"init","step":constraint.task_step}, seed_tag='former' if (constraint.task_step and method=="Iteration") else 'const', ik_backend=ik_backend)
            # self._sample_redundant(constraint, source_tag={"kind":"init","step":constraint.task_step})
            if len(self.q_trajectory[constraint.task_step]) <= 2:
                # q_goal = None
                self._trace_sample_redundant(constraint, source_tag={"kind":"init","step":constraint.task_step}, seed_tag='const', ik_backend=ik_backend)
                # raise RuntimeError(f"constraint{constraint.task_step} has no feasible q at all!")

        joint_weights = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float)

        planned_dict = dp_plan_yaw_path(feasible_by_step=self.q_trajectory, joint_weights=joint_weights)
        
        # Evaluation & Refinemnt!
        sorted_idx = 0
        for j in range(MaxIteration):
            self.path = planned_dict["path"]
            k = int(np.argmax(planned_dict["path_costs"]))
            print(
                f"Iter {j}, best max_edge={planned_dict['max_edge_cost']:.3f}, "
                f"worst_idx={k}, worst_cost={planned_dict['path_costs'][k]:.3f}"
            )

            if planned_dict["max_edge_cost"] < 0.8:
                break    

            # alts = find_counterfactual_alt_for_worst_edge(
            #     feasible_by_step=self.q_trajectory,
            #     best_pd=planned_dict,
            #     joint_weights=joint_weights,
            #     radius_list=(0.20, 0.35, 0.50, 0.80,),
            # )

            # beam_dicts = find_diverse_beam(
            #     feasible_by_step=self.q_trajectory,
            #     q_source_trajectory=self.q_source_trajectory,
            #     joint_weights=joint_weights,
            #     beam_width=2,
            #     min_diff_steps=4,
            #     q_eps=0.3
            # )
            # planned_dict = beam_dicts[0]
            # self.path = planned_dict['path']

            # selected_index, selected_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[sorted_idx%len(self.path)]
            # max_index, max_edge_cost = sorted(enumerate(planned_dict["path_costs"]),key=lambda x: x[1],reverse=True)[0]
            
            # Add some randomness ...
            # sorted_idx += 1
            # if sorted_idx > 3:
            #     sorted_idx = 0
            # import ipdb; ipdb.set_trace()
            # new_q_rest_list = [self.path[selected_index][0].tolist(), self.path[selected_index+1][0].tolist()] # + IK_seed_list # + uniform_q_sampling(10)
            # new_q_rest_list = uniform_q_sampling(7) 
            # q_noisy1 = path[selected_index][0] + np.random.normal(loc=0.0, scale=3, size=(2, 7))
            # q_noisy2 = path[selected_index+1][0] + np.random.normal(loc=0.0, scale=3, size=(2, 7))
            # new_q_rest_list = [self.robot.wrap_into_limits(q.tolist(), self.robot.home_config) for q in q_noisy1] + [self.robot.wrap_into_limits(q.tolist(), self.robot.home_config) for q in q_noisy2] 
            
            # self.q_reset_list = new_q_rest_list
            path_sources = [
                self.q_source_trajectory[step][cand_idx]
                for step, cand_idx in enumerate(planned_dict["indices"])
            ]
            # print(f"Path Sources:")
            # for path_source in path_sources:
            #     print(path_source)

            # print(f"Iter times: {j}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}, selected_index: {selected_index}")
            # if max_edge_cost < 0.8: # set this threashold accordingly
            #     break

            # print(f"\n=== Iter {j}: beam size = {len(beam_dicts)} ===")
            # for beam_id, pd in enumerate(beam_dicts):
            #     print(
            #         f"[beam {beam_id}] max_edge={pd['max_edge_cost']:.3f}, "
            #         f"total={pd['total_cost']:.3f}"
            #     )
            #     if beam_id > 0:
            #         print(
            #             f"         diff_steps={pd['diff_steps']}, "
            #             # f"source_diff_steps={pd['source_diff_steps']}, "
            #             f"q_dev_sum={pd['q_dev_sum']:.3f}"
            #         )

            # if planned_dict["max_edge_cost"] < 0.8:
            #     break


            # Refining...
            # refine_yaw_offset = np.deg2rad(60)

            # for i, constraint in enumerate(constraints):
            #     old_yaw = path[i][1]
            #     low_bound = max(np.deg2rad(30), old_yaw - refine_yaw_offset)
            #     high_bound = min(np.deg2rad(150), old_yaw + refine_yaw_offset)
            #     # self.yaws = np.linspace(low_bound, high_bound, 4).tolist() # It seems that this is not such useful!
            #     # self.yaws = [np.deg2rad(90)]
            #     self._sample_redundant(constraint, source_tag={"kind":"refine", "iter":j+1, "from_edge":max_index})
            # print(alts)
            for beam_id, pd in enumerate([planned_dict]):
                ref_path = pd['path']
                if j == 0:
                    selected_index = int(np.argmax(planned_dict["path_costs"]))
                else:
                    ranked_edges = sorted(enumerate(pd["path_costs"]), key=lambda x: x[1], reverse=True)
                    rank_weights = list(2*i for i in range(len(ranked_edges), 0, -1))
                    selected_index, _ = random.choices(ranked_edges, weights=rank_weights, k=1)[0]
                for i, constraint in enumerate((constraints[:selected_index+1])[::-1]):
                    self._trace_sample_redundant(constraint, source_tag={"kind":"latter_refine", "iter":j+1, "from_edge":constraint.task_step+1, 'beam':beam_id}, seed_tag='latter', reference_path=ref_path, ik_backend=ik_backend)
                for i, constraint in enumerate(constraints[selected_index+1:]):
                    self._trace_sample_redundant(constraint, source_tag={"kind":"former_refine", "iter":j+1, "from_edge":constraint.task_step-1, 'beam':beam_id}, seed_tag='former', reference_path=ref_path, ik_backend=ik_backend)

            # for step in range(len(self.q_trajectory)):
            #     self.q_trajectory[step], self.q_source_trajectory[step] = _dedup_layer(
            #         self.q_trajectory[step],
            #         self.q_source_trajectory[step],
            #     )
        
            planned_dict = dp_plan_yaw_path(feasible_by_step=self.q_trajectory, joint_weights=np.array([1, 1, 1, 1, 1, 1, 1]))


        total_time = time.time()-start_time
        print(f"Total time: {total_time}")
        path = planned_dict['path'] 
        # import ipdb; ipdb.set_trace()
        max_index, max_edge_cost = max(enumerate(planned_dict["path_costs"]), key=lambda x: x[1])
        print(f"Iter times: {MaxIteration}, Max edge cost: {max_edge_cost}, Total cost: {planned_dict['total_cost']}, worst_idx: {np.argmax(planned_dict['path_costs'])}")
        self.robot.set_robot_config(self.current_config)
        # import ipdb; ipdb.set_trace()
        return {
            "success": True,
            "total_cost": float(planned_dict["total_cost"]),
            "max_edge_cost": float(max_edge_cost),
            "planned_dict": planned_dict,
            "path": path,
            "time":total_time
        }

            
            



        
 
