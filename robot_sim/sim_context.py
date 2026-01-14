from __future__ import annotations # let current file's type hints be a "string" and postponed evaluation

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pybullet as p
import pybullet_data

@dataclass(frozen=True)
class PhysicsConfig:
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    time_step: float = 1.0 / 240.0
    num_sub_steps: int = 1
    num_solver_iterations: int = 75

# NOTE: cid, gui, plane_id and physics all belong to the same world! should modified togther!
@dataclass
class SimContext:
    cid: int
    gui: bool
    plane_id: Optional[int]
    physics: PhysicsConfig

def physics_from_config(cfg: Dict[str, Any]) -> PhysicsConfig:
    pb = cfg.get("pybullet", {}) if isinstance(cfg, dict) else {}

    gravity = tuple(pb.get("gravity", (0.0, 0.0, -9.81)))
    time_step = float(pb.get("time_step", 1.0 / 240.0))
    num_sub_steps = int(pb.get("num_sub_steps", 1))
    num_solver_iterations = int(pb.get("num_solver_iterations", 75))

    return PhysicsConfig(
        gravity=gravity,
        time_step=time_step,
        num_sub_steps=num_sub_steps,
        num_solver_iterations=num_solver_iterations,
    )

def connect(gui: bool) -> int:
    return p.connect(p.GUI if gui else p.DIRECT)

def configure_physics(cid: int, physics: PhysicsConfig) -> None:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*physics.gravity, physicsClientId=cid)
    p.setTimeStep(physics.time_step, physicsClientId=cid)
    p.setPhysicsEngineParameter(
        numSubSteps=physics.num_sub_steps,
        numSolverIterations=physics.num_solver_iterations,
        physicsClientId=cid,
    )

def load_plane(cid: int) -> int:
    return p.loadURDF("plane.urdf", physicsClientId=cid)

def make_sim(
    gui: bool,
    physics: Optional[PhysicsConfig] = None,
    load_ground_plane: bool = True,
) -> SimContext:
    cid = connect(gui)
    physics = physics or PhysicsConfig()
    configure_physics(cid, physics)

    plane_id = load_plane(cid) if load_ground_plane else None
    return SimContext(cid=cid, gui=gui, plane_id=plane_id, physics=physics)
