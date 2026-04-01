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
    num_solver_iterations = int(pb.get("num_solver_iterations", 100))

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

def create_pedestal(cid, center_xy, size_xy=(0.40, 0.34), height=0.10, rgba=(0.6, 0.6, 0.6, 1.0)):
    """创建一个静态台子（mass=0），顶面高度=height，底面贴地 z=0。"""
    hx, hy, hz = size_xy[0] * 0.5, size_xy[1] * 0.5, height * 0.5
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba, physicsClientId=cid)
    pedestal_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[center_xy[0], center_xy[1], hz],  # z = height/2
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=cid,
    )
    return pedestal_id

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
