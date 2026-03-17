import random

import pybullet as p

from models import FoldableBox
from planners import PandaGripperPlanner
from scene import create_pedestal
from tasks import Task


class FlapBoxTask(Task):
    def setup_scene(self):
        box_base_pos = self.config.get("foldable_box_pos", [0.7, 0.0, 0.1])
        box_base_orn = self.config.get(
            "foldable_box_orn",
            p.getQuaternionFromEuler([0, 0, random.uniform(-0.3, 0.3)]),
        )

        if self.config.get("robot", "panda") != "panda":
            raise NotImplementedError("ONLY SUPPORT PANDA FRANKA NOW")

        pedestal_h = self.config.get("pedestal_h", 0.20)
        create_pedestal(
            self.sim.cid,
            center_xy=[box_base_pos[0], box_base_pos[1]],
            height=pedestal_h,
        )

        box_base_pos = [box_base_pos[0], box_base_pos[1], pedestal_h + 0.07]
        self.foldable_box = FoldableBox(
            base_pos=box_base_pos,
            base_orn=box_base_orn,
            cid=self.sim.cid,
        )
        self.planner = PandaGripperPlanner(
            oracle_function=self.foldable_box.get_flap_keypoint_pose,
            cid=self.sim.cid,
            box_id=self.foldable_box.body_id,
            plane_id=self.sim.plane_id,
        )

    def run(self):
        self.planner.close_double_flap()
