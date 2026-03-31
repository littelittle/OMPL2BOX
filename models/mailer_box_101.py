import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import random
from utils.path import draw_point
from scene.sim_context import make_sim
from utils.vector import _normalize

class MailerBox:
    def __init__(self, cid, file_path, scaling=1.0, pos=[0.0, 0.0, 0.0], yaw=0.0, closed=False):
        self.cid = cid
        self.body_id = None
        self.lid_id = None
        self.flap_id = None
        self.file_path = file_path
        self.scaling = scaling
        self.pos = pos
        self.yaw = yaw
        
        self._load_urdf()

        # reset C-space
        # set two status: closed or open
        if closed:
            lid_rad = np.deg2rad(90)
            flap_rad = np.deg2rad(90)
        else:
            lid_rad  = np.deg2rad(-90)
            flap_rad = np.deg2rad(-90)

        p.resetJointState(
            bodyUniqueId=self.body_id,
            jointIndex=self.lid_id,
            targetValue=lid_rad,
            targetVelocity=0.0
        )

        p.resetJointState(
            bodyUniqueId=self.body_id,
            jointIndex=self.flap_id,
            targetValue=flap_rad,
            targetVelocity=0.0
        )

        for i in range(p.getNumJoints(self.body_id)):
            self._make_joint_passive(i)

    def _load_urdf(self):
        """
        FOR CJT file_path="assets/103/fixed.urdf"
        FOR ZHW file_path="assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf"
        """
        # self.scaling = 1.0
        pos = self.pos.copy()
        self.body_id = p.loadURDF(
            fileName=self.file_path,
            useFixedBase=True,
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi+np.deg2rad(self.yaw)]), 
            globalScaling=self.scaling,
            physicsClientId=self.cid,
        )
        self.lid_id = self._get_lid_jointId()
        self.flap_id = self._get_flap_jointId()

    def _get_lid_jointId(self,):
        for j in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, j)
            joint_name = info[1].decode("utf-8")
            if joint_name == "mailer_lid_0":
                return j
        raise ValueError("Lid not found!")

    def _get_flap_jointId(self,):
        for j in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, j)
            joint_name = info[1].decode("utf-8")
            if joint_name == "mailer_front_flap_0":
                return j
        raise ValueError("Flap not found!")
    
    def _make_joint_passive(self, joint_id):
        p.setJointMotorControl2(
            bodyUniqueId=self.body_id,
            jointIndex=joint_id,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,          # 关键：force=0 => 不施加任何电机力矩
        )

    def get_flap_keypoint_pose(self, lid_angle: float=None, flap_angle: float=None):
        if lid_angle is None:
            lid_angle = p.getJointState(self.body_id, self.lid_id, physicsClientId=self.cid)[0]
        if flap_angle is None:
            flap_angle = p.getJointState(self.body_id, self.flap_id, physicsClientId=self.cid)[0]
        # print(lid_angle, flap_angle)

        orign_config = p.getJointStates(self.body_id, [self.lid_id, self.flap_id], physicsClientId=self.cid)
        p.resetJointState(self.body_id, self.lid_id, targetValue=lid_angle, physicsClientId=self.cid)
        p.resetJointState(self.body_id, self.flap_id, targetValue=flap_angle, physicsClientId=self.cid)

        ls = p.getLinkState(self.body_id, self.flap_id, computeForwardKinematics=True, physicsClientId=self.cid)
        flap_pos_w, flap_orn_w = ls[4], ls[5]
        key_local = [i * self.scaling for i in [0.13, 0.0, 0.05]] 
        key_world, _ = p.multiplyTransforms(flap_pos_w, flap_orn_w, key_local, [0.0, 0.0, 0.0, 1.0], physicsClientId=self.cid)
        key_world = list(key_world)
        draw_point(key_world, size=0.1)
        # import ipdb; ipdb.set_trace()
        normal_local = [0.0, -1.0, 0.0]
        horizontal_local = [1.0, 0.0, 0.0]
        normal_world = p.multiplyTransforms([0.0, 0.0, 0.0], flap_orn_w, normal_local, [0.0, 0.0, 0.0, 1.0],  physicsClientId=self.cid)[0]
        normal_world = _normalize(list(normal_world))
        horizontal_world = p.multiplyTransforms([0.0, 0.0, 0.0], flap_orn_w, horizontal_local, [0.0, 0.0, 0.0, 1.0],  physicsClientId=self.cid)[0]
        horizontal_world = _normalize(list(horizontal_world))

        p.resetJointState(self.body_id, self.lid_id, targetValue=orign_config[0][0], physicsClientId=self.cid)
        p.resetJointState(self.body_id, self.flap_id, targetValue=orign_config[1][0], physicsClientId=self.cid)

        return key_world, normal_world, horizontal_world

if __name__ == "__main__":

    sim = make_sim(gui=True, load_ground_plane=True)
    cid = sim.cid
    mailerbox = MailerBox(cid)
    flap_link = mailerbox.flap_id
    box_id = mailerbox.body_id

    while True:
        ls = p.getLinkState(box_id, flap_link, computeForwardKinematics=True)
        pos_w, orn_w = ls[:2]
        p_local = [0.4, 0.0, 0.0]
        p_world, _ = p.multiplyTransforms(pos_w, orn_w, p_local, [0,0,0,1])
        draw_point(p_world)
        p.stepSimulation()

        for degree in range(0, 180, 5):
            rad = np.deg2rad(degree)
            pos, _ = mailerbox.get_flap_keypoint_pose(flap_angle=rad, lid_angle=rad)
            draw_point(pos, size=0.05)
            time.sleep(0.1)
        # time.sleep(0.1)

