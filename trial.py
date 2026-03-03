import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

NUM_ITER = 50
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=NUM_ITER)

def find_links_by_keyword(body_id, keywords=("finger", "tip")):
    out = []
    for ji in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, ji)
        link_name = info[12].decode("utf-8")  # linkName
        if any(k in link_name for k in keywords):
            out.append((ji, link_name))
    return out

def squeeze_gripper(panda_id, closing_vel=0.5, force=1000.0):
    jid1 = 9
    jid2 = 10

    # 往里关：速度取负（多数URDF里正方向是张开）
    p.setJointMotorControl2(panda_id, jid1, p.VELOCITY_CONTROL,
                            targetVelocity=-abs(closing_vel), force=force)
    p.setJointMotorControl2(panda_id, jid2, p.VELOCITY_CONTROL,
                            targetVelocity=-abs(closing_vel), force=force)

robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION,
    physicsClientId=cid,
)

link_id, link_name = find_links_by_keyword(robot_id, keywords=("finger",))[0]

squeeze_gripper(robot_id)

# for _ in range(4800):
#     p.stepSimulation()

num_steps = 1000
normal_force_hist = np.zeros(num_steps, dtype=float)
lateral_force_hist = np.zeros(num_steps, dtype=float)

for step in range(num_steps):
    cps = p.getContactPoints(bodyA=robot_id, linkIndexA=link_id)
    Fn_world = np.zeros(3, dtype=float)
    Ft_world = np.zeros(3, dtype=float)

    for cp in cps:
        n_hat = np.asarray(cp[7], dtype=float)
        normal_force = float(cp[9])

        t1_hat = np.asarray(cp[11], dtype=float)
        f1 = float(cp[10])

        t2_hat = np.asarray(cp[13], dtype=float)
        f2 = float(cp[12])

        Fn_world += normal_force * n_hat
        Ft_world += f1 * t1_hat + f2 * t2_hat

    normal_force_hist[step] = np.linalg.norm(Fn_world)
    lateral_force_hist[step] = np.linalg.norm(Ft_world)
    p.stepSimulation()

steps = np.arange(num_steps)
plt.figure(figsize=(9, 4.5))
plt.plot(steps, normal_force_hist, label="Contact force (normal)")
plt.plot(steps, lateral_force_hist, label="Lateral force")
plt.xlabel("Simulation step")
plt.ylabel("Force (N)")
plt.title("Contact and lateral force over 1000 simulation steps")
plt.legend()
plt.tight_layout()
plt.savefig(f"temp/contact_fig_{NUM_ITER}.png")
plt.show()
