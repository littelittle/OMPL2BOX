import pybullet as p

cid1 = p.connect(p.GUI)
p.disconnect(cid1)
cid2 = p.connect(p.GUI)