import pybullet as p

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