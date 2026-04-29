import numpy as np
from typing import Optional, List, Tuple

def pts2obj(pts, filename="pointcloud.obj"):
    with open(filename, "w") as f:
        for p in pts:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

def pybullet_depth_to_pointcloud(
    p, width=160, height=120,
    cam_pos=(0.0, -0.4, 1),
    target=(1.0, 0.0, 0.2),
    up=(0,0,1),
    fov=60, near=0.01, far=3.0,
    *, 
    exclude_body_links: Optional[List[Tuple[int, int]]] = None,   # [(bodyUniqueId, linkIndex), ...]
    exclude_bodies: Optional[List[int]] = None,                   # [bodyUniqueId, ...]
):
    view = p.computeViewMatrix(cam_pos, target, up)
    proj = p.computeProjectionMatrixFOV(fov, width/height, near, far)
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

    _, _, _, depth_buf, seg = p.getCameraImage(width, height, view, proj, flags=flags)
    depth_buf = np.asarray(depth_buf).reshape(height, width)
    seg = np.asarray(seg).reshape(height, width)

    # 先屏蔽背景（环境 seg 是 -1）
    base_valid = (seg >= 0)

    # 解码 objectUniqueId / linkIndex
    obj_uid = np.full_like(seg, -1, dtype=np.int32)
    link_idx = np.full_like(seg, -1, dtype=np.int32)
    seg_v = seg[base_valid]
    obj_uid[base_valid] = seg_v & ((1 << 24) - 1)
    link_idx[base_valid] = (seg_v >> 24) - 1

    # 基础过滤：去掉 plane / 背景
    valid = base_valid & (obj_uid >= 0)

    if exclude_bodies:
        for bid in exclude_bodies:
            valid &= (obj_uid != int(bid))

    # 核心：去掉被抓 flap link 的像素
    if exclude_body_links:
        for bid, lid in exclude_body_links:
            valid &= ~((obj_uid == int(bid)) & (link_idx == int(lid)))


    # OpenGL: pixel -> NDC
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_ndc = (2.0 * (u + 0.5) / width) - 1.0
    y_ndc = 1.0 - (2.0 * (v + 0.5) / height)      # 注意这里把图像y翻到OpenGL
    z_ndc = 2.0 * depth_buf - 1.0                 # depth in [0,1] -> z in [-1,1]

    ones = np.ones_like(z_ndc)
    pts_clip = np.stack([x_ndc, y_ndc, z_ndc, ones], axis=-1)[valid].reshape(-1, 4)

    # inv(P*V): clip -> world
    V = np.array(view).reshape(4, 4).T
    P = np.array(proj).reshape(4, 4).T
    invPV = np.linalg.inv(P @ V)

    pts_world_h = (invPV @ pts_clip.T).T
    pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]

    m = np.isfinite(pts_world).all(axis=1)
    return pts_world[m]

