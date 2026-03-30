import math
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import pybullet as p

from utils.vector import _cross, _dot, _normalize

def _rotate_axis_angle(v, axis, angle: float):
    """Rodrigues 公式，在局部系中绕单位轴 axis 旋转向量 v。"""
    axis = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    axv = _cross(axis, v)
    ax_dot_v = _dot(axis, v)
    return [
        v[i] * c + axv[i] * s + axis[i] * ax_dot_v * (1.0 - c)
        for i in range(3)
    ]

class FoldableBox:
    """A simple foldable box with four top flaps driven by hinge joints (URDF-based)."""

    def __init__(self, base_pos, base_orn, cid):
        self.cid = cid
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.base_half_extents = None # [0.15, 0.12, 0.1]
        self.flap_len = None
        self.flap_width = None
        self.thickness = None
        self.open_angle = -1.35
        self._read_dimensions_from_urdf()
        self.base_pos[-1] += self.base_half_extents[-1]
        # import ipdb; ipdb.set_trace()
        self.body_id = self._load_urdf()
        self.flap_joint_indices = list(range(4))

    # ----------------------------- model build -----------------------------
    def _asset_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent/ "assets" / "foldable_box_small.urdf")

    def _load_urdf(self):
        body_id = p.loadURDF(
            fileName=self._asset_path(),
            basePosition=self.base_pos,
            # flip 90 degrees to have flaps point upwards initially
            # baseOrientation=[0, 0, math.sin(-math.pi / 4), math.cos(-math.pi / 4)],
            baseOrientation=self.base_orn,
            useFixedBase=False,
            physicsClientId=self.cid,
        )
        for j in range(p.getNumJoints(body_id, physicsClientId=self.cid)):
            p.resetJointState(body_id, j, targetValue=0.0, physicsClientId=self.cid)
            
        return body_id

    # ----------------------------- URDF parsing ------------------------------
    def _read_dimensions_from_urdf(self):
        """
        从 URDF 读取：
          - base_half_extents: base_link box size / 2
          - flap_len / flap_width / thickness: 从 flap_* link 的 box size 推断
        """
        urdf_path = Path(self._asset_path())
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        root = ET.parse(str(urdf_path)).getroot()

        def _find_link(name: str):
            for link in root.findall("link"):
                if link.get("name") == name:
                    return link
            return None

        def _read_box_size_from_link(link_name: str) -> Optional[List[float]]:
            """
            优先从 <visual><geometry><box size="..."> 读，
            读不到就从 <collision> 读。
            返回 [sx, sy, sz]（米）。
            """
            link = _find_link(link_name)
            if link is None:
                return None

            def _read_from(tag_name: str) -> Optional[List[float]]:
                tag = link.find(tag_name)
                if tag is None:
                    return None
                geom = tag.find("geometry")
                if geom is None:
                    return None
                box = geom.find("box")
                if box is None:
                    return None
                size_str = box.get("size")
                if not size_str:
                    return None
                vals = [float(x) for x in size_str.strip().split()]
                if len(vals) != 3:
                    return None
                return vals

            return _read_from("visual") or _read_from("collision")

        # --- base ---
        base_size = _read_box_size_from_link("base_link")
        if base_size is None:
            raise RuntimeError("Cannot read base_link <box size=...> from URDF.")
        self.base_half_extents = [0.5 * base_size[0], 0.5 * base_size[1], 0.5 * base_size[2]]

        # --- flaps ---
        flap_px_size = _read_box_size_from_link("flap_px")
        flap_py_size = _read_box_size_from_link("flap_py")

        if flap_px_size is None or flap_py_size is None:
            raise RuntimeError("Cannot read flap_px / flap_py <box size=...> from URDF.")

        # thickness：优先用 flap_px 的 z（通常都一样）
        self.thickness = float(flap_px_size[2])

        # flap_len/flap_width 的语义：
        # - +X/-X flap：延伸方向在 X，铰链方向在 Y
        flap_len_x = float(flap_px_size[0])
        flap_width_x = float(flap_px_size[1])

        # - +Y/-Y flap：延伸方向在 Y，铰链方向在 X
        flap_len_y = float(flap_py_size[1])
        flap_width_y = float(flap_py_size[0])

        # 你原来用单一 flap_len / flap_width（统一规格），这里做一个一致性检查：
        # 如果 x/y flap 尺寸不同（比如你故意做了长短不同的 flap），你可以把下面改成“按 flap_id 返回不同长度”。
        tol = 1e-9
        if abs(flap_len_x - flap_len_y) > tol or abs(flap_width_x - flap_width_y) > tol:
            # 不强行报错：但提示一下，并用 x-flap 作为默认 flap_len/width
            print(
                "[FoldableBox][WARN] flap sizes differ between x-flaps and y-flaps. "
                f"x: (len={flap_len_x}, width={flap_width_x}) "
                f"y: (len={flap_len_y}, width={flap_width_y}). "
                "Using x-flap values for flap_len/flap_width."
            )

        self.flap_len = flap_len_x
        self.flap_width = flap_width_x

    # ----------------------------- control utils -----------------------------
    def set_flap_angle(self, flap_id: int, angle: float):
        p.resetJointState(
            self.body_id,
            int(flap_id),
            targetValue=angle,
            targetVelocity=0.0,
            physicsClientId=self.cid,
        )

    def open_all(self, angle: Optional[float] = None):
        ang = self.open_angle if angle is None else float(angle)
        for i in self.flap_joint_indices:
            self.set_flap_angle(i, ang)

    def get_flap_keypoint_pose(
        self,
        flap_id: int,
        angle: float = None,
        edge_ratio: float = 0.8,
        ) -> Tuple[List[float], List[float], List[float]]:
        """
        在给定 flap 角度下，返回 flap 外侧关键点的世界坐标、法向和铰链轴。
        约定：
        - box 局部坐标系：原点在箱体中心，+z 向上，x/y 对应箱体长宽方向；
        - angle = 0.0 ：flap 与箱体顶面共平面（完全“平”在箱口上）；
        - angle > 0 ：绕铰链轴按右手定则旋转，使 flap 朝“侧面”方向竖起（大约 90° 时竖直）。
        
        参数：
        - flap_id: 0 → +x 侧 flap
                    1 → -x 侧 flap
                    2 → +y 侧 flap
                    3 → -y 侧 flap
        - angle:  flap 绕铰链转动的角度（弧度），从“平放在顶面”的姿态开始计。
        - edge_ratio: 关键点沿 flap 长度方向距离铰链的比例（0~1），接近 1 表示靠近自由边。
        """
        assert 0 <= flap_id < 4

        if angle is None:
            angle = p.getJointState(
                self.body_id, flap_id, physicsClientId=self.cid
            )[0]

        # 当前 box 的基座位姿（注意：不要再只用 self.base_pos，pick-place 会修改 base pose）
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.cid
        )

        hx, hy, hz = self.base_half_extents
        key_dist = float(self.flap_len) * float(edge_ratio)

        # flap 关闭时（贴在箱口上）的法向：朝上
        n_closed = [0.0, 0.0, 1.0]

        # 在 box 局部坐标系下定义各 flap 的铰链位置和轴
        if flap_id == 0:      # +x 边
            hinge_local = [hx, 0.0, hz]
            axis_local = [0.0, 1.0, 0.0]   # 绕 +y 转
        elif flap_id == 1:    # -x 边
            hinge_local = [-hx, 0.0, hz]
            axis_local = [0.0, -1.0, 0.0]  # 绕 -y 转
        elif flap_id == 2:    # +y 边
            hinge_local = [0.0, hy, hz]
            axis_local = [-1.0, 0.0, 0.0]  # 绕 -x 转
        else:                 # -y 边
            hinge_local = [0.0, -hy, hz]
            axis_local = [1.0, 0.0, 0.0]   # 绕 +x 转

        axis_local = _normalize(axis_local)

        # flap 关闭时，铰链到关键点的向量：
        #   - 方向指向箱体内部（而不是外面），否则打开时会把关键点旋到箱体里面去
        t_inward = _cross(axis_local, n_closed)
        t_inward = [x for x in t_inward]          # 取“向内”的方向
        t_inward = _normalize(t_inward)

        offset_closed = [t_inward[i] * key_dist for i in range(3)]

        # 在 flap 平放时，关键点在 box 局部系的位置
        key_local_closed = [
            hinge_local[i] + offset_closed[i] for i in range(3)
        ]

        # 把 offset 和法向都绕铰链轴旋转 angle，得到目标角度下的 offset / normal
        offset_rot = _rotate_axis_angle(offset_closed, axis_local, angle)
        normal_rot = _rotate_axis_angle(n_closed, axis_local, angle)

        key_local = [hinge_local[i] + offset_rot[i] for i in range(3)]

        # 从 box 局部系变到世界系
        key_world, _ = p.multiplyTransforms(
            base_pos,
            base_orn,
            key_local,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )
        # import ipdb; ipdb.set_trace()

        # 法向和铰链轴只需要旋转，不带平移
        normal_world = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            base_orn,
            normal_rot,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )[0]

        axis_world = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            base_orn,
            axis_local,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )[0]

        # compute the third axis based on normal and axis
        extended_world = _cross(axis_world, normal_world)
        extended_world = _normalize(extended_world)

        return key_world, normal_world, axis_world, extended_world, angle

    def get_flap_target_pose(
        self, flap_id: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Return an outer-face contact point (near the free edge), orientation, and outward direction.
        """
        hx, hy, hz = self.base_half_extents
        base_x, base_y, base_z = self.base_pos
        edge_bias = 0.3 * self.flap_len
        height_bias = 0.02

        if flap_id == 0:  # +x
            pt = [base_x + hx + edge_bias, base_y, base_z + hz + height_bias]
            outward = [1, 0, 0]
        elif flap_id == 1:  # -x
            pt = [base_x - hx - edge_bias, base_y, base_z + hz + height_bias]
            outward = [-1, 0, 0]
        elif flap_id == 2:  # +y
            pt = [base_x, base_y + hy + edge_bias, base_z + hz + height_bias]
            outward = [0, 1, 0]
        else:  # -y
            pt = [base_x, base_y - hy - edge_bias, base_z + hz + height_bias]
            outward = [0, -1, 0]

        orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        return pt, orn, outward
