import math

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalize(v, eps: float = 1e-8):
    n = _norm(v)
    if n < eps:
        return [0.0, 0.0, 0.0]
    return [x / n for x in v]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _mat_to_quat(R):
    """
    3x3 旋转矩阵 → quaternion [x, y, z, w]
    R 采用行主序，R[row][col]
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return [x, y, z, w]


def quat_from_normal_and_axis(normal_world, axis_world):
    """
    给定：
      - normal_world: flap 外表面的法向（世界系）
      - axis_world:   flap 铰链轴（世界系）
    构造末端姿态，使得：
      - 末端的 -Z 轴 对齐到 normal_world（吸盘沿 -Z 指向 flap 内部）
      - 末端的 X 轴 尽量对齐 axis_world
    """
    n = _normalize(normal_world)
    a = _normalize(axis_world)

    # 末端 -Z 轴指向 flap 内部：tool_z = -n
    z = [-n[0], -n[1], -n[2]]

    # 先用 a 当作 X 轴，再做一次正交化，避免和 z 太接近
    x = a
    if abs(_dot(x, z)) > 0.99:
        # 几乎共线，随便找一个和 z 不平行的向量做 Gram-Schmidt
        tmp = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
        x = _cross(tmp, z)
    # 正交化并单位化
    proj = _dot(x, z)
    x = [x[i] - proj * z[i] for i in range(3)]
    x = _normalize(x)

    # 再用右手定则算出 Y 轴
    y = _cross(z, x)
    y = _normalize(y)

    # 以列向量形式构造 R: [x, y, z]
    R = [
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
    ]
    return _mat_to_quat(R)