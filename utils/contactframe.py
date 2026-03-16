from typing import List
from dataclasses import dataclass

@dataclass
class ContactFrame:
    # “接触帧”= 关键点 + 局部几何基（normal/axis/extended）+ 当前角度
    key: List[float]
    normal: List[float]
    axis: List[float]
    extended: List[float]
    angle: float
