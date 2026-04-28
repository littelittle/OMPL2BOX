import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pointcloud import pts2obj

if __name__ == "__main__":
    test_data = np.load("perception/data/test.npz")
    print(len(test_data["points"][0]))
    pts2obj(test_data["points"][1], "perception/data/pts.obj")
