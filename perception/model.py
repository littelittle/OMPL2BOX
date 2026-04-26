import numpy as np
import torch
from torch import nn


ANGLE_LABELS = {"box_base_yaw", "lid_angle", "flap_angle"}


def encode_labels(labels, label_names, angle_labels=ANGLE_LABELS):
    """Replace angle labels with cos/sin pairs."""
    is_torch = isinstance(labels, torch.Tensor)
    cos = torch.cos if is_torch else np.cos
    sin = torch.sin if is_torch else np.sin
    concat = torch.cat if is_torch else np.concatenate

    pieces = []
    encoded_names = []
    for i, name in enumerate([str(x) for x in label_names]):
        value = labels[..., i : i + 1]
        if name in angle_labels:
            pieces.extend([cos(value), sin(value)])
            encoded_names.extend([f"{name}_cos", f"{name}_sin"])
        else:
            pieces.append(value)
            encoded_names.append(name)
    return concat(pieces, axis=-1), encoded_names


def decode_labels(labels, label_names):
    """Restore cos/sin angle pairs back to theta labels."""
    is_torch = isinstance(labels, torch.Tensor)
    atan2 = torch.atan2 if is_torch else np.arctan2
    concat = torch.cat if is_torch else np.concatenate

    names = [str(x) for x in label_names]
    pieces = []
    decoded_names = []
    i = 0
    while i < len(names):
        name = names[i]
        if name.endswith("_cos") and i + 1 < len(names):
            base = name[:-4]
            if names[i + 1] == f"{base}_sin":
                theta = atan2(labels[..., i + 1 : i + 2], labels[..., i : i + 1])
                pieces.append(theta)
                decoded_names.append(base)
                i += 2
                continue
        pieces.append(labels[..., i : i + 1])
        decoded_names.append(name)
        i += 1
    return concat(pieces, axis=-1), decoded_names


class TinyPointNetRegressor(nn.Module):
    """Small PointNet-style regressor for fixed-size point clouds."""

    def __init__(self, out_dim=7, width=64):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width * 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, out_dim),
        )

    def forward(self, points):
        features = self.point_mlp(points)
        global_features = features.max(dim=1).values
        return self.head(global_features)
