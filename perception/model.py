import numpy as np
import torch
from torch import nn


ANGLE_LABELS = {"box_base_yaw", "lid_angle", "flap_angle"}
POSITION_LABELS = ("x1", "y1", "z1")
LENGTH_LABELS = {"lid_length"}


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


def normalize_points_and_labels(points, labels=None, label_names=None, eps=1e-6):
    """Center and scale one point cloud, and apply the same transform to coordinate labels."""
    is_torch = isinstance(points, torch.Tensor)
    if is_torch:
        center = points.mean(dim=-2, keepdim=True)
        centered = points - center
        scale = torch.linalg.norm(centered, dim=-1).amax(dim=-1, keepdim=True).clamp_min(eps)
        points_norm = centered / scale.unsqueeze(-1)
        center = center.squeeze(-2)
        scale = scale.squeeze(-1)
    else:
        center = points.mean(axis=-2, keepdims=True)
        centered = points - center
        scale = np.maximum(np.linalg.norm(centered, axis=-1).max(axis=-1, keepdims=True), eps)
        points_norm = centered / np.expand_dims(scale, axis=-1)
        center = np.squeeze(center, axis=-2)
        scale = np.squeeze(scale, axis=-1)

    if labels is None:
        return points_norm, None, center, scale
    if label_names is None:
        raise ValueError("label_names is required when labels are provided")

    labels_norm = labels.clone() if is_torch else labels.copy()
    names = [str(x) for x in label_names]
    for axis, name in enumerate(POSITION_LABELS):
        if name in names:
            i = names.index(name)
            labels_norm[..., i] = (labels_norm[..., i] - center[..., axis]) / scale
    for name in LENGTH_LABELS:
        if name in names:
            i = names.index(name)
            labels_norm[..., i] = labels_norm[..., i] / scale
    return points_norm, labels_norm, center, scale


def denormalize_label_coordinates(labels, label_names, center, scale):
    """Restore coordinate and length labels from per-cloud normalized space."""
    is_torch = isinstance(labels, torch.Tensor)
    labels_world = labels.clone() if is_torch else labels.copy()
    names = [str(x) for x in label_names]
    for axis, name in enumerate(POSITION_LABELS):
        if name in names:
            i = names.index(name)
            labels_world[..., i] = labels_world[..., i] * scale + center[..., axis]
    for name in LENGTH_LABELS:
        if name in names:
            i = names.index(name)
            labels_world[..., i] = labels_world[..., i] * scale
    return labels_world


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


class PointNetPlusRegressor(nn.Module):
    """Stronger PointNet-like regressor with two-stage global feature fusion."""

    def __init__(self, out_dim=7, width=96):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Linear(3, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(width * 3 + 3, width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(width * 2, width * 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 6, width * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(width * 3, width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(width * 2, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, out_dim),
        )

    def forward(self, points):
        local = self.local_mlp(points)
        global_max = local.max(dim=1, keepdim=True).values
        global_mean = local.mean(dim=1, keepdim=True)
        global_context = torch.cat(
            [global_max.expand_as(local), global_mean.expand_as(local), local, points], dim=-1
        )
        fused = self.fusion_mlp(global_context)
        pooled = torch.cat(
            [
                fused.max(dim=1).values,
                fused.mean(dim=1),
                fused.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.head(pooled)


def build_model(name="tiny", out_dim=7, width=64):
    model_name = str(name).lower()
    if model_name in {"tiny", "tinypointnetregressor"}:
        return TinyPointNetRegressor(out_dim=out_dim, width=width)
    if model_name in {"pointnetplus", "pointnet++", "pointnet2", "plus"}:
        return PointNetPlusRegressor(out_dim=out_dim, width=width)
    raise ValueError(f"Unknown model: {name}")
