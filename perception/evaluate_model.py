import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.model import TinyPointNetRegressor, decode_labels, encode_labels


ANGLE_LABELS = {"box_base_yaw", "lid_angle", "flap_angle"}


class PointCloudDataset(Dataset):
    def __init__(self, path, indices):
        data = np.load(path)
        self.points = data["points"].astype(np.float32)[indices]
        self.labels = data["labels"].astype(np.float32)[indices]
        if "label_names" in data.files:
            self.label_names = [str(x) for x in data["label_names"].tolist()]
        else:
            self.label_names = [str(i) for i in range(self.labels.shape[1])]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return torch.from_numpy(self.points[idx]), torch.from_numpy(self.labels[idx])


def split_indices(n, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_size = max(1, int(n * val_ratio))
    return indices[val_size:], indices[:val_size]


def select_indices(n, split, val_ratio, seed):
    train_idx, val_idx = split_indices(n, val_ratio, seed)
    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    return np.arange(n)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate(model, loader, ckpt, device, label_names):
    point_mean = torch.as_tensor(ckpt["point_mean"], dtype=torch.float32, device=device)
    point_std = torch.as_tensor(ckpt["point_std"], dtype=torch.float32, device=device)
    label_mean = torch.as_tensor(ckpt["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.as_tensor(ckpt["label_std"], dtype=torch.float32, device=device)
    output_label_names = ckpt.get("output_label_names", ckpt.get("label_names", label_names))

    pred_batches = []
    target_batches = []
    norm_mse_sum = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            points_norm = (points - point_mean) / point_std
            if labels.shape[-1] == label_mean.shape[-1]:
                labels_out = labels
            else:
                labels_out, _ = encode_labels(labels, label_names)
            labels_norm = (labels_out - label_mean) / label_std

            pred_norm = model(points_norm)
            pred_out = pred_norm * label_std + label_mean
            pred, _ = decode_labels(pred_out, output_label_names)

            norm_mse_sum += torch.mean((pred_norm - labels_norm) ** 2, dim=1).sum().item()
            count += points.shape[0]
            pred_batches.append(pred.cpu().numpy())
            target_batches.append(labels.cpu().numpy())

    preds = np.concatenate(pred_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return preds, targets, norm_mse_sum / count


def print_metrics(label_names, preds, targets, norm_mse):
    print(f"normalized_mse={norm_mse:.6f}")
    print("label                  mae          rmse         extra")
    for i, name in enumerate(label_names):
        err = preds[:, i] - targets[:, i]
        if name in ANGLE_LABELS:
            err = np.arctan2(np.sin(err), np.cos(err))
        mae_i = np.mean(np.abs(err))
        rmse_i = np.sqrt(np.mean(err**2))
        if name in ANGLE_LABELS:
            extra = f"mae_deg={np.rad2deg(mae_i):.3f} rmse_deg={np.rad2deg(rmse_i):.3f}"
        else:
            extra = ""
        print(f"{name:<18} {mae_i:>10.6f} {rmse_i:>10.6f}  {extra}")


def print_examples(label_names, preds, targets, num_examples):
    if num_examples <= 0:
        return
    print()
    print("examples:")
    for i in range(min(num_examples, len(preds))):
        print(f"[{i}]")
        for name, pred, target in zip(label_names, preds[i], targets[i]):
            print(f"  {name:<18} pred={pred: .6f} target={target: .6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="perception/data/mailerbox_poc.npz")
    parser.add_argument("--checkpoint", default="perception/data/pointnet_10k_lr3e-4_sincos.pt")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--save_predictions", default=None)
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    data = np.load(args.data)
    indices = select_indices(len(data["points"]), args.split, args.val_ratio, args.seed)
    dataset = PointCloudDataset(args.data, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = load_checkpoint(args.checkpoint)
    width = int(ckpt.get("width", 64))
    model = TinyPointNetRegressor(out_dim=len(ckpt["label_mean"]), width=width).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets, norm_mse = evaluate(model, loader, ckpt, device, dataset.label_names)
    print(f"device={device} split={args.split} samples={len(dataset)} width={width}")
    print_metrics(dataset.label_names, preds, targets, norm_mse)
    print_examples(dataset.label_names, preds, targets, args.num_examples)

    if args.save_predictions:
        output = Path(args.save_predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            pred=preds.astype(np.float32),
            target=targets.astype(np.float32),
            label_names=np.array(dataset.label_names),
            indices=indices,
        )
        print(f"saved {output}")


if __name__ == "__main__":
    main()
