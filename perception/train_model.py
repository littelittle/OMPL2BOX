import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.model import TinyPointNetRegressor, encode_labels


class PointCloudDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.points = data["points"].astype(np.float32)
        raw_labels = data["labels"].astype(np.float32)
        if "label_names" in data.files:
            self.raw_label_names = [str(x) for x in data["label_names"].tolist()]
        else:
            self.raw_label_names = [str(i) for i in range(raw_labels.shape[1])]
        self.labels, self.label_names = encode_labels(raw_labels, self.raw_label_names)
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]


def split_indices(n, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_size = max(1, int(n * val_ratio))
    return indices[val_size:], indices[:val_size]


def compute_stats(dataset, indices):
    points = dataset.points[indices]
    labels = dataset.labels[indices]
    point_mean = points.reshape(-1, 3).mean(axis=0)
    point_std = points.reshape(-1, 3).std(axis=0) + 1e-6
    label_mean = labels.mean(axis=0)
    label_std = labels.std(axis=0) + 1e-6
    return point_mean, point_std, label_mean, label_std


class NormalizedSubset(Dataset):
    def __init__(self, dataset, indices, point_mean, point_std, label_mean, label_std):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.point_mean = point_mean.astype(np.float32)
        self.point_std = point_std.astype(np.float32)
        self.label_mean = label_mean.astype(np.float32)
        self.label_std = label_std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        points, labels = self.dataset[int(self.indices[idx])]
        points = (points - self.point_mean) / self.point_std
        labels = (labels - self.label_mean) / self.label_std
        return torch.from_numpy(points), torch.from_numpy(labels)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for points, labels in loader:
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(points)
            loss = loss_fn(pred, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * points.shape[0]

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="perception/data/mailerbox_poc.npz")
    parser.add_argument("--output", default="perception/data/tiny_pointnet.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    dataset = PointCloudDataset(args.data)
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    stats = compute_stats(dataset, train_idx)
    train_set = NormalizedSubset(dataset, train_idx, *stats)
    val_set = NormalizedSubset(dataset, val_idx, *stats)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda"
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda"
    )

    model = TinyPointNetRegressor(out_dim=dataset.labels.shape[1], width=args.width).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} train={len(train_set)} val={len(val_set)} params={param_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    point_mean, point_std, label_mean, label_std = stats
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "label_names": dataset.raw_label_names,
            "output_label_names": dataset.label_names,
            "point_mean": point_mean,
            "point_std": point_std,
            "label_mean": label_mean,
            "label_std": label_std,
            "model": "TinyPointNetRegressor",
            "width": args.width,
        },
        output,
    )
    print(f"saved {output} best_val_mse={best_val:.6f}")


if __name__ == "__main__":
    main()
