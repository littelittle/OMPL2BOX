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

from perception.model import build_model, encode_labels, normalize_points_and_labels


def npz_string(data, key, default=None):
    if key not in data.files:
        return default
    return str(data[key].item())


class PointCloudDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.points = data["points"].astype(np.float32)
        raw_labels = data["labels"].astype(np.float32)
        if "label_names" in data.files:
            self.raw_label_names = [str(x) for x in data["label_names"].tolist()]
        else:
            self.raw_label_names = [str(i) for i in range(raw_labels.shape[1])]
        self.point_normalization = npz_string(data, "point_normalization")
        if self.point_normalization != "per_sample_center_scale":
            self.points, raw_labels, self.point_centers, self.point_scales = normalize_points_and_labels(
                self.points, raw_labels, self.raw_label_names
            )
            self.points = self.points.astype(np.float32)
            raw_labels = raw_labels.astype(np.float32)
            self.point_centers = self.point_centers.astype(np.float32)
            self.point_scales = self.point_scales.astype(np.float32)
            self.point_normalization = "per_sample_center_scale"
        else:
            self.point_centers = data["point_centers"].astype(np.float32)
            self.point_scales = data["point_scales"].astype(np.float32)
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
    labels = dataset.labels[np.asarray(indices)]
    label_mean = labels.mean(axis=0)
    label_std = labels.std(axis=0) + 1e-6
    return label_mean, label_std


class NormalizedSubset(Dataset):
    def __init__(self, dataset, indices, label_mean, label_std):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.label_mean = label_mean.astype(np.float32)
        self.label_std = label_std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        points, labels = self.dataset[int(self.indices[idx])]
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


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="perception/data/mailerbox_poc_keypoint_100k.npz")
    parser.add_argument("--output", default="perception/data/keypointNet.pt")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=["tiny", "pointnetplus", "pointnet2", "keypointkinematics", "keypoint", "twohead"],
        default=None,
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no_resume_optimizer",
        action="store_true",
        help="load model weights from --resume but start a fresh optimizer",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    resume_ckpt = load_checkpoint(args.resume) if args.resume else None
    dataset = PointCloudDataset(args.data)
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    if resume_ckpt is None:
        stats = compute_stats(dataset, train_idx)
    else:
        resume_label_names = resume_ckpt.get("output_label_names", resume_ckpt.get("label_names"))
        if resume_label_names is not None:
            resume_label_names = [str(x) for x in resume_label_names]
            if resume_label_names != dataset.label_names:
                raise ValueError(
                    "checkpoint output labels do not match this dataset: "
                    f"checkpoint={resume_label_names} dataset={dataset.label_names}"
                )
        label_mean = np.asarray(resume_ckpt["label_mean"], dtype=np.float32)
        label_std = np.asarray(resume_ckpt["label_std"], dtype=np.float32)
        if label_mean.shape != (dataset.labels.shape[1],) or label_std.shape != (dataset.labels.shape[1],):
            raise ValueError(
                "checkpoint label stats do not match this dataset: "
                f"checkpoint={label_mean.shape}/{label_std.shape} dataset={(dataset.labels.shape[1],)}"
            )
        stats = label_mean, label_std
    train_set = NormalizedSubset(dataset, train_idx, *stats)
    val_set = NormalizedSubset(dataset, val_idx, *stats)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda"
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda"
    )

    label_mean, label_std = stats
    model_name = args.model or (resume_ckpt.get("model", "keypoint") if resume_ckpt is not None else "keypoint")
    width = args.width or int(resume_ckpt.get("width", 96) if resume_ckpt is not None else 96)
    model = build_model(
        model_name,
        out_dim=dataset.labels.shape[1],
        width=width,
        label_names=dataset.label_names,
        label_mean=label_mean,
        label_std=label_std,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} model={model_name} train={len(train_set)} val={len(val_set)} params={param_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    start_epoch = 1
    if resume_ckpt is not None:
        state_key = "last_model_state" if "last_model_state" in resume_ckpt else "model_state"
        model.load_state_dict(resume_ckpt[state_key])
        best_state = state_dict_to_cpu(resume_ckpt.get("model_state", model.state_dict()))
        best_val = float(resume_ckpt.get("best_val_mse", best_val))
        resumed_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        if "optimizer_state" in resume_ckpt and not args.no_resume_optimizer:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                move_optimizer_state(optimizer, device)
            except (RuntimeError, ValueError) as exc:
                print(f"warning: could not load optimizer_state from {args.resume}: {exc}")
        set_optimizer_lr(optimizer, args.lr)
        print(f"resumed {args.resume} from epoch={resumed_epoch} state={state_key} best_val_mse={best_val:.6f}")

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = state_dict_to_cpu(model.state_dict())

        epoch_step = epoch - start_epoch + 1
        if epoch_step == 1 or epoch_step % 10 == 0 or epoch == end_epoch:
            print(f"epoch {epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "label_names": dataset.raw_label_names,
            "output_label_names": dataset.label_names,
            "point_normalization": "per_sample_center_scale",
            "label_mean": label_mean,
            "label_std": label_std,
            "model": model_name,
            "width": width,
            "last_model_state": state_dict_to_cpu(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "epoch": end_epoch,
            "best_val_mse": best_val,
        },
        output,
    )
    print(f"saved {output} best_val_mse={best_val:.6f}")


if __name__ == "__main__":
    main()
