"""
LaTeX checklist Step 5 — MLP baseline for IEEE 8500-node load-type tensors.

Loads `dataset_tensors/X.pt`, `Y.pt`, and optionally `Y_angle.pt` (from
`assemble_dataset_tensors_8500.py`). Flattens inputs to one vector per sample,
z-score normalizes features from the training split, trains a 2-hidden-layer
MLP (512 units), and reports test MAE on voltage magnitude (pu).

Pass criterion (from checklist): test MAE < 0.005 pu on |V| — may require more
samples (5k–10k) than the default notebook smoke run.

Run after tensor assembly (notebook Step 5).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    _REPO = Path(__file__).resolve().parent
except NameError:
    _REPO = Path.cwd()

DEFAULT_DATASET_DIR = _REPO / "datasets_gnn2" / "loadtype_8500"
DEFAULT_TENSOR_SUBDIR = "dataset_tensors"
DEFAULT_OUT_SUBDIR = "mlp_baseline_8500"


class MLP8500(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_baseline_8500(
    dataset_dir: str | os.PathLike | None = None,
    tensor_subdir: str = DEFAULT_TENSOR_SUBDIR,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    angle_loss_weight: float = 0.1,
    seed: int = 20260322,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    dset = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    tdir = dset / tensor_subdir
    path_x = tdir / "X.pt"
    path_y = tdir / "Y.pt"
    path_ya = tdir / "Y_angle.pt"
    if not path_x.is_file() or not path_y.is_file():
        raise FileNotFoundError(f"Need {path_x} and {path_y}. Run assemble_dataset_tensors_8500 first.")

    try:
        X = torch.load(path_x, map_location="cpu", weights_only=True)
    except TypeError:
        X = torch.load(path_x, map_location="cpu")
    try:
        Y = torch.load(path_y, map_location="cpu", weights_only=True)
    except TypeError:
        Y = torch.load(path_y, map_location="cpu")
    X = X.float()
    Y = Y.float()
    if X.dim() != 3:
        raise ValueError(f"Expected X [S,N,F], got {tuple(X.shape)}")
    S, N, F = X.shape
    x_flat = X.reshape(S, N * F)

    use_angle = path_ya.is_file()
    if use_angle:
        try:
            Y_a = torch.load(path_ya, map_location="cpu", weights_only=True)
        except TypeError:
            Y_a = torch.load(path_ya, map_location="cpu")
        Y_a = Y_a.float()
        if Y_a.shape != (S, N):
            raise ValueError(f"Y_angle shape {tuple(Y_a.shape)} != Y {tuple(Y.shape)}")
        y_flat = torch.cat([Y, Y_a], dim=1)
        out_dim = 2 * N
    else:
        y_flat = Y
        out_dim = N

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(S, generator=g)
    n_train = int(S * train_frac)
    n_val = int(S * val_frac)
    n_test = S - n_train - n_val
    if n_test < 1:
        raise RuntimeError(f"Not enough samples S={S} for 80/10/10 split.")

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    x_tr = x_flat[idx_train]
    mean = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_flat_n = (x_flat - mean) / std

    ds_train = TensorDataset(x_flat_n[idx_train], y_flat[idx_train])
    ds_val = TensorDataset(x_flat_n[idx_val], y_flat[idx_val])
    ds_test = TensorDataset(x_flat_n[idx_test], y_flat[idx_test])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP8500(x_flat.shape[1], out_dim, hidden=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    dl_tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    def batch_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if use_angle:
            p_v, p_a = pred[:, :N], pred[:, N:]
            t_v, t_a = tgt[:, :N], tgt[:, N:]
            return torch.mean((p_v - t_v) ** 2) + angle_loss_weight * torch.mean((p_a - t_a) ** 2)
        return torch.mean((pred - tgt) ** 2)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += batch_loss(model(xb), yb).item() * xb.size(0)
                n_val_samples += xb.size(0)
        val_loss /= max(n_val_samples, 1)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{epochs}  val_loss={val_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    x_test, y_test = ds_test.tensors[0].to(device), ds_test.tensors[1].to(device)
    with torch.no_grad():
        pred = model(x_test)
        err_v = (pred[:, :N] - y_test[:, :N]).abs() if use_angle else (pred - y_test).abs()
        mae_pu = float(err_v.mean().cpu())

    out_dir = dset / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "mlp_8500.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "in_dim": int(x_flat.shape[1]),
            "out_dim": out_dim,
            "num_nodes": int(N),
            "num_feat": int(F),
            "use_angle": use_angle,
            "angle_loss_weight": angle_loss_weight,
        },
        ckpt,
    )
    meta = {
        "dataset_dir": str(dset.resolve()),
        "tensor_dir": str(tdir.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "samples_total": S,
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "test_mae_vmag_pu": mae_pu,
        "use_angle_in_loss": use_angle,
        "epochs": epochs,
        "pass_mae_under_0_005_pu": mae_pu < 0.005,
    }
    (out_dir / "mlp_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[train_mlp_8500] device={device!s}")
    print(f"  test MAE |V| (pu): {mae_pu:.6f}  (checklist target < 0.005 pu)")
    print(f"  saved {ckpt}")
    return meta


def main() -> None:
    train_mlp_baseline_8500()


if __name__ == "__main__":
    main()
