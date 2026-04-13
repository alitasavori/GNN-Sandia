"""
Compare a real-valued MLP vs complex-valued MLP for voltage phasor prediction.

Inputs:
  - Node features per sample: p_load_kw, q_load_kvar
  - Flattened to [2 * N] before entering each MLP

Targets:
  - Complex bus voltage per node from vmag_pu and vang_deg
  - Output is V_re + j*V_im for each node

Outputs:
  - Trained checkpoints for both models
  - JSON report with magnitude/angle performance and training time

This script is designed to run on Colab or local Python.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_flattened_pq_and_complex_voltage(
    node_csv: Path,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
      X_flat: [S, 2N] float32 (flattened p/q)
      Y_ri_flat: [S, 2N] float32 (flattened real/imag voltage)
      node_ids: [N] int64
    """
    usecols = ["sample_id", "node_idx", "p_load_kw", "q_load_kvar", "vmag_pu", "vang_deg"]
    sample_ids: set[int] = set()
    node_ids: set[int] = set()
    total_rows = 0

    for chunk in pd.read_csv(node_csv, usecols=usecols, chunksize=500_000):
        sample_ids.update(chunk["sample_id"].astype(np.int64).tolist())
        node_ids.update(chunk["node_idx"].astype(np.int64).tolist())
        total_rows += len(chunk)

    sample_ids_sorted = np.array(sorted(sample_ids), dtype=np.int64)
    node_ids_sorted = np.array(sorted(node_ids), dtype=np.int64)
    S = len(sample_ids_sorted)
    N = len(node_ids_sorted)
    expected = S * N
    if total_rows != expected:
        raise ValueError(f"Row count mismatch: rows={total_rows}, expected S*N={expected}")

    sid_to_i = {int(sid): i for i, sid in enumerate(sample_ids_sorted)}
    nid_to_j = {int(nid): j for j, nid in enumerate(node_ids_sorted)}

    X = np.zeros((S, N, 2), dtype=np.float32)
    Y_ri = np.zeros((S, N, 2), dtype=np.float32)

    for chunk in pd.read_csv(node_csv, usecols=usecols, chunksize=500_000):
        sid = chunk["sample_id"].map(sid_to_i).to_numpy(dtype=np.int64)
        nid = chunk["node_idx"].map(nid_to_j).to_numpy(dtype=np.int64)
        p = chunk["p_load_kw"].to_numpy(dtype=np.float32)
        q = chunk["q_load_kvar"].to_numpy(dtype=np.float32)
        vmag = chunk["vmag_pu"].to_numpy(dtype=np.float32)
        vang_rad = np.deg2rad(chunk["vang_deg"].to_numpy(dtype=np.float32))
        v_re = vmag * np.cos(vang_rad)
        v_im = vmag * np.sin(vang_rad)
        X[sid, nid, 0] = p
        X[sid, nid, 1] = q
        Y_ri[sid, nid, 0] = v_re
        Y_ri[sid, nid, 1] = v_im

    X_flat = X.reshape(S, 2 * N)
    Y_ri_flat = Y_ri.reshape(S, 2 * N)
    return torch.from_numpy(X_flat), torch.from_numpy(Y_ri_flat), node_ids_sorted


class RealMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_re = nn.Parameter(torch.empty(out_features, in_features))
        self.w_im = nn.Parameter(torch.empty(out_features, in_features))
        self.b_re = nn.Parameter(torch.zeros(out_features))
        self.b_im = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.w_re)
        nn.init.xavier_uniform_(self.w_im)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr = z.real
        zi = z.imag
        out_re = zr @ self.w_re.t() - zi @ self.w_im.t() + self.b_re
        out_im = zr @ self.w_im.t() + zi @ self.w_re.t() + self.b_im
        return torch.complex(out_re, out_im)


class CReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))


class ComplexMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024):
        super().__init__()
        self.l1 = ComplexLinear(in_dim, hidden)
        self.l2 = ComplexLinear(hidden, hidden)
        self.l3 = ComplexLinear(hidden, out_dim)
        self.act = CReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.act(self.l1(z))
        z = self.act(self.l2(z))
        return self.l3(z)


@dataclass
class RunResult:
    best_val_mse: float
    test_mae_vmag: float
    test_rmse_vmag: float
    test_mae_angle_deg: float
    test_rmse_angle_deg: float
    train_seconds: float
    checkpoint: str


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _metrics_from_ri_flat(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> dict[str, float]:
    # Both [B, 2N] with layout [v_re(0..N-1), v_im(0..N-1)] after reshape logic below.
    B, two_n = pred_ri.shape
    n_nodes = two_n // 2
    pred = pred_ri.view(B, n_nodes, 2)
    true = true_ri.view(B, n_nodes, 2)

    pred_re, pred_im = pred[..., 0], pred[..., 1]
    true_re, true_im = true[..., 0], true[..., 1]

    pred_mag = torch.sqrt(pred_re * pred_re + pred_im * pred_im + 1e-12)
    true_mag = torch.sqrt(true_re * true_re + true_im * true_im + 1e-12)
    pred_ang = torch.atan2(pred_im, pred_re)
    true_ang = torch.atan2(true_im, true_re)
    ang_err_deg = _angle_diff_deg(pred_ang, true_ang)

    vmag_err = pred_mag - true_mag
    return {
        "mae_vmag_pu": float(vmag_err.abs().mean().item()),
        "rmse_vmag_pu": float(torch.sqrt((vmag_err * vmag_err).mean()).item()),
        "mae_angle_deg": float(ang_err_deg.abs().mean().item()),
        "rmse_angle_deg": float(torch.sqrt((ang_err_deg * ang_err_deg).mean()).item()),
    }


def _evaluate_real(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yp_n = model(xb)
            yp = yp_n * y_std.to(device) + y_mean.to(device)
            preds.append(yp.cpu())
            tgts.append(yb.cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    return _metrics_from_ri_flat(pred, tgt)


def _evaluate_complex(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    y_mean_re: torch.Tensor,
    y_std_re: torch.Tensor,
    y_mean_im: torch.Tensor,
    y_std_im: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            B, two_n = xb.shape
            n_nodes = two_n // 2
            x_ri = xb.view(B, n_nodes, 2)
            z_in = torch.complex(x_ri[..., 0], x_ri[..., 1]).reshape(B, n_nodes)
            z_pred_n = model(z_in)

            pred_re = z_pred_n.real * y_std_re.to(device) + y_mean_re.to(device)
            pred_im = z_pred_n.imag * y_std_im.to(device) + y_mean_im.to(device)
            pred = torch.stack([pred_re, pred_im], dim=-1).reshape(B, 2 * n_nodes)
            preds.append(pred.cpu())
            tgts.append(yb.cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    return _metrics_from_ri_flat(pred, tgt)


def train_real_mlp(
    X_n: torch.Tensor,
    Y: torch.Tensor,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    out_dir: Path,
    *,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> RunResult:
    model = RealMLP(X_n.shape[1], Y.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    mse = nn.MSELoss()

    train_ds = TensorDataset(X_n[idx_train], Y[idx_train])
    val_ds = TensorDataset(X_n[idx_val], Y[idx_val])
    test_ds = TensorDataset(X_n[idx_test], Y[idx_test])
    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    dl_te = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_train = Y[idx_train]
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Y_n = (Y - y_mean) / y_std

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = ((yb - y_mean) / y_std).to(device)
            opt.zero_grad(set_to_none=True)
            loss = mse(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = ((yb - y_mean) / y_std).to(device)
                lv = mse(model(xb), yb)
                val_loss += float(lv.item()) * xb.size(0)
                nv += xb.size(0)
        val_loss /= max(nv, 1)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0:
            print(f"[real] epoch {ep:4d}/{epochs} val_mse_norm={val_loss:.6f} best={best_val:.6f}", flush=True)
        if bad >= patience:
            print(f"[real] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    met = _evaluate_real(model, dl_te, device, y_mean, y_std)
    ckpt_path = out_dir / "real_mlp_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": int(X_n.shape[1]),
            "out_dim": int(Y.shape[1]),
            "hidden": int(hidden),
            "x_norm": {},
            "y_mean": y_mean,
            "y_std": y_std,
        },
        ckpt_path,
    )
    return RunResult(
        best_val_mse=float(best_val),
        test_mae_vmag=met["mae_vmag_pu"],
        test_rmse_vmag=met["rmse_vmag_pu"],
        test_mae_angle_deg=met["mae_angle_deg"],
        test_rmse_angle_deg=met["rmse_angle_deg"],
        train_seconds=float(train_seconds),
        checkpoint=str(ckpt_path.resolve()),
    )


def train_complex_mlp(
    X_n: torch.Tensor,
    Y: torch.Tensor,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    out_dir: Path,
    *,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> RunResult:
    n_nodes = X_n.shape[1] // 2
    model = ComplexMLP(n_nodes, n_nodes, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

    train_ds = TensorDataset(X_n[idx_train], Y[idx_train])
    val_ds = TensorDataset(X_n[idx_val], Y[idx_val])
    test_ds = TensorDataset(X_n[idx_test], Y[idx_test])
    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    dl_te = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_train = Y[idx_train].view(len(idx_train), n_nodes, 2)
    y_mean_re = y_train[..., 0].mean(dim=0, keepdim=True)
    y_std_re = y_train[..., 0].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    y_mean_im = y_train[..., 1].mean(dim=0, keepdim=True)
    y_std_im = y_train[..., 1].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device).view(xb.size(0), n_nodes, 2)
            x_ri = xb.view(xb.size(0), n_nodes, 2)
            z_in = torch.complex(x_ri[..., 0], x_ri[..., 1])
            y_re_n = (yb[..., 0] - y_mean_re.to(device)) / y_std_re.to(device)
            y_im_n = (yb[..., 1] - y_mean_im.to(device)) / y_std_im.to(device)
            z_tgt = torch.complex(y_re_n, y_im_n)

            opt.zero_grad(set_to_none=True)
            z_pred = model(z_in)
            loss = torch.mean((z_pred.real - z_tgt.real) ** 2 + (z_pred.imag - z_tgt.imag) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device).view(xb.size(0), n_nodes, 2)
                x_ri = xb.view(xb.size(0), n_nodes, 2)
                z_in = torch.complex(x_ri[..., 0], x_ri[..., 1])
                y_re_n = (yb[..., 0] - y_mean_re.to(device)) / y_std_re.to(device)
                y_im_n = (yb[..., 1] - y_mean_im.to(device)) / y_std_im.to(device)
                z_tgt = torch.complex(y_re_n, y_im_n)
                z_pred = model(z_in)
                lv = torch.mean((z_pred.real - z_tgt.real) ** 2 + (z_pred.imag - z_tgt.imag) ** 2)
                val_loss += float(lv.item()) * xb.size(0)
                nv += xb.size(0)
        val_loss /= max(nv, 1)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0:
            print(f"[cvnn] epoch {ep:4d}/{epochs} val_mse_complex_norm={val_loss:.6f} best={best_val:.6f}", flush=True)
        if bad >= patience:
            print(f"[cvnn] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    met = _evaluate_complex(model, dl_te, device, y_mean_re, y_std_re, y_mean_im, y_std_im)
    ckpt_path = out_dir / "cvnn_mlp_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim_complex": int(n_nodes),
            "out_dim_complex": int(n_nodes),
            "hidden": int(hidden),
            "y_mean_re": y_mean_re,
            "y_std_re": y_std_re,
            "y_mean_im": y_mean_im,
            "y_std_im": y_std_im,
        },
        ckpt_path,
    )
    return RunResult(
        best_val_mse=float(best_val),
        test_mae_vmag=met["mae_vmag_pu"],
        test_rmse_vmag=met["rmse_vmag_pu"],
        test_mae_angle_deg=met["mae_angle_deg"],
        test_rmse_angle_deg=met["rmse_angle_deg"],
        train_seconds=float(train_seconds),
        checkpoint=str(ckpt_path.resolve()),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare real MLP vs CVNN MLP for complex voltage prediction.")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg")
    p.add_argument(
        "--nodes_csv",
        type=str,
        default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    )
    p.add_argument("--out_dir", type=str, default="mlp_cvnn_compare_8500")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_real", type=int, default=1024)
    p.add_argument("--hidden_cvnn", type=int, default=768)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv)
    if not nodes_path.is_absolute():
        nodes_path = (data_root / nodes_path).resolve()
    if not nodes_path.is_file():
        raise FileNotFoundError(nodes_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading node CSV: {nodes_path}", flush=True)
    X_flat, Y_ri_flat, node_ids = _load_flattened_pq_and_complex_voltage(nodes_path)
    S = X_flat.shape[0]
    N = len(node_ids)
    print(f"Loaded samples={S}, nodes={N}, input_dim={X_flat.shape[1]}, output_dim={Y_ri_flat.shape[1]}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(S)
    n_train = int(S * args.train_frac)
    n_val = int(S * args.val_frac)
    n_test = S - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("Invalid split; require at least one sample in train/val/test.")
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]
    print(f"Split train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}", flush=True)

    x_mean = X_flat[idx_train].mean(dim=0, keepdim=True)
    x_std = X_flat[idx_train].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    X_n = (X_flat - x_mean) / x_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Training real MLP...", flush=True)
    real_result = train_real_mlp(
        X_n,
        Y_ri_flat,
        idx_train,
        idx_val,
        idx_test,
        out_dir,
        hidden=args.hidden_real,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )

    print("Training CVNN MLP...", flush=True)
    cvnn_result = train_complex_mlp(
        X_n,
        Y_ri_flat,
        idx_train,
        idx_val,
        idx_test,
        out_dir,
        hidden=args.hidden_cvnn,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )

    split = {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))}
    report = {
        "task": "Flattened PQ -> complex node voltages (V_re, V_im)",
        "dataset": str(nodes_path),
        "n_samples": int(S),
        "n_nodes": int(N),
        "input_features_per_node": ["p_load_kw", "q_load_kvar"],
        "n_features_per_node": 2,
        "split": split,
        "real_mlp": real_result.__dict__,
        "cvnn_mlp": cvnn_result.__dict__,
        "normalization": {
            "x_mean_path": str((out_dir / "x_mean.pt").resolve()),
            "x_std_path": str((out_dir / "x_std.pt").resolve()),
        },
    }

    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    report_path = out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Comparison complete ===", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(
        f"Real MLP:  |V| MAE={real_result.test_mae_vmag:.6f} pu, angle MAE={real_result.test_mae_angle_deg:.6f} deg, time={real_result.train_seconds:.1f}s",
        flush=True,
    )
    print(
        f"CVNN MLP:  |V| MAE={cvnn_result.test_mae_vmag:.6f} pu, angle MAE={cvnn_result.test_mae_angle_deg:.6f} deg, time={cvnn_result.train_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
