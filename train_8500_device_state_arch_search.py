"""
IEEE 8500 — predict regulator taps + capacitor steps from nodal P/Q only (no graph / no edges).

Inputs:  `datasets_gnn2/loadtype_8500_dailyagg/gnn_node_features_and_targets.csv`
          columns `p_load_kw`, `q_load_kvar` for all nodes per `sample_id`.

Targets: `gnn_sample_meta.csv` device-state columns for the same `sample_id` row order.

Models are plain MLPs on the flattened vector [N * 2] per sample.

Saves checkpoints under `gnn2_architecture_search/<arch_name>/best.pt` and `summary.json`.

Usage (local):
  python train_8500_device_state_arch_search.py
  python train_8500_device_state_arch_search.py --epochs 80 --dataset-dir datasets_gnn2/loadtype_8500_dailyagg

Colab (typical; same as setup cell using REPO_DIR=/content/GNN-Sandia):
  !git clone https://github.com/alitasavori/GNN-Sandia.git /content/GNN-Sandia
  %cd /content/GNN-Sandia
  # Add datasets_gnn2/loadtype_8500_dailyagg, then:
  !python train_8500_device_state_arch_search.py --epochs 100

Or set env before running (any platform):
  export GNN_REPO=/content/GNN-Sandia
  python train_8500_device_state_arch_search.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def resolve_repo_root() -> Path:
    """Project root: GNN2_REPO or GNN_REPO env, else directory containing this file, else cwd."""
    env = (os.environ.get("GNN2_REPO") or os.environ.get("GNN_REPO") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

TARGET_COLS: tuple[str, ...] = (
    "reg_feeder_rega_tap_pu",
    "reg_feeder_regb_tap_pu",
    "reg_feeder_regc_tap_pu",
    "reg_vreg2_a_tap_pu",
    "reg_vreg2_b_tap_pu",
    "reg_vreg2_c_tap_pu",
    "reg_vreg3_a_tap_pu",
    "reg_vreg3_b_tap_pu",
    "reg_vreg3_c_tap_pu",
    "reg_vreg4_a_tap_pu",
    "reg_vreg4_b_tap_pu",
    "reg_vreg4_c_tap_pu",
    "cap_capbank0a_n_steps_on",
    "cap_capbank0b_n_steps_on",
    "cap_capbank0c_n_steps_on",
    "cap_capbank1a_n_steps_on",
    "cap_capbank1b_n_steps_on",
    "cap_capbank1c_n_steps_on",
    "cap_capbank2a_n_steps_on",
    "cap_capbank2b_n_steps_on",
    "cap_capbank2c_n_steps_on",
    "cap_capbank3_n_steps_on",
)

K_OUT = len(TARGET_COLS)


def _infer_n_nodes(node_csv: Path) -> int:
    """Max(node_idx)+1 from the node CSV (no edge file required)."""
    nmax = -1
    for chunk in pd.read_csv(node_csv, usecols=["node_idx"], chunksize=500_000):
        nmax = max(nmax, int(chunk["node_idx"].max()))
    return nmax + 1


def _load_meta_targets(meta_csv: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Meta row order defines training index 0..S-1."""
    df = pd.read_csv(meta_csv)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"gnn_sample_meta.csv missing columns: {missing}")
    sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
    Y = torch.from_numpy(df[list(TARGET_COLS)].to_numpy(dtype=np.float32))
    return sample_ids, Y


def _stream_node_features(
    node_csv: Path,
    S: int,
    n_nodes: int,
    sample_ids_meta: np.ndarray,
) -> torch.Tensor:
    """
    Stream the large node CSV in chunks (avoids loading multi-GB into RAM).
    Fills X[s, node_idx, :] using vectorized indexing per chunk.
    """
    X = np.zeros((S, n_nodes, 2), dtype=np.float32)
    use_direct = bool(
        len(sample_ids_meta) == S
        and sample_ids_meta.min() == 0
        and sample_ids_meta.max() == S - 1
        and np.array_equal(sample_ids_meta, np.arange(S, dtype=np.int64))
    )
    sid_map: dict[int, int] | None
    if use_direct:
        sid_map = None
    else:
        sid_map = {int(s): i for i, s in enumerate(sample_ids_meta)}

    chunksize = 500_000
    total = 0
    for chunk in pd.read_csv(
        node_csv,
        usecols=["sample_id", "node_idx", "p_load_kw", "q_load_kvar"],
        chunksize=chunksize,
    ):
        sid = chunk["sample_id"].to_numpy(np.int64)
        nj = chunk["node_idx"].to_numpy(np.int64)
        p = chunk["p_load_kw"].to_numpy(np.float32)
        q = chunk["q_load_kvar"].to_numpy(np.float32)
        if sid_map is None:
            X[sid, nj, 0] = p
            X[sid, nj, 1] = q
        else:
            row_s = chunk["sample_id"].map(sid_map)
            if row_s.isna().any():
                raise ValueError("Node CSV contains sample_id not present in gnn_sample_meta.csv")
            row = row_s.to_numpy(dtype=np.int64)
            X[row, nj, 0] = p
            X[row, nj, 1] = q
        total += len(chunk)
        if total % (chunksize * 10) == 0:
            print(f"  ... streamed {total} node rows", flush=True)

    expected = S * n_nodes
    if total != expected:
        raise ValueError(f"Node CSV row count {total} != S*N = {expected}")
    return torch.from_numpy(X)


def _mlp_trunk(in_dim: int, hidden: int, num_layers: int, dropout: float) -> nn.Sequential:
    """Stack of `num_layers` Linear+ReLU(+Dropout) blocks, all width `hidden` except first in_dim->hidden."""
    layers: list[nn.Module] = []
    d = in_dim
    for i in range(num_layers):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.ReLU())
        if dropout > 0 and i < num_layers - 1:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


class ArchShared22Heads(nn.Module):
    """Shared MLP trunk + one linear head per output (22 heads)."""

    def __init__(self, flat_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.trunk = _mlp_trunk(flat_dim, hidden, num_layers, dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(K_OUT)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.trunk(x.reshape(x.shape[0], -1))
        outs = [head(g) for head in self.heads]
        return torch.cat(outs, dim=1)


class ArchSingleMLPWide(nn.Module):
    """MLP trunk + single wide readout head to K_OUT."""

    def __init__(self, flat_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.trunk = _mlp_trunk(flat_dim, hidden, num_layers, dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, K_OUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.trunk(x.reshape(x.shape[0], -1))
        return self.readout(g)


class ArchDeepMLP5(nn.Module):
    """Deeper MLP (5 linear blocks in trunk) + linear readout."""

    def __init__(self, flat_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = _mlp_trunk(flat_dim, hidden, num_layers=5, dropout=dropout)
        self.readout = nn.Linear(hidden, K_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.trunk(x.reshape(x.shape[0], -1))
        return self.readout(g)


class ArchWideShallowMLP(nn.Module):
    """Shallow trunk (2 blocks), wide hidden, linear readout."""

    def __init__(self, flat_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = _mlp_trunk(flat_dim, hidden, num_layers=2, dropout=dropout)
        self.readout = nn.Linear(hidden, K_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.trunk(x.reshape(x.shape[0], -1))
        return self.readout(g)


class ArchGrouped5Heads(nn.Module):
    """Shared MLP trunk + 5 heads: feeder(3), vreg2(3), vreg3(3), vreg4(3), caps(10)."""

    def __init__(self, flat_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.trunk = _mlp_trunk(flat_dim, hidden, num_layers, dropout)
        self.h_feeder = nn.Linear(hidden, 3)
        self.h_v2 = nn.Linear(hidden, 3)
        self.h_v3 = nn.Linear(hidden, 3)
        self.h_v4 = nn.Linear(hidden, 3)
        self.h_caps = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.trunk(x.reshape(x.shape[0], -1))
        p1 = self.h_feeder(g)
        p2 = self.h_v2(g)
        p3 = self.h_v3(g)
        p4 = self.h_v4(g)
        p5 = self.h_caps(g)
        return torch.cat([p1, p2, p3, p4, p5], dim=1)


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "a1_shared_22heads": ArchShared22Heads,
    "a2_single_mlp_wide": ArchSingleMLPWide,
    "a3_deep_mlp5": ArchDeepMLP5,
    "a4_wide_shallow_mlp": ArchWideShallowMLP,
    "a5_grouped_5heads": ArchGrouped5Heads,
}


def _build_model(name: str, flat_dim: int, dropout: float) -> nn.Module:
    if name == "a1_shared_22heads":
        return ArchShared22Heads(flat_dim, hidden=64, num_layers=3, dropout=dropout)
    if name == "a2_single_mlp_wide":
        return ArchSingleMLPWide(flat_dim, hidden=96, num_layers=3, dropout=dropout)
    if name == "a3_deep_mlp5":
        return ArchDeepMLP5(flat_dim, hidden=64, dropout=dropout)
    if name == "a4_wide_shallow_mlp":
        return ArchWideShallowMLP(flat_dim, hidden=256, dropout=dropout)
    if name == "a5_grouped_5heads":
        return ArchGrouped5Heads(flat_dim, hidden=80, num_layers=4, dropout=dropout)
    raise ValueError(name)


def _split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr = perm[:n_train]
    va = perm[n_train : n_train + n_val]
    te = perm[n_train + n_val :]
    return tr, va, te


def train_one_arch(
    name: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    log_every: int,
) -> dict:
    flat_dim = int(X.shape[1] * X.shape[2])
    model = _build_model(name, flat_dim=flat_dim, dropout=0.1).to(device)

    Y_tr = Y[idx_train]
    mean = Y_tr.mean(dim=0, keepdim=True)
    std = Y_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Y_n = (Y - mean) / std

    X_tr = X[idx_train]
    xf = X_tr.reshape(-1, X.shape[-1])
    mean_x = xf.mean(dim=0, keepdim=True)
    std_x = xf.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    X_n = (X - mean_x) / std_x

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

    best_val = float("inf")
    best_state = None

    def batch_indices(idxs: np.ndarray, bs: int):
        for i in range(0, len(idxs), bs):
            yield idxs[i : i + bs]

    bs = min(64, max(8, X.shape[0] // 4))

    for ep in range(epochs):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        for bi in batch_indices(idx_train, bs):
            xb = X_n[bi].to(device)
            yb = Y_n[bi].to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_tr += float(loss.item()) * len(bi)
            n_tr += len(bi)
        loss_tr /= max(n_tr, 1)

        model.eval()
        with torch.no_grad():
            loss_va = 0.0
            n_va = 0
            for bi in batch_indices(idx_val, bs):
                xb = X_n[bi].to(device)
                yb = Y_n[bi].to(device)
                pred = model(xb)
                loss_va += float(F.mse_loss(pred, yb).item()) * len(bi)
                n_va += len(bi)
            loss_va /= max(n_va, 1)

        sched.step(loss_va)

        if log_every > 0 and (
            (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1
        ):
            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"    epoch {ep + 1:4d}/{epochs}  train_mse={loss_tr:.6f}  "
                f"val_mse={loss_va:.6f}  lr={cur_lr:.2e}",
                flush=True,
            )

        if loss_va < best_val:
            best_val = loss_va
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    arch_dir = out_dir / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "arch": name,
        "model_type": "mlp_nodal_flat",
        "flat_dim": flat_dim,
        "n_nodes": int(X.shape[1]),
        "model_state_dict": best_state,
        "y_mean": mean.cpu(),
        "y_std": std.cpu(),
        "x_mean": mean_x.cpu(),
        "x_std": std_x.cpu(),
        "target_cols": list(TARGET_COLS),
        "best_val_mse_normalized": float(best_val),
    }
    torch.save(ckpt, arch_dir / "best.pt")

    return {"arch": name, "best_val_mse_normalized": float(best_val), "ckpt": str(arch_dir / "best.pt")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Project root (default: GNN2_REPO or GNN_REPO env, else script directory, else cwd). "
        "Used to build default --dataset-dir / --out-dir when those are omitted.",
    )
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Folder with gnn_node_features_and_targets.csv and gnn_sample_meta.csv "
        "(default: <repo-root>/datasets_gnn2/loadtype_8500_dailyagg).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to save checkpoints and summary.json (default: <repo-root>/gnn2_architecture_search).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260327)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--node-cache",
        type=str,
        default="",
        help="Optional path to a .pt file: load/save stacked node tensor X [S,N,2] to skip re-streaming the large CSV.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print train/val MSE every N epochs (also epoch 1 and the last epoch). Use 0 to disable.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else resolve_repo_root()
    dset = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else (repo / "datasets_gnn2" / "loadtype_8500_dailyagg").resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "gnn2_architecture_search").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    node_csv = dset / "gnn_node_features_and_targets.csv"
    meta_csv = dset / "gnn_sample_meta.csv"
    for p in (node_csv, meta_csv):
        if not p.is_file():
            raise FileNotFoundError(p)

    print(f"[8500 device-state MLP] repo_root={repo}", flush=True)
    print(f"[8500 device-state MLP] dataset_dir={dset}", flush=True)

    sample_ids, Y = _load_meta_targets(meta_csv)
    S = len(sample_ids)
    print(f"  inferring N_nodes from {node_csv.name} (scan node_idx)...", flush=True)
    n_nodes = _infer_n_nodes(node_csv)

    cache_path = Path(args.node_cache).resolve() if args.node_cache else None
    if cache_path and cache_path.is_file():
        print(f"  loading node tensor from cache {cache_path}", flush=True)
        X = torch.load(cache_path, map_location="cpu")
    else:
        print(f"  streaming node features from {node_csv.name} (one pass, chunked)...", flush=True)
        X = _stream_node_features(node_csv, S, n_nodes, sample_ids)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(X, cache_path)
            print(f"  saved node tensor cache -> {cache_path}", flush=True)

    assert X.shape[0] == Y.shape[0] == S
    assert X.shape[1] == n_nodes and X.shape[2] == 2
    assert Y.shape[1] == K_OUT

    idx_train, idx_val, idx_test = _split_indices(len(sample_ids), args.seed, 0.7, 0.15)
    flat_dim = int(X.shape[1] * X.shape[2])
    print(f"  samples={len(sample_ids)} N_nodes={n_nodes} flat_dim={flat_dim} K_out={K_OUT}")
    print(f"  split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    print(f"  log_every={args.log_every}", flush=True)

    results = []
    for name in ARCHITECTURES:
        print(f"\n--- Training {name} ---")
        r = train_one_arch(
            name,
            X,
            Y,
            idx_train,
            idx_val,
            out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            log_every=args.log_every,
        )
        results.append(r)
        print(f"  best_val_mse_norm={r['best_val_mse_normalized']:.6f}")

    best = min(results, key=lambda r: r["best_val_mse_normalized"])
    summary = {
        "dataset_dir": str(dset),
        "out_dir": str(out_dir),
        "model_family": "mlp_nodal_flat",
        "n_samples": int(len(sample_ids)),
        "n_nodes": int(n_nodes),
        "flat_dim": flat_dim,
        "k_outputs": K_OUT,
        "target_cols": list(TARGET_COLS),
        "runs": results,
        "best_arch_by_val": best["arch"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[8500 device-state MLP] Done. Best by val MSE (normalized): {best['arch']}")
    print(f"  summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
