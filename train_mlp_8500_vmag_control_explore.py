"""
IEEE 8500 — five MLP architectures: nodal P/Q + post-solve cap Q + regulator taps -> |V| (pu) per node.

Uses `datasets_gnn2/loadtype_8500_dailyagg/` CSVs:
  - Node CSV: `p_load_kw`, `q_load_kvar`, `vmag_pu` per (sample_id, node_idx)
  - Meta CSV: columns `reg_*_tap_pu` and `cap_*_q_post_kvar` (from run_daily_aggregate_dataset_8500.py)

Design note: a single huge Linear(2N+G -> N) would be parameter-heavy; these variants compress
through a bottleneck before a final Linear(hidden -> N).

Every 10 epochs prints validation MAE and RMSE in **physical pu** (denormalized |V|).

Saves under `<out-dir>/<arch_name>/best.pt` and `summary.json`.

Usage:
  python train_mlp_8500_vmag_control_explore.py --epochs 100
  python train_mlp_8500_vmag_control_explore.py --dataset-dir datasets_gnn2/loadtype_8500_dailyagg --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_repo_root() -> Path:
    env = (os.environ.get("GNN2_REPO") or os.environ.get("GNN_REPO") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


REG_TAP_COLS: tuple[str, ...] = (
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
)


def _cap_q_post_cols(meta_columns: list[str]) -> list[str]:
    pat = re.compile(r"^cap_.*_q_post_kvar$")
    return sorted([c for c in meta_columns if pat.match(c)])


def _infer_n_nodes(node_csv: Path) -> int:
    nmax = -1
    for chunk in pd.read_csv(node_csv, usecols=["node_idx"], chunksize=500_000):
        nmax = max(nmax, int(chunk["node_idx"].max()))
    return nmax + 1


def _load_meta_globals(meta_csv: Path) -> tuple[np.ndarray, torch.Tensor, list[str]]:
    df = pd.read_csv(meta_csv)
    missing = [c for c in REG_TAP_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"gnn_sample_meta.csv missing regulator columns: {missing}. "
            "Regenerate with run_daily_aggregate_dataset_8500.py."
        )
    cap_cols = _cap_q_post_cols(list(df.columns))
    if not cap_cols:
        raise ValueError(
            "No cap_*_q_post_kvar columns in gnn_sample_meta.csv. "
            "Regenerate dataset with the updated run_daily_aggregate_dataset_8500.py."
        )
    gcols = list(REG_TAP_COLS) + cap_cols
    sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
    G = torch.from_numpy(df[gcols].to_numpy(dtype=np.float32))
    return sample_ids, G, gcols


def _stream_pq_v(
    node_csv: Path,
    S: int,
    n_nodes: int,
    sample_ids_meta: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """X [S,N,2] for P,Q and Y [S,N] for vmag_pu."""
    X = np.zeros((S, n_nodes, 2), dtype=np.float32)
    Y = np.zeros((S, n_nodes), dtype=np.float32)
    use_direct = bool(
        len(sample_ids_meta) == S
        and sample_ids_meta.min() == 0
        and sample_ids_meta.max() == S - 1
        and np.array_equal(sample_ids_meta, np.arange(S, dtype=np.int64))
    )
    sid_map: dict[int, int] | None = None
    if not use_direct:
        sid_map = {int(s): i for i, s in enumerate(sample_ids_meta)}

    chunksize = 500_000
    total = 0
    for chunk in pd.read_csv(
        node_csv,
        usecols=["sample_id", "node_idx", "p_load_kw", "q_load_kvar", "vmag_pu"],
        chunksize=chunksize,
    ):
        sid = chunk["sample_id"].to_numpy(np.int64)
        nj = chunk["node_idx"].to_numpy(np.int64)
        p = chunk["p_load_kw"].to_numpy(np.float32)
        q = chunk["q_load_kvar"].to_numpy(np.float32)
        v = chunk["vmag_pu"].to_numpy(np.float32)
        if sid_map is None:
            X[sid, nj, 0] = p
            X[sid, nj, 1] = q
            Y[sid, nj] = v
        else:
            row_s = chunk["sample_id"].map(sid_map)
            if row_s.isna().any():
                raise ValueError("Node CSV sample_id not in gnn_sample_meta.csv")
            row = row_s.to_numpy(dtype=np.int64)
            X[row, nj, 0] = p
            X[row, nj, 1] = q
            Y[row, nj] = v
        total += len(chunk)
        if total % (chunksize * 10) == 0:
            print(f"  ... streamed {total} node rows", flush=True)

    expected = S * n_nodes
    if total != expected:
        raise ValueError(f"Node CSV row count {total} != S*N = {expected}")
    return torch.from_numpy(X), torch.from_numpy(Y)


def _split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr = perm[:n_train]
    va = perm[n_train : n_train + n_val]
    te = perm[n_train + n_val :]
    return tr, va, te


# ----- five architectures (bottleneck before N-wide readout) -----


class ArchBottleneck512(nn.Module):
    """(2N+G) -> 512 -> 512 -> N."""

    def __init__(self, in_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        h = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArchDeepBottleneck384(nn.Module):
    """Deeper squeeze: in -> 768 -> 384 -> 384 -> N."""

    def __init__(self, in_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArchLowRank256(nn.Module):
    """Strong bottleneck: in -> 256 -> N (fewest params in trunk)."""

    def __init__(self, in_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArchResidual640(nn.Module):
    """Project to 640, two residual FC blocks, readout N."""

    def __init__(self, in_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        h = 640
        self.in_proj = nn.Linear(in_dim, h)
        self.b1 = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
        )
        self.b2 = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
        )
        self.out = nn.Linear(h, n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.in_proj(x))
        z = z + self.b1(z)
        z = z + self.b2(z)
        return self.out(z)


class ArchTwinEncoder(nn.Module):
    """Encode PQ_flat and G separately, fuse, then MLP -> N (inductive bias for globals)."""

    def __init__(self, pq_dim: int, g_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        self.pq_dim = int(pq_dim)
        self.enc_pq = nn.Sequential(
            nn.Linear(pq_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.GELU(),
        )
        self.enc_g = nn.Sequential(
            nn.Linear(g_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(384 + 128, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq, g = x[:, : self.pq_dim], x[:, self.pq_dim :]
        u = self.enc_pq(pq)
        v = self.enc_g(g)
        return self.head(torch.cat([u, v], dim=1))


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "e1_bottleneck512": ArchBottleneck512,
    "e2_deep_bottleneck384": ArchDeepBottleneck384,
    "e3_lowrank256": ArchLowRank256,
    "e4_residual640": ArchResidual640,
    "e5_twin_encoder": ArchTwinEncoder,
}


def _build_model(name: str, in_dim: int, pq_dim: int, g_dim: int, n_nodes: int, dropout: float) -> nn.Module:
    if name == "e5_twin_encoder":
        return ArchTwinEncoder(pq_dim, g_dim, n_nodes, dropout)
    cls = ARCHITECTURES[name]
    return cls(in_dim, n_nodes, dropout)


def _val_mae_rmse_pu(
    model: nn.Module,
    X_n: torch.Tensor,
    G_n: torch.Tensor,
    Y: torch.Tensor,
    idx: np.ndarray,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    abs_err = 0.0
    sq_err = 0.0
    n_tot = 0
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            bi = idx[i : i + batch_size]
            xb = torch.cat([X_n[bi].reshape(len(bi), -1), G_n[bi]], dim=1).to(device)
            pred_n = model(xb)
            pred = pred_n * y_std.to(device) + y_mean.to(device)
            yb = Y[bi].to(device)
            d = (pred - yb).abs()
            abs_err += float(d.sum().item())
            sq_err += float((d**2).sum().item())
            n_tot += d.numel()
    mae = abs_err / max(n_tot, 1)
    rmse = (sq_err / max(n_tot, 1)) ** 0.5
    return mae, rmse


def train_one_arch(
    name: str,
    X_n: torch.Tensor,
    G_n: torch.Tensor,
    Y: torch.Tensor,
    Y_n: torch.Tensor,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    pq_dim: int,
    g_dim: int,
    n_nodes: int,
    in_dim: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    report_every: int,
    dropout: float,
) -> dict:
    model = _build_model(name, in_dim, pq_dim, g_dim, n_nodes, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)

    best_val = float("inf")
    best_state = None

    bs = min(32, max(8, X_n.shape[0] // 4))

    def batch_iter(idxs: np.ndarray):
        for j in range(0, len(idxs), bs):
            yield idxs[j : j + bs]

    for ep in range(epochs):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        for bi in batch_iter(idx_train):
            xb = torch.cat([X_n[bi].reshape(len(bi), -1), G_n[bi]], dim=1).to(device)
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
            for bi in batch_iter(idx_val):
                xb = torch.cat([X_n[bi].reshape(len(bi), -1), G_n[bi]], dim=1).to(device)
                yb = Y_n[bi].to(device)
                loss_va += float(F.mse_loss(model(xb), yb).item()) * len(bi)
                n_va += len(bi)
            loss_va /= max(n_va, 1)

        sched.step(loss_va)

        ep1 = ep + 1
        if report_every > 0 and (ep1 == 1 or ep1 % report_every == 0 or ep == epochs - 1):
            mae_pu, rmse_pu = _val_mae_rmse_pu(
                model, X_n, G_n, Y, idx_val, y_mean, y_std, device, bs
            )
            print(
                f"    epoch {ep1:4d}/{epochs}  train_mse_norm={loss_tr:.6f}  val_mse_norm={loss_va:.6f}  "
                f"val_MAE_|V|_pu={mae_pu:.6f}  val_RMSE_|V|_pu={rmse_pu:.6f}  lr={opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if loss_va < best_val:
            best_val = loss_va
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    arch_dir = out_dir / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    return {
        "arch": name,
        "best_val_mse_normalized": float(best_val),
        "ckpt": str(arch_dir / "best.pt"),
        "state_dict": best_state,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Default: <repo>/datasets_gnn2/loadtype_8500_dailyagg",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: <repo>/mlp_vm_control_explore_8500",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260329)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="Print val MAE/RMSE (pu) every N epochs (also epoch 1 and last).",
    )
    ap.add_argument(
        "--node-cache",
        type=str,
        default="",
        help="Optional .pt path to cache (X, Y) tensors after first CSV stream.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else resolve_repo_root()
    dset = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else (repo / "datasets_gnn2" / "loadtype_8500_dailyagg").resolve()
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "mlp_vm_control_explore_8500").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    node_csv = dset / "gnn_node_features_and_targets.csv"
    meta_csv = dset / "gnn_sample_meta.csv"
    for p in (node_csv, meta_csv):
        if not p.is_file():
            raise FileNotFoundError(p)

    print(f"[8500 |V| MLP + controls] repo={repo}", flush=True)
    print(f"  dataset_dir={dset}", flush=True)

    sample_ids, G_raw, gcols = _load_meta_globals(meta_csv)
    S = len(sample_ids)
    g_dim = G_raw.shape[1]
    print(f"  global features: {g_dim} cols ({len(REG_TAP_COLS)} reg taps + {g_dim - len(REG_TAP_COLS)} cap q_post)", flush=True)

    n_nodes = _infer_n_nodes(node_csv)
    cache_path = Path(args.node_cache).resolve() if args.node_cache else None
    if cache_path and cache_path.is_file():
        print(f"  loading X,Y from cache {cache_path}", flush=True)
        X, Y = torch.load(cache_path, map_location="cpu")
    else:
        print(f"  streaming P,Q,|V| from {node_csv.name} ...", flush=True)
        X, Y = _stream_pq_v(node_csv, S, n_nodes, sample_ids)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((X, Y), cache_path)
            print(f"  saved cache -> {cache_path}", flush=True)

    assert X.shape == (S, n_nodes, 2) and Y.shape == (S, n_nodes)

    idx_train, idx_val, idx_test = _split_indices(S, args.seed, 0.7, 0.15)
    pq_dim = 2 * n_nodes
    in_dim = pq_dim + g_dim

    # Normalize PQ per feature (P and Q channels) using train samples
    Xf = X.reshape(S, n_nodes, 2)
    xf_tr = Xf[idx_train].reshape(-1, 2)
    mean_x = xf_tr.mean(dim=0, keepdim=True)
    std_x = xf_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    X_n = (Xf - mean_x) / std_x

    Gf = G_raw.float()
    G_tr = Gf[idx_train]
    mean_g = G_tr.mean(dim=0, keepdim=True)
    std_g = G_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    G_n = (Gf - mean_g) / std_g

    y_mean = Y[idx_train].mean()
    y_std = Y[idx_train].std(unbiased=False).clamp_min(1e-6)
    Y_n = (Y - y_mean) / y_std

    print(f"  samples={S} N={n_nodes} in_dim={in_dim} (pq={pq_dim} + g={g_dim})", flush=True)
    print(f"  split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  report_every={args.report_every}", flush=True)

    results = []
    for name in ARCHITECTURES:
        print(f"\n--- {name} ---", flush=True)
        r = train_one_arch(
            name,
            X_n,
            G_n,
            Y,
            Y_n,
            idx_train,
            idx_val,
            pq_dim,
            g_dim,
            n_nodes,
            in_dim,
            y_mean,
            y_std,
            out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            report_every=args.report_every,
            dropout=args.dropout,
        )
        sd = r.pop("state_dict", None)
        ckpt = {
            "arch": name,
            "model_family": "mlp_pq_capq_reg_to_vmag",
            "in_dim": in_dim,
            "pq_dim": pq_dim,
            "g_dim": g_dim,
            "n_nodes": n_nodes,
            "global_feature_cols": gcols,
            "model_state_dict": sd,
            "y_mean_scalar": float(y_mean.item()),
            "y_std_scalar": float(y_std.item()),
            "pq_mean": mean_x.squeeze(0).cpu(),
            "pq_std": std_x.squeeze(0).cpu(),
            "g_mean": mean_g.squeeze(0).cpu(),
            "g_std": std_g.squeeze(0).cpu(),
            "best_val_mse_normalized": r["best_val_mse_normalized"],
        }
        torch.save(ckpt, r["ckpt"])
        results.append(r)
        print(f"  best_val_mse_norm={r['best_val_mse_normalized']:.6f}", flush=True)

    best = min(results, key=lambda x: x["best_val_mse_normalized"])
    summary = {
        "dataset_dir": str(dset),
        "out_dir": str(out_dir),
        "n_samples": S,
        "n_nodes": n_nodes,
        "in_dim": in_dim,
        "global_cols": gcols,
        "runs": results,
        "best_arch_by_val_mse_norm": best["arch"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Best (val MSE normalized): {best['arch']}")
    print(f"  summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
