"""
IEEE 8500 — 22 separate softmax models (one per device output column).

Input per sample:
  Flattened nodal loads from gnn_node_features_and_targets.csv
  [p_load_kw, q_load_kvar] for all nodes.

Targets per sample:
  22 columns from gnn_sample_meta.csv.
  Each target is treated as a discrete class set and trained with
  CrossEntropyLoss (softmax classifier).

Outputs:
  gnn2_architecture_search_softmax_22/<target_col>/best.pt
  gnn2_architecture_search_softmax_22/summary.json
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


def resolve_repo_root() -> Path:
    env = (os.environ.get("GNN_REPO") or os.environ.get("GNN2_REPO") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _infer_n_nodes(node_csv: Path) -> int:
    nmax = -1
    for chunk in pd.read_csv(node_csv, usecols=["node_idx"], chunksize=500_000):
        nmax = max(nmax, int(chunk["node_idx"].max()))
    return nmax + 1


def _load_meta_targets(meta_csv: Path) -> tuple[np.ndarray, torch.Tensor]:
    df = pd.read_csv(meta_csv)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"gnn_sample_meta.csv missing columns: {missing}")
    sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
    Y = torch.from_numpy(df[list(TARGET_COLS)].to_numpy(dtype=np.float32))
    return sample_ids, Y


def _stream_node_features(node_csv: Path, S: int, n_nodes: int, sample_ids_meta: np.ndarray) -> torch.Tensor:
    X = np.zeros((S, n_nodes, 2), dtype=np.float32)
    use_direct = bool(
        len(sample_ids_meta) == S
        and sample_ids_meta.min() == 0
        and sample_ids_meta.max() == S - 1
        and np.array_equal(sample_ids_meta, np.arange(S, dtype=np.int64))
    )
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


def _split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    return tr, va, te


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float, n_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_target_classes(y: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    # Round regulator tap pu values for stable discrete bins.
    y_q = np.round(y.cpu().numpy().astype(np.float64), 6)
    classes = np.unique(y_q)
    idx = np.searchsorted(classes, y_q)
    return classes.astype(np.float32), torch.from_numpy(idx.astype(np.int64))


def train_one_target(
    target_name: str,
    X_flat_n: torch.Tensor,
    y_idx: torch.Tensor,
    class_values: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    log_every: int,
    device: torch.device,
) -> dict:
    model = MLPClassifier(
        in_dim=int(X_flat_n.shape[1]),
        hidden=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        n_classes=int(len(class_values)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

    best_val_ce = float("inf")
    best_val_acc = 0.0
    best_state = None

    bs = min(128, max(16, X_flat_n.shape[0] // 8))

    def batch_indices(idxs: np.ndarray):
        for i in range(0, len(idxs), bs):
            yield idxs[i:i + bs]

    for ep in range(epochs):
        model.train()
        tr_ce = 0.0
        tr_correct = 0
        tr_n = 0
        for bi in batch_indices(idx_train):
            xb = X_flat_n[bi].to(device)
            yb = y_idx[bi].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_ce += float(loss.item()) * len(bi)
            tr_correct += int((logits.argmax(dim=1) == yb).sum().item())
            tr_n += len(bi)
        tr_ce /= max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)

        model.eval()
        va_ce = 0.0
        va_correct = 0
        va_n = 0
        with torch.no_grad():
            for bi in batch_indices(idx_val):
                xb = X_flat_n[bi].to(device)
                yb = y_idx[bi].to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                va_ce += float(loss.item()) * len(bi)
                va_correct += int((logits.argmax(dim=1) == yb).sum().item())
                va_n += len(bi)
        va_ce /= max(va_n, 1)
        va_acc = va_correct / max(va_n, 1)

        sched.step(va_ce)

        if log_every > 0 and ((ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1):
            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"    epoch {ep + 1:4d}/{epochs}  train_ce={tr_ce:.6f}  train_acc={tr_acc:.4f}  "
                f"val_ce={va_ce:.6f}  val_acc={va_acc:.4f}  lr={cur_lr:.2e}",
                flush=True,
            )

        if (va_ce < best_val_ce) or (abs(va_ce - best_val_ce) < 1e-12 and va_acc > best_val_acc):
            best_val_ce = va_ce
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    target_dir = out_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = target_dir / "best.pt"
    torch.save(
        {
            "target_name": target_name,
            "model_type": "mlp_softmax_single_target",
            "n_classes": int(len(class_values)),
            "class_values": class_values.tolist(),
            "model_state_dict": best_state,
            "best_val_ce": float(best_val_ce),
            "best_val_acc": float(best_val_acc),
        },
        ckpt_path,
    )
    return {
        "target": target_name,
        "n_classes": int(len(class_values)),
        "best_val_ce": float(best_val_ce),
        "best_val_acc": float(best_val_acc),
        "ckpt": str(ckpt_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--dataset-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260329)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--node-cache",
        type=str,
        default="",
        help="Optional path to a .pt tensor cache for X [S,N,2].",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else resolve_repo_root()
    dset = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else (repo / "datasets_gnn2" / "loadtype_8500_dailyagg")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "gnn2_architecture_search_softmax_22")
    out_dir.mkdir(parents=True, exist_ok=True)

    node_csv = dset / "gnn_node_features_and_targets.csv"
    meta_csv = dset / "gnn_sample_meta.csv"
    for p in (node_csv, meta_csv):
        if not p.is_file():
            raise FileNotFoundError(p)

    print(f"[8500 softmax-22] repo_root={repo}", flush=True)
    print(f"[8500 softmax-22] dataset_dir={dset}", flush=True)

    sample_ids, Y = _load_meta_targets(meta_csv)
    S = len(sample_ids)
    print(f"  inferring N_nodes from {node_csv.name}...", flush=True)
    n_nodes = _infer_n_nodes(node_csv)

    cache_path = Path(args.node_cache).resolve() if args.node_cache else None
    if cache_path and cache_path.is_file():
        print(f"  loading node tensor from cache {cache_path}", flush=True)
        X = torch.load(cache_path, map_location="cpu")
    else:
        print(f"  streaming node features from {node_csv.name} (chunked)...", flush=True)
        X = _stream_node_features(node_csv, S, n_nodes, sample_ids)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(X, cache_path)
            print(f"  saved node tensor cache -> {cache_path}", flush=True)

    idx_train, idx_val, idx_test = _split_indices(S, args.seed, 0.7, 0.15)
    x_train = X[idx_train]
    xf = x_train.reshape(-1, 2)
    x_mean = xf.mean(dim=0, keepdim=True)
    x_std = xf.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    X_n = (X - x_mean) / x_std
    X_flat_n = X_n.reshape(S, -1)

    print(f"  samples={S} n_nodes={n_nodes} flat_dim={X_flat_n.shape[1]} targets={len(TARGET_COLS)}")
    print(f"  split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device} log_every={args.log_every}", flush=True)

    results = []
    for j, target_name in enumerate(TARGET_COLS):
        print(f"\n--- Training target {j + 1}/{len(TARGET_COLS)}: {target_name} ---", flush=True)
        class_values, y_idx = _build_target_classes(Y[:, j])
        r = train_one_target(
            target_name=target_name,
            X_flat_n=X_flat_n,
            y_idx=y_idx,
            class_values=class_values,
            idx_train=idx_train,
            idx_val=idx_val,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            log_every=args.log_every,
            device=device,
        )
        results.append(r)
        print(
            f"  done {target_name}: val_ce={r['best_val_ce']:.6f}  val_acc={r['best_val_acc']:.4f}  "
            f"n_classes={r['n_classes']}",
            flush=True,
        )

    mean_acc = float(np.mean([r["best_val_acc"] for r in results]))
    mean_ce = float(np.mean([r["best_val_ce"] for r in results]))
    summary = {
        "dataset_dir": str(dset),
        "out_dir": str(out_dir),
        "model_family": "mlp_softmax_single_target",
        "n_samples": int(S),
        "n_nodes": int(n_nodes),
        "flat_dim": int(X_flat_n.shape[1]),
        "n_models": len(TARGET_COLS),
        "targets": list(TARGET_COLS),
        "mean_best_val_acc": mean_acc,
        "mean_best_val_ce": mean_ce,
        "runs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[8500 softmax-22] Done. mean_best_val_acc={mean_acc:.4f}, mean_best_val_ce={mean_ce:.6f}")
    print(f"  summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
