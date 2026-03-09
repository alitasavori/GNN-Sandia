"""
Train a weighted-loss Load-type PF-identity GNN without regenerating the dataset.

The script uses the existing `datasets_gnn2/loadtype` snapshots plus
`gnn_sample_meta.csv` to identify samples in a target hour window and multiply
their training loss.

Default behavior:
- Architecture: light_emb_h96_phase_onehot_depth3_h112
- Dataset: datasets_gnn2/loadtype
- Target: vmag_pu
- Weighted window: 15:00-17:00
- Weight multiplier inside window: 3.0x

Example:
    python train_weighted_loadtype_gnn.py
    python train_weighted_loadtype_gnn.py --window-start 14.5 --window-end 17.5 --boost 4.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from run_gnn3_best7_train import (
    BATCH_SIZE,
    DATA_FRAC,
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LOADTYPE_FEAT,
    LR,
    MIN_DELTA,
    MIN_EPOCHS_BEFORE_STOP,
    PFIdentityGNN,
    SEED,
    TEST_FRAC,
    WEIGHT_DECAY,
    _parse_phase_from_node_name,
    compute_metrics,
    seed_all,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "loadtype")
OUTPUT_DIR = os.path.join(BASE_DIR, "gnn3_best7_output")

CFG_NAME = "light_emb_h96_phase_onehot_depth3_h112_weighted_loss"
BLOCK_ID = 104


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train weighted-loss load-type GNN.")
    p.add_argument("--window-start", type=float, default=15.0, help="Start hour of weighted window.")
    p.add_argument("--window-end", type=float, default=17.0, help="End hour of weighted window.")
    p.add_argument("--boost", type=float, default=3.0, help="Loss multiplier inside the weighted window.")
    p.add_argument("--block-id", type=int, default=BLOCK_ID, help="Checkpoint block id.")
    p.add_argument("--cfg-name", type=str, default=CFG_NAME, help="Checkpoint/config name.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    mae_sum = torch.tensor(0.0, device=DEVICE)
    rmse_sum = torch.tensor(0.0, device=DEVICE)
    n_batches = 0
    for data in loader:
        data = data.to(DEVICE)
        mae, rmse = compute_metrics(model(data), data.y)
        mae_sum += mae
        rmse_sum += rmse
        n_batches += 1
    return (mae_sum / max(1, n_batches)).item(), (rmse_sum / max(1, n_batches)).item()


def weighted_graph_mse(yhat: torch.Tensor, ytrue: torch.Tensor, ptr: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    err2 = (yhat - ytrue).squeeze(-1).pow(2)
    graph_losses = []
    for g in range(ptr.numel() - 1):
        n0 = int(ptr[g].item())
        n1 = int(ptr[g + 1].item())
        graph_losses.append(err2[n0:n1].mean())
    graph_losses = torch.stack(graph_losses)
    weights = sample_weight.view(-1)
    return (weights * graph_losses).sum() / weights.sum().clamp_min(1e-12)


def load_weighted_dataset(window_start: float, window_end: float, boost: float):
    edge_csv = os.path.join(DATASET_DIR, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DATASET_DIR, "gnn_node_features_and_targets.csv")
    sample_csv = os.path.join(DATASET_DIR, "gnn_sample_meta.csv")
    master_csv = os.path.join(DATASET_DIR, "gnn_node_index_master.csv")
    for path in (edge_csv, node_csv, sample_csv, master_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    df_s = pd.read_csv(sample_csv)

    required = {"sample_id", "node_idx", "vmag_pu"} | set(LOADTYPE_FEAT)
    if missing := (required - set(df_n.columns)):
        raise RuntimeError(f"Missing required node columns: {missing}")
    if missing := ({"sample_id", "t_minutes"} - set(df_s.columns)):
        raise RuntimeError(f"Missing required sample-meta columns: {missing}")

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    df_s["sample_id"] = pd.to_numeric(df_s["sample_id"], errors="raise").astype(int)
    df_s["t_minutes"] = pd.to_numeric(df_s["t_minutes"], errors="raise").astype(float)

    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in LOADTYPE_FEAT + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")

    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    counts = df_n.groupby("sample_id")["node_idx"].count()
    N = int(counts.mode()[0])
    if N != 89:
        raise RuntimeError(f"Expected 89 nodes per sample after excluding upstream buses, got {N}.")

    kept_node_ids = sorted(df_n["node_idx"].unique())
    if len(kept_node_ids) != N:
        raise RuntimeError(f"Expected {N} unique kept node ids, found {len(kept_node_ids)}.")
    old_to_new = {old: new for new, old in enumerate(kept_node_ids)}

    df_e = df_e[df_e["u_idx"].isin(kept_node_ids) & df_e["v_idx"].isin(kept_node_ids)].copy()
    df_e["u_idx"] = df_e["u_idx"].map(old_to_new)
    df_e["v_idx"] = df_e["v_idx"].map(old_to_new)
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    E = int(len(df_e))

    df_n["node_idx"] = df_n["node_idx"].map(old_to_new)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    df_s = df_s[df_s["sample_id"].isin(good_ids)].copy()

    all_ids = df_n["sample_id"].unique()
    n_keep = max(1, int(len(all_ids) * DATA_FRAC))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_keep, replace=False)

    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    df_s = df_s[df_s["sample_id"].isin(keep_ids)].copy()

    sample_ids = df_n["sample_id"].drop_duplicates().to_numpy()
    S = len(sample_ids)
    X_all = df_n[LOADTYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    # Phase one-hot, same as existing best load-type model.
    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    phase_old = {}
    for _, row in master_df.iterrows():
        idx = int(row["node_idx"])
        if idx in old_to_new:
            phase_old[idx] = _parse_phase_from_node_name(row["node"]) - 1
    phase_vec = np.empty(N, dtype=np.int64)
    for old_idx, new_idx in old_to_new.items():
        phase_vec[new_idx] = phase_old[old_idx]
    phase_tensor = torch.tensor(phase_vec, dtype=torch.long)
    ph_oh = F.one_hot(phase_tensor, num_classes=3).numpy().astype(np.float32)
    X_all = np.concatenate([X_all, np.broadcast_to(ph_oh[None, :, :], (S, N, 3))], axis=-1)

    # Sample weights from time-of-day.
    meta_by_id = (
        df_s[["sample_id", "t_minutes"]]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
        .sort_index()
    )
    if len(meta_by_id) != S:
        missing = sorted(set(sample_ids) - set(meta_by_id.index))
        raise RuntimeError(f"Sample meta mismatch; missing sample_ids in gnn_sample_meta.csv: {missing[:10]}")
    t_hours = meta_by_id.loc[sample_ids, "t_minutes"].to_numpy(dtype=np.float32) / 60.0
    sample_weights = np.ones(S, dtype=np.float32)
    sample_weights[(t_hours >= window_start) & (t_hours <= window_end)] = float(boost)

    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float32)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx: np.ndarray):
        ds = []
        for k in idx:
            ds.append(
                Data(
                    x=torch.tensor(X_all[k], dtype=torch.float32),
                    y=torch.tensor(Y_all[k], dtype=torch.float32),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_id=edge_id,
                    num_nodes=N,
                    sample_weight=torch.tensor([sample_weights[k]], dtype=torch.float32),
                    sample_hour=torch.tensor([t_hours[k]], dtype=torch.float32),
                )
            )
        return ds

    return {
        "N": N,
        "E": E,
        "node_in_dim": int(X_all.shape[-1]),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_id": edge_id,
        "train_loader": DataLoader(make_ds(train_idx), batch_size=BATCH_SIZE, shuffle=True),
        "test_loader": DataLoader(make_ds(test_idx), batch_size=BATCH_SIZE, shuffle=False),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "sample_weights": sample_weights,
        "t_hours": t_hours,
    }


def main() -> None:
    args = parse_args()
    os.chdir(BASE_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    seed_all(SEED)

    print("=" * 72)
    print("Weighted-loss Load-type training")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Weighted window: {args.window_start:.2f}h to {args.window_end:.2f}h")
    print(f"Boost inside window: {args.boost:.2f}x")
    print("=" * 72)

    data = load_weighted_dataset(args.window_start, args.window_end, args.boost)
    train_hours = data["t_hours"][data["train_idx"]]
    train_weights = data["sample_weights"][data["train_idx"]]
    frac_weighted = float(np.mean(train_weights > 1.0))
    print(
        f"Train samples: {len(data['train_idx'])} | Test samples: {len(data['test_idx'])} | "
        f"Weighted-train fraction: {frac_weighted:.3f}"
    )
    print(
        "Train hour range: {:.2f}h to {:.2f}h".format(
            float(np.min(train_hours)),
            float(np.max(train_hours)),
        )
    )

    model = PFIdentityGNN(
        num_nodes=data["N"],
        num_edges=data["E"],
        node_in_dim=data["node_in_dim"],
        edge_in_dim=2,
        out_dim=1,
        node_emb_dim=16,
        edge_emb_dim=8,
        h_dim=112,
        num_layers=3,
        use_norm=False,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test = float("inf")
    best_state = None
    best_epoch = None
    best_mae = None
    best_rmse = None
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        for batch in data["train_loader"]:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            yhat = model(batch)
            loss = weighted_graph_mse(yhat, batch.y, batch.ptr, batch.sample_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item())
            nb += 1

        mae_t, rmse_t = evaluate(model, data["test_loader"])
        if (best_test - rmse_t) > MIN_DELTA:
            best_test = rmse_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_mae = mae_t
            best_rmse = rmse_t
            patience_left = EARLY_STOP_PATIENCE
        elif epoch >= MIN_EPOCHS_BEFORE_STOP:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:02d} | train_weighted_loss={total_loss / max(1, nb):.6f} | "
                f"test_RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}"
            )
        if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(model, data["test_loader"])

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "N": data["N"],
            "E": data["E"],
            "node_in_dim": data["node_in_dim"],
            "edge_in_dim": 2,
            "out_dim": 1,
            "node_emb_dim": 16,
            "edge_emb_dim": 8,
            "h_dim": 112,
            "num_layers": 3,
            "use_norm": False,
            "target_col": "vmag_pu",
            "dataset": DATASET_DIR,
            "use_phase_onehot": True,
            "loss_name": "weighted_graph_mse",
            "weighted_window_hours": [float(args.window_start), float(args.window_end)],
            "weighted_boost": float(args.boost),
        },
        "edge_index": data["edge_index"],
        "edge_attr": data["edge_attr"],
        "edge_id": data["edge_id"],
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "best_epoch": best_epoch,
    }

    ckpt_path = os.path.join(OUTPUT_DIR, f"block{args.block_id}.pt")
    out_path = os.path.join(MODELS_DIR, f"{args.cfg_name}_best.pt")
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, out_path)

    print(f"\n[SAVED] Weighted checkpoint -> {ckpt_path}")
    print(f"[SAVED] Weighted Load-type model -> {out_path}")
    print(f"[FINAL] MAE={mae_f:.6f} RMSE={rmse_f:.6f} | best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
