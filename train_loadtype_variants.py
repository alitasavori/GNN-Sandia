"""
Train loadtype model variants with the best architecture (light_emb_h96_depth2):
  n_emb=16, e_emb=8, h=96, L=2, use_norm=False.

1) Best loadtype WITHOUT phase one-hot (14 features) → light_emb_h96_depth2_best.pt
2) Best loadtype WITH phase one-hot (17 features)   → light_emb_h96_depth2_phase_onehot_best.pt
3) Three per-phase models (one per phase 1,2,3)     → light_emb_h96_depth2_phase1_best.pt, _phase2_, _phase3_

Uses datasets_gnn2/loadtype. Run from repo root.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from run_gnn3_best7_train import (
    BASE_DIR,
    SEED,
    TEST_FRAC,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    MIN_EPOCHS_BEFORE_STOP,
    MIN_DELTA,
    LR,
    WEIGHT_DECAY,
    DEVICE,
    OUTPUT_DIR,
    LOADTYPE_FEAT,
    seed_all,
    _parse_phase_from_node_name,
    PFIdentityGNN,
    evaluate,
)

os.chdir(BASE_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "loadtype")
os.makedirs(MODELS_DIR, exist_ok=True)

# Best loadtype arch (from search)
N_EMB = 16
E_EMB = 8
H_DIM = 96
N_LAYERS = 2


def train_one_phase(block_id: int, phase_1based: int, early_stop: bool = True) -> str | None:
    """Train a GNN on loadtype data restricted to nodes of one phase. Returns path to saved checkpoint or None."""
    seed_all(SEED)
    phase_0based = phase_1based - 1  # 0, 1, 2
    edge_csv = os.path.join(DATASET_DIR, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DATASET_DIR, "gnn_node_features_and_targets.csv")
    master_path = os.path.join(DATASET_DIR, "gnn_node_index_master.csv")
    for p in (edge_csv, node_csv, master_path):
        if not os.path.exists(p):
            print(f"  [SKIP] Block {block_id} phase{phase_1based}: Missing {p}")
            return None

    required = {"sample_id", "node_idx", "vmag_pu"} | set(LOADTYPE_FEAT)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    master_df = pd.read_csv(master_path)
    if required - set(df_n.columns):
        print(f"  [SKIP] Block {block_id}: Missing columns {required - set(df_n.columns)}")
        return None

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in LOADTYPE_FEAT + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()

    # Global 89-node mapping (same as train_one)
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    N_full = int(counts.mode()[0])
    if N_full != 89:
        raise RuntimeError(f"Expected 89 nodes per sample, got {N_full}.")
    kept_node_ids = sorted(df_n["node_idx"].unique())
    old_to_new = {old: new for new, old in enumerate(kept_node_ids)}
    df_n["node_idx_new"] = df_n["node_idx"].map(old_to_new)

    # Phase per node (0,1,2) in full 89-node index
    node_idx_to_phase = {}
    for _, row in master_df.iterrows():
        idx = int(row["node_idx"])
        if idx in old_to_new:
            node_idx_to_phase[old_to_new[idx]] = _parse_phase_from_node_name(row["node"]) - 1
    phase_vec = np.array([node_idx_to_phase.get(i, 0) for i in range(N_full)], dtype=np.int64)
    phase_mask = phase_vec == phase_0based
    node_indices_phase = np.where(phase_mask)[0]
    N_phase = int(phase_mask.sum())
    if N_phase == 0:
        print(f"  [SKIP] Block {block_id}: No nodes for phase {phase_1based}")
        return None
    global_to_local = {g: i for i, g in enumerate(node_indices_phase)}

    # Edges with both endpoints in this phase; remap to 0..N_phase-1
    df_e_full = df_e.copy()
    df_e_full["u_new"] = df_e_full["u_idx"].map(old_to_new)
    df_e_full["v_new"] = df_e_full["v_idx"].map(old_to_new)
    df_e_full = df_e_full.dropna(subset=["u_new", "v_new"]).astype({"u_new": int, "v_new": int})
    df_e_phase = df_e_full[
        df_e_full["u_new"].isin(global_to_local) & df_e_full["v_new"].isin(global_to_local)
    ].copy()
    df_e_phase["u_local"] = df_e_phase["u_new"].map(global_to_local)
    df_e_phase["v_local"] = df_e_phase["v_new"].map(global_to_local)
    df_e_phase = df_e_phase.reset_index(drop=True)
    df_e_phase["edge_id"] = np.arange(len(df_e_phase), dtype=int)
    E_phase = len(df_e_phase)

    # Keep only samples that have all N_full nodes; then take phase subset
    good_ids = counts[counts == N_full].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx_new"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    all_ids = df_n["sample_id"].unique()
    n_keep = max(1, int(len(all_ids) * 1.0))
    keep_ids = rng.choice(all_ids, size=n_keep, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx_new"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()

    # Full-grid features/targets then slice to phase
    X_full = df_n[LOADTYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N_full, -1)
    Y_full = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N_full, 1)
    X_phase = X_full[:, node_indices_phase, :]
    Y_phase = Y_full[:, node_indices_phase, :]
    node_in_dim = X_phase.shape[-1]

    edge_index = torch.tensor(df_e_phase[["u_local", "v_local"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e_phase[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e_phase["edge_id"].to_numpy(), dtype=torch.long)

    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        return [
            Data(
                x=torch.tensor(X_phase[k], dtype=torch.float),
                y=torch.tensor(Y_phase[k], dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_id=edge_id,
                num_nodes=N_phase,
            )
            for k in idx
        ]

    train_loader = DataLoader(make_ds(train_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(make_ds(test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(
        num_nodes=N_phase,
        num_edges=E_phase,
        node_in_dim=node_in_dim,
        edge_in_dim=2,
        out_dim=1,
        node_emb_dim=N_EMB,
        edge_emb_dim=E_EMB,
        h_dim=H_DIM,
        num_layers=N_LAYERS,
        use_norm=False,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_test, best_state, best_epoch, best_mae, best_rmse = float("inf"), None, None, None, None
    patience_left = EARLY_STOP_PATIENCE if early_stop else 1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        mae_t, rmse_t = evaluate(model, test_loader)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test, best_state, best_epoch = rmse_t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
            best_mae, best_rmse, patience_left = mae_t, rmse_t, (EARLY_STOP_PATIENCE if early_stop else 1)
        else:
            if early_stop and epoch >= MIN_EPOCHS_BEFORE_STOP:
                patience_left -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}")
        if early_stop and epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break

    if best_state is None:
        return None
    model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(model, test_loader)

    ckpt_path = os.path.join(OUTPUT_DIR, f"block{block_id}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "N": N_phase,
            "E": E_phase,
            "node_in_dim": node_in_dim,
            "edge_in_dim": 2,
            "out_dim": 1,
            "node_emb_dim": N_EMB,
            "edge_emb_dim": E_EMB,
            "h_dim": H_DIM,
            "num_layers": N_LAYERS,
            "use_norm": False,
            "target_col": "vmag_pu",
            "dataset": DATASET_DIR,
            "use_phase_onehot": False,
            "phase_1based": phase_1based,
        },
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_id": edge_id,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "best_epoch": best_epoch,
        "node_indices_phase": node_indices_phase,
    }
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] Block {block_id} phase{phase_1based} -> {ckpt_path} | MAE={mae_f:.6f} RMSE={rmse_f:.6f}")
    return ckpt_path


def main() -> None:
    os.chdir(BASE_DIR)
    from run_gnn3_best7_train import train_one

    print("=" * 72)
    print("Loadtype variants: best arch (light_emb_h96_depth2) — 2 full-graph + 3 per-phase")
    print("=" * 72)

    # 1) Best without phase one-hot
    print("\n[1/5] Best loadtype (no phase one-hot)...")
    ckpt1 = train_one(
        block_id=107,
        cfg_name="light_emb_h96_depth2",
        out_dir=DATASET_DIR,
        feature_cols=LOADTYPE_FEAT,
        target_col="vmag_pu",
        n_emb=N_EMB,
        e_emb=E_EMB,
        h_dim=H_DIM,
        n_layers=N_LAYERS,
        use_norm=False,
        use_phase_onehot=False,
        early_stop=True,
    )
    if ckpt1:
        p = os.path.join(MODELS_DIR, "light_emb_h96_depth2_best.pt")
        torch.save(torch.load(ckpt1, map_location="cpu", weights_only=False), p)
        print(f"[SAVED] {p}")

    # 2) Best with phase one-hot
    print("\n[2/5] Best loadtype WITH phase one-hot...")
    ckpt2 = train_one(
        block_id=108,
        cfg_name="light_emb_h96_depth2_phase_onehot",
        out_dir=DATASET_DIR,
        feature_cols=LOADTYPE_FEAT,
        target_col="vmag_pu",
        n_emb=N_EMB,
        e_emb=E_EMB,
        h_dim=H_DIM,
        n_layers=N_LAYERS,
        use_norm=False,
        use_phase_onehot=True,
        early_stop=True,
    )
    if ckpt2:
        p = os.path.join(MODELS_DIR, "light_emb_h96_depth2_phase_onehot_best.pt")
        torch.save(torch.load(ckpt2, map_location="cpu", weights_only=False), p)
        print(f"[SAVED] {p}")

    # 3–5) Per-phase models
    for idx, ph in enumerate((1, 2, 3), start=3):
        print(f"\n[{idx}/5] Per-phase model (phase {ph})...")
        ckpt = train_one_phase(block_id=108 + ph, phase_1based=ph, early_stop=True)
        if ckpt:
            dest = os.path.join(MODELS_DIR, f"light_emb_h96_depth2_phase{ph}_best.pt")
            torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), dest)
            print(f"[SAVED] {dest}")

    print("\n" + "=" * 72)
    print("Done. Checkpoints in", MODELS_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
