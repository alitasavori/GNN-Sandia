"""
Train two models on the same 10k samples from gnn_samples_loadtype_full:
  (A) Injection features: p_inj_kw, q_inj_kvar (2 feat)
  (B) Load-type per-type: m1_p_kw, m1_q_kvar, m2_p_kw, m2_q_kvar, m4_p_kw, m4_q_kvar,
      m5_p_kw, m5_q_kvar, q_cap_kvar, p_pv_kw (10 feat)
Same data, same graph, same architecture. Saves models for plot_two_models_worst_node_profile.py.
Run from repo root. Requires: gnn_samples_loadtype_full
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from run_gnn3_overlay_7 import BASE_DIR
from run_gnn3_best7_train import PFIdentityGNN

os.chdir(BASE_DIR)

DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
CKPT_INJECTION = os.path.join(OUTPUT_DIR, "block_injection_features.pt")
CKPT_LOADTYPE_PER_TYPE = os.path.join(OUTPUT_DIR, "block_loadtype_per_type.pt")

TARGET_SAMPLES = 10000
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INJECTION_FEAT = ["p_inj_kw", "q_inj_kvar"]
LOADTYPE_PER_TYPE_FEAT = [
    "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
    "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar",
    "q_cap_kvar", "p_pv_kw",
]
LOADTYPE_COLS = LOADTYPE_PER_TYPE_FEAT + ["p_sys_balance_kw", "q_sys_balance_kvar"]


def derive_features_from_loadtype(df_n):
    """Derive injection features from loadtype columns."""
    df = df_n.copy()
    if "bus" not in df.columns and "node" in df.columns:
        df["bus"] = df["node"].astype(str).str.split(".").str[0]
    df["p_load_kw"] = df["m1_p_kw"] + df["m2_p_kw"] + df["m4_p_kw"] + df["m5_p_kw"]
    df["q_load_kvar"] = df["m1_q_kvar"] + df["m2_q_kvar"] + df["m4_q_kvar"] + df["m5_q_kvar"]
    P_grid = df.groupby("sample_id")["p_sys_balance_kw"].first().to_dict()
    Q_grid = df.groupby("sample_id")["q_sys_balance_kvar"].first().to_dict()
    P_grid_ph = {sid: P_grid[sid] / 3.0 for sid in P_grid}
    Q_grid_ph = {sid: Q_grid[sid] / 3.0 for sid in Q_grid}
    p_inj, q_inj = [], []
    for _, row in df.iterrows():
        sid, bus = row["sample_id"], str(row["bus"])
        if bus in ("sourcebus", "800"):
            p_inj.append(P_grid_ph[sid])
            q_inj.append(Q_grid_ph[sid])
        else:
            p_inj.append(float(row["p_pv_kw"]) - float(row["p_load_kw"]))
            q_inj.append(-float(row["q_load_kvar"]) + float(row["q_cap_kvar"]))
    df["p_inj_kw"] = p_inj
    df["q_inj_kvar"] = q_inj
    return df


def train_one(X_all, Y_all, edge_index, edge_attr, edge_id, N, E, node_in_dim, ckpt_path, name):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    S = X_all.shape[0]
    n_test = int(np.floor(TEST_FRAC * S))
    perm = np.random.default_rng(SEED).permutation(S)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                     edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in train_idx]
    test_ds = [Data(x=torch.tensor(X_all[k], dtype=torch.float), y=torch.tensor(Y_all[k], dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N) for k in test_idx]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                         node_emb_dim=8, edge_emb_dim=4, h_dim=32, num_layers=4, use_norm=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_rmse, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(data), data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = [((model(data.to(DEVICE)) - data.y) ** 2).mean().sqrt().item() for data in test_loader]
            rmse = np.mean(errs)
        if rmse < best_rmse:
            best_rmse, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{name}] Epoch {epoch:02d} | test RMSE={rmse:.5f} | best={best_rmse:.5f}")

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"N": N, "E": E, "node_in_dim": node_in_dim, "edge_in_dim": 2, "out_dim": 1,
                   "node_emb_dim": 8, "edge_emb_dim": 4, "h_dim": 32, "num_layers": 4, "use_norm": False,
                   "target_col": "vmag_pu", "dataset": DIR_LOADTYPE, "use_phase_onehot": False},
        "edge_index": edge_index, "edge_attr": edge_attr, "edge_id": edge_id,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"  [SAVED] {ckpt_path}")
    return best_rmse


def main():
    print("=" * 70)
    print("INJECTION vs LOADTYPE-PER-TYPE: same 10k samples, same arch (medium)")
    print("=" * 70)

    edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(DIR_LOADTYPE, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        raise FileNotFoundError(f"Missing {DIR_LOADTYPE}. Run run_loadtype_dataset.py first.")

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    for c in LOADTYPE_COLS + ["vmag_pu", "sample_id", "node_idx"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    if "bus" in df_n.columns:
        df_n["bus"] = df_n["bus"].astype(str)
    df_n = df_n.dropna(subset=LOADTYPE_PER_TYPE_FEAT + ["vmag_pu"])
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == counts.max()].index
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    all_ids = df_n["sample_id"].unique()
    n_want = min(TARGET_SAMPLES, len(all_ids))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_ids, size=n_want, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    df_n = derive_features_from_loadtype(df_n)

    N = int(df_n["node_idx"].max()) + 1
    E = len(df_e)
    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    S = df_n["sample_id"].nunique()
    X_inj = df_n[INJECTION_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    X_lt = df_n[LOADTYPE_PER_TYPE_FEAT].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    print(f"\n>>> Training on {S} samples (medium: 4L, h=32)...")
    train_one(X_inj, Y_all, edge_index, edge_attr, edge_id, N, E, 2, CKPT_INJECTION, "Injection (p_inj, q_inj)")
    train_one(X_lt, Y_all, edge_index, edge_attr, edge_id, N, E, 10, CKPT_LOADTYPE_PER_TYPE, "Loadtype per-type (m1..m5, q_cap, p_pv)")

    print("\n>>> Done. Run: %run plot_two_models_worst_node_profile.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
