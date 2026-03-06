# ============================================================
# Comprehensive error inspection for vmag-only GNN
# - rebuilds test split exactly like training (SEED, TEST_FRAC)
# - runs inference on test set
# - produces:
#   (1) worst node ranking (bus-phase)
#   (2) worst bus ranking (aggregate phases)
#   (3) worst snapshot ranking (sample_id)
#   (4) system-wide histogram
#   (5) error vs time-of-day curve (if sample meta available)
#   (6) per-node histograms (paged grids, worst-first)
# - saves: gnn_samples_out/test_node_errors_detailed.csv
# ============================================================

import os, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ----------------------------
# User-config / paths
# ----------------------------
try:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _ROOT = os.getcwd()

# Dataset directory: unified location used by run_original_dataset.py
OUT_DIR = os.path.join(_ROOT, "datasets_gnn2", "original")
EDGE_CSV   = os.path.join(OUT_DIR, "gnn_edges_phase_static.csv")
NODE_CSV   = os.path.join(OUT_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DIR, "gnn_sample_meta.csv")

# Model directory: new convention. Default checkpoint name is unchanged.
_MODELS_DIR = os.path.join(_ROOT, "models_gnn2", "original")
_DEFAULT_CKPT_NAME = "pf_identity_gnn_vmag_only_best.pt"
_NEW_CKPT_PATH = os.path.join(_MODELS_DIR, _DEFAULT_CKPT_NAME)
_LEGACY_CKPT_PATH = os.path.join("gnn_samples_out", "checkpoints", _DEFAULT_CKPT_NAME)
CKPT_PATH = _NEW_CKPT_PATH if os.path.exists(_NEW_CKPT_PATH) else _LEGACY_CKPT_PATH

SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64

# Per-node histogram paging
NODES_LIMIT = None   # e.g., 30 to only plot top-30 worst nodes; None => all
BINS = 60
COLS, ROWS = 5, 4     # hist grid per page
PER_FIG = COLS * ROWS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ----------------------------
# Safety checks
# ----------------------------
for p in [EDGE_CSV, NODE_CSV, CKPT_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

df_s = pd.read_csv(SAMPLE_CSV) if os.path.exists(SAMPLE_CSV) else None

# ============================================================
# Load model for inference (re-declare architecture from ckpt config)
# ============================================================
def load_model_for_inference(path, device=DEVICE):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]

    import torch.nn as nn
    from torch_geometric.nn import MessagePassing

    HIDDEN = int(cfg["h_dim"])
    NUM_LAYERS = int(cfg["num_layers"])
    NODE_EMB_DIM = int(cfg["node_emb_dim"])
    EDGE_EMB_DIM = int(cfg["edge_emb_dim"])
    DROPOUT = float(cfg.get("dropout", 0.0))

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=HIDDEN, dropout=DROPOUT):
            super().__init__()
            layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, out_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    class EdgeIdentityMP(MessagePassing):
        # m_ji = psi([h_j || e_ji || r_ji]), aggr add
        def __init__(self, h_dim, edge_feat_dim, edge_emb_dim):
            super().__init__(aggr="add")
            self.psi = MLP(in_dim=h_dim + edge_feat_dim + edge_emb_dim, out_dim=h_dim)
        def forward(self, h, edge_index, edge_attr, edge_emb):
            return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr, edge_emb=edge_emb)
        def message(self, h_j, edge_attr, edge_emb):
            return self.psi(torch.cat([h_j, edge_attr, edge_emb], dim=-1))

    class PFIdentityGNN(nn.Module):
        def __init__(self, num_nodes, num_edges,
                     node_in_dim=3, edge_in_dim=2, out_dim=1,
                     node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM,
                     h_dim=HIDDEN, num_layers=NUM_LAYERS):
            super().__init__()
            self.num_nodes = int(num_nodes)
            self.node_emb = nn.Embedding(num_nodes, node_emb_dim)  # z_i
            self.edge_emb = nn.Embedding(num_edges, edge_emb_dim)  # r_ij
            self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim)

            self.mps = nn.ModuleList([
                EdgeIdentityMP(h_dim=h_dim, edge_feat_dim=edge_in_dim, edge_emb_dim=edge_emb_dim)
                for _ in range(num_layers)
            ])
            self.updates = nn.ModuleList([
                MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim)
                for _ in range(num_layers)
            ])
            self.readout = MLP(in_dim=h_dim, out_dim=out_dim)

        def _local_node_ids_from_ptr(self, ptr: torch.Tensor) -> torch.Tensor:
            ids = []
            for g in range(ptr.numel() - 1):
                n0 = int(ptr[g].item())
                n1 = int(ptr[g + 1].item())
                n = n1 - n0
                if n != self.num_nodes:
                    raise RuntimeError(f"Batch graph {g} has {n} nodes != expected {self.num_nodes}.")
                ids.append(torch.arange(n, device=ptr.device))
            return torch.cat(ids, dim=0)

        def forward(self, data: Data) -> torch.Tensor:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            edge_id = data.edge_id

            if not hasattr(data, "ptr") or data.ptr is None:
                node_ids = torch.arange(data.num_nodes, device=x.device)
            else:
                node_ids = self._local_node_ids_from_ptr(data.ptr.to(x.device))

            z = self.node_emb(node_ids)
            h = self.phi0(torch.cat([x, z], dim=-1))

            r = self.edge_emb(edge_id)

            for mp, upd in zip(self.mps, self.updates):
                m = mp(h=h, edge_index=edge_index, edge_attr=edge_attr, edge_emb=r)
                h = upd(torch.cat([h, m, z], dim=-1))

            return self.readout(h)

    mdl = PFIdentityGNN(
        num_nodes=int(cfg["N"]),
        num_edges=int(cfg["E"]),
        node_in_dim=int(cfg["node_in_dim"]),
        edge_in_dim=int(cfg["edge_in_dim"]),
        out_dim=int(cfg["out_dim"]),  # should be 1 for vmag-only
        node_emb_dim=int(cfg["node_emb_dim"]),
        edge_emb_dim=int(cfg["edge_emb_dim"]),
        h_dim=int(cfg["h_dim"]),
        num_layers=int(cfg["num_layers"]),
    ).to(device)

    mdl.load_state_dict(ckpt["state_dict"])
    mdl.eval()

    static = dict(
        N=int(cfg["N"]),
        E=int(cfg["E"]),
        edge_index=ckpt["edge_index"].to(device),
        edge_attr=ckpt["edge_attr"].to(device),
        edge_id=ckpt["edge_id"].to(device),
        node_idx_to_name=ckpt.get("node_idx_to_name", [f"node_{i}" for i in range(int(cfg["N"]))]),
        config=cfg,
    )
    return mdl, static

model, static = load_model_for_inference(CKPT_PATH, device=DEVICE)
N = static["N"]
edge_index = static["edge_index"]
edge_attr  = static["edge_attr"]
edge_id    = static["edge_id"]

# ============================================================
# Load node CSV and reconstruct (S, N, features) like training
# ============================================================
df_n = pd.read_csv(NODE_CSV)

needed_cols = ["sample_id", "node_idx", "p_load_kw", "q_load_kvar", "p_pv_kw", "vmag_pu"]
if "node" in df_n.columns:
    needed_cols.append("node")
df_n = df_n[needed_cols].copy()

# numeric cleanup
df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="coerce").astype("Int64")
df_n["node_idx"]  = pd.to_numeric(df_n["node_idx"],  errors="coerce").astype("Int64")
for c in ["p_load_kw","q_load_kvar","p_pv_kw","vmag_pu"]:
    df_n[c] = pd.to_numeric(df_n[c], errors="coerce")

df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna().copy()
df_n["sample_id"] = df_n["sample_id"].astype(int)
df_n["node_idx"]  = df_n["node_idx"].astype(int)

# enforce sample/node ordering
df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

# keep only samples that have exactly N node rows
counts = df_n.groupby("sample_id")["node_idx"].count()
good_ids = counts[counts == N].index.to_numpy()
df_n = df_n[df_n["sample_id"].isin(good_ids)].copy()
df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

sample_ids = df_n["sample_id"].unique()
S = int(len(sample_ids))

# sanity check contiguity for a few samples
for sid in sample_ids[:5]:
    idxs = df_n.loc[df_n["sample_id"] == sid, "node_idx"].to_numpy()
    if not (idxs[0] == 0 and idxs[-1] == N - 1 and np.all(np.diff(idxs) == 1)):
        raise RuntimeError(f"Sample {sid} node_idx is not contiguous 0..N-1. Mapping/order mismatch persists.")

# build tensors
X_all = df_n[["p_load_kw","q_load_kvar","p_pv_kw"]].to_numpy(np.float32).reshape(S, N, 3)
Y_all = df_n[["vmag_pu"]].to_numpy(np.float32).reshape(S, N, 1)

# add node name if missing
if "node" not in df_n.columns:
    names = static["node_idx_to_name"]
    df_n["node"] = df_n["node_idx"].map(lambda i: names[int(i)])

# ============================================================
# Rebuild EXACT test split by sample index (same as training)
# ============================================================
rng = np.random.default_rng(SEED)
perm = rng.permutation(S)
n_test = int(np.floor(TEST_FRAC * S))
test_pos = perm[:n_test]   # positions in 0..S-1
train_pos = perm[n_test:]

test_sample_ids = [int(sample_ids[k]) for k in test_pos]

print(f"N nodes: {N}, E edges: {static['E']}, total samples: {S}")
print(f"Train samples: {len(train_pos)}, Test samples: {len(test_pos)}")

# ============================================================
# Build PyG test dataset (each graph = one snapshot)
# ============================================================
test_graphs = []
for k in test_pos:
    x = torch.tensor(X_all[k], dtype=torch.float32)
    y = torch.tensor(Y_all[k], dtype=torch.float32)
    g = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N)
    test_graphs.append(g)

loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# Predict on test set and assemble df_test aligned to rows
# ============================================================
preds = []
trues = []

model.eval()
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        yhat = model(batch)                 # [B*N, 1]
        preds.append(yhat.detach().cpu().numpy())
        trues.append(batch.y.detach().cpu().numpy())

pred = np.vstack(preds).reshape(-1)
true = np.vstack(trues).reshape(-1)
err  = pred - true
abs_err = np.abs(err)

# Build df_test aligned with (sample_id,node_idx) order
df_test = df_n[df_n["sample_id"].isin(test_sample_ids)].copy()
df_test = df_test.sort_values(["sample_id","node_idx"]).reset_index(drop=True)

if len(df_test) != len(err):
    raise RuntimeError(f"Row mismatch: df_test rows={len(df_test)} vs predictions={len(err)}")

df_test["vmag_pred"] = pred
df_test["vmag_err"]  = err
df_test["vmag_abs_err"] = abs_err

# derive bus/phase
df_test["node"] = df_test["node"].astype(str)
df_test["bus"] = df_test["node"].str.split(".").str[0]
df_test["phase"] = df_test["node"].str.split(".").str[1].astype(int)

# join sample meta (time-of-day, profiles) if available
if df_s is not None and "sample_id" in df_s.columns:
    meta_cols = [c for c in ["sample_id","t_index","t_minutes","m_loadshape","m_irradshape","prof_net",
                             "P_load_time_kw","Q_load_time_kvar","P_pv_time_kw"] if c in df_s.columns]
    if meta_cols:
        df_test = df_test.merge(df_s[meta_cols], on="sample_id", how="left")

# save detailed errors
OUT_ERR = os.path.join(OUT_DIR, "test_node_errors_detailed.csv")
df_test.to_csv(OUT_ERR, index=False)
print(f"[SAVED] Detailed per-node test errors -> {OUT_ERR}")

# ============================================================
# (A) Rankings: worst nodes, worst buses, worst snapshots
# ============================================================
node_stats = (
    df_test.groupby("node")["vmag_abs_err"]
    .agg(["mean","median","max","count"])
    .sort_values("mean", ascending=False)
)
print("\n=== WORST 15 NODES (by mean abs error) ===")
print(node_stats.head(15))

bus_stats = (
    df_test.groupby("bus")["vmag_abs_err"]
    .agg(["mean","median","max","count"])
    .sort_values("mean", ascending=False)
)
print("\n=== WORST 15 BUSES (by mean abs error) ===")
print(bus_stats.head(15))

snap_stats = (
    df_test.groupby("sample_id")["vmag_abs_err"]
    .mean()
    .sort_values(ascending=False)
)
print("\n=== WORST 10 SNAPSHOTS (by mean node abs error) ===")
print(snap_stats.head(10))

print("\n=== 99th percentile abs error (test) ===")
print(float(np.quantile(df_test["vmag_abs_err"].values, 0.99)))

# Optional: inspect worst snapshot’s top-10 worst nodes
if len(snap_stats) > 0:
    worst_sid = int(snap_stats.index[0])
    print(f"\nInspecting worst sample_id={worst_sid}")
    cols_show = ["sample_id","node","p_load_kw","q_load_kvar","p_pv_kw","vmag_pu","vmag_pred","vmag_abs_err"]
    cols_show += [c for c in ["t_index","t_minutes","m_loadshape","m_irradshape","prof_net"] if c in df_test.columns]
    print(df_test[df_test["sample_id"] == worst_sid].sort_values("vmag_abs_err", ascending=False).head(10)[cols_show])

# ============================================================
# (B) Plots: system-wide histogram, error vs time-of-day
# ============================================================
plt.figure()
plt.hist(df_test["vmag_abs_err"].values, bins=60)
plt.xlabel("|V| abs error (pu)")
plt.ylabel("Count")
plt.title("Distribution of node-level abs error (test) — system-wide")
plt.show()

if "t_minutes" in df_test.columns and df_test["t_minutes"].notna().any():
    tod = (
        df_test.groupby("t_minutes")["vmag_abs_err"]
        .mean()
        .sort_index()
    )
    plt.figure()
    plt.plot(tod.index, tod.values)
    plt.xlabel("t_minutes")
    plt.ylabel("Mean |V| abs error (pu)")
    plt.title("Test error vs time-of-day")
    plt.show()
else:
    print("[INFO] No t_minutes found in df_test; skipping time-of-day plot.")

# ============================================================
# (C) Per-node histograms (paged grids, worst-first)
# ============================================================
# Order nodes by mean error descending (worst first)
node_order = node_stats.index.tolist()
if NODES_LIMIT is not None:
    node_order = node_order[:int(NODES_LIMIT)]

# Use a robust xmax so plots are comparable and not dominated by outliers
XMAX = float(np.quantile(df_test["vmag_abs_err"].values, 0.999))
XMAX = max(XMAX, 0.02)

n_nodes = len(node_order)
n_figs = int(math.ceil(n_nodes / PER_FIG))
print(f"\nPlotting {n_nodes} node histograms across {n_figs} figure(s). XMAX={XMAX:.4f} pu")

for fidx in range(n_figs):
    start = fidx * PER_FIG
    end = min((fidx + 1) * PER_FIG, n_nodes)
    chunk = node_order[start:end]

    fig, axes = plt.subplots(ROWS, COLS, figsize=(18, 10))
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(chunk):
            ax.axis("off")
            continue

        node = chunk[ax_i]
        vals = df_test.loc[df_test["node"] == node, "vmag_abs_err"].to_numpy(dtype=float)

        ax.hist(vals, bins=BINS, range=(0.0, XMAX))
        ax.set_title(f"{node} (mean={vals.mean():.3f})", fontsize=10)
        ax.set_xlim(0.0, XMAX)

        if ax_i % COLS == 0:
            ax.set_ylabel("Count")
        if ax_i >= (ROWS - 1) * COLS:
            ax.set_xlabel("|V| abs error (pu)")

    fig.suptitle(f"Node abs error histograms (test) — nodes {start+1}..{end} of {n_nodes} (worst-first)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\nDone.")
