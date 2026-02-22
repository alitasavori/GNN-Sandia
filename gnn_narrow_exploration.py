"""
Narrow exploration: 10 promising architectures from Blocks 1–2 rankings.
Uses 30% data, corrected MAE/RMSE. Explores:
- Increasing depth (3–4 layers on best 2-layer configs)
- Node/edge feature normalization (LayerNorm)
- Phase-aware: one-hot phase or 3 separate subgraphs (one per phase)
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DATA_FRAC = 0.30
SEED = 20260130
TEST_FRAC = 0.20
BATCH_SIZE = 64
EPOCHS = 40
EARLY_STOP_PATIENCE = 8
MIN_EPOCHS_BEFORE_STOP = 10
MIN_DELTA = 1e-6
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MAE = 0.001

# 10 configs: (name, n_emb, e_emb, h_dim, n_layers, use_norm, use_phase_subgraphs, use_phase_onehot)
# use_phase_subgraphs: 3 separate subgraphs (one per phase)
# use_phase_onehot: add one-hot phase to node features
CANDIDATES = [
    ("light_emb_h96_depth3", 16, 8, 96, 3, False, False, False),
    ("light_emb_h96_depth4", 16, 8, 96, 4, False, False, False),
    ("light_xwide_emb_depth3", 16, 8, 128, 3, False, False, False),
    ("light_emb_h96_norm", 16, 8, 96, 2, True, False, False),
    ("light_xwide_emb_norm", 16, 8, 128, 2, True, False, False),
    ("wide_shallow_h160_depth3", 8, 4, 160, 3, False, False, False),
    ("light_wide_emb12_norm", 12, 6, 64, 2, True, False, False),
    ("light_emb_h96_phase_onehot", 16, 8, 96, 2, False, False, True),
    ("light_xwide_emb_phase_onehot", 16, 8, 128, 2, False, False, True),
    ("light_emb_h96_phase_subgraph", 16, 8, 96, 2, False, True, False),
]

DATASETS = [
    ("gnn_samples_out", ["p_load_kw", "q_load_kvar", "p_pv_kw"], "Original"),
    ("gnn_samples_inj_full", ["p_inj_kw", "q_inj_kvar"], "Derived"),
    ("gnn_samples_loadtype_full", [
        "electrical_distance_ohm", "m1_p_kw", "m1_q_kvar", "m2_p_kw", "m2_q_kvar",
        "m4_p_kw", "m4_q_kvar", "m5_p_kw", "m5_q_kvar", "q_cap_kvar", "p_pv_kw",
        "p_sys_balance_kw", "q_sys_balance_kvar"
    ], "Load-type"),
]


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_phase_from_node_name(name):
    """Parse phase (1,2,3) from node name like '800.1' or '810.2'."""
    m = re.search(r"\.(\d+)$", str(name))
    return int(m.group(1)) if m else 1


def load_phase_mapping(out_dir):
    """Load node_idx -> phase (0,1,2 for phases 1,2,3)."""
    master = os.path.join(out_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master):
        return None
    df = pd.read_csv(master)
    phase = np.array([_parse_phase_from_node_name(n) - 1 for n in df["node"]], dtype=np.int64)
    return torch.tensor(phase, dtype=torch.long)


def load_phase_subgraph_edges(edge_csv):
    """Build 3 edge_index, edge_attr, edge_id per phase."""
    df = pd.read_csv(edge_csv)
    df["u_idx"] = pd.to_numeric(df["u_idx"], errors="raise").astype(int)
    df["v_idx"] = pd.to_numeric(df["v_idx"], errors="raise").astype(int)
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce").fillna(1).astype(int)
    for c in ["R_full", "X_full"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["u_idx", "v_idx", "R_full", "X_full"])
    result = []
    for p in (1, 2, 3):
        sub = df[df["phase"] == p].reset_index(drop=True)
        if len(sub) == 0:
            result.append((torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)))
        else:
            ei = torch.tensor(sub[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
            ea = torch.tensor(sub[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
            eid = torch.arange(len(sub), dtype=torch.long)
            result.append((ei, ea, eid))
    return result


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EdgeIdentityMP(MessagePassing):
    def __init__(self, h_dim, edge_feat_dim, edge_emb_dim):
        super().__init__(aggr="add")
        self.psi = MLP(in_dim=h_dim + edge_feat_dim + edge_emb_dim, out_dim=h_dim, hidden=h_dim)

    def forward(self, h, edge_index, edge_attr, edge_emb):
        return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr, edge_emb=edge_emb)

    def message(self, h_j, edge_attr, edge_emb):
        return self.psi(torch.cat([h_j, edge_attr, edge_emb], dim=-1))


class PFIdentityGNN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_in_dim, edge_in_dim=2, out_dim=1,
                 node_emb_dim=8, edge_emb_dim=4, h_dim=64, num_layers=4, use_norm=False):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(num_edges, edge_emb_dim)
        self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim)
        self.use_norm = use_norm
        if use_norm:
            self.node_norm = nn.LayerNorm(node_in_dim + node_emb_dim)
            self.edge_norm = nn.LayerNorm(edge_in_dim)
        self.mps = nn.ModuleList([EdgeIdentityMP(h_dim, edge_in_dim, edge_emb_dim) for _ in range(num_layers)])
        self.updates = nn.ModuleList([MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim) for _ in range(num_layers)])
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def forward(self, data):
        x, edge_index, edge_attr, edge_id = data.x, data.edge_index, data.edge_attr, data.edge_id
        node_ids = self._local_node_ids_from_ptr(data.ptr.to(x.device)) if hasattr(data, "ptr") and data.ptr is not None else torch.arange(data.num_nodes, device=x.device)
        z = self.node_emb(node_ids)
        inp = torch.cat([x, z], dim=-1)
        if self.use_norm:
            inp = self.node_norm(inp)
            edge_attr = self.edge_norm(edge_attr)
        h = self.phi0(inp)
        r = self.edge_emb(edge_id)
        for mp, upd in zip(self.mps, self.updates):
            m = mp(h=h, edge_index=edge_index, edge_attr=edge_attr, edge_emb=r)
            h = upd(torch.cat([h, m, z], dim=-1))
        return self.readout(h)

    def _local_node_ids_from_ptr(self, ptr):
        ids = []
        for g in range(ptr.numel() - 1):
            n0, n1 = int(ptr[g].item()), int(ptr[g + 1].item())
            n = n1 - n0
            if n != self.num_nodes:
                raise RuntimeError(f"Batch graph {g} has {n} nodes != expected {self.num_nodes}.")
            ids.append(torch.arange(n, device=ptr.device))
        return torch.cat(ids, dim=0)


class PhaseSubgraphGNN(nn.Module):
    """3 separate subgraphs (one per phase). Each phase has its own MP; outputs combined."""
    def __init__(self, num_nodes, node_in_dim, edge_in_dim=2, out_dim=1,
                 node_emb_dim=8, edge_emb_dim=4, h_dim=64, num_layers=2,
                 max_edges_per_phase=100):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.max_edges_per_phase = max_edges_per_phase
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(max_edges_per_phase * 3, edge_emb_dim)
        self.phi0 = MLP(in_dim=node_in_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim)
        self.mps = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(num_layers):
            self.mps.append(nn.ModuleList([EdgeIdentityMP(h_dim, edge_in_dim, edge_emb_dim) for _ in range(3)]))
            self.updates.append(MLP(in_dim=h_dim + h_dim + node_emb_dim, out_dim=h_dim, hidden=h_dim))
        self.readout = MLP(in_dim=h_dim, out_dim=out_dim, hidden=h_dim)

    def forward(self, data):
        x = data.x
        node_ids = self._local_node_ids_from_ptr(data.ptr.to(x.device)) if hasattr(data, "ptr") and data.ptr is not None else torch.arange(data.num_nodes, device=x.device)
        z = self.node_emb(node_ids)
        h = self.phi0(torch.cat([x, z], dim=-1))
        for mp_list, upd in zip(self.mps, self.updates):
            ms = []
            for p in range(3):
                ei = data[f"edge_index_{p}"]
                ea = data[f"edge_attr_{p}"]
                eid = data[f"edge_id_{p}"]
                if ei.numel() == 0:
                    ms.append(torch.zeros_like(h, device=h.device))
                else:
                    eid_offset = eid + p * self.max_edges_per_phase
                    r = self.edge_emb(eid_offset)
                    m = mp_list[p](h=h, edge_index=ei, edge_attr=ea, edge_emb=r)
                    ms.append(m)
            m_combined = ms[0] + ms[1] + ms[2]
            h = upd(torch.cat([h, m_combined, z], dim=-1))
        return self.readout(h)

    def _local_node_ids_from_ptr(self, ptr):
        ids = []
        for g in range(ptr.numel() - 1):
            n0, n1 = int(ptr[g].item()), int(ptr[g + 1].item())
            ids.append(torch.arange(n1 - n0, device=ptr.device))
        return torch.cat(ids, dim=0)


def compute_metrics(yhat, ytrue):
    err = (yhat - ytrue).squeeze(-1)
    return err.abs().mean(), torch.sqrt((err ** 2).mean())


@torch.no_grad()
def evaluate(model, loader, batch_transform=None):
    model.eval()
    mae_sum = torch.tensor(0.0, device=DEVICE)
    rmse_sum = torch.tensor(0.0, device=DEVICE)
    n_batches = 0
    for data in loader:
        data = data.to(DEVICE)
        if batch_transform is not None:
            data = batch_transform(data)
        mae, rmse = compute_metrics(model(data), data.y)
        mae_sum += mae
        rmse_sum += rmse
        n_batches += 1
    return (mae_sum / max(1, n_batches)).item(), (rmse_sum / max(1, n_batches)).item()


def _log_skip_reason(out_dir, edge_csv, node_csv):
    """Log once why data is missing (avoids spam across 30 configs)."""
    if not hasattr(_log_skip_reason, "_logged"):
        missing = []
        if not os.path.exists(edge_csv):
            missing.append(edge_csv)
        if not os.path.exists(node_csv):
            missing.append(node_csv)
        print(f"  [SKIP] Missing data files. Expected under {os.path.abspath(out_dir)}:")
        for p in missing:
            print(f"    - {p}")
        print("  Run GNN2 notebook to generate gnn_samples_out, gnn_samples_inj_full, gnn_samples_loadtype_full.")
        _log_skip_reason._logged = True


def train_one(out_dir, feature_cols, ds_label, cfg_name, n_emb, e_emb, h_dim, n_layers,
              use_norm, use_phase_subgraphs, use_phase_onehot, data_frac=0.30):
    seed_all(SEED)
    edge_csv = os.path.join(out_dir, "gnn_edges_phase_static.csv")
    node_csv = os.path.join(out_dir, "gnn_node_features_and_targets.csv")
    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        _log_skip_reason(out_dir, edge_csv, node_csv)
        return None, None, None
    required_node_cols = {"sample_id", "node_idx", "vmag_pu"} | set(feature_cols)
    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)
    missing = required_node_cols - set(df_n.columns)
    if missing:
        if not hasattr(train_one, "_warned_missing"):
            print(f"  [SKIP] {out_dir}: missing columns {missing}")
            train_one._warned_missing = True
        return None, None, None

    df_e["u_idx"] = pd.to_numeric(df_e["u_idx"], errors="raise").astype(int)
    df_e["v_idx"] = pd.to_numeric(df_e["v_idx"], errors="raise").astype(int)
    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    for c in ["R_full", "X_full"]:
        df_e[c] = pd.to_numeric(df_e[c], errors="coerce")
    for c in feature_cols + ["vmag_pu"]:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_e = df_e.replace([np.inf, -np.inf], np.nan).dropna(subset=["u_idx", "v_idx", "R_full", "X_full"]).copy()
    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_node_cols)).copy()
    df_e = df_e.reset_index(drop=True).copy()
    df_e["edge_id"] = np.arange(len(df_e), dtype=int)

    E = int(len(df_e))
    N = int(df_n["node_idx"].max()) + 1
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    all_sample_ids = df_n["sample_id"].unique()
    S_full = len(all_sample_ids)
    n_keep = max(1, int(S_full * data_frac))
    rng = np.random.default_rng(SEED)
    keep_ids = rng.choice(all_sample_ids, size=n_keep, replace=False)
    df_n = df_n[df_n["sample_id"].isin(keep_ids)].copy().sort_values(["sample_id", "node_idx"]).reset_index(drop=True)
    S = df_n["sample_id"].nunique()
    X_all = df_n[feature_cols].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y_all = df_n["vmag_pu"].to_numpy(dtype=np.float32).reshape(S, N, 1)

    if use_phase_onehot or use_phase_subgraphs:
        phase_map = load_phase_mapping(out_dir)
        if phase_map is None:
            if not hasattr(train_one, "_warned_phase"):
                print(f"  [SKIP] {out_dir}: missing gnn_node_index_master.csv (required for phase_onehot/phase_subgraph)")
                train_one._warned_phase = True
            return None, None, None
        phase_onehot = F.one_hot(phase_map, num_classes=3).numpy().astype(np.float32)

    if use_phase_onehot:
        X_all = np.concatenate([X_all, np.broadcast_to(phase_onehot[None, :, :], (S, N, 3))], axis=-1)
        feature_cols = list(feature_cols) + ["phase_1", "phase_2", "phase_3"]

    node_in_dim = X_all.shape[-1]
    edge_index = torch.tensor(df_e[["u_idx", "v_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(df_e[["R_full", "X_full"]].to_numpy(), dtype=torch.float)
    edge_id = torch.tensor(df_e["edge_id"].to_numpy(), dtype=torch.long)

    if use_phase_subgraphs:
        phase_edges = load_phase_subgraph_edges(edge_csv)
        max_e = max(pe[2].numel() for pe in phase_edges)
        if max_e == 0:
            return None, None, None
    else:
        phase_edges = None

    perm = rng.permutation(S)
    n_test = int(np.floor(TEST_FRAC * S))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def make_ds(idx):
        out = []
        for k in idx:
            x = torch.tensor(X_all[k], dtype=torch.float)
            y = torch.tensor(Y_all[k], dtype=torch.float)
            if use_phase_subgraphs and phase_edges is not None:
                d = Data(x=x, y=y, num_nodes=N, phase_idx=phase_map)
                for p, (ei, ea, eid) in enumerate(phase_edges):
                    d[f"edge_index_{p}"] = ei
                    d[f"edge_attr_{p}"] = ea
                    d[f"edge_id_{p}"] = eid
            else:
                d = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N)
            out.append(d)
        return out

    train_data, test_data = make_ds(train_idx), make_ds(test_idx)
    batch_sz = 1 if use_phase_subgraphs else BATCH_SIZE
    train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    if use_phase_subgraphs:
        max_edges = max(pe[2].numel() for pe in phase_edges) * 3
        model = PhaseSubgraphGNN(num_nodes=N, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                                node_emb_dim=n_emb, edge_emb_dim=e_emb, h_dim=h_dim, num_layers=n_layers,
                                max_edges_per_phase=max(pe[2].numel() for pe in phase_edges)).to(DEVICE)
    else:
        model = PFIdentityGNN(num_nodes=N, num_edges=E, node_in_dim=node_in_dim, edge_in_dim=2, out_dim=1,
                             node_emb_dim=n_emb, edge_emb_dim=e_emb, h_dim=h_dim, num_layers=n_layers,
                             use_norm=use_norm).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_test, best_state, best_epoch = float("inf"), None, None
    patience_left = EARLY_STOP_PATIENCE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        for data in train_loader:
            data = data.to(DEVICE)
            if use_phase_subgraphs:
                data = _batch_phase_subgraph(data, phase_edges)
            opt.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        train_loss = total_loss / max(1, nb)
        batch_fn = (lambda d: _batch_phase_subgraph(d, phase_edges)) if use_phase_subgraphs else None
        mae_t, rmse_t = evaluate(model, test_loader, batch_transform=batch_fn)
        if (best_test - rmse_t) > MIN_DELTA:
            best_test, best_state, best_epoch = rmse_t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
            patience_left = EARLY_STOP_PATIENCE
        else:
            if epoch >= MIN_EPOCHS_BEFORE_STOP:
                patience_left -= 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | train_loss={train_loss:.6f} | RMSE={rmse_t:.5f} | best={best_test:.5f} | patience={patience_left}")
        if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    mae_f, rmse_f = evaluate(model, test_loader, batch_transform=batch_fn)
    return mae_f, rmse_f, best_epoch


def _batch_phase_subgraph(data, phase_edges):
    """Batch phase subgraph Data: offset edge indices for each graph in batch."""
    from torch_geometric.data import Batch
    if isinstance(data, Batch):
        ptr = data.ptr
        n_graphs = ptr.numel() - 1
        N = int((ptr[1] - ptr[0]).item())
        out = Data(x=data.x, y=data.y, ptr=ptr, phase_idx=data.phase_idx)
        for p in range(3):
            ei, ea, eid = phase_edges[p]
            if ei.numel() == 0:
                out[f"edge_index_{p}"] = ei.to(data.x.device)
                out[f"edge_attr_{p}"] = ea.to(data.x.device)
                out[f"edge_id_{p}"] = eid.to(data.x.device)
            else:
                ei_list = [ei + g * N for g in range(n_graphs)]
                out[f"edge_index_{p}"] = torch.cat(ei_list, dim=1)
                out[f"edge_attr_{p}"] = ea.repeat(n_graphs, 1)
                out[f"edge_id_{p}"] = eid.repeat(n_graphs)
        return out
    return data


def main():
    print("=" * 70)
    print(f"NARROW EXPLORATION: {int(DATA_FRAC*100)}% data, 10 configs × 3 datasets")
    print(f"Working dir: {os.path.abspath('.')}")
    for d, _, _ in DATASETS:
        exists = "OK" if os.path.isdir(d) else "MISSING"
        ec = os.path.join(d, "gnn_edges_phase_static.csv")
        nc = os.path.join(d, "gnn_node_features_and_targets.csv")
        print(f"  {d}: {exists} (edge_csv={os.path.exists(ec)}, node_csv={os.path.exists(nc)})")
    print("Corrected MAE/RMSE | Depth | Norm | Phase subgraph/onehot")
    print("=" * 70)
    results = []
    for cfg in CANDIDATES:
        cfg_name, n_emb, e_emb, h_dim, n_layers, use_norm, use_phase_subgraphs, use_phase_onehot = cfg
        for out_dir, feature_cols, ds_label in DATASETS:
            print(f"\n>>> {cfg_name} + {ds_label} (n={n_layers} h={h_dim} norm={use_norm} phase_sg={use_phase_subgraphs} phase_oh={use_phase_onehot})")
            try:
                mae, rmse, best_ep = train_one(out_dir, feature_cols, ds_label, cfg_name, n_emb, e_emb, h_dim, n_layers,
                                              use_norm, use_phase_subgraphs, use_phase_onehot, data_frac=DATA_FRAC)
                if mae is not None:
                    hit = " *** TARGET HIT ***" if mae < TARGET_MAE else ""
                    results.append((cfg_name, ds_label, mae, rmse, best_ep, hit))
                    print(f"  FINAL: MAE={mae:.6f} RMSE={rmse:.6f} best_epoch={best_ep}{hit}")
                else:
                    print(f"  SKIP (missing data)")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by MAE)")
    print("=" * 70)
    results.sort(key=lambda r: r[2])
    df_res = pd.DataFrame(results, columns=["Config", "Dataset", "MAE", "RMSE", "Best_epoch", "Target_hit"])
    if len(df_res) > 0:
        print(df_res.to_string())
        best = df_res.iloc[0]
        print(f"\nBest: {best['Config']} + {best['Dataset']} | MAE={best['MAE']:.6f} | RMSE={best['RMSE']:.6f}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
