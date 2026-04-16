"""
Train GINE+MLP (no local/global split) with optional auxiliary heads.

Main task:
  PQ graph snapshot -> flattened complex voltage [V_re, V_im] for all nodes.

Aux tasks (training-time supervision):
  - 12 regulator tap classifications
  - 10 capacitor step classifications

Aux heads read the GINE+MLP voltage output vector directly.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges, _load_nodes_pq_target

TARGET_REG_COLS: tuple[str, ...] = (
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

TARGET_CAP_COLS: tuple[str, ...] = (
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


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm_sid(s: object) -> int:
    try:
        return int(float(s))
    except Exception:
        return int(str(s).strip())


def _build_classes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yq = np.round(y.astype(np.float64), 6)
    classes = np.unique(yq)
    cls_to_i = {float(c): i for i, c in enumerate(classes.tolist())}
    idx = np.array([cls_to_i[float(v)] for v in yq.tolist()], dtype=np.int64)
    return classes.astype(np.float32), idx


def _load_aux_targets(meta_csv: Path, sample_ids: list[int]) -> dict:
    import pandas as pd

    usecols = ["sample_id", *TARGET_REG_COLS, *TARGET_CAP_COLS]
    df = pd.read_csv(meta_csv, usecols=usecols)
    lk = {_norm_sid(k): i for i, k in enumerate(df["sample_id"].tolist())}
    miss = [sid for sid in sample_ids if _norm_sid(sid) not in lk]
    if miss:
        raise KeyError(f"{len(miss)} sample_id values from nodes CSV not found in {meta_csv}")
    order = [lk[_norm_sid(sid)] for sid in sample_ids]
    out: dict = {"reg": [], "cap": []}
    for c in TARGET_REG_COLS:
        y = df[c].to_numpy(dtype=np.float64)[order]
        cls, yi = _build_classes(y)
        out["reg"].append({"name": c, "classes": cls, "y_idx": torch.from_numpy(yi)})
    for c in TARGET_CAP_COLS:
        y = df[c].to_numpy(dtype=np.float64)[order]
        cls, yi = _build_classes(y)
        out["cap"].append({"name": c, "classes": cls, "y_idx": torch.from_numpy(yi)})
    return out


def _build_complex_targets(nodes_csv: Path, sample_ids: list[int], node_to_local: dict[str, int]) -> torch.Tensor:
    import pandas as pd

    sid_to_i = {int(s): i for i, s in enumerate(sample_ids)}
    y_ri = np.zeros((len(sample_ids), len(node_to_local), 2), dtype=np.float32)
    usecols = ["sample_id", "node", "vmag_pu", "vang_deg"]
    for chunk in pd.read_csv(nodes_csv, usecols=usecols, chunksize=500_000):
        row_s = chunk["sample_id"].map(lambda v: sid_to_i.get(int(float(v)), -1)).to_numpy(dtype=np.int64)
        row_n = chunk["node"].map(lambda v: node_to_local.get(str(v).strip(), -1)).to_numpy(dtype=np.int64)
        valid = (row_s >= 0) & (row_n >= 0)
        if not np.any(valid):
            continue
        s = row_s[valid]
        n = row_n[valid]
        vmag = chunk.loc[valid, "vmag_pu"].to_numpy(dtype=np.float32)
        vang_rad = np.deg2rad(chunk.loc[valid, "vang_deg"].to_numpy(dtype=np.float32))
        y_ri[s, n, 0] = vmag * np.cos(vang_rad)
        y_ri[s, n, 1] = vmag * np.sin(vang_rad)
    return torch.from_numpy(y_ri)


class AuxGraphDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y_ri: torch.Tensor,
        y_reg: list[torch.Tensor],
        y_cap: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        self.x = x
        self.y_ri = y_ri
        self.y_reg = y_reg
        self.y_cap = y_cap
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Data:
        d = Data(x=self.x[i], y=self.y_ri[i], edge_index=self.edge_index, edge_attr=self.edge_attr)
        d.y_reg = torch.stack([yr[i] for yr in self.y_reg], dim=0).long()
        d.y_cap = torch.stack([yc[i] for yc in self.y_cap], dim=0).long()
        return d


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


class GINEEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_nodes: int,
        num_edges: int,
        hidden: int,
        n_layers: int,
        state_dim: int,
        node_emb_dim: int,
        edge_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.num_edges = int(num_edges)
        self.node_emb_dim = max(0, int(node_emb_dim))
        self.edge_emb_dim = max(0, int(edge_emb_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim) if self.edge_emb_dim > 0 else None
        eff_in = in_dim + self.node_emb_dim
        eff_edge = 2 + self.edge_emb_dim
        self.input_proj = nn.Linear(eff_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(n_layers)):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=eff_edge))
            self.norms.append(nn.LayerNorm(hidden))
        self.state_head = nn.Linear(hidden, int(state_dim))

    def _node_ids(self, n_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_nodes, device=device, dtype=torch.long).repeat(n_total // self.n_nodes)

    def _edge_ids(self, e_total: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_edges, device=device, dtype=torch.long).repeat(e_total // self.num_edges)

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        ea = batch.edge_attr
        if self.node_emb is not None:
            x = torch.cat([x, self.node_emb(self._node_ids(x.size(0), x.device))], dim=-1)
        if self.edge_emb is not None:
            ea = torch.cat([ea, self.edge_emb(self._edge_ids(ea.size(0), ea.device))], dim=-1)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_msg = F.relu(conv(h, batch.edge_index, ea))
            h = h + self.dropout(norm(h_msg))
        state = self.state_head(h)
        return state.view(batch.num_graphs, self.n_nodes, -1).reshape(batch.num_graphs, 2 * self.n_nodes)


class GINEPlusMLPAux(nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        encoder: GINEEncoder,
        hidden_mlp: int,
        aux_hidden: int,
        reg_nclasses: list[int],
        cap_nclasses: list[int],
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.encoder = encoder
        self.voltage_mlp = RealMLP(in_dim=2 * self.n_nodes, out_dim=2 * self.n_nodes, hidden=hidden_mlp)
        aux_in = 2 * self.n_nodes
        self.aux_proj = nn.Linear(aux_in, aux_hidden)
        self.aux_reg_heads = nn.ModuleList([nn.Linear(aux_hidden, c) for c in reg_nclasses])
        self.aux_cap_heads = nn.ModuleList([nn.Linear(aux_hidden, c) for c in cap_nclasses])

    def forward_train(self, batch: Data) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        state_flat = self.encoder(batch)
        v_pred_n = self.voltage_mlp(state_flat)
        h_aux = F.relu(self.aux_proj(v_pred_n))
        reg_logits = [head(h_aux) for head in self.aux_reg_heads]
        cap_logits = [head(h_aux) for head in self.aux_cap_heads]
        return v_pred_n, reg_logits, cap_logits

    def forward(self, batch: Data) -> torch.Tensor:
        v_pred_n, _r, _c = self.forward_train(batch)
        return v_pred_n


def _aux_loss(reg_logits: list[torch.Tensor], cap_logits: list[torch.Tensor], y_reg: torch.Tensor, y_cap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reg_losses = [F.cross_entropy(lg, y_reg[:, j]) for j, lg in enumerate(reg_logits)]
    cap_losses = [F.cross_entropy(lg, y_cap[:, j]) for j, lg in enumerate(cap_logits)]
    lreg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=y_reg.device)
    lcap = torch.stack(cap_losses).mean() if cap_losses else torch.tensor(0.0, device=y_reg.device)
    return lreg, lcap


def _aux_lambda_scale(epoch_1based: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if warmup_epochs <= 0 and ramp_epochs <= 0:
        return 1.0
    if epoch_1based <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    t = epoch_1based - warmup_epochs
    if t > ramp_epochs:
        return 1.0
    return float(t) / float(ramp_epochs)


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _metrics_from_ri_flat(pred_ri: torch.Tensor, true_ri: torch.Tensor) -> dict[str, float]:
    bsz, two_n = pred_ri.shape
    n_nodes = two_n // 2
    pred = pred_ri.view(bsz, n_nodes, 2)
    true = true_ri.view(bsz, n_nodes, 2)
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


@torch.no_grad()
def _evaluate_voltage(model: nn.Module, dl: DataLoader, device: torch.device, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    for batch in dl:
        batch = batch.to(device)
        yp_n = model(batch)
        yp = yp_n * y_std.to(device) + y_mean.to(device)
        preds.append(yp.cpu())
        tgts.append(batch.y.view(batch.num_graphs, -1).cpu())
    return _metrics_from_ri_flat(torch.cat(preds, dim=0), torch.cat(tgts, dim=0))


@torch.no_grad()
def _evaluate_aux_accuracy(model: GINEPlusMLPAux, dl: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    reg_correct = reg_total = 0
    cap_correct = cap_total = 0
    for batch in dl:
        batch = batch.to(device)
        _v, reg_logits, cap_logits = model.forward_train(batch)
        yr = batch.y_reg.view(batch.num_graphs, -1).long()
        yc = batch.y_cap.view(batch.num_graphs, -1).long()
        for j, lg in enumerate(reg_logits):
            pred = lg.argmax(dim=1)
            reg_correct += int((pred == yr[:, j]).sum().item())
            reg_total += int(pred.numel())
        for j, lg in enumerate(cap_logits):
            pred = lg.argmax(dim=1)
            cap_correct += int((pred == yc[:, j]).sum().item())
            cap_total += int(pred.numel())
    return {
        "reg_acc": float(reg_correct / max(reg_total, 1)),
        "cap_acc": float(cap_correct / max(cap_total, 1)),
    }


@dataclass
class RunResult:
    best_val_mse: float
    test_mae_vmag: float
    test_rmse_vmag: float
    test_mae_angle_deg: float
    test_rmse_angle_deg: float
    test_reg_acc: float
    test_cap_acc: float
    train_seconds: float
    checkpoint: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GINE+MLP with aux heads on voltage output.")
    p.add_argument("--data_root", type=str, default="datasets_gnn2/loadtype_8500_dailyagg")
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument("--out_dir", type=str, default="gine_plus_mlp_aux_complex_8500")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_gnn", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--state_dim", type=int, default=2)
    p.add_argument("--hidden_mlp", type=int, default=1024)
    p.add_argument("--aux_hidden", type=int, default=512)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--edge_emb_dim", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--disable_dropout", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--cache_tensor", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lambda_reg", type=float, default=0.02)
    p.add_argument("--lambda_cap", type=float, default=0.01)
    p.add_argument("--aux_warmup_epochs", type=int, default=30)
    p.add_argument("--aux_ramp_epochs", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    if int(args.state_dim) != 2:
        raise ValueError("This script expects state_dim=2 to match the baseline GINE+MLP trunk.")

    repo = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    nodes_path = Path(args.nodes_csv) if Path(args.nodes_csv).is_absolute() else (data_root / args.nodes_csv).resolve()
    edges_path = Path(args.edge_catalog_csv) if Path(args.edge_catalog_csv).is_absolute() else (data_root / args.edge_catalog_csv).resolve()
    meta_path = Path(args.meta_csv) if Path(args.meta_csv).is_absolute() else (data_root / args.meta_csv).resolve()
    for p in (nodes_path, edges_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_tensor).resolve() if args.cache_tensor else None
    node_to_local = None
    if cache_path and cache_path.is_file():
        print(f"Loading cache: {cache_path}", flush=True)
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = pack["x"]
        edge_index = pack["edge_index"]
        edge_attr = pack["edge_attr"]
        sample_ids = pack["sample_ids"]
    else:
        x, _y_unused, sample_ids, _node_order, node_to_local = _load_nodes_pq_target(nodes_path)
        edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "sample_ids": sample_ids}, cache_path)
            print(f"Wrote cache: {cache_path}", flush=True)

    if node_to_local is None:
        _x_tmp, _y_tmp, _sid_tmp, _n_tmp, node_to_local = _load_nodes_pq_target(nodes_path)
        del _x_tmp, _y_tmp, _sid_tmp, _n_tmp

    y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)
    aux = _load_aux_targets(meta_path, sample_ids)
    y_reg = [d["y_idx"] for d in aux["reg"]]
    y_cap = [d["y_idx"] for d in aux["cap"]]
    reg_nclasses = [len(d["classes"]) for d in aux["reg"]]
    cap_nclasses = [len(d["classes"]) for d in aux["cap"]]

    if args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0,1].")
        k = max(1, int(round(len(sample_ids) * args.sample_frac)))
        x = x[:k]
        y_ri = y_ri[:k]
        y_reg = [yy[:k] for yy in y_reg]
        y_cap = [yy[:k] for yy in y_cap]
        sample_ids = sample_ids[:k]
        print(f"Using sample_frac={args.sample_frac} => {k} samples", flush=True)

    n = int(x.shape[0])
    n_nodes = int(x.shape[1])
    perm = np.random.default_rng(args.seed).permutation(n)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("Invalid split; require at least one sample in train/val/test.")
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]
    print(f"Split train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}", flush=True)

    xt = x[idx_train].reshape(-1, 2)
    x_mean = xt.mean(dim=0, keepdim=True)
    x_std = xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_n = (x - x_mean) / x_std
    y_train = y_ri[idx_train].reshape(len(idx_train), -1)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    ds = AuxGraphDataset(x_n, y_ri, y_reg, y_cap, edge_index, edge_attr)
    dl_tr = DataLoader(Subset(ds, idx_train.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(Subset(ds, idx_val.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(Subset(ds, idx_test.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GINEEncoder(
        in_dim=2,
        n_nodes=n_nodes,
        num_edges=int(edge_index.shape[1]),
        hidden=int(args.hidden_gnn),
        n_layers=int(args.layers),
        state_dim=2,
        node_emb_dim=int(args.node_emb_dim),
        edge_emb_dim=int(args.edge_emb_dim),
        dropout=0.0 if args.disable_dropout else float(args.dropout),
    )
    model = GINEPlusMLPAux(
        n_nodes=n_nodes,
        encoder=encoder,
        hidden_mlp=int(args.hidden_mlp),
        aux_hidden=int(args.aux_hidden),
        reg_nclasses=reg_nclasses,
        cap_nclasses=cap_nclasses,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        aux_scale = _aux_lambda_scale(ep, int(args.aux_warmup_epochs), int(args.aux_ramp_epochs))
        eff_reg = float(args.lambda_reg) * aux_scale
        eff_cap = float(args.lambda_cap) * aux_scale

        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            yv = batch.y.view(batch.num_graphs, -1)
            yr = batch.y_reg.view(batch.num_graphs, -1).long()
            yc = batch.y_cap.view(batch.num_graphs, -1).long()
            yv_n = (yv - y_mean.to(device)) / y_std.to(device)
            v_pred_n, reg_logits, cap_logits = model.forward_train(batch)
            lv = mse(v_pred_n, yv_n)
            lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
            loss = lv + eff_reg * lr_aux + eff_cap * lc_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in dl_va:
                batch = batch.to(device)
                yv = batch.y.view(batch.num_graphs, -1)
                yr = batch.y_reg.view(batch.num_graphs, -1).long()
                yc = batch.y_cap.view(batch.num_graphs, -1).long()
                yv_n = (yv - y_mean.to(device)) / y_std.to(device)
                v_pred_n, reg_logits, cap_logits = model.forward_train(batch)
                lv = mse(v_pred_n, yv_n)
                lr_aux, lc_aux = _aux_loss(reg_logits, cap_logits, yr, yc)
                ltot = lv + eff_reg * lr_aux + eff_cap * lc_aux
                val_loss += float(ltot.item()) * batch.num_graphs
                nv += int(batch.num_graphs)
        val_loss /= max(nv, 1)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            print(
                f"[gine+mlp+aux] epoch {ep:4d}/{args.epochs} aux_scale={aux_scale:.4f} "
                f"eff_lambda_reg={eff_reg:.5f} eff_lambda_cap={eff_cap:.5f} "
                f"val_obj={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )
        if bad >= args.patience:
            print(f"[gine+mlp+aux] early stopping at epoch {ep}", flush=True)
            break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    met = _evaluate_voltage(model, dl_te, device, y_mean, y_std)
    aux_acc = _evaluate_aux_accuracy(model, dl_te, device)

    ckpt_path = out_dir / "gine_plus_mlp_aux_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_nodes": int(n_nodes),
            "hidden_gnn": int(args.hidden_gnn),
            "layers": int(args.layers),
            "state_dim": int(args.state_dim),
            "hidden_mlp": int(args.hidden_mlp),
            "aux_hidden": int(args.aux_hidden),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        ckpt_path,
    )
    torch.save(x_mean, out_dir / "x_mean.pt")
    torch.save(x_std, out_dir / "x_std.pt")
    torch.save(y_mean, out_dir / "y_mean.pt")
    torch.save(y_std, out_dir / "y_std.pt")

    result = RunResult(
        best_val_mse=float(best_val),
        test_mae_vmag=met["mae_vmag_pu"],
        test_rmse_vmag=met["rmse_vmag_pu"],
        test_mae_angle_deg=met["mae_angle_deg"],
        test_rmse_angle_deg=met["rmse_angle_deg"],
        test_reg_acc=float(aux_acc["reg_acc"]),
        test_cap_acc=float(aux_acc["cap_acc"]),
        train_seconds=float(train_seconds),
        checkpoint=str(ckpt_path.resolve()),
    )
    report = {
        "task": "GINE+MLP with aux heads on voltage output (no local/global voltage decomposition)",
        "dataset_nodes_csv": str(nodes_path),
        "dataset_edges_csv": str(edges_path),
        "dataset_meta_csv": str(meta_path),
        "n_samples": int(n),
        "n_nodes": int(n_nodes),
        "split": {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
        "model": {
            "type": "gine_plus_mlp_aux",
            "hidden_gnn": int(args.hidden_gnn),
            "layers": int(args.layers),
            "state_dim": int(args.state_dim),
            "hidden_mlp": int(args.hidden_mlp),
            "aux_hidden": int(args.aux_hidden),
            "node_emb_dim": int(args.node_emb_dim),
            "edge_emb_dim": int(args.edge_emb_dim),
        },
        "aux_schedule": {
            "lambda_reg": float(args.lambda_reg),
            "lambda_cap": float(args.lambda_cap),
            "aux_warmup_epochs": int(args.aux_warmup_epochs),
            "aux_ramp_epochs": int(args.aux_ramp_epochs),
        },
        "result": result.__dict__,
        "aux_targets": {
            "reg": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["reg"]],
            "cap": [{"name": d["name"], "n_classes": len(d["classes"])} for d in aux["cap"]],
        },
    }
    report_path = out_dir / "gine_plus_mlp_aux_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Training complete ===", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(
        f"GINE+MLP+aux: |V| MAE={result.test_mae_vmag:.6f} pu, angle MAE={result.test_mae_angle_deg:.6f} deg, "
        f"reg_acc={result.test_reg_acc:.4f}, cap_acc={result.test_cap_acc:.4f}, time={result.train_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
