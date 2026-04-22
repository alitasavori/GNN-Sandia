"""
Evaluate frozen GNN-only vs GINE+MLP checkpoints on an OOD daily-aggregate bundle.

Uses the *training-run* normalization tensors (x_mean/std, y_mean/std) shipped next
to each checkpoint — not statistics recomputed on OOD — so metrics reflect true
out-of-distribution generalization.

Also reports:
  - |V| error percentiles (p50, p90, p95, p99, max) and angle MAE/RMSE
  - Histogram of |V| errors (JSON-friendly bin edges + counts)
  - Min positive electrical distance to any regulator column (from
    load_electrical_distance_to_each_regulator.csv), binned into quantiles
  - Mean |V| error per distance bin (shows whether errors grow far from regs)
  - Spearman correlation between per-(sample,node) |V| error and distance
    (subsampled if very large)

Example::

  python eval_ood_complex_voltage_checkpoints.py \\
    --ood_data_root datasets_gnn2/loadtype_8500_dailyagg/loadtype_8500_dailyagg_ood_stress \\
    --electrical_distance_csv 8500-node/load_electrical_distance_to_each_regulator.csv \\
    --eval_dirs gnn2_architecture_search/homo_mv_8500/comparison/gnn_only_compare_complex_8500 \\
                gnn2_architecture_search/homo_mv_8500/comparison/gine_plus_mlp_to_overcome_mlp_only \\
    --out_json gnn2_architecture_search/homo_mv_8500/comparison/ood_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from train_gine_plus_mlp_complex_voltage import (
    GINEEncoder,
    GINEPlusMLP,
    GraphVoltageDataset,
    _build_complex_targets,
)
from train_gine_plus_mlp_aux_complex_voltage import GINEPlusMLPAux as GINEPlusMLPAuxVoltage
from train_gine_plus_mlp_aux_complex_voltage import GINEEncoder as GINEEncoderAux
from train_gnn_only_compare_complex_voltage import GNNOnlyVoltageModel
from train_compare_mlp_cvnn_complex_voltage import ComplexMLP, RealMLP
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges, _load_nodes_pq_target


class _RealMLPAsGraph(nn.Module):
    """Adapter: RealMLP on flattened [B,2N] but called with PyG batch."""

    def __init__(self, mlp: RealMLP, n_nodes: int):
        super().__init__()
        self.mlp = mlp
        self.n_nodes = int(n_nodes)

    def forward(self, batch: Data) -> torch.Tensor:
        b = int(batch.num_graphs)
        x_flat = batch.x.view(b, self.n_nodes, 2).reshape(b, 2 * self.n_nodes)
        return self.mlp(x_flat)  # [B, 2N] normalized [V_re, V_im]


class _ComplexMLPAsGraph(nn.Module):
    """Adapter: ComplexMLP on complex [B,N] but called with PyG batch."""

    def __init__(self, cvnn: ComplexMLP, n_nodes: int):
        super().__init__()
        self.cvnn = cvnn
        self.n_nodes = int(n_nodes)

    def forward(self, batch: Data) -> torch.Tensor:
        b = int(batch.num_graphs)
        x_ri = batch.x.view(b, self.n_nodes, 2)
        z_in = torch.complex(x_ri[..., 0], x_ri[..., 1])  # [B, N]
        z_pred_n = self.cvnn(z_in)  # complex normalized
        return torch.stack([z_pred_n.real, z_pred_n.imag], dim=-1).reshape(b, 2 * self.n_nodes)


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _load_ckpt_dict(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_norm_tensors(ckpt: dict[str, Any], ckpt_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = ckpt_path.parent
    # Real MLP checkpoint stores x/y normalization in-checkpoint.
    if all(k in ckpt for k in ("x_mean", "x_std", "y_mean", "y_std")):
        return ckpt["x_mean"], ckpt["x_std"], ckpt["y_mean"], ckpt["y_std"]
    # Real MLP checkpoint from train_compare_mlp_cvnn_complex_voltage.py:
    # y stats are in-checkpoint, x stats are sidecar x_mean.pt/x_std.pt.
    if all(k in ckpt for k in ("y_mean", "y_std")):
        xm = torch.load(parent / "x_mean.pt", map_location="cpu", weights_only=True)
        xs = torch.load(parent / "x_std.pt", map_location="cpu", weights_only=True)
        ym = torch.as_tensor(ckpt["y_mean"], dtype=torch.float32).view(1, -1)
        ys = torch.as_tensor(ckpt["y_std"], dtype=torch.float32).view(1, -1)
        return xm, xs, ym, ys
    # CVNN checkpoint stores per-component y normalization in-checkpoint; x stats beside ckpt.
    if all(k in ckpt for k in ("y_mean_re", "y_std_re", "y_mean_im", "y_std_im")):
        xm = torch.load(parent / "x_mean.pt", map_location="cpu", weights_only=True)
        xs = torch.load(parent / "x_std.pt", map_location="cpu", weights_only=True)
        ymr = torch.as_tensor(ckpt["y_mean_re"], dtype=torch.float32).view(1, -1)
        ysr = torch.as_tensor(ckpt["y_std_re"], dtype=torch.float32).view(1, -1)
        ymi = torch.as_tensor(ckpt["y_mean_im"], dtype=torch.float32).view(1, -1)
        ysi = torch.as_tensor(ckpt["y_std_im"], dtype=torch.float32).view(1, -1)
        ym = torch.stack([ymr, ymi], dim=-1).reshape(1, -1)
        ys = torch.stack([ysr, ysi], dim=-1).reshape(1, -1)
        return xm, xs, ym, ys
    xm = torch.load(parent / "x_mean.pt", map_location="cpu", weights_only=True)
    xs = torch.load(parent / "x_std.pt", map_location="cpu", weights_only=True)
    ym = torch.load(parent / "y_mean.pt", map_location="cpu", weights_only=True)
    ys = torch.load(parent / "y_std.pt", map_location="cpu", weights_only=True)
    return xm, xs, ym, ys


def _infer_num_edges_from_state(sd: dict[str, torch.Tensor]) -> Optional[int]:
    w = sd.get("edge_emb.weight")
    if w is None:
        return None
    return int(w.shape[0])


def _infer_aux_head_nclasses(sd: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """
    Infer per-head class counts from aux head final linear weights.
    Supports nn.Sequential heads where final layer index may vary.
    """
    import re

    by_head: dict[int, list[tuple[int, int]]] = {}
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(\d+)\.weight$")
    for k, w in sd.items():
        m = pat.match(k)
        if m is None:
            continue
        h_idx = int(m.group(1))
        layer_idx = int(m.group(2))
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        out_dim = int(w.shape[0])
        by_head.setdefault(h_idx, []).append((layer_idx, out_dim))
    if not by_head:
        return []
    out: list[int] = []
    for h in range(max(by_head.keys()) + 1):
        layers = sorted(by_head.get(h, []), key=lambda t: t[0])
        if not layers:
            raise ValueError(f"Missing aux head index {h} for prefix {prefix}")
        out.append(int(layers[-1][1]))
    return out


def _classify_and_build_model(ckpt_path: Path, ckpt: dict[str, Any], n_nodes_ood: int, n_edges_ood: int) -> tuple[nn.Module, str]:
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{ckpt_path}: expected a dict with 'model_state_dict'")
    sd = ckpt["model_state_dict"]
    if "model_type" in ckpt:
        mt = str(ckpt["model_type"]).lower()
        n_nodes = int(ckpt.get("n_nodes", n_nodes_ood))
        if n_nodes != n_nodes_ood:
            raise ValueError(f"{ckpt_path}: ckpt n_nodes={n_nodes} != OOD n_nodes={n_nodes_ood}")
        n_e_ckpt = _infer_num_edges_from_state(sd)
        n_e = int(n_e_ckpt if n_e_ckpt is not None else n_edges_ood)
        if n_e != n_edges_ood:
            raise ValueError(
                f"{ckpt_path}: edge count mismatch ckpt/inferred num_edges={n_e} vs OOD graph={n_edges_ood} "
                "(topology must match training)."
            )
        model = GNNOnlyVoltageModel(
            model_type=mt,
            in_dim=2,
            hidden=int(ckpt["hidden"]),
            layers=int(ckpt["layers"]),
            n_nodes=n_nodes,
            num_edges=n_e,
            node_emb_dim=int(ckpt["node_emb_dim"]),
            edge_emb_dim=int(ckpt["edge_emb_dim"]),
            dropout=0.0,
        )
        model.load_state_dict(sd, strict=True)
        tag = f"gnn_only:{mt}"
        return model, tag
    if "hidden_mlp" in ckpt and any(k.startswith("encoder.") for k in sd):
        if any(k.startswith("aux_reg_heads.") or k.startswith("aux_cap_heads.") for k in sd):
            n_nodes = int(ckpt.get("n_nodes", n_nodes_ood))
            if n_nodes != n_nodes_ood:
                raise ValueError(f"{ckpt_path}: ckpt n_nodes={n_nodes} != OOD n_nodes={n_nodes_ood}")
            n_e_ckpt = _infer_num_edges_from_state(sd)
            n_e = int(n_e_ckpt if n_e_ckpt is not None else n_edges_ood)
            if n_e != n_edges_ood:
                raise ValueError(
                    f"{ckpt_path}: GINE+MLP+aux ckpt num_edges={n_e} vs OOD={n_edges_ood}; graphs must align."
                )
            reg_nclasses = _infer_aux_head_nclasses(sd, "aux_reg_heads")
            cap_nclasses = _infer_aux_head_nclasses(sd, "aux_cap_heads")
            enc = GINEEncoderAux(
                in_dim=2,
                n_nodes=n_nodes,
                num_edges=n_e,
                hidden=int(ckpt["hidden_gnn"]),
                n_layers=int(ckpt["layers"]),
                state_dim=int(ckpt.get("state_dim", 2)),
                node_emb_dim=int(ckpt["node_emb_dim"]),
                edge_emb_dim=int(ckpt["edge_emb_dim"]),
                dropout=0.0,
            )
            model = GINEPlusMLPAuxVoltage(
                n_nodes=n_nodes,
                encoder=enc,
                hidden_mlp=int(ckpt["hidden_mlp"]),
                aux_head_depth=int(ckpt.get("aux_head_depth", 1)),
                aux_head_dropout=0.0,
                aux_head_first_hidden=int(ckpt.get("aux_head_first_hidden", ckpt.get("aux_hidden", 512))),
                reg_nclasses=reg_nclasses,
                cap_nclasses=cap_nclasses,
            )
            model.load_state_dict(sd, strict=True)
            return model, "gine_plus_mlp_aux"
        n_nodes = int(ckpt.get("n_nodes", n_nodes_ood))
        if n_nodes != n_nodes_ood:
            raise ValueError(f"{ckpt_path}: ckpt n_nodes={n_nodes} != OOD n_nodes={n_nodes_ood}")
        n_e_ckpt = _infer_num_edges_from_state(sd)
        n_e = int(n_e_ckpt if n_e_ckpt is not None else n_edges_ood)
        if n_e != n_edges_ood:
            raise ValueError(
                f"{ckpt_path}: GINE+MLP ckpt num_edges={n_e} vs OOD={n_edges_ood}; graphs must align."
            )
        enc = GINEEncoder(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=n_e,
            hidden=int(ckpt["hidden_gnn"]),
            n_layers=int(ckpt["layers"]),
            state_dim=int(ckpt.get("state_dim", 2)),
            node_emb_dim=int(ckpt["node_emb_dim"]),
            edge_emb_dim=int(ckpt["edge_emb_dim"]),
            dropout=0.0,
        )
        model = GINEPlusMLP(encoder=enc, mlp_hidden=int(ckpt["hidden_mlp"]), n_nodes=n_nodes)
        model.load_state_dict(sd, strict=True)
        return model, "gine_plus_mlp"
    # Real MLP checkpoint from train_compare_mlp_cvnn_complex_voltage.py
    if all(k in ckpt for k in ("in_dim", "out_dim", "hidden")):
        in_dim = int(ckpt["in_dim"])
        out_dim = int(ckpt["out_dim"])
        n_nodes = in_dim // 2
        if in_dim != 2 * n_nodes_ood or out_dim != 2 * n_nodes_ood:
            raise ValueError(
                f"{ckpt_path}: RealMLP dim mismatch in/out=({in_dim},{out_dim}) vs OOD expects {2*n_nodes_ood}."
            )
        mlp = RealMLP(in_dim=in_dim, out_dim=out_dim, hidden=int(ckpt["hidden"]))
        mlp.load_state_dict(sd, strict=True)
        return _RealMLPAsGraph(mlp=mlp, n_nodes=n_nodes_ood), "real_mlp_flat"
    # CVNN MLP checkpoint from train_compare_mlp_cvnn_complex_voltage.py
    if all(k in ckpt for k in ("in_dim_complex", "out_dim_complex", "hidden")):
        n_in = int(ckpt["in_dim_complex"])
        n_out = int(ckpt["out_dim_complex"])
        if n_in != n_nodes_ood or n_out != n_nodes_ood:
            raise ValueError(
                f"{ckpt_path}: CVNN dim mismatch in/out=({n_in},{n_out}) vs OOD expects {n_nodes_ood}."
            )
        cvnn = ComplexMLP(in_dim=n_in, out_dim=n_out, hidden=int(ckpt["hidden"]))
        cvnn.load_state_dict(sd, strict=True)
        return _ComplexMLPAsGraph(cvnn=cvnn, n_nodes=n_nodes_ood), "cvnn_mlp_flat"
    raise ValueError(
        f"{ckpt_path}: unrecognized checkpoint (supported: GNN-only, GINE+MLP(+aux), real_mlp, cvnn_mlp)."
    )


def _min_positive_regulator_distance(dist_csv: Path, node_order: list[str]) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(dist_csv)
    if "node" not in df.columns:
        raise ValueError(f"{dist_csv} must have a 'node' column")
    reg_cols = [c for c in df.columns if c != "node"]
    lk = {str(r["node"]).strip(): r for _, r in df.iterrows()}
    out = np.full(len(node_order), np.nan, dtype=np.float64)
    for i, nod in enumerate(node_order):
        key = str(nod).strip()
        if key not in lk:
            continue
        row = lk[key]
        pos: list[float] = []
        for c in reg_cols:
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if v > 0.0:
                pos.append(v)
        if pos:
            out[i] = float(min(pos))
    return out


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(x, y, nan_policy="omit")
        return float(r) if not math.isnan(float(r)) else float("nan")
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        c = np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1]
        return float(c) if not math.isnan(float(c)) else float("nan")


@torch.no_grad()
def _run_model_collect_errors(
    model: nn.Module,
    dl: PyGDataLoader,
    device: torch.device,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    dist_per_node: np.ndarray,
    corr_subsample: int,
) -> dict[str, Any]:
    model.eval()
    vmag_abs_list: list[np.ndarray] = []
    ang_abs_list: list[np.ndarray] = []
    dist_tile_list: list[np.ndarray] = []

    for batch in dl:
        batch = batch.to(device)
        b = int(batch.num_graphs)
        n = dist_per_node.shape[0]
        x_mean_dev = x_mean.to(device)
        x_std_dev = x_std.to(device)
        # Support both training normalization layouts:
        # 1) per-feature (P,Q): x_mean shape [1,2] or [2]
        # 2) flattened per-node feature: x_mean shape [1,2N] or [2N] (MLP/CVNN checkpoints)
        if int(x_mean_dev.numel()) == 2:
            xm2 = x_mean_dev.view(1, 2)
            xs2 = x_std_dev.view(1, 2)
            xn = (batch.x - xm2) / xs2
        elif int(x_mean_dev.numel()) == 2 * n:
            x_flat = batch.x.view(b, n, 2).reshape(b, 2 * n)
            xm = x_mean_dev.view(1, 2 * n)
            xs = x_std_dev.view(1, 2 * n)
            x_flat_n = (x_flat - xm) / xs
            xn = x_flat_n.view(b, n, 2).reshape(b * n, 2)
        else:
            raise ValueError(
                f"Unsupported x normalization shape: numel={int(x_mean_dev.numel())}, expected 2 or {2*n}."
            )
        batch_n = batch.clone()
        batch_n.x = xn
        pred_n = model(batch_n)
        pred = pred_n * y_std.to(device) + y_mean.to(device)
        tgt = batch.y.view(b, n, 2)
        pr = pred.view(b, n, 2)
        pred_re, pred_im = pr[..., 0], pr[..., 1]
        true_re, true_im = tgt[..., 0], tgt[..., 1]
        pred_mag = torch.sqrt(pred_re * pred_re + pred_im * pred_im + 1e-12)
        true_mag = torch.sqrt(true_re * true_re + true_im * true_im + 1e-12)
        pred_ang = torch.atan2(pred_im, pred_re)
        true_ang = torch.atan2(true_im, true_re)
        ang_err = _angle_diff_deg(pred_ang, true_ang)
        vmag_err = (pred_mag - true_mag).abs().cpu().numpy().reshape(-1)
        ang_abs = ang_err.abs().cpu().numpy().reshape(-1)
        vmag_abs_list.append(vmag_err)
        ang_abs_list.append(ang_abs)
        d_tile = np.tile(dist_per_node.astype(np.float64), b)
        dist_tile_list.append(d_tile)

    vmag_abs = np.concatenate(vmag_abs_list)
    ang_abs = np.concatenate(ang_abs_list)
    dist_all = np.concatenate(dist_tile_list)
    valid = np.isfinite(dist_all) & np.isfinite(vmag_abs)
    vmag_f = vmag_abs[valid]
    dist_f = dist_all[valid]

    p = np.percentile(vmag_f, [50, 90, 95, 99]).tolist()
    pmax = float(np.max(vmag_f))
    hist_counts, hist_edges = np.histogram(vmag_f, bins=64, range=(0.0, max(float(np.percentile(vmag_f, 99.9)), 1e-6)))

    n_bins = 10
    if dist_f.size > n_bins * 5:
        qs = np.quantile(dist_f, np.linspace(0, 1, n_bins + 1))
        qs[0] = dist_f.min()
        qs[-1] = dist_f.max()
        bin_means: list[float] = []
        bin_counts: list[int] = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (dist_f >= lo) & (dist_f <= hi) if hi == qs[-1] else (dist_f >= lo) & (dist_f < hi)
            bin_counts.append(int(np.sum(m)))
            bin_means.append(float(np.mean(vmag_f[m])) if np.any(m) else float("nan"))
        bin_edges = [float(x) for x in qs.tolist()]
    else:
        bin_means, bin_counts, bin_edges = [], [], []

    if corr_subsample > 0 and vmag_f.size > corr_subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(vmag_f.size, size=corr_subsample, replace=False)
        rho = _spearman_rho(dist_f[idx], vmag_f[idx])
    else:
        rho = _spearman_rho(dist_f, vmag_f)

    q_dist = [float(np.percentile(dist_f, q)) for q in (25, 50, 75)]
    near = dist_f <= q_dist[0]
    far = dist_f >= q_dist[2]
    mae_near = float(np.mean(vmag_f[near])) if np.any(near) else float("nan")
    mae_far = float(np.mean(vmag_f[far])) if np.any(far) else float("nan")

    return {
        "n_points_valid_dist": int(vmag_f.size),
        "n_points_missing_dist": int(np.sum(~np.isfinite(dist_all))),
        "mae_vmag_pu": float(np.mean(vmag_abs)),
        "rmse_vmag_pu": float(np.sqrt(np.mean(vmag_abs**2))),
        "mae_angle_deg": float(np.mean(ang_abs)),
        "rmse_angle_deg": float(np.sqrt(np.mean(ang_abs**2))),
        "vmag_abs_p50_p90_p95_p99": p,
        "vmag_abs_max": pmax,
        "hist_vmag_abs_counts": hist_counts.astype(int).tolist(),
        "hist_vmag_abs_edges": hist_edges.astype(float).tolist(),
        "dist_stratum_quantile_edges": bin_edges,
        "dist_stratum_mean_vmag_abs": bin_means,
        "dist_stratum_counts": bin_counts,
        "spearman_vmag_abs_vs_min_regulator_distance": rho,
        "mae_vmag_far_vs_near": {"near_q25_mae": mae_near, "far_q75_mae": mae_far, "dist_q25_q50_q75": q_dist},
    }


def _training_bundle_peer(ood_root: Path) -> Path:
    """Sibling ``loadtype_8500_dailyagg`` next to ``..._ood_stress`` (static topology)."""
    p = ood_root.parent / "loadtype_8500_dailyagg"
    return p if p.is_dir() else Path()


_NODES_REQUIRED = frozenset(
    {"sample_id", "node", "node_idx", "vmag_pu", "vang_deg", "p_load_kw", "q_load_kvar"},
)


def _nodes_csv_usable(path: Path) -> bool:
    import pandas as pd

    try:
        cols = set(pd.read_csv(path, nrows=0).columns.astype(str).str.strip())
    except Exception:
        return False
    return _NODES_REQUIRED.issubset(cols)


def _resolve_nodes_csv(ood_root: Path, nodes_arg: str) -> Path:
    """Resolve OOD node table: hetero tap-only, or pre-merge MV/GNN CSVs under ``ood_root``."""
    rel = Path(nodes_arg)
    if rel.is_file() and _nodes_csv_usable(rel):
        return rel.resolve()
    name = rel.name
    candidates: list[Path] = [
        ood_root / rel,
        ood_root / "Heterogenous GNN dataset" / "nodes" / name,
        ood_root / name,
        ood_root / "gnn_node_features_and_targets_mv_only.csv",
        ood_root / "gnn_node_features_and_targets.csv",
    ]
    parent = ood_root.parent
    if parent.is_dir():
        candidates.extend(
            [
                parent / "gnn_node_features_and_targets_mv_only.csv",
                parent / "gnn_node_features_and_targets.csv",
                parent / "Heterogenous GNN dataset" / "nodes" / name,
            ]
        )
    tried_lines: list[str] = []
    for c in candidates:
        if not c.is_file():
            tried_lines.append(f"{c}  (missing)")
            continue
        if not _nodes_csv_usable(c):
            tried_lines.append(f"{c}  (wrong columns; need {_NODES_REQUIRED})")
            continue
        print(f"Using nodes CSV: {c}", flush=True)
        if c.name.startswith("gnn_node_features"):
            print(
                "  (MV / full GNN node CSV — OK if node count matches checkpoints; "
                "prefer hetero tap-only after merge for exact training layout.)",
                flush=True,
            )
        return c.resolve()
    tried = "\n  ".join(tried_lines)
    raise FileNotFoundError(
        "Could not find a usable OOD nodes CSV (need sample_id, node, node_idx, vmag_pu, vang_deg, p_load_kw, q_load_kvar). Tried:\n  "
        + tried
        + "\n\nEither:\n"
        "  A) Run aggregate_mv_node_dataset_8500.py → build_hetero_mv_node_type_datasets.py → "
        "merge_load_transformer_reg_tap_only.py --dataset-root \"<OOD_ROOT>\" (see generate_ood_daily_aggregate_dataset_8500.py), or\n"
        "  B) Ensure OOD root contains gnn_node_features_and_targets_mv_only.csv (after step 1 of that pipeline), or\n"
        "  C) Pass --nodes_csv with an absolute path to a compatible CSV.\n"
    )


def _resolve_edges_csv(ood_root: Path, edges_arg: str) -> tuple[Path, str]:
    rel = Path(edges_arg)
    if rel.is_file():
        return rel.resolve(), "explicit"
    name = rel.name
    candidates: list[tuple[Path, str]] = [
        (ood_root / rel, "ood_data_root"),
        (ood_root / "Heterogenous GNN dataset" / "edges" / name, "ood_data_root"),
        (ood_root / name, "ood_data_root"),
    ]
    peer = _training_bundle_peer(ood_root)
    if peer.is_dir():
        candidates.append((peer / rel, "training_bundle_peer (static topology)"))
        candidates.append((peer / "Heterogenous GNN dataset" / "edges" / name, "training_bundle_peer (static topology)"))
    for c, tag in candidates:
        if c.is_file():
            return c.resolve(), tag
    tried = "\n  ".join(str(c[0]) for c in candidates)
    raise FileNotFoundError(
        "Could not find compacted line-edge CSV. Tried:\n  "
        + tried
        + "\nPass --edge_catalog_csv as an absolute path, or sync edges from training (see generate_ood --sync-static-from)."
    )


def _check_nodes_match_ood_meta(nodes_path: Path, ood_root: Path) -> None:
    """
    Guardrail: if OOD meta exists, ensure node sample_id universe matches it.
    This prevents accidental evaluation on parent/training node CSVs.
    """
    import pandas as pd

    meta_path = ood_root / "gnn_sample_meta.csv"
    if not meta_path.is_file():
        return
    meta = pd.read_csv(meta_path, usecols=["sample_id"])
    meta_ids = set(meta["sample_id"].astype(str).tolist())
    if not meta_ids:
        return
    node_ids = set(pd.read_csv(nodes_path, usecols=["sample_id"])["sample_id"].astype(str).unique().tolist())
    if node_ids != meta_ids:
        only_nodes = len(node_ids - meta_ids)
        only_meta = len(meta_ids - node_ids)
        raise ValueError(
            "Node CSV sample_id set does not match OOD gnn_sample_meta.csv.\n"
            f"  nodes_csv: {nodes_path}\n"
            f"  ood_meta:  {meta_path}\n"
            f"  unique sample_id counts -> nodes: {len(node_ids)}, meta: {len(meta_ids)}\n"
            f"  ids only in nodes_csv: {only_nodes}\n"
            f"  ids only in ood_meta:  {only_meta}\n"
            "Use a nodes CSV generated inside --ood_data_root (e.g. OOD "
            "gnn_node_features_and_targets_mv_only.csv or merged hetero tap-only file)."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD eval for GNN-only vs GINE+MLP complex voltage checkpoints.")
    p.add_argument(
        "--ood_data_root",
        type=str,
        default="datasets_gnn2/loadtype_8500_dailyagg/loadtype_8500_dailyagg_ood_stress",
    )
    p.add_argument("--nodes_csv", type=str, default="Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    p.add_argument(
        "--edge_catalog_csv",
        type=str,
        default="Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
    )
    p.add_argument(
        "--electrical_distance_csv",
        type=str,
        default="",
        help="CSV with 'node' + regulator distance columns (0 = not on path). "
        "Default: <ood_data_root>/Heterogenous GNN dataset/load_electrical_distance_to_each_regulator.csv "
        "then repo 8500-node/...",
    )
    p.add_argument(
        "--eval_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Folders containing *.pt checkpoints (each run's x_mean.pt etc. live beside the ckpt).",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--corr_subsample", type=int, default=500_000, help="Max pairs for Spearman (0 = no subsampling).")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--plot_dir", type=str, default="", help="If set, save simple PNG histograms (requires matplotlib).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent
    ood_root = Path(args.ood_data_root)
    if not ood_root.is_absolute():
        ood_root = (repo / ood_root).resolve()
    if not ood_root.is_dir():
        raise FileNotFoundError(f"OOD data root is not a directory: {ood_root}")

    nodes_path = _resolve_nodes_csv(ood_root, str(args.nodes_csv))
    try:
        if not str(nodes_path.resolve()).startswith(str(ood_root.resolve())):
            print(
                "WARNING: node features CSV is outside --ood_data_root. "
                "If that file is the in-distribution (training) bundle, metrics are not true OOD. "
                "Place gnn_node_features_and_targets*.csv inside the OOD stress folder, or pass an "
                "absolute --nodes_csv pointing at the OOD-generated node table.",
                flush=True,
            )
    except Exception:
        pass
    edges_path, edges_source = _resolve_edges_csv(ood_root, str(args.edge_catalog_csv))
    if edges_source.startswith("training_bundle"):
        print(
            f"Note: using edge catalog from training bundle (static line topology): {edges_path}",
            flush=True,
        )

    dist_arg = str(args.electrical_distance_csv).strip()
    if dist_arg:
        dist_csv = Path(dist_arg)
        if not dist_csv.is_absolute():
            cand = ood_root / dist_arg
            dist_csv = cand if cand.is_file() else (repo / dist_arg).resolve()
    else:
        dist_csv = ood_root / "Heterogenous GNN dataset" / "load_electrical_distance_to_each_regulator.csv"
        if not dist_csv.is_file():
            dist_csv = repo / "8500-node" / "load_electrical_distance_to_each_regulator.csv"
    if not dist_csv.is_file():
        raise FileNotFoundError(
            f"Electrical distance CSV not found. Tried OOD hetero copy and 8500-node. Pass --electrical_distance_csv."
        )

    print(f"OOD root: {ood_root}", flush=True)
    print(f"Nodes:    {nodes_path}", flush=True)
    print(f"Edges:    {edges_path}", flush=True)
    print(f"Dist CSV: {dist_csv}", flush=True)
    _check_nodes_match_ood_meta(nodes_path, ood_root)

    x, _y_unused, sample_ids, node_order, node_to_local = _load_nodes_pq_target(nodes_path)
    edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
    n_nodes = int(x.shape[1])
    n_edges = int(edge_index.shape[1])

    y_ri = _build_complex_targets(nodes_path, sample_ids, node_to_local)

    dist_per_node = _min_positive_regulator_distance(dist_csv, list(node_order))
    n_miss = int(np.sum(~np.isfinite(dist_per_node)))
    if n_miss:
        print(f"Warning: {n_miss} nodes missing from distance CSV; excluded from distance-stratified stats.", flush=True)

    ds = GraphVoltageDataset(x, y_ri, edge_index, edge_attr)
    dl = PyGDataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} OOD samples={len(ds)} n_nodes={n_nodes}", flush=True)

    ckpt_files: list[Path] = []
    eval_dirs_resolved: list[Path] = []
    for d in args.eval_dirs:
        dpath = Path(d)
        if not dpath.is_absolute():
            dpath = (repo / dpath).resolve()
        eval_dirs_resolved.append(dpath)
        if not dpath.is_dir():
            raise FileNotFoundError(dpath)
        for f in sorted(dpath.glob("*.pt")):
            if f.name.endswith("_mean.pt") or f.name.endswith("_std.pt"):
                continue
            ckpt_files.append(f)

    if not ckpt_files:
        raise RuntimeError("No .pt checkpoints found under --eval_dirs (excluding *_mean.pt / *_std.pt).")

    results: dict[str, Any] = {"ood_data_root": str(ood_root), "checkpoints": {}}

    for ckpt_path in ckpt_files:
        print(f"\n--- {ckpt_path.name} ({ckpt_path.parent.name}) ---", flush=True)
        ckpt = _load_ckpt_dict(ckpt_path)
        x_mean, x_std, y_mean, y_std = _load_norm_tensors(ckpt, ckpt_path)
        model, tag = _classify_and_build_model(ckpt_path, ckpt, n_nodes, n_edges)
        model = model.to(device)
        metrics = _run_model_collect_errors(
            model,
            dl,
            device,
            x_mean,
            x_std,
            y_mean,
            y_std,
            dist_per_node,
            corr_subsample=int(args.corr_subsample),
        )
        key = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        results["checkpoints"][key] = {
            "path": str(ckpt_path.resolve()),
            "family": tag,
            "metrics": metrics,
        }
        m = metrics
        print(
            f"  {tag}: MAE|V|={m['mae_vmag_pu']:.6f} RMSE|V|={m['rmse_vmag_pu']:.6f} "
            f"MAEang={m['mae_angle_deg']:.4f}  P99|err|={m['vmag_abs_p50_p90_p95_p99'][3]:.6f}  max={m['vmag_abs_max']:.6f}",
            flush=True,
        )
        print(
            f"  Spearman(|err|, d_reg)={m['spearman_vmag_abs_vs_min_regulator_distance']:.4f}  "
            f"MAE near(q25)/far(q75)={m['mae_vmag_far_vs_near']['near_q25_mae']:.6f} / "
            f"{m['mae_vmag_far_vs_near']['far_q75_mae']:.6f}",
            flush=True,
        )

        plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else None
        if plot_dir is not None:
            try:
                import matplotlib.pyplot as plt

                plot_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{ckpt_path.parent.name}__{ckpt_path.stem}"
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(
                    (np.array(m["hist_vmag_abs_edges"][:-1]) + np.array(m["hist_vmag_abs_edges"][1:])) / 2.0,
                    m["hist_vmag_abs_counts"],
                    width=np.diff(m["hist_vmag_abs_edges"]).mean(),
                    align="center",
                    edgecolor="none",
                )
                ax.set_xlabel("|V| absolute error (pu)")
                ax.set_ylabel("count")
                ax.set_title(f"{stem} OOD")
                fig.tight_layout()
                fig.savefig(plot_dir / f"{stem}_vmag_err_hist.png", dpi=150)
                plt.close(fig)

                if m["dist_stratum_mean_vmag_abs"]:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(range(1, len(m["dist_stratum_mean_vmag_abs"]) + 1), m["dist_stratum_mean_vmag_abs"], "o-")
                    ax2.set_xlabel("distance decile (1=near … 10=far)")
                    ax2.set_ylabel("mean |V| err (pu)")
                    ax2.set_title(f"{stem} error vs electrical distance")
                    fig2.tight_layout()
                    fig2.savefig(plot_dir / f"{stem}_err_vs_dist_decile.png", dpi=150)
                    plt.close(fig2)
            except Exception as ex:
                print(f"  (plot skipped: {ex})", flush=True)

    out_path = Path(args.out_json) if args.out_json else None
    if out_path is not None:
        if not out_path.is_absolute():
            out_path = (repo / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
