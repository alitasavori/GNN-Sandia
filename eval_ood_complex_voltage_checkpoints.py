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
from train_gnn_only_compare_complex_voltage import GNNOnlyVoltageModel
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges, _load_nodes_pq_target


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _load_ckpt_dict(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_norm_tensors(ckpt: dict[str, Any], ckpt_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = ckpt_path.parent
    if all(k in ckpt for k in ("x_mean", "x_std", "y_mean", "y_std")):
        return ckpt["x_mean"], ckpt["x_std"], ckpt["y_mean"], ckpt["y_std"]
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
    raise ValueError(
        f"{ckpt_path}: unrecognized checkpoint (need model_type for GNN-only or hidden_mlp+encoder.* for GINE+MLP)"
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
        xn = (batch.x - x_mean.to(device)) / x_std.to(device)
        batch_n = Data(x=xn, y=batch.y, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
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
    nodes_path = Path(args.nodes_csv)
    if not nodes_path.is_absolute():
        nodes_path = (ood_root / nodes_path).resolve()
    edges_path = Path(args.edge_catalog_csv)
    if not edges_path.is_absolute():
        edges_path = (ood_root / edges_path).resolve()
    for pth in (nodes_path, edges_path):
        if not pth.is_file():
            raise FileNotFoundError(pth)

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
