"""
Daily OpenDSS vs load-only homo GNN with local+global residual readout.

Compares checkpoint from train_homo_gine_global_localres_pq_loadonly.py and
plots, per monitored node:
  - V_pred (final)
  - V_local
  - DeltaV_global
  - OpenDSS voltage (reference)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_daily_aggregate_dataset_8500 as rd8500
from compare_mv_daily_timing import print_mv_daily_timing_summary, resolve_inference_device, sync_inference_device
from compare_opendss_snapshot_helpers import force_snapshot_mode_for_compare_timing, reassert_snapshot_before_each_solve
from train_homo_gine_global_localres_pq_loadonly import (
    HomoGCNGlobalLocalRes,
    HomoGINEGlobalLocalRes,
    _load_compacted_edges,
    _load_nodes_pq_target,
)


def _load_mv_sx_mapping(path: Path) -> list[dict[str, str]]:
    import csv

    rules: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mv = (row.get("mv_node") or "").strip()
            lv1 = (row.get("lv_x_node_1") or "").strip()
            lv2 = (row.get("lv_x_node_2") or "").strip()
            sx1 = (row.get("sx_node_1") or "").strip()
            sx2 = (row.get("sx_node_2") or "").strip()
            if not mv or not lv1 or not lv2:
                continue
            la, lb = (sx1, sx2) if sx1 and sx2 else (lv1, lv2)
            rules.append({"mv_key": mv.lower(), "load_a": la.lower(), "load_b": lb.lower()})
    return rules


def run_compare_homo_global_localres(
    *,
    checkpoint: Path,
    dataset_dir: Path,
    out_dir: Path,
    plot_nodes: list[str],
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    mv_sx_mapping: Path | None = None,
    nodes_csv: Path | None = None,
    edge_csv: Path | None = None,
    train_metrics_path: Path | None = None,
    feature_norm_path: Path | None = None,
    device: str | None = None,
    show_plots: bool = False,
) -> None:
    device = resolve_inference_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ds = Path(dataset_dir).resolve()
    nodes_path = Path(nodes_csv).resolve() if nodes_csv else (ds / "nodes" / "hetero_mv_nodes_load_transformer_reg_tap_only.csv")
    edges_path = Path(edge_csv).resolve() if edge_csv else (ds / "edges" / "hetero_mv_line_edges_load_only_compacted.csv")
    for p in (nodes_path, edges_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    meta_path = Path(train_metrics_path).resolve() if train_metrics_path else (ckpt.parent / "train_metrics_global_localres.json")
    norm_path = Path(feature_norm_path).resolve() if feature_norm_path else (ckpt.parent / "feature_norm_pq.pt")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing train_metrics_global_localres.json: {meta_path}")
    if not norm_path.is_file():
        raise FileNotFoundError(f"Missing feature_norm_pq.pt: {norm_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_kind = str(meta.get("model", "gine")).lower()
    hidden = int(meta["hidden"])
    n_layers = int(meta["layers"])
    node_out_dim = int(meta.get("node_out_dim", 2))
    node_emb_dim = int(meta.get("node_emb_dim", 0))
    edge_emb_dim = int(meta.get("edge_emb_dim", 0))
    dropout = float(meta.get("dropout", 0.15))

    # Build canonical node ordering from first sample in nodes CSV.
    x_tmp, _y_tmp, _sids, node_order, node_to_local = _load_nodes_pq_target(nodes_path)
    n_nodes = int(x_tmp.shape[1])
    del x_tmp, _y_tmp
    edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    norm_pack = torch.load(norm_path, map_location="cpu", weights_only=False)
    feat_mean = torch.as_tensor(norm_pack["mean"], dtype=torch.float32).view(1, 1, -1)
    feat_std = torch.as_tensor(norm_pack["std"], dtype=torch.float32).clamp_min(1e-8).view(1, 1, -1)

    if model_kind == "gine":
        model = HomoGINEGlobalLocalRes(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges_directed=int(edge_index.shape[1]),
            hidden=hidden,
            n_layers=n_layers,
            node_out_dim=node_out_dim,
            dropout=dropout,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
        ).to(device)
    else:
        model = HomoGCNGlobalLocalRes(
            in_dim=2,
            n_nodes=n_nodes,
            hidden=hidden,
            n_layers=n_layers,
            node_out_dim=node_out_dim,
            dropout=dropout,
            node_emb_dim=node_emb_dim,
        ).to(device)

    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # OpenDSS setup.
    rd8500._compile_8500_daily_setup()
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    mL = rd8500._daily_profile_5min(npts=npts)

    repo_root = Path(__file__).resolve().parent
    mpath = Path(mv_sx_mapping).resolve() if mv_sx_mapping else (repo_root / "8500-node" / "mv_x_sx_node_mapping_8500.csv")
    mv_sx_rules = _load_mv_sx_mapping(mpath) if mpath.is_file() else []

    node_order_l = [n.strip().lower() for n in node_order]
    node_to_idx = {n: i for i, n in enumerate(node_order_l)}
    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)

    v_dss = np.full((npts, n_nodes), np.nan, dtype=np.float32)
    v_pred = np.full((npts, n_nodes), np.nan, dtype=np.float32)
    v_local = np.full((npts, n_nodes), np.nan, dtype=np.float32)
    v_delta = np.full((npts, n_nodes), np.nan, dtype=np.float32)

    open_apply_s = open_reassert_s = open_solve_s = open_get_s = feat_build_s = gnn_s = 0.0
    nonconv = 0

    for i in range(npts):
        hr = int(i // 12)
        sec = int((i % 12) * (step_min * 60))
        m_t = float(mL[i])
        kw_set = base_kw * m_t
        kvar_set = base_kvar * m_t

        t0 = time.perf_counter()
        dss.Text.Command(f"set hour={hr} sec={sec}")
        for j, name in enumerate(base_names):
            dss.Loads.Name(name)
            dss.Loads.kW(float(kw_set[j]))
            dss.Loads.kvar(float(kvar_set[j]))
        open_apply_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        reassert_snapshot_before_each_solve()
        open_reassert_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        dss.Solution.Solve()
        open_solve_s += time.perf_counter() - t0
        if not dss.Solution.Converged():
            nonconv += 1
            continue

        # DSS truth
        t0 = time.perf_counter()
        all_nodes, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_order_l)
        v_dss[i, :] = np.asarray(all_nodes, dtype=np.float32)
        open_get_s += time.perf_counter() - t0

        # Build P/Q features for load-only MV nodes.
        t0 = time.perf_counter()
        busph_p: dict[tuple[str, int], float] = {}
        busph_q: dict[tuple[str, int], float] = {}
        for j, name in enumerate(base_names):
            for (bus, ph, w) in load_to_busph[name]:
                bk = str(bus).strip().lower()
                busph_p[(bk, int(ph))] = busph_p.get((bk, int(ph)), 0.0) + float(kw_set[j]) * float(w)
                busph_q[(bk, int(ph))] = busph_q.get((bk, int(ph)), 0.0) + float(kvar_set[j]) * float(w)

        node_p: dict[str, float] = {}
        node_q: dict[str, float] = {}
        for (bus, ph), val in busph_p.items():
            node_p[f"{bus}.{int(ph)}"] = float(val)
        for (bus, ph), val in busph_q.items():
            node_q[f"{bus}.{int(ph)}"] = float(val)

        # Optional MV <- (sx/lv pair) rollup.
        if mv_sx_rules:
            for rec in mv_sx_rules:
                mvk = rec["mv_key"]
                pa = node_p.get(rec["load_a"], 0.0)
                pb = node_p.get(rec["load_b"], 0.0)
                qa = node_q.get(rec["load_a"], 0.0)
                qb = node_q.get(rec["load_b"], 0.0)
                node_p[mvk] = pa + pb
                node_q[mvk] = qa + qb

        x = np.zeros((n_nodes, 2), dtype=np.float32)
        for ni, nk in enumerate(node_order_l):
            x[ni, 0] = float(node_p.get(nk, 0.0))
            x[ni, 1] = float(node_q.get(nk, 0.0))
        feat_build_s += time.perf_counter() - t0

        # GNN inference.
        t0 = time.perf_counter()
        xb = torch.from_numpy(x).view(1, n_nodes, 2)
        xb = ((xb - feat_mean) / feat_std).squeeze(0).to(device)
        data = Data(x=xb, edge_index=edge_index, edge_attr=edge_attr)
        with torch.no_grad():
            pred_t, local_t, delta_t = model.forward_components(data)
        v_pred[i, :] = pred_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        v_local[i, :] = local_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        v_delta[i, :] = delta_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        gnn_s += time.perf_counter() - t0

    # Metrics and outputs.
    m = np.isfinite(v_dss) & np.isfinite(v_pred)
    mae = float(np.nanmean(np.abs(v_pred[m] - v_dss[m]))) if np.any(m) else float("nan")
    rmse = float(np.sqrt(np.nanmean((v_pred[m] - v_dss[m]) ** 2))) if np.any(m) else float("nan")

    print(f"[compare_homo_mv_daily_global_localres] model={model_kind} hidden={hidden} L={n_layers} N={n_nodes} device={device}")
    print(f"[compare_homo_mv_daily_global_localres] daily MAE={mae:.6f} RMSE={rmse:.6f} nonconv={nonconv}/{npts}")
    print_mv_daily_timing_summary(
        title="Daily Timing Summary (homo global+localres vs OpenDSS)",
        n_ok=(npts - nonconv),
        npts=npts,
        n_nonconv=nonconv,
        open_apply_s_total=open_apply_s,
        open_reassert_s_total=open_reassert_s,
        open_solve_only_s_total=open_solve_s,
        open_get_s_total=open_get_s,
        feature_build_s_total=feat_build_s,
        gnn_forward_only_s_total=gnn_s,
        gnn_bucket_s_total=gnn_s,
        device=str(device),
    )

    out_metrics = {
        "mae_pu": mae,
        "rmse_pu": rmse,
        "npts": npts,
        "nonconv": nonconv,
        "timing": {
            "n_ok": int(npts - nonconv),
            "open_apply_s_total": float(open_apply_s),
            "open_reassert_s_total": float(open_reassert_s),
            "open_solve_only_s_total": float(open_solve_s),
            "open_get_s_total": float(open_get_s),
            "feature_build_s_total": float(feat_build_s),
            "gnn_forward_only_s_total": float(gnn_s),
            "gnn_bucket_s_total": float(gnn_s),
        },
        "checkpoint": str(ckpt),
        "train_metrics": str(meta_path),
        "feature_norm": str(norm_path),
    }
    (out_dir / "daily_metrics_global_localres.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    plots_dir = out_dir / "monitoring_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for raw in plot_nodes:
        nk = str(raw).strip().lower()
        if nk not in node_to_idx:
            print(f"[compare_homo_mv_daily_global_localres] skip plot node not in load set: {raw}")
            continue
        j = node_to_idx[nk]
        fig, ax1 = plt.subplots(figsize=(10, 4))
        l1, = ax1.plot(t_hours, v_dss[:, j], lw=2.0, label="OpenDSS V")
        l2, = ax1.plot(t_hours, v_pred[:, j], lw=1.8, label="V_pred")
        l3, = ax1.plot(t_hours, v_local[:, j], lw=1.2, label="V_local")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Voltage (p.u.)")
        ax1.set_ylim(ymin, ymax)
        ax1.grid(True, alpha=0.3)

        # ΔV is typically around 0 and gets clipped on the voltage axis; show it on a second axis.
        ax2 = ax1.twinx()
        l4, = ax2.plot(t_hours, v_delta[:, j], lw=1.4, ls="--", color="tab:red", label="ΔV_global")
        ax2.set_ylabel("ΔV_global (p.u.)")
        max_abs_dv = float(np.nanmax(np.abs(v_delta[:, j]))) if np.isfinite(v_delta[:, j]).any() else 0.01
        dv_lim = max(0.01, max_abs_dv * 1.15)
        ax2.set_ylim(-dv_lim, dv_lim)

        ax1.set_title(f"{nk} | V, V_local, ΔV_global")
        lines = [l1, l2, l3, l4]
        ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
        out_png = plots_dir / f"{nk.replace('.', '_')}_v_local_delta.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Daily compare for homo global+local residual checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True, help=".../Heterogenous GNN dataset")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--plot-node", action="append", default=[])
    p.add_argument("--npts", type=int, default=288)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--ymin", type=float, default=0.85)
    p.add_argument("--ymax", type=float, default=1.10)
    p.add_argument("--mv-sx-mapping", type=Path, default=None)
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    run_compare_homo_global_localres(
        checkpoint=args.checkpoint.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        plot_nodes=list(args.plot_node),
        npts=args.npts,
        step_min=args.step_min,
        ymin=args.ymin,
        ymax=args.ymax,
        mv_sx_mapping=args.mv_sx_mapping.resolve() if args.mv_sx_mapping else None,
        device=args.device,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()

