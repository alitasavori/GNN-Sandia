"""
Daily OpenDSS vs load-only homo GNN with local+global residual readout.

Compares checkpoint from train_homo_gine_global_localres_pq_loadonly.py (or aux
checkpoint with aux_* weights stripped) and plots, per ``--plot-node``:
  V_pred, V_local, ΔV_global, OpenDSS V.

For ``voltage_target_mode: complex_ri`` (train_metrics JSON), the model outputs
V_re,V_im; compare plots only OpenDSS |V| vs predicted |V| (no local/ΔV split).

Optional ``--worst-k K``: rank all load-type nodes by time-mean |V_pred−V_dss|,
save ``daily_per_node_mae.csv``, print top-K, and save figures under
``monitoring_plots/worst_by_mae/``. JSON adds ``mean_mae_pu_per_node`` and
``worst_nodes_by_mae``.
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
from train_homo_gine_global_localres_pq_aux import HomoGCNGlobalLocalAux, HomoGINEGlobalLocalAux


def _infer_dropout_from_meta(meta: dict) -> float:
    if "dropout" in meta:
        return float(meta["dropout"])
    return max(
        float(meta.get("dropout_trunk", 0.0)),
        float(meta.get("dropout_global", 0.0)),
    )


def _state_dict_voltage_only(state_dict: dict) -> tuple[dict, int]:
    """Strip aux-head weights from train_homo_gine_global_localres_pq_aux checkpoints."""
    n_strip = 0
    out: dict = {}
    for k, v in state_dict.items():
        if k.startswith("aux_"):
            n_strip += 1
            continue
        out[k] = v
    return out, n_strip


def _per_node_mae_rmse(v_pred: np.ndarray, v_dss: np.ndarray, node_order_l: list[str]) -> pd.DataFrame:
    """One row per GNN node: MAE/RMSE over timesteps where both pred and DSS are finite."""
    rows: list[dict[str, object]] = []
    _, n_nodes = v_pred.shape
    for j in range(n_nodes):
        a = v_pred[:, j]
        b = v_dss[:, j]
        mask = np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n == 0:
            rows.append({"node": node_order_l[j], "mae_pu": np.nan, "rmse_pu": np.nan, "n_valid": 0})
            continue
        err = a[mask] - b[mask]
        rows.append(
            {
                "node": node_order_l[j],
                "mae_pu": float(np.mean(np.abs(err))),
                "rmse_pu": float(np.sqrt(np.mean(err**2))),
                "n_valid": n,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("mae_pu", ascending=False, na_position="last").reset_index(drop=True)


def _plot_one_node_global_localres(
    *,
    nk: str,
    j: int,
    t_hours: np.ndarray,
    v_dss: np.ndarray,
    v_pred: np.ndarray,
    v_local: np.ndarray,
    v_delta: np.ndarray,
    ymin: float,
    ymax: float,
    out_png: Path,
    show_plots: bool,
    title_extra: str = "",
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    l1, = ax1.plot(t_hours, v_dss[:, j], lw=2.0, label="OpenDSS V")
    l2, = ax1.plot(t_hours, v_pred[:, j], lw=1.8, label="V_pred")
    l3, = ax1.plot(t_hours, v_local[:, j], lw=1.2, label="V_local")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l4, = ax2.plot(t_hours, v_delta[:, j], lw=1.4, ls="--", color="tab:red", label="ΔV_global")
    ax2.set_ylabel("ΔV_global (p.u.)")
    max_abs_dv = float(np.nanmax(np.abs(v_delta[:, j]))) if np.isfinite(v_delta[:, j]).any() else 0.01
    dv_lim = max(0.01, max_abs_dv * 1.15)
    ax2.set_ylim(-dv_lim, dv_lim)

    ax1.set_title(f"{nk} | V, V_local, ΔV_global{title_extra}")
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_one_node_complex_ri_magnitude(
    *,
    nk: str,
    j: int,
    t_hours: np.ndarray,
    v_dss: np.ndarray,
    v_pred: np.ndarray,
    ymin: float,
    ymax: float,
    out_png: Path,
    show_plots: bool,
    title_extra: str = "",
) -> None:
    """complex_ri checkpoints predict V_re,V_im; compare uses |V| only — no local/ΔV split."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_hours, v_dss[:, j], lw=2.0, label="OpenDSS |V|")
    ax.plot(t_hours, v_pred[:, j], lw=1.8, label="GNN |V| from (V_re, V_im)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage magnitude (p.u.)")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(f"{nk} | complex_ri magnitude{title_extra}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


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
    daily_profile_csv: str | Path | None = None,
    mv_sx_mapping: Path | None = None,
    nodes_csv: Path | None = None,
    edge_csv: Path | None = None,
    train_metrics_path: Path | None = None,
    feature_norm_path: Path | None = None,
    device: str | None = None,
    show_plots: bool = False,
    worst_k: int = 0,
    save_per_node_mae_csv: bool = False,
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

    if train_metrics_path is not None:
        meta_path = Path(train_metrics_path).resolve()
    else:
        p0 = (ckpt.parent / "train_metrics_global_localres.json").resolve()
        p1 = (ckpt.parent / "train_metrics_global_localres_aux.json").resolve()
        # Prefer aux JSON when present: non-aux JSON is easy to leave stale beside *_aux_* checkpoints
        # and would force vmag mode + wrong model class (local/ΔV plots break or look zero).
        if p1.is_file():
            meta_path = p1
            print(f"[compare_homo_mv_daily_global_localres] using train metrics: {meta_path}", flush=True)
            if p0.is_file():
                print(
                    f"[compare_homo_mv_daily_global_localres] (ignored non-aux metrics: {p0.name}; remove it or pass --train-metrics explicitly)",
                    flush=True,
                )
        elif p0.is_file():
            meta_path = p0
        else:
            meta_path = p0
    norm_path = Path(feature_norm_path).resolve() if feature_norm_path else (ckpt.parent / "feature_norm_pq.pt")
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing train metrics JSON: {meta_path}. "
            "Pass --train-metrics or place train_metrics_global_localres.json or train_metrics_global_localres_aux.json next to the checkpoint."
        )
    if not norm_path.is_file():
        raise FileNotFoundError(f"Missing feature_norm_pq.pt: {norm_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_kind = str(meta.get("model", "gine")).lower()
    voltage_target_mode = str(meta.get("voltage_target_mode", "vmag")).lower()
    hidden = int(meta["hidden"])
    n_layers = int(meta["layers"])
    node_out_dim = int(meta.get("node_out_dim", 2))
    node_emb_dim = int(meta.get("node_emb_dim", 0))
    edge_emb_dim = int(meta.get("edge_emb_dim", 0))
    dropout = _infer_dropout_from_meta(meta)

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

    if voltage_target_mode == "complex_ri":
        aux_targets = meta.get("aux_targets", {})
        reg_nclasses = [int(d.get("n_classes", 1)) for d in aux_targets.get("reg", [])]
        cap_nclasses = [int(d.get("n_classes", 1)) for d in aux_targets.get("cap", [])]
        if not reg_nclasses:
            reg_nclasses = [1] * 12
        if not cap_nclasses:
            cap_nclasses = [1] * 10
        if model_kind == "gine":
            model = HomoGINEGlobalLocalAux(
                in_dim=2,
                n_nodes=n_nodes,
                num_edges=int(edge_index.shape[1]),
                hidden=hidden,
                n_layers=n_layers,
                node_out_dim=node_out_dim,
                voltage_out_components=2,
                dropout_trunk=float(meta.get("dropout_trunk", 0.0)),
                dropout_global=float(meta.get("dropout_global", 0.0)),
                dropout_aux=float(meta.get("dropout_aux", 0.0)),
                node_emb_dim=node_emb_dim,
                edge_emb_dim=edge_emb_dim,
                reg_nclasses=reg_nclasses,
                cap_nclasses=cap_nclasses,
            ).to(device)
        else:
            model = HomoGCNGlobalLocalAux(
                in_dim=2,
                n_nodes=n_nodes,
                hidden=hidden,
                n_layers=n_layers,
                node_out_dim=node_out_dim,
                voltage_out_components=2,
                dropout_trunk=float(meta.get("dropout_trunk", 0.0)),
                dropout_global=float(meta.get("dropout_global", 0.0)),
                dropout_aux=float(meta.get("dropout_aux", 0.0)),
                node_emb_dim=node_emb_dim,
                reg_nclasses=reg_nclasses,
                cap_nclasses=cap_nclasses,
            ).to(device)
    else:
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
    n_aux = 0
    if voltage_target_mode != "complex_ri":
        state_dict, n_aux = _state_dict_voltage_only(state_dict)
        if n_aux:
            print(f"[compare_homo_mv_daily_global_localres] stripped {n_aux} aux_* keys from checkpoint (voltage-only inference)", flush=True)
    missing_unexpected = model.load_state_dict(state_dict, strict=False)
    if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
        print(
            "[compare_homo_mv_daily_global_localres] load_state_dict(strict=False): "
            f"missing={missing_unexpected.missing_keys} unexpected={missing_unexpected.unexpected_keys}",
            flush=True,
        )
    model.eval()

    # OpenDSS setup.
    rd8500._compile_8500_daily_setup()
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    prof_path = rd8500._resolve_daily_profile_csv(daily_profile_csv)
    print(f"[compare_homo_mv_daily_global_localres] daily profile: {prof_path}", flush=True)
    mL = rd8500._daily_profile_5min(npts=npts, profile_csv=daily_profile_csv)

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
            if voltage_target_mode == "complex_ri":
                pred_t = model(data)  # [1, N, 2]
                pr = pred_t[..., 0]
                pi = pred_t[..., 1]
                vmag_t = torch.sqrt(pr * pr + pi * pi + 1e-12)
                v_pred_i = vmag_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                v_pred[i, :] = v_pred_i
                # No local/ΔV decomposition for complex_ri — leave unset (NaN).
            else:
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

    df_node = _per_node_mae_rmse(v_pred, v_dss, node_order_l)
    if save_per_node_mae_csv or worst_k > 0:
        csv_path = out_dir / "daily_per_node_mae.csv"
        df_node.to_csv(csv_path, index=False)
        print(f"[compare_homo_mv_daily_global_localres] wrote per-node MAE/RMSE: {csv_path}", flush=True)

    worst_entries: list[dict[str, object]] = []
    if worst_k > 0:
        top = df_node.head(int(worst_k))
        for _, row in top.iterrows():
            worst_entries.append(
                {
                    "node": str(row["node"]),
                    "mae_pu": float(row["mae_pu"]) if pd.notna(row["mae_pu"]) else None,
                    "rmse_pu": float(row["rmse_pu"]) if pd.notna(row["rmse_pu"]) else None,
                    "n_valid": int(row["n_valid"]),
                }
            )
        print(f"[compare_homo_mv_daily_global_localres] worst {worst_k} nodes by MAE (pu):", flush=True)
        for w in worst_entries:
            print(f"  {w['node']}: MAE={w['mae_pu']}", flush=True)

    mean_mae_nodes = float(np.nanmean(df_node["mae_pu"].to_numpy(dtype=np.float64))) if len(df_node) else float("nan")

    out_metrics = {
        "voltage_target_mode": voltage_target_mode,
        "mae_pu": mae,
        "rmse_pu": rmse,
        "mean_mae_pu_per_node": mean_mae_nodes,
        "n_nodes": int(n_nodes),
        "npts": npts,
        "nonconv": nonconv,
        "daily_profile_csv": str(prof_path),
        "worst_k": int(worst_k),
        "worst_nodes_by_mae": worst_entries,
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
    worst_dir = plots_dir / "worst_by_mae"
    if worst_k > 0:
        worst_dir.mkdir(parents=True, exist_ok=True)

    for raw in plot_nodes:
        nk = str(raw).strip().lower()
        if nk not in node_to_idx:
            print(f"[compare_homo_mv_daily_global_localres] skip plot node not in load set: {raw}")
            continue
        j = node_to_idx[nk]
        sub_m = df_node.loc[df_node["node"] == nk, "mae_pu"]
        m_nk = float(sub_m.iloc[0]) if len(sub_m) else float("nan")
        title_ex = "" if not np.isfinite(m_nk) else f" | node MAE={m_nk:.5f} pu"
        if voltage_target_mode == "complex_ri":
            _plot_one_node_complex_ri_magnitude(
                nk=nk,
                j=j,
                t_hours=t_hours,
                v_dss=v_dss,
                v_pred=v_pred,
                ymin=ymin,
                ymax=ymax,
                out_png=plots_dir / f"{nk.replace('.', '_')}_complex_ri_vmag.png",
                show_plots=show_plots,
                title_extra=title_ex,
            )
        else:
            _plot_one_node_global_localres(
                nk=nk,
                j=j,
                t_hours=t_hours,
                v_dss=v_dss,
                v_pred=v_pred,
                v_local=v_local,
                v_delta=v_delta,
                ymin=ymin,
                ymax=ymax,
                out_png=plots_dir / f"{nk.replace('.', '_')}_v_local_delta.png",
                show_plots=show_plots,
                title_extra=title_ex,
            )

    for rank, w in enumerate(worst_entries, start=1):
        nk = str(w["node"])
        if nk not in node_to_idx:
            continue
        j = node_to_idx[nk]
        mae_w = w.get("mae_pu")
        title_ex = f" | rank #{rank} by MAE"
        if mae_w is not None and np.isfinite(mae_w):
            title_ex += f" (MAE={float(mae_w):.5f} pu)"
        if voltage_target_mode == "complex_ri":
            _plot_one_node_complex_ri_magnitude(
                nk=nk,
                j=j,
                t_hours=t_hours,
                v_dss=v_dss,
                v_pred=v_pred,
                ymin=ymin,
                ymax=ymax,
                out_png=worst_dir / f"rank{rank:02d}_{nk.replace('.', '_')}_complex_ri_vmag.png",
                show_plots=show_plots,
                title_extra=title_ex,
            )
        else:
            _plot_one_node_global_localres(
                nk=nk,
                j=j,
                t_hours=t_hours,
                v_dss=v_dss,
                v_pred=v_pred,
                v_local=v_local,
                v_delta=v_delta,
                ymin=ymin,
                ymax=ymax,
                out_png=worst_dir / f"rank{rank:02d}_{nk.replace('.', '_')}_v_local_delta.png",
                show_plots=show_plots,
                title_extra=title_ex,
            )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Daily compare for homo global+local residual checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True, help=".../Heterogenous GNN dataset")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--plot-node", action="append", default=[])
    p.add_argument(
        "--daily-profile",
        type=str,
        default="5minDayShape.csv",
        metavar="CSV",
        help="Load-shape file under 8500-node/ (e.g. 5minDayShape.csv, 5minDayShape2.csv, 5minDayShape3.csv) or absolute path.",
    )
    p.add_argument("--npts", type=int, default=288)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--ymin", type=float, default=0.85)
    p.add_argument("--ymax", type=float, default=1.10)
    p.add_argument("--mv-sx-mapping", type=Path, default=None)
    p.add_argument(
        "--train-metrics",
        type=Path,
        default=None,
        help="train_metrics JSON (default: next to checkpoint). Use ..._aux.json for aux-trained checkpoints.",
    )
    p.add_argument("--feature-norm", type=Path, default=None, help="feature_norm_pq.pt path (default: next to checkpoint).")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--worst-k",
        type=int,
        default=0,
        metavar="K",
        help="Plot K highest-MAE nodes under monitoring_plots/worst_by_mae/ and list them in JSON.",
    )
    p.add_argument(
        "--save-per-node-mae-csv",
        action="store_true",
        help="Write daily_per_node_mae.csv (even if --worst-k is 0).",
    )
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
        daily_profile_csv=str(args.daily_profile),
        mv_sx_mapping=args.mv_sx_mapping.resolve() if args.mv_sx_mapping else None,
        train_metrics_path=args.train_metrics.resolve() if args.train_metrics else None,
        feature_norm_path=args.feature_norm.resolve() if args.feature_norm else None,
        device=args.device,
        show_plots=args.show_plots,
        worst_k=int(args.worst_k),
        save_per_node_mae_csv=bool(args.save_per_node_mae_csv),
    )


if __name__ == "__main__":
    main()

