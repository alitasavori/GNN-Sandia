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
from torch_geometric.data import Batch, Data

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
from train_gnn_only_compare_complex_voltage import GNNOnlyVoltageModel
from train_gine_plus_mlp_complex_voltage import GINEEncoder as GINEEncoderPlain, GINEPlusMLP
from train_gine_plus_mlp_global_local_complex_voltage import (
    GINEEncoder as GINEEncoderGlobalLocal,
    GINEPlusMLPGlobalLocal,
)
from train_gine_plus_mlp_aux_complex_voltage import GINEEncoder as GINEEncoderAux, GINEPlusMLPAux


def _min_positive_regulator_distance(dist_csv: Path, node_order: list[str]) -> np.ndarray:
    """Return per-node min positive distance across regulator columns; NaN if unavailable."""
    df = pd.read_csv(dist_csv)
    if "node" not in df.columns:
        raise ValueError(f"{dist_csv} must have a 'node' column")
    reg_cols = [c for c in df.columns if c != "node"]
    lk = {str(r["node"]).strip().lower(): r for _, r in df.iterrows()}
    out = np.full(len(node_order), np.nan, dtype=np.float64)
    for i, nod in enumerate(node_order):
        row = lk.get(str(nod).strip().lower())
        if row is None:
            continue
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


def _resolve_distance_csv(path_arg: Path | None) -> Path | None:
    if path_arg is not None:
        p = Path(path_arg).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Distance CSV not found: {p}")
        return p
    repo_root = Path(__file__).resolve().parent
    default = (repo_root / "8500-node" / "load_electrical_distance_to_each_regulator.csv").resolve()
    return default if default.is_file() else None


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


def _validate_feature_norm_pack(norm_pack: dict, norm_path: Path) -> None:
    if "mean" not in norm_pack or "std" not in norm_pack:
        raise RuntimeError(f"Invalid feature norm pack (missing mean/std): {norm_path}")
    mean = np.asarray(norm_pack["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(norm_pack["std"], dtype=np.float32).reshape(-1)
    if mean.size != 2 or std.size != 2:
        raise RuntimeError(
            f"Expected 2 feature norm entries (P,Q), got mean={mean.size} std={std.size} from {norm_path}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"Non-finite feature norm stats in {norm_path}")
    if np.any(std <= 0):
        raise RuntimeError(f"Non-positive feature std in {norm_path}")


def _infer_aux_head_nclasses(sd: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """Infer per-head class counts from aux head final linear weights."""
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


def _plot_one_node_complex_ri_global_local_components(
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
    """
    Complex-ri global+local model:
    - v_pred: |V_pred|
    - v_local: |V_local component|
    - v_delta: Δ|V|_global = |V_pred| - |V_local|  (additive in magnitude domain)
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    l1, = ax1.plot(t_hours, v_dss[:, j], lw=2.0, label="OpenDSS |V|")
    l2, = ax1.plot(t_hours, v_pred[:, j], lw=1.8, label="|V_pred|")
    l3, = ax1.plot(t_hours, v_local[:, j], lw=1.2, label="|V_local|")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Voltage magnitude (p.u.)")
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l4, = ax2.plot(t_hours, v_delta[:, j], lw=1.4, ls="--", color="tab:red", label="Δ|V|_global")
    ax2.set_ylabel("Global correction on |V| (p.u.)")
    max_abs_dv = float(np.nanmax(np.abs(v_delta[:, j]))) if np.isfinite(v_delta[:, j]).any() else 0.01
    dv_lim = max(0.01, max_abs_dv * 1.15)
    ax2.set_ylim(-dv_lim, dv_lim)

    ax1.set_title(f"{nk} | complex_ri global/local components{title_extra}")
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
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
    worst_k_per_dist_bin: int = 0,
    dist_bins: int = 10,
    electrical_distance_csv: Path | None = None,
    save_per_node_mae_csv: bool = False,
    debug_features: int = 0,
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

    raw_ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_dict = raw_ckpt if isinstance(raw_ckpt, dict) else {}
    is_gnn_only_ckpt = isinstance(ckpt_dict, dict) and ("model_type" in ckpt_dict) and ("model_state_dict" in ckpt_dict)
    has_sd = isinstance(ckpt_dict, dict) and ("model_state_dict" in ckpt_dict)
    sd0 = ckpt_dict.get("model_state_dict", {}) if isinstance(ckpt_dict, dict) else {}
    is_gine_plus_mlp_ckpt = bool(
        has_sd
        and ("hidden_mlp" in ckpt_dict)
        and isinstance(sd0, dict)
        and any(str(k).startswith("encoder.") for k in sd0.keys())
        and not any(str(k).startswith("aux_reg_heads.") or str(k).startswith("aux_cap_heads.") for k in sd0.keys())
        and not any(str(k).startswith("local_head.") or str(k).startswith("global_mlp.") for k in sd0.keys())
    )
    is_gine_plus_mlp_global_local_ckpt = bool(
        has_sd
        and ("hidden_mlp" in ckpt_dict)
        and isinstance(sd0, dict)
        and any(str(k).startswith("encoder.") for k in sd0.keys())
        and any(str(k).startswith("local_head.") for k in sd0.keys())
        and any(str(k).startswith("global_mlp.") for k in sd0.keys())
    )
    is_gine_plus_mlp_aux_ckpt = bool(
        has_sd
        and ("hidden_mlp" in ckpt_dict)
        and isinstance(sd0, dict)
        and any(str(k).startswith("encoder.") for k in sd0.keys())
        and any(str(k).startswith("aux_reg_heads.") or str(k).startswith("aux_cap_heads.") for k in sd0.keys())
    )

    meta_path = None
    norm_path = None
    model_kind = ""
    voltage_target_mode = "vmag"
    hidden = 0
    n_layers = 0
    node_out_dim = 2
    node_emb_dim = 0
    edge_emb_dim = 0
    dropout = 0.0
    gnn_only_y_mean = None
    gnn_only_y_std = None

    if is_gnn_only_ckpt:
        # Checkpoint format from train_gnn_only_compare_complex_voltage.py
        model_kind = str(ckpt_dict.get("model_type", "gine")).lower()
        voltage_target_mode = "complex_ri"
        hidden = int(ckpt_dict["hidden"])
        n_layers = int(ckpt_dict["layers"])
        node_emb_dim = int(ckpt_dict.get("node_emb_dim", 0))
        edge_emb_dim = int(ckpt_dict.get("edge_emb_dim", 0))
        dropout = float(ckpt_dict.get("dropout", 0.0))

        if all(k in ckpt_dict for k in ("x_mean", "x_std", "y_mean", "y_std")):
            norm_pack = {"mean": ckpt_dict["x_mean"], "std": ckpt_dict["x_std"]}
            gnn_only_y_mean = torch.as_tensor(ckpt_dict["y_mean"], dtype=torch.float32).view(1, -1)
            gnn_only_y_std = torch.as_tensor(ckpt_dict["y_std"], dtype=torch.float32).view(1, -1)
            _validate_feature_norm_pack(norm_pack, ckpt)
        else:
            x_mean_p = (ckpt.parent / "x_mean.pt").resolve()
            x_std_p = (ckpt.parent / "x_std.pt").resolve()
            y_mean_p = (ckpt.parent / "y_mean.pt").resolve()
            y_std_p = (ckpt.parent / "y_std.pt").resolve()
            for p_need in (x_mean_p, x_std_p, y_mean_p, y_std_p):
                if not p_need.is_file():
                    raise FileNotFoundError(f"Missing normalization tensor for gnn-only checkpoint: {p_need}")
            norm_pack = {
                "mean": torch.load(x_mean_p, map_location="cpu", weights_only=True),
                "std": torch.load(x_std_p, map_location="cpu", weights_only=True),
            }
            _validate_feature_norm_pack(norm_pack, x_mean_p)
            gnn_only_y_mean = torch.load(y_mean_p, map_location="cpu", weights_only=True).view(1, -1)
            gnn_only_y_std = torch.load(y_std_p, map_location="cpu", weights_only=True).view(1, -1)
    elif is_gine_plus_mlp_ckpt or is_gine_plus_mlp_global_local_ckpt or is_gine_plus_mlp_aux_ckpt:
        model_kind = "gine"
        voltage_target_mode = "complex_ri"
        hidden = int(ckpt_dict["hidden_gnn"])
        n_layers = int(ckpt_dict["layers"])
        node_emb_dim = int(ckpt_dict.get("node_emb_dim", 0))
        edge_emb_dim = int(ckpt_dict.get("edge_emb_dim", 0))
        dropout = float(ckpt_dict.get("dropout", 0.0))

        if all(k in ckpt_dict for k in ("x_mean", "x_std", "y_mean", "y_std")):
            norm_pack = {"mean": ckpt_dict["x_mean"], "std": ckpt_dict["x_std"]}
            gnn_only_y_mean = torch.as_tensor(ckpt_dict["y_mean"], dtype=torch.float32).view(1, -1)
            gnn_only_y_std = torch.as_tensor(ckpt_dict["y_std"], dtype=torch.float32).view(1, -1)
            _validate_feature_norm_pack(norm_pack, ckpt)
        else:
            x_mean_p = (ckpt.parent / "x_mean.pt").resolve()
            x_std_p = (ckpt.parent / "x_std.pt").resolve()
            y_mean_p = (ckpt.parent / "y_mean.pt").resolve()
            y_std_p = (ckpt.parent / "y_std.pt").resolve()
            for p_need in (x_mean_p, x_std_p, y_mean_p, y_std_p):
                if not p_need.is_file():
                    raise FileNotFoundError(f"Missing normalization tensor for GINE+MLP checkpoint: {p_need}")
            norm_pack = {
                "mean": torch.load(x_mean_p, map_location="cpu", weights_only=True),
                "std": torch.load(x_std_p, map_location="cpu", weights_only=True),
            }
            _validate_feature_norm_pack(norm_pack, x_mean_p)
            gnn_only_y_mean = torch.load(y_mean_p, map_location="cpu", weights_only=True).view(1, -1)
            gnn_only_y_std = torch.load(y_std_p, map_location="cpu", weights_only=True).view(1, -1)
    else:
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
        if voltage_target_mode == "complex_ri":
            vm = meta.get("val_voltage_metrics", {}) if isinstance(meta.get("val_voltage_metrics"), dict) else {}
            mae_vmag = float(vm.get("mae_vmag_pu", float("nan")))
            mae_ang = float(vm.get("mae_angle_deg", float("nan")))
            # Defensive guard for a known bad-training signature:
            # near-zero vmag MAE together with ~random angle MAE often means complex targets were malformed.
            if np.isfinite(mae_vmag) and np.isfinite(mae_ang) and mae_vmag < 1e-3 and mae_ang > 30.0:
                raise RuntimeError(
                    "Suspicious complex_ri checkpoint: val MAE(vmag) is near zero but angle MAE is very large. "
                    "This is a known sign of malformed complex targets (often near-zero V_re/V_im labels). "
                    "Retrain with corrected complex target construction."
                )

    # Build canonical node ordering from first sample in nodes CSV.
    x_tmp, _y_tmp, _sids, node_order, node_to_local = _load_nodes_pq_target(nodes_path)
    n_nodes = int(x_tmp.shape[1])
    del x_tmp, _y_tmp
    edge_index, edge_attr = _load_compacted_edges(edges_path, node_to_local)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    if not (is_gnn_only_ckpt or is_gine_plus_mlp_ckpt or is_gine_plus_mlp_global_local_ckpt or is_gine_plus_mlp_aux_ckpt):
        norm_pack = torch.load(norm_path, map_location="cpu", weights_only=False)
        _validate_feature_norm_pack(norm_pack, norm_path)
    feat_mean = torch.as_tensor(norm_pack["mean"], dtype=torch.float32).view(1, 1, -1)
    feat_std = torch.as_tensor(norm_pack["std"], dtype=torch.float32).clamp_min(1e-8).view(1, 1, -1)

    if is_gnn_only_ckpt:
        model = GNNOnlyVoltageModel(
            model_type=model_kind,
            in_dim=2,
            hidden=hidden,
            layers=n_layers,
            n_nodes=n_nodes,
            num_edges=int(edge_index.shape[1]),
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            dropout=dropout,
        ).to(device)
    elif is_gine_plus_mlp_ckpt:
        enc = GINEEncoderPlain(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            n_layers=n_layers,
            state_dim=int(ckpt_dict.get("state_dim", 2)),
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            dropout=dropout,
        )
        model = GINEPlusMLP(encoder=enc, mlp_hidden=int(ckpt_dict["hidden_mlp"]), n_nodes=n_nodes).to(device)
    elif is_gine_plus_mlp_aux_ckpt:
        state_dict_for_aux = ckpt_dict["model_state_dict"]
        reg_nclasses = _infer_aux_head_nclasses(state_dict_for_aux, "aux_reg_heads")
        cap_nclasses = _infer_aux_head_nclasses(state_dict_for_aux, "aux_cap_heads")
        enc = GINEEncoderAux(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            n_layers=n_layers,
            state_dim=int(ckpt_dict.get("state_dim", 2)),
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            dropout=dropout,
        )
        model = GINEPlusMLPAux(
            n_nodes=n_nodes,
            encoder=enc,
            hidden_mlp=int(ckpt_dict["hidden_mlp"]),
            aux_head_depth=int(ckpt_dict.get("aux_head_depth", 1)),
            aux_head_dropout=float(ckpt_dict.get("aux_head_dropout", 0.0)),
            aux_head_first_hidden=int(ckpt_dict.get("aux_head_first_hidden", ckpt_dict.get("aux_hidden", 512))),
            reg_nclasses=reg_nclasses,
            cap_nclasses=cap_nclasses,
        ).to(device)
    elif is_gine_plus_mlp_global_local_ckpt:
        enc = GINEEncoderGlobalLocal(
            in_dim=2,
            n_nodes=n_nodes,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            n_layers=n_layers,
            state_dim=int(ckpt_dict.get("state_dim", 2)),
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            dropout=dropout,
        )
        model = GINEPlusMLPGlobalLocal(encoder=enc, mlp_hidden=int(ckpt_dict["hidden_mlp"]), n_nodes=n_nodes).to(device)
    elif voltage_target_mode == "complex_ri":
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

    state_dict = raw_ckpt
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    n_aux = 0
    if voltage_target_mode != "complex_ri":
        state_dict, n_aux = _state_dict_voltage_only(state_dict)
        if n_aux:
            print(f"[compare_homo_mv_daily_global_localres] stripped {n_aux} aux_* keys from checkpoint (voltage-only inference)", flush=True)
    missing_unexpected = model.load_state_dict(state_dict, strict=False)
    if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
        missing_n = len(missing_unexpected.missing_keys)
        unexpected_n = len(missing_unexpected.unexpected_keys)
        sample_missing = missing_unexpected.missing_keys[:8]
        sample_unexpected = missing_unexpected.unexpected_keys[:8]
        raise RuntimeError(
            "Checkpoint/model mismatch during inference. "
            f"missing_keys={missing_n} unexpected_keys={unexpected_n}. "
            f"Examples missing={sample_missing} unexpected={sample_unexpected}. "
            "This usually means checkpoint, train-metrics JSON, and/or model settings are not from the same run."
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

    debug_nodes_l = [str(n).strip().lower() for n in plot_nodes] if plot_nodes else []

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

        if debug_features > 0 and i < int(debug_features):
            nnz = int(np.sum(np.abs(x) > 0.0))
            nnz_rows = int(np.sum((np.abs(x[:, 0]) + np.abs(x[:, 1])) > 0.0))
            print(
                f"[compare_homo_mv_daily_global_localres][debug] t={i} mult={m_t:.6g} "
                f"raw_x nnz_entries={nnz}/{x.size} nnz_rows={nnz_rows}/{n_nodes} "
                f"p_range=[{float(x[:,0].min()):.3g},{float(x[:,0].max()):.3g}] "
                f"q_range=[{float(x[:,1].min()):.3g},{float(x[:,1].max()):.3g}]",
                flush=True,
            )
            for dn in debug_nodes_l:
                jdn = node_to_idx.get(dn)
                if jdn is None:
                    continue
                print(
                    f"[compare_homo_mv_daily_global_localres][debug] node {dn}: "
                    f"P={x[jdn,0]:.6g} Q={x[jdn,1]:.6g} | "
                    f"DSS_V={float(v_dss[i,jdn]):.6g}",
                    flush=True,
                )

        # GNN inference.
        t0 = time.perf_counter()
        xb = torch.from_numpy(x).view(1, n_nodes, 2)
        xb = ((xb - feat_mean) / feat_std).squeeze(0).to(device)
        if debug_features > 0 and i < int(debug_features):
            xb_cpu = xb.detach().cpu()
            print(
                f"[compare_homo_mv_daily_global_localres][debug] xb_norm "
                f"mean={float(xb_cpu.mean()):.6g} std={float(xb_cpu.std(unbiased=False)):.6g} "
                f"min={float(xb_cpu.min()):.6g} max={float(xb_cpu.max()):.6g}",
                flush=True,
            )
        data = Data(x=xb, edge_index=edge_index, edge_attr=edge_attr)
        with torch.no_grad():
            if is_gnn_only_ckpt or is_gine_plus_mlp_ckpt or is_gine_plus_mlp_aux_ckpt:
                data_b = Batch.from_data_list([data]).to(device)
                pred_n = model(data_b)  # [1, 2N] normalized [V_re, V_im]
                pred_flat = pred_n * gnn_only_y_std.to(device) + gnn_only_y_mean.to(device)
                pred_ri = pred_flat.view(1, n_nodes, 2)
                pr = pred_ri[..., 0]
                pi = pred_ri[..., 1]
                vmag_t = torch.sqrt(pr * pr + pi * pi + 1e-12)
                v_pred[i, :] = vmag_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                # No local/ΔV decomposition for gnn-only complex_ri.
            elif is_gine_plus_mlp_global_local_ckpt:
                data_b = Batch.from_data_list([data]).to(device)
                pred_n, local_n, delta_n = model(data_b)  # all [1, 2N] in normalized target space
                y_std_dev = gnn_only_y_std.to(device)
                y_mean_dev = gnn_only_y_mean.to(device)
                pred_flat = pred_n * y_std_dev + y_mean_dev
                local_flat = local_n * y_std_dev + y_mean_dev
                delta_flat = delta_n * y_std_dev

                pred_ri = pred_flat.view(1, n_nodes, 2)
                local_ri = local_flat.view(1, n_nodes, 2)

                pr, pi = pred_ri[..., 0], pred_ri[..., 1]
                lr, li = local_ri[..., 0], local_ri[..., 1]
                vmag_pred = torch.sqrt(pr * pr + pi * pi + 1e-12)
                vmag_local = torch.sqrt(lr * lr + li * li + 1e-12)
                # Use additive magnitude decomposition for plotting:
                # |V_pred| = |V_local| + (|V_pred| - |V_local|)
                vmag_delta = vmag_pred - vmag_local
                v_pred[i, :] = vmag_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
                v_local[i, :] = vmag_local.squeeze(0).detach().cpu().numpy().astype(np.float32)
                v_delta[i, :] = vmag_delta.squeeze(0).detach().cpu().numpy().astype(np.float32)
            elif voltage_target_mode == "complex_ri":
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
        if debug_features > 0 and i < int(debug_features):
            vp = v_pred[i, :]
            print(
                f"[compare_homo_mv_daily_global_localres][debug] v_pred range "
                f"[{float(np.nanmin(vp)):.6g},{float(np.nanmax(vp)):.6g}] mean={float(np.nanmean(vp)):.6g}",
                flush=True,
            )

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

    worst_per_bin_entries: list[dict[str, object]] = []
    if worst_k_per_dist_bin > 0:
        dist_csv = _resolve_distance_csv(electrical_distance_csv)
        if dist_csv is None:
            raise FileNotFoundError(
                "Cannot run --worst-k-per-dist-bin without an electrical distance CSV. "
                "Pass --electrical-distance-csv explicitly."
            )
        dmin = _min_positive_regulator_distance(dist_csv, node_order_l)
        ddf = pd.DataFrame({"node": node_order_l, "min_regulator_distance": dmin})
        df_rank = df_node.merge(ddf, on="node", how="left")
        valid = df_rank[np.isfinite(df_rank["mae_pu"].to_numpy(dtype=np.float64))]
        valid = valid[np.isfinite(valid["min_regulator_distance"].to_numpy(dtype=np.float64))].copy()
        nb = int(max(1, dist_bins))
        if len(valid) >= nb:
            valid["dist_bin"] = pd.qcut(valid["min_regulator_distance"], q=nb, labels=False, duplicates="drop")
        elif len(valid) > 0:
            valid["dist_bin"] = 0
        else:
            valid["dist_bin"] = np.nan
        if len(valid):
            valid["dist_bin"] = valid["dist_bin"].astype(int) + 1
            for b in sorted(valid["dist_bin"].unique().tolist()):
                sub = valid.loc[valid["dist_bin"] == b].sort_values("mae_pu", ascending=False).head(int(worst_k_per_dist_bin))
                for _, row in sub.iterrows():
                    worst_per_bin_entries.append(
                        {
                            "bin_index": int(b),
                            "node": str(row["node"]),
                            "mae_pu": float(row["mae_pu"]) if pd.notna(row["mae_pu"]) else None,
                            "rmse_pu": float(row["rmse_pu"]) if pd.notna(row["rmse_pu"]) else None,
                            "n_valid": int(row["n_valid"]),
                            "min_regulator_distance": (
                                float(row["min_regulator_distance"]) if pd.notna(row["min_regulator_distance"]) else None
                            ),
                        }
                    )
            print(
                f"[compare_homo_mv_daily_global_localres] worst {worst_k_per_dist_bin} per distance bin "
                f"(bins={int(valid['dist_bin'].max())})",
                flush=True,
            )
            for w in worst_per_bin_entries:
                print(
                    f"  bin{w['bin_index']:02d} {w['node']}: MAE={w['mae_pu']} dmin={w['min_regulator_distance']}",
                    flush=True,
                )
        else:
            print(
                "[compare_homo_mv_daily_global_localres] no valid node distances found; "
                "skip worst-per-distance-bin selection.",
                flush=True,
            )

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
        "worst_k_per_dist_bin": int(worst_k_per_dist_bin),
        "dist_bins_requested": int(dist_bins),
        "electrical_distance_csv": str(_resolve_distance_csv(electrical_distance_csv)) if worst_k_per_dist_bin > 0 else None,
        "worst_nodes_by_mae_per_dist_bin": worst_per_bin_entries,
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
        "train_metrics": str(meta_path) if meta_path is not None else None,
        "feature_norm": str(norm_path) if norm_path is not None else "from checkpoint (x_mean/x_std) or sibling x_mean.pt/x_std.pt",
        "checkpoint_family": (
            "gnn_only_compare_complex_voltage"
            if is_gnn_only_ckpt
            else (
                "gine_plus_mlp_aux_complex_voltage"
                if is_gine_plus_mlp_aux_ckpt
                else (
                    "gine_plus_mlp_global_local_complex_voltage"
                    if is_gine_plus_mlp_global_local_ckpt
                    else ("gine_plus_mlp_complex_voltage" if is_gine_plus_mlp_ckpt else "homo_global_localres")
                )
            )
        ),
    }
    (out_dir / "daily_metrics_global_localres.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    plots_dir = out_dir / "monitoring_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    worst_dir = plots_dir / "worst_by_mae"
    worst_per_bin_dir = plots_dir / "worst_by_mae_per_dist_bin"
    if worst_k > 0:
        worst_dir.mkdir(parents=True, exist_ok=True)
    if worst_k_per_dist_bin > 0:
        worst_per_bin_dir.mkdir(parents=True, exist_ok=True)

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
            if is_gine_plus_mlp_global_local_ckpt:
                _plot_one_node_complex_ri_global_local_components(
                    nk=nk,
                    j=j,
                    t_hours=t_hours,
                    v_dss=v_dss,
                    v_pred=v_pred,
                    v_local=v_local,
                    v_delta=v_delta,
                    ymin=ymin,
                    ymax=ymax,
                    out_png=plots_dir / f"{nk.replace('.', '_')}_complex_ri_vmag_components.png",
                    show_plots=show_plots,
                    title_extra=title_ex,
                )
            else:
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
            if is_gine_plus_mlp_global_local_ckpt:
                _plot_one_node_complex_ri_global_local_components(
                    nk=nk,
                    j=j,
                    t_hours=t_hours,
                    v_dss=v_dss,
                    v_pred=v_pred,
                    v_local=v_local,
                    v_delta=v_delta,
                    ymin=ymin,
                    ymax=ymax,
                    out_png=worst_dir / f"rank{rank:02d}_{nk.replace('.', '_')}_complex_ri_vmag_components.png",
                    show_plots=show_plots,
                    title_extra=title_ex,
                )
            else:
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

    for rank, w in enumerate(worst_per_bin_entries, start=1):
        nk = str(w["node"])
        if nk not in node_to_idx:
            continue
        j = node_to_idx[nk]
        mae_w = w.get("mae_pu")
        bidx = int(w.get("bin_index", -1))
        dmin = w.get("min_regulator_distance")
        title_ex = f" | dist-bin #{bidx} rank-list"
        if mae_w is not None and np.isfinite(mae_w):
            title_ex += f" (MAE={float(mae_w):.5f} pu)"
        if dmin is not None and np.isfinite(dmin):
            title_ex += f" dmin={float(dmin):.4f}"
        if voltage_target_mode == "complex_ri":
            if is_gine_plus_mlp_global_local_ckpt:
                _plot_one_node_complex_ri_global_local_components(
                    nk=nk,
                    j=j,
                    t_hours=t_hours,
                    v_dss=v_dss,
                    v_pred=v_pred,
                    v_local=v_local,
                    v_delta=v_delta,
                    ymin=ymin,
                    ymax=ymax,
                    out_png=worst_per_bin_dir
                    / f"bin{bidx:02d}_rank{rank:02d}_{nk.replace('.', '_')}_complex_ri_vmag_components.png",
                    show_plots=show_plots,
                    title_extra=title_ex,
                )
            else:
                _plot_one_node_complex_ri_magnitude(
                    nk=nk,
                    j=j,
                    t_hours=t_hours,
                    v_dss=v_dss,
                    v_pred=v_pred,
                    ymin=ymin,
                    ymax=ymax,
                    out_png=worst_per_bin_dir / f"bin{bidx:02d}_rank{rank:02d}_{nk.replace('.', '_')}_complex_ri_vmag.png",
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
                out_png=worst_per_bin_dir / f"bin{bidx:02d}_rank{rank:02d}_{nk.replace('.', '_')}_v_local_delta.png",
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
    p.add_argument(
        "--worst-k-per-dist-bin",
        type=int,
        default=0,
        metavar="K",
        help="Select/plot top-K MAE nodes inside each distance-quantile bin (by min positive regulator distance).",
    )
    p.add_argument(
        "--dist-bins",
        type=int,
        default=10,
        metavar="B",
        help="Number of quantile bins for --worst-k-per-dist-bin (default: 10).",
    )
    p.add_argument(
        "--electrical-distance-csv",
        type=Path,
        default=None,
        help="Path to load_electrical_distance_to_each_regulator.csv (default: repo/8500-node/... if present).",
    )
    p.add_argument(
        "--debug-features",
        type=int,
        default=0,
        metavar="N",
        help="Print feature+prediction stats for first N timesteps (helps diagnose near-zero predictions due to mapping/norm mismatch).",
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
        worst_k_per_dist_bin=int(args.worst_k_per_dist_bin),
        dist_bins=int(args.dist_bins),
        electrical_distance_csv=args.electrical_distance_csv.resolve() if args.electrical_distance_csv else None,
        save_per_node_mae_csv=bool(args.save_per_node_mae_csv),
        debug_features=int(args.debug_features),
    )


if __name__ == "__main__":
    main()

