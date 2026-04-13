"""
Daily OpenDSS comparison for real MLP vs CVNN MLP checkpoints.

Evaluates checkpoints produced by train_compare_mlp_cvnn_complex_voltage.py:
  - real_mlp_best.pt
  - cvnn_mlp_best.pt
  - x_mean.pt / x_std.pt

Outputs:
  - daily_metrics_mlp_cvnn.json
  - daily_per_node_mae_magnitude.csv
  - per-node plots for selected --plot-node entries
  - auto-plots for --worst-k nodes by real-model |V| MAE
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss

import run_injection_dataset as inj
import run_daily_aggregate_dataset_8500 as rd8500
from compare_opendss_snapshot_helpers import force_snapshot_mode_for_compare_timing, reassert_snapshot_before_each_solve
from train_homo_gine_global_localres_pq_loadonly import _load_nodes_pq_target
from train_compare_mlp_cvnn_complex_voltage import RealMLP, ComplexMLP


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


def _angle_diff_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    d = pred_deg - true_deg
    return (d + 180.0) % 360.0 - 180.0


def _metrics(vmag_pred: np.ndarray, vang_pred_deg: np.ndarray, vmag_true: np.ndarray, vang_true_deg: np.ndarray) -> dict[str, float]:
    m = np.isfinite(vmag_pred) & np.isfinite(vmag_true) & np.isfinite(vang_pred_deg) & np.isfinite(vang_true_deg)
    if not np.any(m):
        return {
            "mae_vmag_pu": float("nan"),
            "rmse_vmag_pu": float("nan"),
            "mae_angle_deg": float("nan"),
            "rmse_angle_deg": float("nan"),
        }
    dv = vmag_pred[m] - vmag_true[m]
    da = _angle_diff_deg(vang_pred_deg[m], vang_true_deg[m])
    return {
        "mae_vmag_pu": float(np.mean(np.abs(dv))),
        "rmse_vmag_pu": float(np.sqrt(np.mean(dv * dv))),
        "mae_angle_deg": float(np.mean(np.abs(da))),
        "rmse_angle_deg": float(np.sqrt(np.mean(da * da))),
    }


def _plot_node(
    *,
    nk: str,
    j: int,
    t_hours: np.ndarray,
    vmag_dss: np.ndarray,
    vang_dss: np.ndarray,
    vmag_real: np.ndarray,
    vang_real: np.ndarray,
    vmag_cvnn: np.ndarray,
    vang_cvnn: np.ndarray,
    out_png: Path,
    ymin: float,
    ymax: float,
    title_extra: str = "",
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(t_hours, vmag_dss[:, j], lw=2.0, label="OpenDSS |V|")
    ax1.plot(t_hours, vmag_real[:, j], lw=1.4, label="Real MLP |V|")
    ax1.plot(t_hours, vmag_cvnn[:, j], lw=1.4, label="CVNN MLP |V|")
    ax1.set_ylabel("Voltage (pu)")
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(t_hours, vang_dss[:, j], lw=2.0, label="OpenDSS angle")
    ax2.plot(t_hours, vang_real[:, j], lw=1.4, label="Real MLP angle")
    ax2.plot(t_hours, vang_cvnn[:, j], lw=1.4, label="CVNN MLP angle")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Angle (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(f"{nk} daily comparison{title_extra}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Daily compare OpenDSS vs real MLP vs CVNN MLP.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="Directory containing real_mlp_best.pt, cvnn_mlp_best.pt, x_mean.pt, x_std.pt")
    ap.add_argument("--dataset-dir", type=Path, required=True, help=".../Heterogenous GNN dataset")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--daily-profile", type=str, default="5minDayShape.csv")
    ap.add_argument("--plot-node", action="append", default=[])
    ap.add_argument("--npts", type=int, default=288)
    ap.add_argument("--step-min", type=int, default=5)
    ap.add_argument("--ymin", type=float, default=0.85)
    ap.add_argument("--ymax", type=float, default=1.10)
    ap.add_argument("--worst-k", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.ckpt_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_ckpt_path = ckpt_dir / "real_mlp_best.pt"
    cvnn_ckpt_path = ckpt_dir / "cvnn_mlp_best.pt"
    x_mean_path = ckpt_dir / "x_mean.pt"
    x_std_path = ckpt_dir / "x_std.pt"
    for p in (real_ckpt_path, cvnn_ckpt_path, x_mean_path, x_std_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    real_ckpt = torch.load(real_ckpt_path, map_location="cpu", weights_only=False)
    cvnn_ckpt = torch.load(cvnn_ckpt_path, map_location="cpu", weights_only=False)
    x_mean = torch.load(x_mean_path, map_location="cpu", weights_only=False).float()
    x_std = torch.load(x_std_path, map_location="cpu", weights_only=False).float().clamp_min(1e-8)

    real_model = RealMLP(
        in_dim=int(real_ckpt["in_dim"]),
        out_dim=int(real_ckpt["out_dim"]),
        hidden=int(real_ckpt["hidden"]),
    ).to(device)
    real_model.load_state_dict(real_ckpt["model_state_dict"])
    real_model.eval()
    real_y_mean = torch.as_tensor(real_ckpt["y_mean"], dtype=torch.float32)
    real_y_std = torch.as_tensor(real_ckpt["y_std"], dtype=torch.float32).clamp_min(1e-8)

    cvnn_model = ComplexMLP(
        in_dim=int(cvnn_ckpt["in_dim_complex"]),
        out_dim=int(cvnn_ckpt["out_dim_complex"]),
        hidden=int(cvnn_ckpt["hidden"]),
    ).to(device)
    cvnn_model.load_state_dict(cvnn_ckpt["model_state_dict"])
    cvnn_model.eval()
    y_mean_re = torch.as_tensor(cvnn_ckpt["y_mean_re"], dtype=torch.float32)
    y_std_re = torch.as_tensor(cvnn_ckpt["y_std_re"], dtype=torch.float32).clamp_min(1e-8)
    y_mean_im = torch.as_tensor(cvnn_ckpt["y_mean_im"], dtype=torch.float32)
    y_std_im = torch.as_tensor(cvnn_ckpt["y_std_im"], dtype=torch.float32).clamp_min(1e-8)

    ds = args.dataset_dir.resolve()
    nodes_path = ds / "nodes" / "hetero_mv_nodes_load_transformer_reg_tap_only.csv"
    if not nodes_path.is_file():
        raise FileNotFoundError(nodes_path)
    x_tmp, _y_tmp, _sids, node_order, _node_to_local = _load_nodes_pq_target(nodes_path)
    n_nodes = int(x_tmp.shape[1])
    del x_tmp, _y_tmp, _sids, _node_to_local
    node_order_l = [str(n).strip().lower() for n in node_order]
    node_to_idx = {n: i for i, n in enumerate(node_order_l)}

    rd8500._compile_8500_daily_setup()
    rd8500._detach_daily_loadshape_from_loads()
    force_snapshot_mode_for_compare_timing()
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    profile_path = rd8500._resolve_daily_profile_csv(args.daily_profile)
    mL = rd8500._daily_profile_5min(npts=args.npts, profile_csv=args.daily_profile)

    repo_root = Path(__file__).resolve().parent
    mapping_path = repo_root / "8500-node" / "mv_x_sx_node_mapping_8500.csv"
    mv_sx_rules = _load_mv_sx_mapping(mapping_path) if mapping_path.is_file() else []

    t_hours = np.arange(args.npts, dtype=np.float32) * (args.step_min / 60.0)
    vmag_dss = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)
    vang_dss = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)
    vmag_real = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)
    vang_real = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)
    vmag_cvnn = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)
    vang_cvnn = np.full((args.npts, n_nodes), np.nan, dtype=np.float32)

    n_nonconv = 0
    real_forward_s = 0.0
    cvnn_forward_s = 0.0
    for i in range(args.npts):
        hr = int(i // 12)
        sec = int((i % 12) * (args.step_min * 60))
        m_t = float(mL[i])
        kw_set = base_kw * m_t
        kvar_set = base_kvar * m_t
        dss.Text.Command(f"set hour={hr} sec={sec}")
        for j, name in enumerate(base_names):
            dss.Loads.Name(name)
            dss.Loads.kW(float(kw_set[j]))
            dss.Loads.kvar(float(kvar_set[j]))
        reassert_snapshot_before_each_solve()
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            n_nonconv += 1
            continue

        vd, ad = inj.get_all_node_voltage_pu_and_angle_filtered(node_order_l)
        vmag_dss[i, :] = np.asarray(vd, dtype=np.float32)
        vang_dss[i, :] = np.asarray(ad, dtype=np.float32)

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
        if mv_sx_rules:
            for rec in mv_sx_rules:
                node_p[rec["mv_key"]] = float(node_p.get(rec["load_a"], 0.0) + node_p.get(rec["load_b"], 0.0))
                node_q[rec["mv_key"]] = float(node_q.get(rec["load_a"], 0.0) + node_q.get(rec["load_b"], 0.0))

        x = np.zeros((n_nodes, 2), dtype=np.float32)
        for ni, nk in enumerate(node_order_l):
            x[ni, 0] = float(node_p.get(nk, 0.0))
            x[ni, 1] = float(node_q.get(nk, 0.0))

        x_flat = torch.from_numpy(x.reshape(1, -1)).float()
        x_n = (x_flat - x_mean) / x_std

        with torch.no_grad():
            t0 = time.perf_counter()
            y_real_n = real_model(x_n.to(device))
            real_forward_s += time.perf_counter() - t0
            y_real = y_real_n * real_y_std.to(device) + real_y_mean.to(device)
            yr = y_real.view(1, n_nodes, 2).squeeze(0).cpu().numpy()
            r_re = yr[:, 0]
            r_im = yr[:, 1]
            vmag_real[i, :] = np.sqrt(r_re * r_re + r_im * r_im).astype(np.float32)
            vang_real[i, :] = np.rad2deg(np.arctan2(r_im, r_re)).astype(np.float32)

            t0 = time.perf_counter()
            xri = x_n.view(1, n_nodes, 2).to(device)
            z_in = torch.complex(xri[..., 0], xri[..., 1])
            z_pred_n = cvnn_model(z_in)
            cvnn_forward_s += time.perf_counter() - t0
            c_re = z_pred_n.real * y_std_re.to(device) + y_mean_re.to(device)
            c_im = z_pred_n.imag * y_std_im.to(device) + y_mean_im.to(device)
            c_re_np = c_re.squeeze(0).cpu().numpy()
            c_im_np = c_im.squeeze(0).cpu().numpy()
            vmag_cvnn[i, :] = np.sqrt(c_re_np * c_re_np + c_im_np * c_im_np).astype(np.float32)
            vang_cvnn[i, :] = np.rad2deg(np.arctan2(c_im_np, c_re_np)).astype(np.float32)

    met_real = _metrics(vmag_real, vang_real, vmag_dss, vang_dss)
    met_cvnn = _metrics(vmag_cvnn, vang_cvnn, vmag_dss, vang_dss)

    rows = []
    for j, nk in enumerate(node_order_l):
        m = np.isfinite(vmag_dss[:, j]) & np.isfinite(vmag_real[:, j]) & np.isfinite(vmag_cvnn[:, j])
        if not np.any(m):
            rows.append({"node": nk, "mae_real_pu": np.nan, "mae_cvnn_pu": np.nan, "n_valid": 0})
            continue
        mae_r = float(np.mean(np.abs(vmag_real[m, j] - vmag_dss[m, j])))
        mae_c = float(np.mean(np.abs(vmag_cvnn[m, j] - vmag_dss[m, j])))
        rows.append({"node": nk, "mae_real_pu": mae_r, "mae_cvnn_pu": mae_c, "n_valid": int(m.sum())})
    df_node = pd.DataFrame(rows).sort_values("mae_real_pu", ascending=False, na_position="last").reset_index(drop=True)
    df_node.to_csv(out_dir / "daily_per_node_mae_magnitude.csv", index=False)

    plots_dir = out_dir / "monitoring_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.plot_node:
        nk = str(raw).strip().lower()
        if nk not in node_to_idx:
            print(f"skip plot node not in model node set: {raw}", flush=True)
            continue
        j = node_to_idx[nk]
        _plot_node(
            nk=nk,
            j=j,
            t_hours=t_hours,
            vmag_dss=vmag_dss,
            vang_dss=vang_dss,
            vmag_real=vmag_real,
            vang_real=vang_real,
            vmag_cvnn=vmag_cvnn,
            vang_cvnn=vang_cvnn,
            out_png=plots_dir / f"{nk.replace('.', '_')}_daily_compare_mlp_cvnn.png",
            ymin=args.ymin,
            ymax=args.ymax,
        )

    worst_entries: list[dict[str, object]] = []
    if args.worst_k > 0 and len(df_node) > 0:
        worst_dir = plots_dir / "worst_by_mae"
        worst_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df_node.head(int(args.worst_k)).iterrows():
            nk = str(row["node"])
            worst_entries.append(
                {
                    "node": nk,
                    "mae_real_pu": float(row["mae_real_pu"]) if pd.notna(row["mae_real_pu"]) else None,
                    "mae_cvnn_pu": float(row["mae_cvnn_pu"]) if pd.notna(row["mae_cvnn_pu"]) else None,
                    "n_valid": int(row["n_valid"]),
                }
            )
            if nk in node_to_idx:
                j = node_to_idx[nk]
                title_extra = ""
                if pd.notna(row["mae_real_pu"]):
                    title_extra = f" | worst-by-real-MAE={float(row['mae_real_pu']):.5f} pu"
                _plot_node(
                    nk=nk,
                    j=j,
                    t_hours=t_hours,
                    vmag_dss=vmag_dss,
                    vang_dss=vang_dss,
                    vmag_real=vmag_real,
                    vang_real=vang_real,
                    vmag_cvnn=vmag_cvnn,
                    vang_cvnn=vang_cvnn,
                    out_png=worst_dir / f"{nk.replace('.', '_')}_daily_compare_mlp_cvnn.png",
                    ymin=args.ymin,
                    ymax=args.ymax,
                    title_extra=title_extra,
                )

    out = {
        "daily_profile_csv": str(profile_path),
        "npts": int(args.npts),
        "n_nonconv": int(n_nonconv),
        "n_nodes": int(n_nodes),
        "checkpoint_dir": str(ckpt_dir),
        "real_checkpoint": str(real_ckpt_path),
        "cvnn_checkpoint": str(cvnn_ckpt_path),
        "metrics_real": met_real,
        "metrics_cvnn": met_cvnn,
        "timing_seconds": {
            "real_forward_total_s": float(real_forward_s),
            "cvnn_forward_total_s": float(cvnn_forward_s),
        },
        "worst_k": int(args.worst_k),
        "worst_nodes_by_real_mae": worst_entries,
    }
    (out_dir / "daily_metrics_mlp_cvnn.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved:", out_dir, flush=True)
    print("Real:", met_real, flush=True)
    print("CVNN:", met_cvnn, flush=True)


if __name__ == "__main__":
    main()
