"""
Paper §auxiliary_performance (IEEE 8500): aux metrics + device-day trajectories + token attention.

Wraps existing pieces:
  - ``notebook_cell_attention_aux_voltage_nodes.py`` (aux + hop attention ratios / histograms)
  - optional Method A daily compare for regulator / capacitor trajectories
  - hop-distance ratio curve + LaTeX-ready table printout

Colab / notebook: prefer the launcher cell in ``906.ipynb`` (sets knobs then runs this).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent


def _find_repo() -> Path:
    cands: list[Path] = []
    env = os.environ.get("GNN2_REPO_ROOT", "").strip()
    if env:
        cands.append(Path(env))
    if Path("/content").is_dir():
        cands.extend([Path("/content/GNN2"), Path("/content/GNN-Sandia")])
    cands.append(REPO_ROOT)
    cands.append(Path.cwd())
    seen: set[Path] = set()
    for raw in cands:
        p = raw.expanduser().resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / "notebook_cell_attention_aux_voltage_nodes.py").is_file():
            return p
    raise FileNotFoundError("GNN2 repo not found (need notebook_cell_attention_aux_voltage_nodes.py)")


def _compose_device_trajectories_figure(
    day_dir: Path,
    out_path: Path,
    *,
    reg_col: str = "reg_vreg2_a_tap_pu",
    cap_col: str = "cap_capbank0a",
) -> Path | None:
    """Build a 2-panel paper figure from Method A daily CSVs (OD vs DA-GPS)."""
    reg_csv = next(day_dir.glob("daily_regulator_tap_*.csv"), None)
    cap_csv = next(day_dir.glob("daily_cap_bank_status_*.csv"), None)
    if reg_csv is None or cap_csv is None:
        return None
    reg = pd.read_csv(reg_csv)
    cap = pd.read_csv(cap_csv)
    # Flexible column names from daily compare exports
    tcol = "hour" if "hour" in reg.columns else ("t_hour" if "t_hour" in reg.columns else None)
    if tcol is None and "step" in reg.columns:
        reg = reg.copy()
        reg["hour"] = reg["step"].astype(float) * (24.0 / max(len(reg), 1))
        tcol = "hour"
    if tcol is None:
        return None

    def _pair(df: pd.DataFrame, stem: str) -> tuple[str | None, str | None]:
        """Match Method A exports like ``{stem}__dss_tap_pu`` / ``{stem}__gnn_tap_pu``."""
        od = gnn = None
        stem_l = stem.lower()
        # Prefer exact stem prefixes used by run_da_gps_daily_opendss_compare
        for c in df.columns:
            cl = str(c)
            if not cl.lower().startswith(stem_l):
                continue
            low = cl.lower()
            if "__dss_" in low or low.endswith("__dss") or "_opendss" in low:
                od = c
            elif "__gnn_" in low or low.endswith("__gnn") or "_da_gps" in low:
                gnn = c
        if od is None or gnn is None:
            for c in df.columns:
                cl = str(c).lower()
                if stem_l not in cl:
                    continue
                if "dss" in cl or "opendss" in cl:
                    od = od or c
                elif "gnn" in cl or "da_gps" in cl:
                    gnn = gnn or c
        return od, gnn

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)

    # --- regulator ---
    ax = axes[0]
    od_c, gnn_c = _pair(reg, reg_col)
    if od_c and gnn_c and od_c in reg.columns and gnn_c in reg.columns:
        ax.plot(reg[tcol], reg[od_c], lw=2.0, label="OpenDSS")
        ax.plot(reg[tcol], reg[gnn_c], ls="--", lw=1.6, label="DA-GPS")
    else:
        # Long format fallback: filter by reg_col
        key = "reg_col" if "reg_col" in reg.columns else ("device" if "device" in reg.columns else None)
        if key is not None:
            sub = reg[reg[key].astype(str) == reg_col]
            y_od = next((c for c in sub.columns if "opendss" in c.lower() or c.lower() in ("tap_od", "y_true")), None)
            y_gn = next((c for c in sub.columns if "da_gps" in c.lower() or "gnn" in c.lower() or c.lower() in ("tap_gnn", "y_pred")), None)
            if y_od and y_gn and tcol in sub.columns:
                ax.plot(sub[tcol], sub[y_od], lw=2.0, label="OpenDSS")
                ax.plot(sub[tcol], sub[y_gn], ls="--", lw=1.6, label="DA-GPS")
            else:
                ax.text(0.5, 0.5, f"reg columns not found\n({reg_col})", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f"reg columns not found\n({reg_col})", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Tap (p.u.)")
    ax.set_title(f"(a) Regulator tap — {reg_col.replace('reg_', '').replace('_tap_pu', '')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- capacitor ---
    ax = axes[1]
    tcol_c = tcol if tcol in cap.columns else ("hour" if "hour" in cap.columns else None)
    od_c, gnn_c = _pair(cap, cap_col)
    plotted = False
    if tcol_c and od_c and gnn_c and od_c in cap.columns and gnn_c in cap.columns:
        ax.step(cap[tcol_c], cap[od_c], where="post", lw=2.0, label="OpenDSS")
        ax.step(cap[tcol_c], cap[gnn_c], where="post", ls="--", lw=1.6, label="DA-GPS")
        plotted = True
    else:
        key = "cap_col" if "cap_col" in cap.columns else ("device" if "device" in cap.columns else None)
        if key is not None and tcol_c is not None:
            sub = cap[cap[key].astype(str).str.contains(cap_col.replace("cap_", ""), case=False, regex=False)]
            if len(sub) == 0:
                sub = cap[cap[key].astype(str) == cap_col]
            y_od = next((c for c in sub.columns if "opendss" in c.lower() or c.lower() in ("status_od", "y_true", "on_od")), None)
            y_gn = next((c for c in sub.columns if "da_gps" in c.lower() or "gnn" in c.lower() or c.lower() in ("status_gnn", "y_pred", "on_gnn")), None)
            if y_od and y_gn and len(sub):
                ax.step(sub[tcol_c], sub[y_od], where="post", lw=2.0, label="OpenDSS")
                ax.step(sub[tcol_c], sub[y_gn], where="post", ls="--", lw=1.6, label="DA-GPS")
                plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f"cap columns not found\n({cap_col})", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Status (on/off)")
    ax.set_title(f"(b) Switched capacitor — {cap_col}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def _print_paper_table(aux_meta: dict, reg_df: pd.DataFrame, cap_df: pd.DataFrame, meta_aux_df: pd.DataFrame) -> None:
    print("\n========== Paper Table: tab:auxiliary_results ==========")
    tap_acc = aux_meta.get("reg_tap_acc_all", float("nan"))
    if not np.isfinite(tap_acc) and "exact_tap_acc" in reg_df.columns:
        tap_acc = float(reg_df["exact_tap_acc"].mean())
    tap_mae = aux_meta.get("reg_tap_mae_pu_all", float("nan"))
    if not np.isfinite(tap_mae) and "mae_tap_pu" in reg_df.columns:
        tap_mae = float(reg_df["mae_tap_pu"].mean())
    cap_bce = aux_meta.get("cap_bce_all", float("nan"))
    print(f"  Regulator taps | Exact-tap accuracy | {100.0 * tap_acc:.2f} %")
    print(f"  Regulator taps | Tap-position MAE   | {tap_mae:.6g} p.u.")
    print(f"  Switched caps  | Binary cross-entropy| {cap_bce:.6g}")

    def _row(name_keys: tuple[str, ...], label: str, unit: str) -> None:
        if meta_aux_df is None or len(meta_aux_df) == 0:
            print(f"  {label} | MAE / RMSE | [--] / [--] {unit}")
            return
        hit = None
        for _, r in meta_aux_df.iterrows():
            mc = str(r.get("meta_col", "")).lower()
            if any(k.lower() in mc for k in name_keys):
                hit = r
                break
        if hit is None:
            print(f"  {label} | MAE / RMSE | [--] / [--] {unit}")
            return
        print(
            f"  {label} | MAE / RMSE | {float(hit['mae_raw']):.4g} / {float(hit['rmse_raw']):.4g} {unit}"
        )

    _row(("pv_pv2_p", "pv2_p", "p_post_kw"), "PV2 active power", "kW")
    _row(("pv_pv2_q", "pv2_q", "q_post_kvar"), "PV2 reactive power", "kvar")
    _row(("p_loss", "P_loss"), "Feeder active-power loss", "kW")
    _row(("q_loss", "Q_loss"), "Feeder reactive-power loss", "kvar")
    print("=======================================================\n")


def main() -> None:
    # ---- knobs (override via globals()/env before import, or edit here) ----
    SMOKE = bool(globals().get("SMOKE", False))
    RUN_DEVICE_DAY = bool(globals().get("RUN_DEVICE_DAY", True))
    N_SAMPLES_AVG = globals().get("N_SAMPLES_AVG", 50 if SMOKE else None)
    DEVICE = str(globals().get("DEVICE", "auto"))
    DAY = int(globals().get("DAY", 4))
    NPTS = int(globals().get("NPTS", 12 if SMOKE else 288))
    STEP_MIN = int(globals().get("STEP_MIN", 5))
    RUN_DIR_OVERRIDE = str(globals().get("RUN_DIR_OVERRIDE", "") or "")
    CACHE_PT_OVERRIDE = str(globals().get("CACHE_PT_OVERRIDE", "") or "")
    CKPT_OVERRIDE = str(globals().get("CKPT_OVERRIDE", "") or "")
    SKIP_ATTENTION = bool(globals().get("SKIP_ATTENTION", False))
    SKIP_WORST_BUS_PLOTS = bool(globals().get("SKIP_WORST_BUS_PLOTS", True))  # keep paper run lighter
    REG_FIG_COL = str(globals().get("REG_FIG_COL", "reg_vreg2_a_tap_pu"))
    CAP_FIG_COL = str(globals().get("CAP_FIG_COL", "cap_capbank0a_n_steps_on"))

    repo = _find_repo()
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("GNN2_REPO_ROOT", str(repo))

    for _m in (
        "nonunique_notebook_bootstrap",
        "da_gps_hop_attention_ratios",
        "extract_da_gps_attention",
        "notebook_cell_attention_aux_voltage_nodes",
    ):
        sys.modules.pop(_m, None)

    from nonunique_notebook_bootstrap import (  # noqa: E402
        is_colab,
        resolve_cache_pt,
        resolve_feeder_checkpoint,
        resolve_feeder_run_dir,
        resolve_inference_device,
    )

    if is_colab():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # noqa: BLE001
            print(f"[aux_paper] Drive mount skipped/failed: {exc}", flush=True)

    run_dir = (
        Path(RUN_DIR_OVERRIDE).expanduser().resolve()
        if RUN_DIR_OVERRIDE.strip()
        else resolve_feeder_run_dir(repo, "8500")
    )
    cache_pt = (
        Path(CACHE_PT_OVERRIDE).expanduser().resolve()
        if CACHE_PT_OVERRIDE.strip()
        else resolve_cache_pt(repo, "8500")
    )
    ckpt = (
        Path(CKPT_OVERRIDE).expanduser().resolve()
        if CKPT_OVERRIDE.strip()
        else resolve_feeder_checkpoint(run_dir)
    )
    device = resolve_inference_device(DEVICE)
    out_root = run_dir / "auxiliary_performance_paper"
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    hop_cands = [
        repo / "datasets_gnn2_from pc" / "load_hop_distance_to_each_regulator_all_index_nodes.csv",
        Path("/content/drive/MyDrive/datasets_gnn2/load_hop_distance_to_each_regulator_all_index_nodes.csv"),
        Path("/content/drive/MyDrive/datasets_gnn2 (1)/load_hop_distance_to_each_regulator_all_index_nodes.csv"),
    ]
    # shared Drive shortcut used by other Colab cells
    _drive = Path("/content/drive")
    if _drive.is_dir():
        for root in _drive.glob(".shortcut-targets-by-id/*/datasets_gnn2"):
            hop_cands.append(root / "load_hop_distance_to_each_regulator_all_index_nodes.csv")
    hop_csv = next((p for p in hop_cands if p.is_file()), None)

    edges_cands = [
        repo / "datasets_gnn2_from pc" / "gnn_edges_phase_static.csv",
        run_dir / "gnn_edges_phase_static.csv",
    ]
    edges_csv = next((p for p in edges_cands if p.is_file()), None)

    print("[aux_paper] REPO =", repo, flush=True)
    print("[aux_paper] RUN_DIR =", run_dir, flush=True)
    print("[aux_paper] CKPT =", ckpt, flush=True)
    print("[aux_paper] CACHE =", cache_pt, flush=True)
    print("[aux_paper] HOP_CSV =", hop_csv, flush=True)
    print("[aux_paper] EDGES_CSV =", edges_csv, flush=True)
    print("[aux_paper] OUT_DIR =", out_dir, flush=True)
    print("[aux_paper] DEVICE =", device, "SMOKE =", SMOKE, flush=True)

    # ----- 1) optional representative-day device trajectories (Method A) -----
    day_out: Path | None = None
    if RUN_DEVICE_DAY:
        day_out = out_dir / f"device_day_{DAY:03d}"
        day_out.mkdir(parents=True, exist_ok=True)
        day_dir = repo / "a representativ days"
        load_p = day_dir / f"load_day_{DAY:03d}.csv"
        irr_p = day_dir / f"irr_day_{DAY:03d}.csv"
        cmd = [
            sys.executable,
            "-u",
            str(repo / "run_da_gps_daily_opendss_compare.py"),
            "--feeder",
            "8500",
            "--run-dir",
            str(run_dir),
            "--cache-pt",
            str(cache_pt),
            "--checkpoint",
            str(ckpt),
            "--load-profile-path",
            str(load_p),
            "--load-profile-filename",
            load_p.name,
            "--pv-irradiance-profile-path",
            str(irr_p),
            "--pv-irradiance-filename",
            irr_p.name,
            "--npts",
            str(NPTS),
            "--step-min",
            str(STEP_MIN),
            "--daily-stress",
            "0.0",
            "--scenario-scale",
            "1.0",
            "--ref-sample-index",
            "0",
            "--device",
            device,
            "--out-dir",
            str(day_out),
        ]
        print("[aux_paper] Method A device day:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(repo))
        if rc != 0:
            print(f"[aux_paper] WARNING: daily compare exit={rc}", flush=True)
        fig_dev = _compose_device_trajectories_figure(
            day_out,
            out_dir / "device_trajectories.png",
            reg_col=REG_FIG_COL,
            cap_col=CAP_FIG_COL,
        )
        if fig_dev is not None:
            print(f"[aux_paper] wrote {fig_dev}", flush=True)
        else:
            print("[aux_paper] device_trajectories.png not composed (CSV schema mismatch or missing).", flush=True)
    else:
        # reuse newest prior day-compare under run_dir if present
        cands = sorted(run_dir.glob("da_gps_daily_compare_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            fig_dev = _compose_device_trajectories_figure(
                cands[0],
                out_dir / "device_trajectories.png",
                reg_col=REG_FIG_COL,
                cap_col=CAP_FIG_COL,
            )
            print(f"[aux_paper] reused day-compare {cands[0].name}; fig={fig_dev}", flush=True)

    if SKIP_ATTENTION:
        print("[aux_paper] SKIP_ATTENTION=True — done after device day.", flush=True)
        return

    # ----- 2) aux + attention (existing notebook cell script) -----
    attn_out = out_dir / "attention_extract"
    knobs = {
        "N_SAMPLES_AVG": N_SAMPLES_AVG,
        "SAMPLE_IDX_START": 0,
        "RUN_DIR": run_dir,
        "CACHE_PT": cache_pt,
        "CKPT_PATH": ckpt,
        "OUT_DIR": attn_out,
        "DEVICE": device,
        "DOWNSTREAM_RULE": "hop_gt_0",
        "HIST_DPI": 200,
        "SAVE_HIST_COMBINED_GRID": False,
        "TOP_K_WORST_BUSES": 20 if SKIP_WORST_BUS_PLOTS else 1000,
        "N_ROWS_PRINT_R2_TABLE": 10 if SKIP_WORST_BUS_PLOTS else 30,
    }
    if edges_csv is not None:
        knobs["EDGES_CSV"] = edges_csv
    if hop_csv is not None:
        knobs["HOP_CSV"] = hop_csv
    g = {"__name__": "__main__", "__NOTEBOOK_KNOBS__": knobs}
    cell_path = repo / "notebook_cell_attention_aux_voltage_nodes.py"
    print(f"[aux_paper] exec {cell_path.name} …", flush=True)
    exec(compile(cell_path.read_text(encoding="utf-8"), str(cell_path), "exec"), g)

    # pull results from exec namespace
    reg_df = g.get("reg_df", pd.DataFrame())
    cap_df = g.get("cap_df", pd.DataFrame())
    aux_meta = g.get("aux_meta", {})
    meta_aux_df = g.get("meta_aux_df", pd.DataFrame())
    ratios = g.get("ratios", pd.DataFrame())
    mh = g.get("mh")
    res = g.get("res")
    hop_df = g.get("hop_df")
    _n_avg = int(g.get("_n_avg", 0) or 0)

    _print_paper_table(aux_meta, reg_df, cap_df, meta_aux_df)

    # ----- 3) hop-distance ratio curve (paper fig token_attention panel b) -----
    if mh is not None and res is not None and hop_df is not None:
        from da_gps_hop_attention_ratios import attention_ratio_vs_hop_distance

        man = res["manifest"]
        hop_vs = attention_ratio_vs_hop_distance(
            mh,
            reg_target_cols=list(man["reg_target_cols"]),
            n_cap=int(res["n_cap"]),
            node_names=list(man["node_names"]),
            hop_df=hop_df,
            layer=None,
            direction="node_to_token",
        )
        hop_csv_out = attn_out / f"attention_ratio_vs_hop_avg{_n_avg}.csv"
        hop_vs.to_csv(hop_csv_out, index=False)
        print(f"[aux_paper] wrote {hop_csv_out}", flush=True)

        if len(ratios):
            last = int(ratios["layer"].max())
            r_last = ratios[ratios["layer"] == last]
            mean_r = float(r_last["ratio"].mean())
            n_gt1 = int((r_last["ratio"] > 1.0).sum())
            n_tok = int(len(r_last))
            print(
                f"[aux_paper] Final-layer node→token: mean downstream/non-downstream ratio={mean_r:.4f}; "
                f"{n_gt1}/{n_tok} tokens with ratio>1",
                flush=True,
            )

        if len(hop_vs):
            last_l = int(hop_vs["layer"].max())
            sub = hop_vs[hop_vs["layer"] == last_l]
            by_h = sub.groupby("hop", as_index=True)["ratio"].mean()
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
            # (a) one representative regulator attention mass vs hop (final layer)
            ax = axes[0]
            reg0 = str(REG_FIG_COL) if REG_FIG_COL in set(sub["reg_col"]) else str(sub["reg_col"].iloc[0])
            s0 = sub[sub["reg_col"] == reg0].sort_values("hop")
            ax.plot(s0["hop"], s0["mu_at_hop"], marker="o", ms=4, label="downstream hop h")
            ax.axhline(float(s0["mu_other"].iloc[0]), color="k", ls="--", lw=1, label="non-downstream mean")
            ax.set_xlabel("Hop distance from regulator")
            ax.set_ylabel("Mean node→token attention")
            ax.set_title(f"(a) {reg0.replace('reg_', '').replace('_tap_pu', '')} (layer {last_l})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            # (b) ratio vs hop averaged over 12 regulators
            ax = axes[1]
            ax.plot(by_h.index.to_numpy(), by_h.to_numpy(), marker="o", ms=4, color="C1")
            ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
            ax.set_xlabel("Hop distance from regulator")
            ax.set_ylabel("Downstream / non-downstream ratio")
            ax.set_title(f"(b) Mean over {sub['reg_col'].nunique()} regulator tokens (layer {last_l})")
            ax.grid(True, alpha=0.3)
            fig_path = out_dir / "token_attention.png"
            fig.savefig(fig_path, dpi=200)
            fig.savefig(fig_path.with_suffix(".pdf"))
            plt.close(fig)
            print(f"[aux_paper] wrote {fig_path}", flush=True)
            # prose helpers
            near = by_h[by_h.index <= 5]
            far = by_h[by_h.index >= 15]
            if len(near):
                print(
                    f"[aux_paper] ratio stays above {float(near.min()):.3f} within 5 hops; "
                    f"mean at hops≥15 = {float(far.mean()) if len(far) else float('nan'):.3f}",
                    flush=True,
                )

    summary = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt),
        "cache_pt": str(cache_pt),
        "out_dir": str(out_dir),
        "n_samples_avg": _n_avg,
        "aux_meta": aux_meta,
        "device_day": str(day_out) if day_out is not None else None,
    }
    (out_dir / "auxiliary_performance_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"[aux_paper] summary -> {out_dir / 'auxiliary_performance_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
