"""
DA-GPS: mean attention over N cache rows + hop ratios + per-device reg/cap
+ **per-bus voltage error ranking** + **worst-bus downstream regulator** table/heatmap (same cache indices).

Requires ``extract_da_gps_attention`` helpers (reload after edits).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib

import da_gps_hop_attention_ratios
import extract_da_gps_attention

importlib.reload(extract_da_gps_attention)
importlib.reload(da_gps_hop_attention_ratios)

from da_gps_hop_attention_ratios import (
    REG_COL_TO_HOP_COL,
    attention_downstream_ratio_table,
    attention_downstream_ratio_table_tn,
    load_hop_frame,
    worst_nodes_downstream_regulator_table,
)
from extract_da_gps_attention import (
    eval_aux_per_device_on_cache_indices,
    eval_voltage_per_node_errors_on_cache_indices,
    run_attention_extract,
)
from plot_da_gps_attention_hop_histograms_all import plot_all_regulator_layer_hop_histograms


def _resolve_da_gps_ckpt(run_dir: Path, explicit) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CKPT_PATH not found: {p}")
        return p
    rd = Path(run_dir).resolve()
    best = rd / "da_gps_multitask_best.pt"
    last = rd / "training_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(
        f"No checkpoint in {rd}: expected {best.name} or {last.name} (or set CKPT_PATH)."
    )


# =============================================================================
# Knobs — edit here
# =============================================================================
# How many cache rows to average (max = cache size).
#   None → all rows from SAMPLE_IDX_START; int → at most that many consecutive rows.
N_SAMPLES_AVG = None

SAMPLE_IDX_START = 0

# Worst-bus plots / CSV / heatmap: at most this many buses (clipped to n_nodes).
TOP_K_WORST_BUSES = 1000

# Log: how many rows of the R²-sorted table (lowest R² first).
N_ROWS_PRINT_R2_TABLE = 30

TOP_K_PLOT_H_CAP_IN = 42.0

# --- paths ---
RUN_DIR = REPO_ROOT / r"gnn2_architecture_search\attention checkpoints\da_gps_chunked_l4_mvagg_20260510_134709"
CACHE_PT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\run_001_scen_0000_0049_seed_20360133__full.pt")
EDGES_CSV = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\gnn_edges_phase_static.csv")
HOP_CSV = REPO_ROOT / r"datasets_gnn2_from pc\load_hop_distance_to_each_regulator_all_index_nodes.csv"
OUT_DIR = RUN_DIR / "attention_extract"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNSTREAM_RULE = "hop_gt_0"
CKPT_PATH = None
# =============================================================================

print("DEVICE =", DEVICE)
ckpt = _resolve_da_gps_ckpt(RUN_DIR, CKPT_PATH)
print("ckpt =", ckpt)

_z = torch.load(CACHE_PT, map_location="cpu", weights_only=False)
_n_cache = int(_z["x"].shape[0])
if N_SAMPLES_AVG is None:
    _n_avg = _n_cache
else:
    _n_avg = max(1, min(int(N_SAMPLES_AVG), _n_cache))
_sample_indices = [SAMPLE_IDX_START + k for k in range(_n_avg)]
if not _sample_indices or _sample_indices[-1] >= _n_cache:
    raise IndexError(
        f"sample range {SAMPLE_IDX_START}..{SAMPLE_IDX_START + _n_avg - 1} out of cache [0,{_n_cache})"
    )
print(f"cache_pt={CACHE_PT}")
print(f"  total_samples_in_cache (x batch dim) = {_n_cache}")
print(
    f"  n_used_for_average = {_n_avg}  (indices {_sample_indices[0]}..{_sample_indices[-1]} inclusive)"
)
print(
    f"knobs: N_SAMPLES_AVG={N_SAMPLES_AVG!r} → n_used={_n_avg}  |  "
    f"TOP_K_WORST_BUSES={TOP_K_WORST_BUSES}  |  N_ROWS_PRINT_R2_TABLE={N_ROWS_PRINT_R2_TABLE}"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
hop_df = load_hop_frame(HOP_CSV)

# --- per-device reg (pu) + cap: same cache rows, model loaded once inside helper ---
reg_df, cap_df, aux_meta, meta_aux_df = eval_aux_per_device_on_cache_indices(
    ckpt,
    RUN_DIR,
    CACHE_PT,
    EDGES_CSV,
    _sample_indices,
    device=DEVICE,
    dropout=0.0,
)
reg_csv = OUT_DIR / f"aux_reg_per_device_avg{_n_avg}.csv"
cap_csv = OUT_DIR / f"aux_cap_per_device_avg{_n_avg}.csv"
reg_df.to_csv(reg_csv, index=False)
cap_df.to_csv(cap_csv, index=False)
(OUT_DIR / f"aux_metrics_meta_avg{_n_avg}.json").write_text(json.dumps(aux_meta, indent=2), encoding="utf-8")
if len(meta_aux_df):
    _mcsv = OUT_DIR / f"aux_meta_per_col_avg{_n_avg}.csv"
    meta_aux_df.to_csv(_mcsv, index=False)
    print(f"wrote {_mcsv}")
print(f"wrote {reg_csv}")
print(f"wrote {cap_csv}")
print(f"aggregate reg_mse_tap_pu={aux_meta['reg_mse_tap_pu_all']:.8f}  cap_bce_all={aux_meta['cap_bce_all']:.6f}")
print("\n--- per-regulator (tap pu) ---")
print(reg_df.to_string(index=False))
print("\n--- per-cap ---")
print(cap_df.to_string(index=False))


def _short_reg_col(name: str) -> str:
    return str(name).replace("reg_", "").replace("_tap_pu", "")


if len(reg_df):
    fig_reg, axes_reg = plt.subplots(1, 3, figsize=(15, max(4.0, 0.22 * len(reg_df))), constrained_layout=True)
    su = f"Regulator tap prediction vs cache targets (n={_n_avg} rows)\nagg MSE(pu)={aux_meta['reg_mse_tap_pu_all']:.6e}"
    fig_reg.suptitle(su, fontsize=11)

    r_mae = reg_df.sort_values("mae_tap_pu", ascending=True)
    ax0 = axes_reg[0]
    ax0.barh([_short_reg_col(c) for c in r_mae["reg_col"]], r_mae["mae_tap_pu"], color="steelblue")
    ax0.set_xlabel("MAE (tap pu)")
    ax0.set_title("Mean absolute error")

    r_rmse = reg_df.sort_values("rmse_tap_pu", ascending=True)
    ax1 = axes_reg[1]
    ax1.barh([_short_reg_col(c) for c in r_rmse["reg_col"]], r_rmse["rmse_tap_pu"], color="darkorange")
    ax1.set_xlabel("RMSE (tap pu)")
    ax1.set_title("Root mean square error")

    r_mse = reg_df.sort_values("mse_tap_pu", ascending=True)
    ax2 = axes_reg[2]
    ax2.barh([_short_reg_col(c) for c in r_mse["reg_col"]], r_mse["mse_tap_pu"], color="seagreen")
    ax2.set_xlabel("MSE (tap pu²)")
    ax2.set_title("Mean square error")

    for ax_ in axes_reg:
        ax_.grid(True, axis="x", alpha=0.3)
    fig_reg.savefig(OUT_DIR / f"aux_reg_metrics_bar_avg{_n_avg}.png", dpi=150)
    plt.show()
    print(f"wrote {OUT_DIR / f'aux_reg_metrics_bar_avg{_n_avg}.png'}")

# ----- per-node voltage errors (mean |V| MAE & angle MAE over same cache rows) -----
node_err_df, node_err_meta = eval_voltage_per_node_errors_on_cache_indices(
    ckpt,
    RUN_DIR,
    CACHE_PT,
    EDGES_CSV,
    _sample_indices,
    device=DEVICE,
    dropout=0.0,
)
_node_csv = OUT_DIR / f"voltage_node_errors_ranked_avg{_n_avg}.csv"
node_err_df.to_csv(_node_csv, index=False)
(OUT_DIR / f"voltage_node_errors_meta_avg{_n_avg}.json").write_text(
    json.dumps(node_err_meta, indent=2), encoding="utf-8"
)
print(f"wrote {_node_csv}")
print(
    f"voltage pooled over all nodes×cache rows (n={node_err_meta.get('n_points_vmag_finite_overlap', 0)}): "
    f"|V| MAE={node_err_meta.get('mae_global_vmag_pu', float('nan')):.6f} pu  "
    f"RMSE={node_err_meta.get('rmse_global_vmag_pu', float('nan')):.6f} pu  "
    f"R²={node_err_meta.get('r2_global_vmag_pu', float('nan')):.6f}"
)
print(
    f"angle pooled (circular MAE, naive linear R² vs true angle in deg): "
    f"MAE={node_err_meta.get('mae_global_angle_deg', float('nan')):.4f} deg  "
    f"R²={node_err_meta.get('r2_global_vang_deg_naive', float('nan')):.6f}"
)
print(
    f"worst bus (mean |V| MAE over n={_n_avg}): {node_err_meta['worst_node']}  "
    f"mean_mae_vmag_pu={node_err_meta['worst_mean_mae_vmag_pu']:.6f}"
)
_r2s = node_err_df["r2_vmag"].to_numpy(dtype=np.float64)
_r2ok = _r2s[np.isfinite(_r2s)]
print(
    f"|V| R² over nodes (finite {len(_r2ok)}/{node_err_meta['n_nodes']}): "
    f"mean={node_err_meta.get('r2_vmag_mean', float('nan')):.4f}  "
    f"median={node_err_meta.get('r2_vmag_median', float('nan')):.4f}  "
    f"min={node_err_meta.get('r2_vmag_min', float('nan')):.4f}  "
    f"max={node_err_meta.get('r2_vmag_max', float('nan')):.4f}"
)
print(
    f"worst |V| R² bus (finite): {node_err_meta.get('worst_r2_node', '')!r}  "
    f"r2_vmag={node_err_meta.get('worst_r2_vmag', float('nan')):.4f}"
)
_r2_sorted = (
    node_err_df.sort_values("r2_vmag", ascending=True, na_position="last").reset_index(drop=True)
)
_r2_sorted.insert(0, "rank_r2_table", np.arange(1, len(_r2_sorted) + 1, dtype=np.int32))
print(
    "--- lowest R² buses (first {0} rows; rank_r2_table 1 = lowest R²) ---".format(
        int(N_ROWS_PRINT_R2_TABLE)
    )
)
print(_r2_sorted.head(int(N_ROWS_PRINT_R2_TABLE)).to_string(index=False))

fig_nv, axes_nv = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes_nv[0].hist(node_err_df["mean_mae_vmag_pu"], bins=50, color="steelblue", alpha=0.85, edgecolor="white")
axes_nv[0].set_xlabel("mean |V| MAE (pu) per bus")
axes_nv[0].set_ylabel("count")
axes_nv[0].set_title(f"Distribution over {node_err_meta['n_nodes']} buses")
axes_nv[1].hist(node_err_df["mean_mae_angle_deg"], bins=50, color="darkorange", alpha=0.85, edgecolor="white")
axes_nv[1].set_xlabel("mean |Δangle| (deg) per bus")
axes_nv[1].set_ylabel("count")
axes_nv[1].set_title("Angle error distribution")
_fig_hist = OUT_DIR / f"voltage_node_error_histograms_avg{_n_avg}.png"
fig_nv.savefig(_fig_hist, dpi=150)
plt.show()
print(f"wrote {_fig_hist}")

fig_r2, ax_r2 = plt.subplots(figsize=(8, 4), constrained_layout=True)
if len(_r2ok):
    ax_r2.hist(_r2ok, bins=50, color="seagreen", alpha=0.85, edgecolor="white")
    ax_r2.axvline(float(np.mean(_r2ok)), color="k", ls="--", lw=1.2, label=f"mean={np.mean(_r2ok):.3f}")
    ax_r2.legend(loc="upper left", fontsize=8)
else:
    ax_r2.text(0.5, 0.5, "no finite r2_vmag", ha="center", va="center", transform=ax_r2.transAxes)
ax_r2.set_xlabel(r"per-node $R^2$ of $|V|$ over cache rows")
ax_r2.set_ylabel("count (buses)")
ax_r2.set_title(
    f"Distribution of |V| magnitude R² (n_nodes finite={node_err_meta.get('n_nodes_r2_vmag_finite', 0)}, "
    f"n_samples={_n_avg})"
)
_fig_r2 = OUT_DIR / f"voltage_node_r2_vmag_distribution_avg{_n_avg}.png"
fig_r2.savefig(_fig_r2, dpi=150)
plt.show()
print(f"wrote {_fig_r2}")

_sy = node_err_df["std_vmag_true_pu"].to_numpy(dtype=np.float64)
_r2col = node_err_df["r2_vmag"].to_numpy(dtype=np.float64)
_mae_col = node_err_df["mean_mae_vmag_pu"].to_numpy(dtype=np.float64)
_xlog = np.maximum(_sy, 1e-12)
mask_r2 = np.isfinite(_r2col)
fig_diag, axes_diag = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axd0, axd1 = axes_diag
axd0.scatter(_xlog[mask_r2], _r2col[mask_r2], s=10, alpha=0.35, c="tab:blue", edgecolors="none")
axd0.set_xscale("log")
axd0.set_xlabel(r"std($|V|_{\mathrm{true}}$) (pu) over cache rows [log scale]")
axd0.set_ylabel(r"$R^2$ of $|V|$ (finite only)")
axd0.set_title("small std + low R² → touchy metric; large std + low R² → hard / misfit")
axd0.axhline(0.0, color="k", lw=0.5, alpha=0.4)
axd0.grid(True, which="both", alpha=0.25)
axd1.scatter(_xlog, _mae_col, s=10, alpha=0.35, c="tab:orange", edgecolors="none")
axd1.set_xscale("log")
axd1.set_xlabel(r"std($|V|_{\mathrm{true}}$) (pu) [log scale]")
axd1.set_ylabel("mean |V| MAE (pu)")
axd1.set_title("MAE vs spread: small MAE at tiny std can still give odd R²")
axd1.grid(True, which="both", alpha=0.25)
_fig_diag = OUT_DIR / f"voltage_node_std_vmag_vs_r2_mae_avg{_n_avg}.png"
fig_diag.savefig(_fig_diag, dpi=150)
plt.show()
print(f"wrote {_fig_diag}")
_sy_fin = _sy[mask_r2]
_r2_fin = _r2col[mask_r2]
_low_thr = 1e-4
_touchy = int(np.sum((_sy_fin < _low_thr) & (_r2_fin < 0.5)))
_hardish = int(np.sum((_sy_fin >= _low_thr) & (_r2_fin < 0.5)))
print(
    f"std-vs-R² hint (finite R² only): n(sy<{_low_thr:g} pu & R²<0.5)={_touchy} "
    f"(metric touchy); n(sy>={_low_thr:g} & R²<0.5)={_hardish} (likely hard / misfit)"
)

_top_k = min(int(TOP_K_WORST_BUSES), len(node_err_df))
print(
    f"worst-bus analysis: TOP_K_WORST_BUSES={TOP_K_WORST_BUSES}, rows_used={_top_k} (n_nodes={len(node_err_df)})"
)
_h_bar = min(TOP_K_PLOT_H_CAP_IN, max(5.0, 0.06 * _top_k))
worst = node_err_df.head(_top_k).iloc[::-1]
fig_top, ax_top = plt.subplots(figsize=(10, _h_bar), constrained_layout=True)
ax_top.barh(worst["node"].astype(str), worst["mean_mae_vmag_pu"], color="crimson", alpha=0.85)
ax_top.set_xlabel("mean |V| MAE (pu)")
ax_top.set_title(f"Top {_top_k} worst buses (mean over {_n_avg} samples)")
ax_top.grid(True, axis="x", alpha=0.3)
_fig_top = OUT_DIR / f"voltage_node_worst_{_top_k}_avg{_n_avg}.png"
fig_top.savefig(_fig_top, dpi=150)
plt.show()
print(f"wrote {_fig_top}")

worst_raw = node_err_df.head(_top_k).copy()
worst_ann, hop_sub_worst = worst_nodes_downstream_regulator_table(
    worst_raw,
    hop_df,
    node_err_meta["node_names"],
    node_err_meta["reg_target_cols"],
    reg_col_to_hop_col=REG_COL_TO_HOP_COL,
    downstream_rule=DOWNSTREAM_RULE,
)
worst_ann.to_csv(OUT_DIR / f"voltage_worst_{_top_k}_with_downstream_regs_avg{_n_avg}.csv", index=False)
M = np.ma.masked_invalid(hop_sub_worst.to_numpy(dtype=float))
_h_hm = min(48.0, max(5.5, 0.038 * float(M.shape[0])))
fig_hm, ax_hm = plt.subplots(
    figsize=(max(10.0, 0.42 * M.shape[1]), _h_hm),
    constrained_layout=True,
)
im = ax_hm.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0)
ax_hm.set_xticks(np.arange(M.shape[1]))
ax_hm.set_xticklabels(list(hop_sub_worst.columns), rotation=55, ha="right", fontsize=8)
ax_hm.set_yticks(np.arange(M.shape[0]))
_labels_y = [f"#{int(r['rank_vmag'])}  {r['node']}" for _, r in worst_raw.iterrows()]
_ylab_fs = 6 if _top_k > 400 else 7
ax_hm.set_yticklabels(_labels_y, fontsize=_ylab_fs)
ax_hm.set_xlabel("regulator (hop CSV)")
ax_hm.set_ylabel("worst buses by mean |V| MAE")
fig_hm.colorbar(im, ax=ax_hm, label="hop count (0 = not downstream for hop>0 rule)")
fig_hm.suptitle(
    f"Hop distance to each regulator — top {_top_k} worst voltage buses ({DOWNSTREAM_RULE})",
    fontsize=10,
)
fig_hm.savefig(OUT_DIR / f"voltage_worst_{_top_k}_hop_heatmap_avg{_n_avg}.png", dpi=150)
plt.show()
print(f"wrote {OUT_DIR / f'voltage_worst_{_top_k}_hop_heatmap_avg{_n_avg}.png'}")

_ypos = np.arange(len(worst_ann))
fig_nd, ax_nd = plt.subplots(figsize=(10, _h_bar), constrained_layout=True)
ax_nd.barh(_ypos, worst_ann["n_downstream_regs"].to_numpy(), color="teal", alpha=0.85)
ax_nd.set_yticks(_ypos)
ax_nd.set_yticklabels(_labels_y, fontsize=_ylab_fs)
ax_nd.invert_yaxis()
ax_nd.set_xlabel("number of regulators with hop>0 (strictly downstream)")
ax_nd.set_title(f"Downstream regulator count ({DOWNSTREAM_RULE})")
fig_nd.savefig(OUT_DIR / f"voltage_worst_{_top_k}_n_downstream_bar_avg{_n_avg}.png", dpi=150)
plt.show()
print(f"wrote {OUT_DIR / f'voltage_worst_{_top_k}_n_downstream_bar_avg{_n_avg}.png'}")

# --- attention mean over same indices ---
mh_acc = None
mhtn_acc = None
res = None
for si in _sample_indices:
    res = run_attention_extract(
        ckpt,
        RUN_DIR,
        CACHE_PT,
        EDGES_CSV,
        sample_idx=si,
        sample_id=None,
        out_dir=OUT_DIR,
        device=DEVICE,
        dropout=0.0,
        head_mean=True,
        save_outputs=False,
    )
    mh = np.asarray(res["mean_heads"], dtype=np.float64)
    mhtn = np.asarray(res["mean_heads_tn"], dtype=np.float64)
    while mh.ndim == 4 and mh.shape[1] == 1:
        mh = mh[:, 0, :, :]
    while mhtn.ndim == 4 and mhtn.shape[1] == 1:
        mhtn = mhtn[:, 0, :, :]
    if mh_acc is None:
        mh_acc = mh
        mhtn_acc = mhtn
    else:
        mh_acc += mh
        mhtn_acc += mhtn

mh = (mh_acc / float(_n_avg)).astype(np.float32)
mhtn = (mhtn_acc / float(_n_avg)).astype(np.float32)
L, N, T = mh.shape
Lt, Tt, Nt = mhtn.shape

np.savez_compressed(
    OUT_DIR / f"attention_mean_heads_avg{_n_avg}.npz",
    probs_nt=mh,
    probs_tn=mhtn,
    n_samples_averaged=np.int64(_n_avg),
    sample_indices=np.asarray(_sample_indices, dtype=np.int64),
)
_manifest_avg = {
    **res["manifest"],
    "attention_averaging": {
        "n_samples_averaged": _n_avg,
        "sample_indices": _sample_indices,
        "note": "mean_heads are arithmetic mean of head-mean attention over listed cache indices",
    },
}
(OUT_DIR / f"manifest_attention_avg{_n_avg}.json").write_text(
    json.dumps(_manifest_avg, indent=2), encoding="utf-8"
)
print(f"wrote {OUT_DIR / f'attention_mean_heads_avg{_n_avg}.npz'}")

ratios = attention_downstream_ratio_table(
    mh,
    reg_target_cols=list(res["manifest"]["reg_target_cols"]),
    n_cap=int(res["n_cap"]),
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
)
ratios_tn = attention_downstream_ratio_table_tn(
    mhtn,
    reg_target_cols=list(res["manifest"]["reg_target_cols"]),
    n_cap=int(res["n_cap"]),
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
)

out_csv = OUT_DIR / f"attention_hop_ratios_avg{_n_avg}.csv"
out_csv_tn = OUT_DIR / f"attention_hop_ratios_token_to_node_avg{_n_avg}.csv"
ratios.to_csv(out_csv, index=False)
ratios_tn.to_csv(out_csv_tn, index=False)

reg_cols = list(res["manifest"]["reg_target_cols"])
n_cap = int(res["n_cap"])


def _print_ratio_block(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    if len(df) == 0:
        print("  No ratio rows.")
        return
    last_layer = int(df["layer"].max())
    r_last = df[df["layer"] == last_layer]
    print(f"  Final layer ({last_layer}): median={r_last['ratio'].median():.4f}  mean={r_last['ratio'].mean():.4f}")
    print(f"  share ratio>1: {(r_last['ratio'] > 1).mean():.1%}  ratio>1.5: {(r_last['ratio'] > 1.5).mean():.1%}")
    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer]
        gt1 = (sub["ratio"] > 1).sum()
        print(f"  layer {int(layer)}: median={sub['ratio'].median():.4f}  count(ratio>1)={gt1}/{len(sub)}")


print("\n========== DA-GPS attention + hop ratios ==========")
print(
    f"averaged over n={_n_avg} of {_n_cache} cache rows (indices {_sample_indices[0]}..{_sample_indices[-1]})"
)
print(f"out_dir={OUT_DIR}")
print(f"node→token tensors: layers={L}  nodes={N}  tokens={T}")
print(f"downstream_rule={DOWNSTREAM_RULE!r}")

_print_ratio_block(ratios, "Hop ratios: node → token")
_print_ratio_block(ratios_tn, "Hop ratios: token → node")

_ratio_title = "downstream vs non-downstream (hop≤0)"
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

ax00 = axes[0, 0]
if len(ratios):
    for reg_col in reg_cols:
        sub = ratios[ratios["reg_col"] == reg_col]
        if len(sub) == 0:
            continue
        lab = reg_col.replace("reg_", "").replace("_tap_pu", "")
        ax00.plot(sub["layer"], sub["ratio"], marker="o", ms=4, label=lab)
    ax00.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax00.set_xlabel("GPS block (layer)")
ax00.set_ylabel("ratio")
ax00.set_title(f"Node → token: {_ratio_title} (mean over {_n_avg} samples)")
ax00.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax00.grid(True, alpha=0.3)

ax01 = axes[0, 1]
if len(ratios_tn):
    for reg_col in reg_cols:
        sub = ratios_tn[ratios_tn["reg_col"] == reg_col]
        if len(sub) == 0:
            continue
        lab = reg_col.replace("reg_", "").replace("_tap_pu", "")
        ax01.plot(sub["layer"], sub["ratio"], marker="o", ms=4, label=lab)
    ax01.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax01.set_xlabel("GPS block (layer)")
ax01.set_ylabel("ratio")
ax01.set_title(f"Token → node: {_ratio_title} (mean over {_n_avg} samples)")
ax01.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax01.grid(True, alpha=0.3)

ax10 = axes[1, 0]
_last_nt = "—"
if len(ratios):
    _last_nt = str(int(ratios["layer"].max()))
    r_last = ratios[ratios["layer"] == int(_last_nt)].sort_values("ratio", ascending=True)
    if len(r_last):
        labels = [c.replace("reg_", "").replace("_tap_pu", "") for c in r_last["reg_col"]]
        ax10.barh(labels, r_last["ratio"], color="steelblue")
        ax10.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax10.set_xlabel("ratio")
ax10.set_title(f"Node→token final layer ({_last_nt}), mean n={_n_avg}")

ax11 = axes[1, 1]
_last_tn = "—"
if len(ratios_tn):
    _last_tn = str(int(ratios_tn["layer"].max()))
    r2 = ratios_tn[ratios_tn["layer"] == int(_last_tn)].sort_values("ratio", ascending=True)
    if len(r2):
        labels2 = [c.replace("reg_", "").replace("_tap_pu", "") for c in r2["reg_col"]]
        ax11.barh(labels2, r2["ratio"], color="darkorange")
        ax11.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
ax11.set_xlabel("ratio")
ax11.set_title(f"Token→node final layer ({_last_tn}), mean n={_n_avg}")

plt.show()

_hist_suffix = f"mean over {_n_avg} cache samples"
plot_all_regulator_layer_hop_histograms(
    mh,
    mhtn,
    reg_target_cols=reg_cols,
    reg_col_to_hop_col=REG_COL_TO_HOP_COL,
    n_cap=n_cap,
    node_names=list(res["manifest"]["node_names"]),
    hop_df=hop_df,
    downstream_rule=DOWNSTREAM_RULE,
    downstream_label=f"downstream ({DOWNSTREAM_RULE})",
    non_downstream_label="non-downstream (hop≤0): ref bus / outside subtree / other laterals / per CSV",
    suptitle_suffix=_hist_suffix,
    out_dir=OUT_DIR,
    show=True,
)
