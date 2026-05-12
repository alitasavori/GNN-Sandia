# Full notebook cell: aux + reg bars + per-bus voltage rank + attention / hop / hist.
# Source of truth: this file (copy entire file into one Jupyter cell, or %run it).

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2").resolve()
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

# --- edit ---
RUN_DIR = REPO_ROOT / r"gnn2_architecture_search\attention checkpoints\da_gps_chunked_l4_mvagg_20260510_134709"
CACHE_PT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\run_001_scen_0000_0049_seed_20360133__full.pt")
EDGES_CSV = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\gnn_edges_phase_static.csv")
HOP_CSV = REPO_ROOT / r"datasets_gnn2_from pc\load_hop_distance_to_each_regulator_all_index_nodes.csv"
OUT_DIR = RUN_DIR / "attention_extract"
SAMPLE_IDX_START = 0
N_SAMPLES_AVG = 100
TOP_K_NODE_PLOT = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNSTREAM_RULE = "hop_gt_0"
# import matplotlib
# matplotlib.use("Agg")
# ------------

print("DEVICE =", DEVICE)
ckpt = RUN_DIR / "da_gps_multitask_best.pt"

_z = torch.load(CACHE_PT, map_location="cpu", weights_only=False)
_n_cache = int(_z["x"].shape[0])
_n_avg = max(1, min(int(N_SAMPLES_AVG), _n_cache))
_sample_indices = [SAMPLE_IDX_START + k for k in range(_n_avg)]
if _sample_indices[-1] >= _n_cache:
    raise IndexError(f"sample range {_sample_indices[0]}..{_sample_indices[-1]} out of cache [0,{_n_cache})")

OUT_DIR.mkdir(parents=True, exist_ok=True)
hop_df = load_hop_frame(HOP_CSV)

# ----- same cache rows: per-regulator MAE/MSE/RMS in tap pu + per-cap BCE & accuracy -----
reg_df, cap_df, aux_meta = eval_aux_per_device_on_cache_indices(
    ckpt,
    RUN_DIR,
    CACHE_PT,
    EDGES_CSV,
    _sample_indices,
    device=DEVICE,
    dropout=0.0,
)
reg_df.to_csv(OUT_DIR / f"aux_reg_per_device_avg{_n_avg}.csv", index=False)
cap_df.to_csv(OUT_DIR / f"aux_cap_per_device_avg{_n_avg}.csv", index=False)
(OUT_DIR / f"aux_metrics_meta_avg{_n_avg}.json").write_text(
    json.dumps(aux_meta, indent=2), encoding="utf-8"
)
print(f"wrote {OUT_DIR / f'aux_reg_per_device_avg{_n_avg}.csv'}")
print(f"wrote {OUT_DIR / f'aux_cap_per_device_avg{_n_avg}.csv'}")
print(f"wrote {OUT_DIR / f'aux_metrics_meta_avg{_n_avg}.json'}")
print(f"aggregate reg_mse_tap_pu_all={aux_meta['reg_mse_tap_pu_all']:.8f}  cap_bce_all={aux_meta['cap_bce_all']:.6f}")
print("\n--- per-regulator (tap pu) ---")
print(reg_df.to_string(index=False))
print("\n--- per-cap ---")
print(cap_df.to_string(index=False))


def _short_reg_col(name: str) -> str:
    return str(name).replace("reg_", "").replace("_tap_pu", "")


# ----- regulator metrics: horizontal bars (MAE, RMSE, MSE in tap pu) -----
if len(reg_df):
    fig_reg, axes_reg = plt.subplots(1, 3, figsize=(15, max(4.0, 0.22 * len(reg_df))), constrained_layout=True)
    fig_reg.suptitle(
        f"Regulator tap vs cache (n={_n_avg} rows)  |  agg MSE(pu)={aux_meta['reg_mse_tap_pu_all']:.6e}",
        fontsize=11,
    )

    r_mae = reg_df.sort_values("mae_tap_pu", ascending=True)
    axes_reg[0].barh([_short_reg_col(c) for c in r_mae["reg_col"]], r_mae["mae_tap_pu"], color="steelblue")
    axes_reg[0].set_xlabel("MAE (tap pu)")
    axes_reg[0].set_title("Mean absolute error")
    axes_reg[0].grid(True, axis="x", alpha=0.3)

    r_rmse = reg_df.sort_values("rmse_tap_pu", ascending=True)
    axes_reg[1].barh([_short_reg_col(c) for c in r_rmse["reg_col"]], r_rmse["rmse_tap_pu"], color="darkorange")
    axes_reg[1].set_xlabel("RMSE (tap pu)")
    axes_reg[1].set_title("Root mean square error")
    axes_reg[1].grid(True, axis="x", alpha=0.3)

    r_mse = reg_df.sort_values("mse_tap_pu", ascending=True)
    axes_reg[2].barh([_short_reg_col(c) for c in r_mse["reg_col"]], r_mse["mse_tap_pu"], color="seagreen")
    axes_reg[2].set_xlabel("MSE (tap pu²)")
    axes_reg[2].set_title("Mean square error")
    axes_reg[2].grid(True, axis="x", alpha=0.3)

    _reg_png = OUT_DIR / f"aux_reg_metrics_bar_avg{_n_avg}.png"
    fig_reg.savefig(_reg_png, dpi=150)
    plt.show()
    print(f"wrote {_reg_png}")

# ----- per-bus voltage errors (mean over same samples); rank all nodes -----
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
    f"worst bus (mean |V| MAE over n={_n_avg}): {node_err_meta['worst_node']}  "
    f"mean_mae_vmag_pu={node_err_meta['worst_mean_mae_vmag_pu']:.6f}"
)

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

worst = node_err_df.head(TOP_K_NODE_PLOT).iloc[::-1]
fig_top, ax_top = plt.subplots(figsize=(10, max(5.0, 0.22 * TOP_K_NODE_PLOT)), constrained_layout=True)
ax_top.barh(worst["node"].astype(str), worst["mean_mae_vmag_pu"], color="crimson", alpha=0.85)
ax_top.set_xlabel("mean |V| MAE (pu)")
ax_top.set_title(f"Top {TOP_K_NODE_PLOT} worst buses (mean over {_n_avg} samples)")
ax_top.grid(True, axis="x", alpha=0.3)
_fig_top = OUT_DIR / f"voltage_node_worst_{TOP_K_NODE_PLOT}_avg{_n_avg}.png"
fig_top.savefig(_fig_top, dpi=150)
plt.show()
print(f"wrote {_fig_top}")

# ----- worst buses: regulators they are strictly downstream of (hop CSV) + heatmap -----
worst_raw = node_err_df.head(TOP_K_NODE_PLOT).copy()
worst_ann, hop_sub_worst = worst_nodes_downstream_regulator_table(
    worst_raw,
    hop_df,
    node_err_meta["node_names"],
    node_err_meta["reg_target_cols"],
    reg_col_to_hop_col=REG_COL_TO_HOP_COL,
    downstream_rule=DOWNSTREAM_RULE,
)
_csv_w = OUT_DIR / f"voltage_worst_{TOP_K_NODE_PLOT}_with_downstream_regs_avg{_n_avg}.csv"
worst_ann.to_csv(_csv_w, index=False)
print(f"wrote {_csv_w}")

M = np.ma.masked_invalid(hop_sub_worst.to_numpy(dtype=float))
fig_hm, ax_hm = plt.subplots(
    figsize=(max(10.0, 0.42 * M.shape[1]), max(5.5, 0.2 * M.shape[0])),
    constrained_layout=True,
)
im = ax_hm.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0)
ax_hm.set_xticks(np.arange(M.shape[1]))
ax_hm.set_xticklabels(list(hop_sub_worst.columns), rotation=55, ha="right", fontsize=8)
ax_hm.set_yticks(np.arange(M.shape[0]))
_labels_y = [f"#{int(r['rank_vmag'])}  {r['node']}" for _, r in worst_raw.iterrows()]
ax_hm.set_yticklabels(_labels_y, fontsize=7)
ax_hm.set_xlabel("regulator (hop CSV)")
ax_hm.set_ylabel("worst buses by mean |V| MAE")
fig_hm.colorbar(im, ax=ax_hm, label="hop count (0 = not downstream for hop>0 rule)")
fig_hm.suptitle(
    f"Hop distance to each regulator — top {TOP_K_NODE_PLOT} worst voltage buses\n"
    f"Same downstream rule as attention ratios: {DOWNSTREAM_RULE!r}",
    fontsize=10,
)
_png_hm = OUT_DIR / f"voltage_worst_{TOP_K_NODE_PLOT}_hop_heatmap_avg{_n_avg}.png"
fig_hm.savefig(_png_hm, dpi=150)
plt.show()
print(f"wrote {_png_hm}")

_ypos = np.arange(len(worst_ann))
fig_nd, ax_nd = plt.subplots(figsize=(10, max(4.0, 0.18 * TOP_K_NODE_PLOT)), constrained_layout=True)
ax_nd.barh(_ypos, worst_ann["n_downstream_regs"].to_numpy(), color="teal", alpha=0.85)
ax_nd.set_yticks(_ypos)
ax_nd.set_yticklabels(_labels_y, fontsize=7)
ax_nd.invert_yaxis()
ax_nd.set_xlabel("number of regulators with hop>0 (strictly downstream)")
ax_nd.set_title(f"Downstream regulator count (same {DOWNSTREAM_RULE!r} as hop ratios)")
_png_nd = OUT_DIR / f"voltage_worst_{TOP_K_NODE_PLOT}_n_downstream_bar_avg{_n_avg}.png"
fig_nd.savefig(_png_nd, dpi=150)
plt.show()
print(f"wrote {_png_nd}")

# ----- mean attention over same indices -----
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
print(f"wrote {OUT_DIR / f'manifest_attention_avg{_n_avg}.json'}")

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
print(f"averaged over n={_n_avg} cache rows (indices {_sample_indices[0]}..{_sample_indices[-1]})")
print(f"out_dir={OUT_DIR}")
print(f"wrote {out_csv}")
print(f"wrote {out_csv_tn}")
print(f"node→token tensors: layers={L}  nodes={N}  tokens={T}")
print(f"token→node tensors: layers={Lt}  tokens={Tt}  nodes={Nt}  regulators={len(reg_cols)}")
print(f"downstream_rule={DOWNSTREAM_RULE!r}")

_print_ratio_block(ratios, "Hop ratios: node → token (each node attends regulators)")
_print_ratio_block(ratios_tn, "Hop ratios: token → node (each regulator token attends nodes)")

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
print(f"wrote {OUT_DIR / 'attention_hop_hist_all_layers_regs_nt.png'}")
print(f"wrote {OUT_DIR / 'attention_hop_hist_all_layers_regs_tn.png'}")

try:
    from IPython.display import display

    print("\n--- worst buses + downstream regs (head) ---")
    display(worst_ann.head(15))
    print("\n--- worst buses voltage only (head) ---")
    display(node_err_df.head(20))
    print("\n--- node→token ratios (head) ---")
    display(ratios.head(24))
    print("\n--- token→node ratios (head) ---")
    display(ratios_tn.head(24))
except Exception:
    print(worst_ann.head(15).to_string())
    print(node_err_df.head(20).to_string())
    print(ratios.head(24).to_string())
    print(ratios_tn.head(24).to_string())
