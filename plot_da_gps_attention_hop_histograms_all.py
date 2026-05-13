"""
Plot downstream vs other attention histograms for every regulator and every layer.

Run after attention extract (same paths / tensors as your notebook snippet), or import
``plot_all_regulator_layer_hop_histograms`` and call with mh, mhtn, etc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from da_gps_hop_attention_ratios import (
    REG_COL_TO_HOP_COL,
    downstream_mask,
    hops_for_manifest_nodes,
    load_hop_frame,
    non_downstream_catalog_mask,
)


def _as_lnt(x: object) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]
    return x


def _as_ltn(x: object) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]
    return x


def plot_all_regulator_layer_hop_histograms(
    mh: np.ndarray | object,
    mhtn: np.ndarray | object,
    *,
    reg_target_cols: list[str],
    reg_col_to_hop_col: dict[str, str],
    n_cap: int,
    node_names: list[str],
    hop_df: pd.DataFrame,
    downstream_rule: str = "hop_gt_0",
    downstream_label: str | None = None,
    non_downstream_label: str | None = None,
    suptitle_suffix: str = "",
    out_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """
    mh: (L, N, T) node→token mean over heads.
    mhtn: (L, T, N) token→node mean over heads.

    One figure: rows = layers, cols = regulators (only cols with hop mapping).
    Second figure: same for token→node.
    """
    mh = _as_lnt(mh)
    mhtn = _as_ltn(mhtn)
    L, N, T = mh.shape
    Lt, Tt, Nt = mhtn.shape
    if L != Lt or N != Nt or T != Tt:
        raise ValueError(f"shape mismatch nt={mh.shape} tn={mhtn.shape}")
    if N != len(node_names):
        raise ValueError(f"N={N} vs len(node_names)={len(node_names)}")

    regs = [c for c in reg_target_cols if reg_col_to_hop_col.get(c)]
    if not regs:
        raise ValueError("No regulator columns with hop CSV mapping")

    lab_d = downstream_label or f"downstream ({downstream_rule})"
    lab_o = non_downstream_label or (
        "non-downstream (hop≤0): subst / outside subtree / other laterals per CSV"
    )

    n_r = len(regs)
    fig_w = 3.0 * n_r
    fig_h = 2.4 * L

    def _one_figure(weights_fn, title_prefix: str, fname: str) -> None:
        fig, axes = plt.subplots(L, n_r, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
        for j, reg_col in enumerate(regs):
            hop_col = reg_col_to_hop_col[reg_col]
            tok = n_cap + int(reg_target_cols.index(reg_col))
            hvec, _ = hops_for_manifest_nodes(hop_df, list(node_names), hop_col)
            dmask = downstream_mask(hvec, rule=downstream_rule)
            omask = non_downstream_catalog_mask(hvec, rule=downstream_rule)
            lab_short = reg_col.replace("reg_", "").replace("_tap_pu", "")
            for ell in range(L):
                ax = axes[ell, j]
                w = weights_fn(ell, tok)
                if np.any(dmask) and np.any(omask):
                    ax.hist(
                        w[dmask],
                        bins=40,
                        alpha=0.65,
                        label=lab_d,
                        color="C0",
                        density=True,
                    )
                    ax.hist(w[omask], bins=40, alpha=0.65, label=lab_o, color="C1", density=True)
                elif np.any(dmask):
                    ax.hist(w[dmask], bins=40, alpha=0.65, color="C0", density=True)
                elif np.any(omask):
                    ax.hist(w[omask], bins=40, alpha=0.65, color="C1", density=True)
                ax.set_title(f"L{ell} {lab_short}", fontsize=8)
                if j == 0:
                    ax.set_ylabel("density", fontsize=8)
                if ell == L - 1:
                    ax.set_xlabel("attn", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                if ell == L - 1 and (np.any(dmask) and np.any(omask)):
                    ax.legend(fontsize=6, loc="upper right")
        suf = f" {suptitle_suffix}".rstrip() if suptitle_suffix else ""
        fig.suptitle(f"{title_prefix} ({downstream_rule}){suf}", fontsize=11)
        if out_dir is not None:
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            fig.savefig(p / fname, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    _one_figure(
        lambda ell, tok: mh[ell, :, tok],
        "Node→token (mean heads): weight on regulator token",
        "attention_hop_hist_all_layers_regs_nt.png",
    )
    _one_figure(
        lambda ell, tok: mhtn[ell, tok, :],
        "Token→node (mean heads): regulator token over nodes",
        "attention_hop_hist_all_layers_regs_tn.png",
    )


# --- runnable example: same defaults as your notebook snippet ---
if __name__ == "__main__":
    import os
    import sys

    import torch

    REPO_ROOT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2").resolve()
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from extract_da_gps_attention import run_attention_extract

    RUN_DIR = REPO_ROOT / r"gnn2_architecture_search\attention checkpoints\da_gps_chunked_l4_mvagg_20260510_134709"
    CACHE_PT = Path(
        r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\run_001_scen_0000_0049_seed_20360133__full.pt"
    )
    EDGES_CSV = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2\datasets_gnn2_from pc\gnn_edges_phase_static.csv")
    HOP_CSV = REPO_ROOT / r"datasets_gnn2_from pc\load_hop_distance_to_each_regulator_all_index_nodes.csv"
    OUT_DIR = RUN_DIR / "attention_extract"
    ckpt = RUN_DIR / "da_gps_multitask_best.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DOWNSTREAM_RULE = "hop_gt_0"

    hop_df = load_hop_frame(HOP_CSV)

    res = run_attention_extract(
        ckpt,
        RUN_DIR,
        CACHE_PT,
        EDGES_CSV,
        sample_idx=0,
        sample_id=None,
        out_dir=OUT_DIR,
        device=DEVICE,
        dropout=0.0,
        head_mean=True,
    )

    plot_all_regulator_layer_hop_histograms(
        res["mean_heads"],
        res["mean_heads_tn"],
        reg_target_cols=list(res["manifest"]["reg_target_cols"]),
        reg_col_to_hop_col=REG_COL_TO_HOP_COL,
        n_cap=int(res["n_cap"]),
        node_names=list(res["manifest"]["node_names"]),
        hop_df=hop_df,
        downstream_rule=DOWNSTREAM_RULE,
        out_dir=OUT_DIR,
        show=True,
    )
