"""
Plot downstream vs other attention **distributions** (density histograms) per GPS layer and regulator.

Also saves optional summary ratio line/bar figures (same content as the notebook 2×2 panel).

See module docstring in ``da_gps_hop_attention_ratios`` for hop masks and ratio definitions.
"""
from __future__ import annotations

import json
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

# Short text for figure footers (slide-ready).
FORMULATION_NT = (
    "Node→token attention ᾱ_{ℓ,i→t} (mean over heads, cache samples). "
    "Split nodes i by hop h_{i,r} from the regulator subtree CSV: "
    "downstream = h>0; other = in MV catalog with h≤0 (ref bus, outside subtree, other laterals)."
)
FORMULATION_TN = (
    "Token→node attention ᾱ_{ℓ,t→i} (mean over heads, cache samples). "
    "Same hop split per regulator token t."
)
FORMULATION_RATIO_NT = (
    "Per layer ℓ and regulator token: "
    "R_{ℓ,r} = mean_{i∈𝒟_r} ᾱ_{ℓ,i→t(r)} / (mean_{i∈𝒪_r} ᾱ_{ℓ,i→t(r)} + ε) "
    "with 𝒟_r downstream (hop>0), 𝒪_r other in-catalog (hop≤0)."
)
FORMULATION_RATIO_TN = (
    "Per layer ℓ: "
    "R_{ℓ,r} = mean_{i∈𝒟_r} ᾱ_{ℓ,t(r)→i} / (mean_{i∈𝒪_r} ᾱ_{ℓ,t(r)→i} + ε)."
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


def _reg_short(reg_col: str) -> str:
    return reg_col.replace("reg_", "").replace("_tap_pu", "")


def _plot_single_hist(
    ax: plt.Axes,
    w: np.ndarray,
    dmask: np.ndarray,
    omask: np.ndarray,
    *,
    lab_d: str,
    lab_o: str,
    bins: int = 40,
) -> None:
    if np.any(dmask):
        ax.hist(w[dmask], bins=bins, alpha=0.65, label=lab_d, color="C0", density=True)
    if np.any(omask):
        ax.hist(w[omask], bins=bins, alpha=0.65, label=lab_o, color="C1", density=True)
    ax.set_ylabel("density")
    ax.set_xlabel("attention weight")
    ax.grid(True, alpha=0.3)
    if np.any(dmask) and np.any(omask):
        ax.legend(fontsize=8, loc="upper right")


def _save_one_panel(
    w: np.ndarray,
    dmask: np.ndarray,
    omask: np.ndarray,
    *,
    title: str,
    subtitle: str,
    lab_d: str,
    lab_o: str,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    _plot_single_hist(ax, w, dmask, omask, lab_d=lab_d, lab_o=lab_o)
    ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=7.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


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
    show: bool = False,
    save_separate: bool = True,
    save_combined_grid: bool = False,
    hist_dpi: int = 300,
    panel_figsize: tuple[float, float] = (5.5, 4.0),
    hist_bins: int = 40,
) -> list[Path]:
    """
    mh: (L, N, T) node→token mean over heads.
    mhtn: (L, T, N) token→node mean over heads.

    By default writes **one PNG per (layer, regulator, direction)** under::

        {out_dir}/hop_hist_nt/L{ell:02d}_{reg}_node_to_token.png
        {out_dir}/hop_hist_tn/L{ell:02d}_{reg}_token_to_node.png

    Set ``save_combined_grid=True`` to also emit the legacy L×R mosaic PNGs.
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
    lab_o = non_downstream_label or "other in-catalog (hop≤0)"
    suf = f" — {suptitle_suffix}" if suptitle_suffix else ""
    saved: list[Path] = []
    out_root = Path(out_dir).resolve() if out_dir is not None else None

    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for reg_col in regs:
        hop_col = reg_col_to_hop_col[reg_col]
        hvec, _ = hops_for_manifest_nodes(hop_df, list(node_names), hop_col)
        masks[reg_col] = (
            downstream_mask(hvec, rule=downstream_rule),
            non_downstream_catalog_mask(hvec, rule=downstream_rule),
        )

    if out_root is not None and save_separate:
        sub_nt = out_root / "hop_hist_nt"
        sub_tn = out_root / "hop_hist_tn"
        for ell in range(L):
            for reg_col in regs:
                rs = _reg_short(reg_col)
                tok = n_cap + int(reg_target_cols.index(reg_col))
                dmask, omask = masks[reg_col]
                w_nt = mh[ell, :, tok]
                p_nt = sub_nt / f"L{ell:02d}_{rs}_node_to_token.png"
                _save_one_panel(
                    w_nt,
                    dmask,
                    omask,
                    title=f"Layer {ell} · {rs} · node→token{suf}",
                    subtitle=FORMULATION_NT,
                    lab_d=lab_d,
                    lab_o=lab_o,
                    out_path=p_nt,
                    dpi=hist_dpi,
                    figsize=panel_figsize,
                )
                saved.append(p_nt)
                w_tn = mhtn[ell, tok, :]
                p_tn = sub_tn / f"L{ell:02d}_{rs}_token_to_node.png"
                _save_one_panel(
                    w_tn,
                    dmask,
                    omask,
                    title=f"Layer {ell} · {rs} · token→node{suf}",
                    subtitle=FORMULATION_TN,
                    lab_d=lab_d,
                    lab_o=lab_o,
                    out_path=p_tn,
                    dpi=hist_dpi,
                    figsize=panel_figsize,
                )
                saved.append(p_tn)
        manifest = {
            "downstream_rule": downstream_rule,
            "n_layers": L,
            "n_regulators": len(regs),
            "regulators": [_reg_short(c) for c in regs],
            "formulation_nt": FORMULATION_NT,
            "formulation_tn": FORMULATION_TN,
            "files": [str(p.relative_to(out_root)) for p in saved],
        }
        (out_root / "hop_hist_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    if out_root is not None and save_combined_grid:
        n_r = len(regs)
        fig_w, fig_h = 3.0 * n_r, 2.4 * L

        def _one_grid(weights_fn, title_prefix: str, fname: str) -> None:
            fig, axes = plt.subplots(L, n_r, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
            for j, reg_col in enumerate(regs):
                dmask, omask = masks[reg_col]
                rs = _reg_short(reg_col)
                tok = n_cap + int(reg_target_cols.index(reg_col))
                for ell in range(L):
                    ax = axes[ell, j]
                    _plot_single_hist(ax, weights_fn(ell, tok), dmask, omask, lab_d=lab_d, lab_o=lab_o, bins=hist_bins)
                    ax.set_title(f"L{ell} {rs}", fontsize=8)
                    if j == 0:
                        ax.set_ylabel("density", fontsize=8)
                    if ell == L - 1:
                        ax.set_xlabel("attn", fontsize=8)
                    ax.tick_params(axis="both", labelsize=7)
            fig.suptitle(f"{title_prefix} ({downstream_rule}){suf}", fontsize=11)
            p = out_root / fname
            fig.savefig(p, dpi=min(int(hist_dpi), 200), bbox_inches="tight")
            saved.append(p)
            if show:
                plt.show()
            else:
                plt.close(fig)

        _one_grid(lambda ell, tok: mh[ell, :, tok], "Node→token (mean heads)", "attention_hop_hist_all_layers_regs_nt.png")
        _one_grid(lambda ell, tok: mhtn[ell, tok, :], "Token→node (mean heads)", "attention_hop_hist_all_layers_regs_tn.png")

    return saved


def save_attention_ratio_summary_figures(
    ratios: pd.DataFrame,
    ratios_tn: pd.DataFrame,
    *,
    reg_cols: list[str],
    n_avg: int,
    out_dir: Path | str,
    dpi: int = 300,
    show: bool = False,
) -> list[Path]:
    """Save the four panels from the notebook 2×2 ratio figure as separate high-res PNGs."""
    out_root = Path(out_dir).resolve()
    sub = out_root / "attention_ratio_summary"
    sub.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    ratio_title = "downstream vs other (hop≤0)"

    # 1) node→token ratio vs layer
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(ratios):
        for reg_col in reg_cols:
            subdf = ratios[ratios["reg_col"] == reg_col]
            if len(subdf) == 0:
                continue
            ax.plot(
                subdf["layer"],
                subdf["ratio"],
                marker="o",
                ms=4,
                label=_reg_short(reg_col),
            )
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("GPS block (layer)")
    ax.set_ylabel("ratio")
    ax.set_title(f"Node → token: {ratio_title}\n(mean over {n_avg} samples)")
    fig.text(0.5, 0.01, FORMULATION_RATIO_NT, ha="center", fontsize=8, color="0.35")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    p = sub / "ratio_nt_vs_layer.png"
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 2) token→node ratio vs layer
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(ratios_tn):
        for reg_col in reg_cols:
            subdf = ratios_tn[ratios_tn["reg_col"] == reg_col]
            if len(subdf) == 0:
                continue
            ax.plot(
                subdf["layer"],
                subdf["ratio"],
                marker="o",
                ms=4,
                label=_reg_short(reg_col),
            )
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("GPS block (layer)")
    ax.set_ylabel("ratio")
    ax.set_title(f"Token → node: {ratio_title}\n(mean over {n_avg} samples)")
    fig.text(0.5, 0.01, FORMULATION_RATIO_TN, ha="center", fontsize=8, color="0.35")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    p = sub / "ratio_tn_vs_layer.png"
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 3) final layer bar — node→token
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(reg_cols))))
    _last_nt = "—"
    if len(ratios):
        _last_nt = str(int(ratios["layer"].max()))
        r_last = ratios[ratios["layer"] == int(_last_nt)].sort_values("ratio", ascending=True)
        if len(r_last):
            labels = [_reg_short(c) for c in r_last["reg_col"]]
            ax.barh(labels, r_last["ratio"], color="steelblue")
            ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("ratio")
    ax.set_title(f"Node→token final layer ({_last_nt}), mean n={n_avg}")
    p = sub / f"ratio_nt_final_layer_{_last_nt}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 4) final layer bar — token→node
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(reg_cols))))
    _last_tn = "—"
    if len(ratios_tn):
        _last_tn = str(int(ratios_tn["layer"].max()))
        r2 = ratios_tn[ratios_tn["layer"] == int(_last_tn)].sort_values("ratio", ascending=True)
        if len(r2):
            labels2 = [_reg_short(c) for c in r2["reg_col"]]
            ax.barh(labels2, r2["ratio"], color="darkorange")
            ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("ratio")
    ax.set_title(f"Token→node final layer ({_last_tn}), mean n={n_avg}")
    p = sub / f"ratio_tn_final_layer_{_last_tn}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    if show:
        plt.show()
    return saved


# --- runnable example ---
if __name__ == "__main__":
    import os
    import sys

    import torch

    REPO_ROOT = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2").resolve()
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from extract_da_gps_attention import run_attention_extract

    RUN_DIR = REPO_ROOT / r"gnn2_architecture_search\attention checkpoints\da_gps_chunked_l4_mvagg_gine_metaaux_20260510_134709"
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
    paths = plot_all_regulator_layer_hop_histograms(
        res["mean_heads"],
        res["mean_heads_tn"],
        reg_target_cols=list(res["manifest"]["reg_target_cols"]),
        reg_col_to_hop_col=REG_COL_TO_HOP_COL,
        n_cap=int(res["n_cap"]),
        node_names=list(res["manifest"]["node_names"]),
        hop_df=hop_df,
        downstream_rule=DOWNSTREAM_RULE,
        out_dir=OUT_DIR,
        save_separate=True,
        save_combined_grid=False,
        hist_dpi=300,
        show=False,
    )
    print(f"wrote {len(paths)} histogram PNGs under {OUT_DIR}")
