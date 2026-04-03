"""
Preset wrappers around `compare_hetero_mv_daily.run_compare` for the two checkpoints you use.

Imports from `compare_hetero_mv_daily` are **lazy** (inside each function) so
`from compare_hetero_mv_daily_wrappers import run_juxtapose_sage_vs_gine` works
even when PyG isn't loaded until you actually call a runner.

If you **do not** see ``=== Daily Timing Summary (hetero MV vs OpenDSS) ===`` after
``run_sage_not_edge_aware`` / ``run_gine_edge_aware``, the kernel had a stale
``compare_hetero_mv_daily``. Wrappers now ``reload`` it each call; if needed **Restart kernel**.

After ``git pull``, **restart the Jupyter kernel** so ``CKPT_HOMO_*`` paths update.
Homo checkpoints under ``homo_mv_8500/``: ``GINE-128-4``, ``GINE-64-2``, ``GINE-64-2-EMB-16-8``, ``GINE-64-3``, ``GINE-64-3-EMB-16-8``, ``GCN-128-4``, ``GCN-64-3``
(see ``run_homo_gine_mv``, ``run_homo_gine_64_2_mv``, ``run_homo_gine_64_2_emb_16_8_mv``, ``run_homo_gine_64_3_mv``, ``run_homo_gine_64_3_emb_16_8_mv``, ``run_homo_gcn_mv``, ``run_homo_gcn_64_3_mv``).

Timing: OpenDSS **solve** in the daily compare uses **snapshot** solves per timestep; GNN timing
may use ``torch.compile`` when ``GNN_TORCH_COMPILE=1`` (default off on Windows).

GNN **device**: set env ``GNN_COMPARE_DEVICE`` to ``auto`` (default), ``cpu``, or ``cuda``, or pass
``device="cpu"`` / ``device="cuda"`` into any runner (overrides env). See ``compare_mv_daily_timing.resolve_inference_device``.

If Jupyter says ``cannot import name …`` (e.g. ``run_homo_gine_64_2_mv``), the kernel
cached an **old** copy of this module. Fix: **Restart kernel**, or::

    import importlib, sys
    sys.modules.pop("compare_hetero_mv_daily_wrappers", None)
    import compare_hetero_mv_daily_wrappers as w
    importlib.reload(w)
    from compare_hetero_mv_daily_wrappers import run_homo_gine_64_2_mv  # etc.

Or use ``%run juxtapose_sage_gine_daily.py``, ``juxtapose_both_fail_vs_dss_daily.py``,
``juxtapose_lowest_min_v_top5_daily.py`` from the repo root.

From a notebook:
    from compare_hetero_mv_daily_wrappers import run_juxtapose_sage_vs_gine
    run_juxtapose_sage_vs_gine()

CLI:
    python compare_hetero_mv_daily_wrappers.py sage
    python compare_hetero_mv_daily_wrappers.py gine
    python compare_hetero_mv_daily_wrappers.py homo-gine
    python compare_hetero_mv_daily_wrappers.py homo-gcn
    python compare_hetero_mv_daily_wrappers.py homo-gcn-64-3
    python compare_hetero_mv_daily_wrappers.py homo-gine-64-2
    python compare_hetero_mv_daily_wrappers.py homo-gine-64-3
    python compare_hetero_mv_daily_wrappers.py homo-gine-64-3-emb-16-8
    python compare_hetero_mv_daily_wrappers.py juxtapose
    python compare_hetero_mv_daily_wrappers.py both-fail
    python compare_hetero_mv_daily_wrappers.py lowest-min-v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Curated monitoring buses (see comments). Used by run_sage / run_gine / run_homo_*.
DEFAULT_NODES = [
    "l2917359.1",  # Type A — Near-regulator, GINE succeeds
    "l2730108.3",  # Type B — Deep feeder, both models flat
    "l3067506.3",  # Type C — High-voltage cap-driven
    "l3101782.1",  # Type D — Low-voltage ohmic-dominated
    "l2786204.3",  # Type E — Edge-attr critical (SAGE fails)
]

DATASET_DIR = Path("datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset")
NODE_INDEX = Path("datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv")
MV_SX_DEFAULT = Path("8500-node/mv_x_sx_node_mapping_8500.csv")

CKPT_SAGE = Path("gnn2_architecture_search/hetero_mv_8500/NOT EDGE AWARE/hetero_sage_4x64_ln_drop_best.pt")
OUT_SAGE = Path("gnn2_daily_compare_8500_output")

CKPT_GINE = Path("gnn2_architecture_search/hetero_mv_8500/EDGE AWARE/hetero_gine_3x80_best.pt")
OUT_GINE = Path("gnn2_daily_compare_8500_output_gine_edgeaware")

# Homogeneous GINE / GCN from train_homo_gine_csv.py (each run folder: *_best.pt + train_metrics.json + feature_norm.pt)
CKPT_HOMO_GINE = Path("gnn2_architecture_search/homo_mv_8500/GINE-128-4/homo_gine_h128_L4_best.pt")
CKPT_HOMO_GINE_64_2 = Path("gnn2_architecture_search/homo_mv_8500/GINE-64-2/homo_gine_h64_L2_best.pt")
# Same h64 L2 + learned ID embeddings (train_homo_gine_csv: --node_emb_dim 16 --edge_emb_dim 8)
CKPT_HOMO_GINE_64_2_EMB_16_8 = Path(
    "gnn2_architecture_search/homo_mv_8500/GINE-64-2-EMB-16-8/homo_gine_h64_L2_ne16_ee8_best.pt"
)
CKPT_HOMO_GINE_64_3 = Path("gnn2_architecture_search/homo_mv_8500/GINE-64-3/homo_gine_h64_L3_best.pt")
CKPT_HOMO_GINE_64_3_EMB_16_8 = Path(
    "gnn2_architecture_search/homo_mv_8500/GINE-64-3-EMB-16-8/homo_gine_h64_L3_ne16_ee8_best.pt"
)
CKPT_HOMO_GCN = Path("gnn2_architecture_search/homo_mv_8500/GCN-128-4/homo_gcn_h128_L4_best.pt")
CKPT_HOMO_GCN_64_3 = Path("gnn2_architecture_search/homo_mv_8500/GCN-64-3/homo_gcn_h64_L3_best.pt")
OUT_HOMO_GINE = Path("gnn2_daily_compare_8500_output_homo_gine")
OUT_HOMO_GINE_64_2 = Path("gnn2_daily_compare_8500_output_homo_gine_64_2")
OUT_HOMO_GINE_64_2_EMB_16_8 = Path("gnn2_daily_compare_8500_output_homo_gine_64_2_emb_16_8")
OUT_HOMO_GINE_64_3 = Path("gnn2_daily_compare_8500_output_homo_gine_64_3")
OUT_HOMO_GINE_64_3_EMB_16_8 = Path("gnn2_daily_compare_8500_output_homo_gine_64_3_emb_16_8")
OUT_HOMO_GCN = Path("gnn2_daily_compare_8500_output_homo_gcn")
OUT_HOMO_GCN_64_3 = Path("gnn2_daily_compare_8500_output_homo_gcn_64_3")

OUT_JUXTAPOSE = Path("gnn2_daily_compare_8500_output_sage_vs_gine")
OUT_JUXTAPOSE_BOTH_FAIL = Path("gnn2_daily_compare_8500_output_both_fail_vs_dss")
OUT_JUXTAPOSE_LOWEST_MIN_V = Path("gnn2_daily_compare_8500_output_lowest_min_v_top5")

# If ``ImportError: cannot import name 'run_homo_gine_64_2_emb_16_8_mv'`` (or similar),
# the kernel cached an *old* copy of this file. Before importing, run:
#   import sys; sys.modules.pop("compare_hetero_mv_daily_wrappers", None)
# then import again, or use Kernel → Restart. Fresh load should show __version__ >= "16".
__version__ = "16"  # bump when exports change (helps debug stale notebook imports)

__all__ = (
    "run_sage_not_edge_aware",
    "run_gine_edge_aware",
    "run_homo_gine_mv",
    "run_homo_gine_64_2_mv",
    "run_homo_gine_64_2_emb_16_8_mv",
    "run_homo_gine_64_3_mv",
    "run_homo_gine_64_3_emb_16_8_mv",
    "run_homo_gcn_mv",
    "run_homo_gcn_64_3_mv",
    "run_juxtapose_sage_vs_gine",
    "run_juxtapose_both_fail_vs_dss",
    "run_juxtapose_lowest_min_v_top5",
)


def _repo() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _resolve_mv_sx_mapping(root: Path, mv_sx_mapping: Path | None) -> Path | None:
    if mv_sx_mapping is not None:
        p = Path(mv_sx_mapping)
        return p.resolve() if p.is_absolute() else (root / p).resolve()
    default = (root / MV_SX_DEFAULT).resolve()
    return default if default.is_file() else None


def _reload_compare_hetero_mv_daily():
    """
    Reload `compare_hetero_mv_daily` from disk if already imported.

    Jupyter keeps a stale copy (e.g. missing wall-clock timing prints). Reload fixes that
    without requiring a kernel restart when the repo file was updated.

    Also reload dependencies first: ``compare_opendss_snapshot_helpers``, ``run_daily_aggregate_dataset_8500``,
    ``compare_gnn_inference_utils``, and ``compare_mv_daily_timing`` — otherwise ``reload(compare_hetero_mv_daily)``
    keeps stale submodules from an earlier import.
    """
    import importlib

    if "compare_opendss_snapshot_helpers" in sys.modules:
        importlib.reload(sys.modules["compare_opendss_snapshot_helpers"])
    if "run_daily_aggregate_dataset_8500" in sys.modules:
        importlib.reload(sys.modules["run_daily_aggregate_dataset_8500"])
    if "compare_gnn_inference_utils" in sys.modules:
        importlib.reload(sys.modules["compare_gnn_inference_utils"])
    if "compare_mv_daily_timing" in sys.modules:
        importlib.reload(sys.modules["compare_mv_daily_timing"])
    name = "compare_hetero_mv_daily"
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    import compare_hetero_mv_daily as ch

    return ch


def run_sage_not_edge_aware(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    node_index: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """``show_plots=False`` saves PNGs only (no ``plt.show()``) — default for notebook batch runs."""
    ch = _reload_compare_hetero_mv_daily()
    root = repo_root or _repo()
    ch.run_compare(
        checkpoint=(root / (checkpoint or CKPT_SAGE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        node_index=(root / (node_index or NODE_INDEX)).resolve(),
        out_dir=(root / (out_dir or OUT_SAGE)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def _reload_compare_homo_mv_daily():
    import importlib

    if "compare_opendss_snapshot_helpers" in sys.modules:
        importlib.reload(sys.modules["compare_opendss_snapshot_helpers"])
    if "run_daily_aggregate_dataset_8500" in sys.modules:
        importlib.reload(sys.modules["run_daily_aggregate_dataset_8500"])
    if "compare_gnn_inference_utils" in sys.modules:
        importlib.reload(sys.modules["compare_gnn_inference_utils"])
    if "compare_mv_daily_timing" in sys.modules:
        importlib.reload(sys.modules["compare_mv_daily_timing"])
    name = "compare_homo_mv_daily"
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    import compare_homo_mv_daily as ch

    return ch


def run_homo_gine_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """Daily OpenDSS vs HomoGINE h128 L4 under ``homo_mv_8500/GINE-128-4/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GINE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GINE)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gcn_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGCN h128 L4 checkpoint under ``homo_mv_8500/GCN-128-4/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GCN)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GCN)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gcn_64_3_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGCN h64 L3 checkpoint under ``homo_mv_8500/GCN-64-3/`` (separate default out_dir from h128 L4)."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GCN_64_3)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GCN_64_3)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gine_64_2_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGINE h64 L2 checkpoint under ``homo_mv_8500/GINE-64-2/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GINE_64_2)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GINE_64_2)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gine_64_2_emb_16_8_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGINE h64 L2 with node/edge embeddings (16/8) under ``homo_mv_8500/GINE-64-2-EMB-16-8/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GINE_64_2_EMB_16_8)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GINE_64_2_EMB_16_8)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gine_64_3_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGINE h64 L3 checkpoint under ``homo_mv_8500/GINE-64-3/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GINE_64_3)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GINE_64_3)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_homo_gine_64_3_emb_16_8_mv(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    """HomoGINE h64 L3 with node/edge embeddings (16/8) under ``homo_mv_8500/GINE-64-3-EMB-16-8/``."""
    ch = _reload_compare_homo_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_homo(
        checkpoint=(root / (checkpoint or CKPT_HOMO_GINE_64_3_EMB_16_8)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        out_dir=(root / (out_dir or OUT_HOMO_GINE_64_3_EMB_16_8)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_gine_edge_aware(
    *,
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    dataset_dir: Path | None = None,
    node_index: Path | None = None,
    out_dir: Path | None = None,
    plot_nodes: list[str] | None = None,
    mv_sx_mapping: Path | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
    show_plots: bool = False,
    monitoring_plots_subfolders: bool = True,
) -> None:
    ch = _reload_compare_hetero_mv_daily()
    root = repo_root or _repo()
    ch.run_compare(
        checkpoint=(root / (checkpoint or CKPT_GINE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        node_index=(root / (node_index or NODE_INDEX)).resolve(),
        out_dir=(root / (out_dir or OUT_GINE)).resolve(),
        plot_nodes=list(plot_nodes or DEFAULT_NODES),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        show_plots=show_plots,
        monitoring_plots_subfolders=monitoring_plots_subfolders,
    )


def run_juxtapose_sage_vs_gine(
    *,
    repo_root: Path | None = None,
    checkpoint_sage: Path | None = None,
    checkpoint_gine: Path | None = None,
    dataset_dir: Path | None = None,
    node_index: Path | None = None,
    out_dir: Path | None = None,
    mv_sx_mapping: Path | None = None,
    top_disagree: int = 10,
    disagree_scope: str = "load",
    also_nodes: list[str] | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
) -> None:
    """One daily solve; forward SAGE + GINE; plot nodes with largest mean |V_sage − V_gine|."""
    ch = _reload_compare_hetero_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_juxtapose(
        checkpoint_a=(root / (checkpoint_sage or CKPT_SAGE)).resolve(),
        checkpoint_b=(root / (checkpoint_gine or CKPT_GINE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        node_index=(root / (node_index or NODE_INDEX)).resolve(),
        out_dir=(root / (out_dir or OUT_JUXTAPOSE)).resolve(),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        top_disagree=top_disagree,
        disagree_scope=disagree_scope,
        also_plot_nodes=list(also_nodes or []),
        device=device,
    )


def run_juxtapose_both_fail_vs_dss(
    *,
    repo_root: Path | None = None,
    checkpoint_sage: Path | None = None,
    checkpoint_gine: Path | None = None,
    dataset_dir: Path | None = None,
    node_index: Path | None = None,
    out_dir: Path | None = None,
    mv_sx_mapping: Path | None = None,
    top_disagree: int = 10,
    disagree_scope: str = "load",
    also_nodes: list[str] | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
) -> None:
    """One daily solve; forward SAGE + GINE; plot loads with largest min(MAE vs OpenDSS) across the two models."""
    ch = _reload_compare_hetero_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_juxtapose(
        checkpoint_a=(root / (checkpoint_sage or CKPT_SAGE)).resolve(),
        checkpoint_b=(root / (checkpoint_gine or CKPT_GINE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        node_index=(root / (node_index or NODE_INDEX)).resolve(),
        out_dir=(root / (out_dir or OUT_JUXTAPOSE_BOTH_FAIL)).resolve(),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        top_disagree=top_disagree,
        disagree_scope=disagree_scope,
        also_plot_nodes=list(also_nodes or []),
        juxtapose_mode="both_fail_dss",
        device=device,
    )


def run_juxtapose_lowest_min_v_top5(
    *,
    repo_root: Path | None = None,
    checkpoint_sage: Path | None = None,
    checkpoint_gine: Path | None = None,
    dataset_dir: Path | None = None,
    node_index: Path | None = None,
    out_dir: Path | None = None,
    mv_sx_mapping: Path | None = None,
    top_disagree: int = 5,
    disagree_scope: str = "load",
    also_nodes: list[str] | None = None,
    npts: int = 288,
    step_min: int = 5,
    ymin: float = 0.85,
    ymax: float = 1.10,
    device: str | None = None,
) -> None:
    """One daily solve; SAGE + GINE; plot loads with the lowest daily minimum OpenDSS |V| (top-K)."""
    ch = _reload_compare_hetero_mv_daily()
    root = repo_root or _repo()
    ch.run_compare_juxtapose(
        checkpoint_a=(root / (checkpoint_sage or CKPT_SAGE)).resolve(),
        checkpoint_b=(root / (checkpoint_gine or CKPT_GINE)).resolve(),
        dataset_dir=(root / (dataset_dir or DATASET_DIR)).resolve(),
        node_index=(root / (node_index or NODE_INDEX)).resolve(),
        out_dir=(root / (out_dir or OUT_JUXTAPOSE_LOWEST_MIN_V)).resolve(),
        npts=npts,
        step_min=step_min,
        ymin=ymin,
        ymax=ymax,
        mv_sx_mapping=_resolve_mv_sx_mapping(root, mv_sx_mapping),
        device=device,
        top_disagree=top_disagree,
        disagree_scope=disagree_scope,
        also_plot_nodes=list(also_nodes or []),
        juxtapose_mode="lowest_min_v_dss",
    )


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Run compare_hetero_mv_daily with fixed presets")
    p.add_argument(
        "which",
        choices=(
            "sage",
            "gine",
            "homo-gine",
            "homo-gine-64-2",
            "homo-gine-64-2-emb-16-8",
            "homo-gine-64-3",
            "homo-gine-64-3-emb-16-8",
            "homo-gcn",
            "homo-gcn-64-3",
            "juxtapose",
            "both-fail",
            "lowest-min-v",
        ),
        help="hetero sage/gine; homo GINE/GCN presets; juxtapose; both-fail; lowest-min-v",
    )
    args = p.parse_args(argv)
    if args.which == "sage":
        run_sage_not_edge_aware()
    elif args.which == "gine":
        run_gine_edge_aware()
    elif args.which == "homo-gine":
        run_homo_gine_mv()
    elif args.which == "homo-gine-64-2":
        run_homo_gine_64_2_mv()
    elif args.which == "homo-gine-64-2-emb-16-8":
        run_homo_gine_64_2_emb_16_8_mv()
    elif args.which == "homo-gine-64-3":
        run_homo_gine_64_3_mv()
    elif args.which == "homo-gine-64-3-emb-16-8":
        run_homo_gine_64_3_emb_16_8_mv()
    elif args.which == "homo-gcn":
        run_homo_gcn_mv()
    elif args.which == "homo-gcn-64-3":
        run_homo_gcn_64_3_mv()
    elif args.which == "juxtapose":
        run_juxtapose_sage_vs_gine()
    elif args.which == "both-fail":
        run_juxtapose_both_fail_vs_dss()
    else:
        run_juxtapose_lowest_min_v_top5()


if __name__ == "__main__":
    main()
