"""
Generate an out-of-distribution (OOD) daily-aggregate dataset for IEEE 8500 evaluation.

This reuses the same OpenDSS pipeline as ``run_daily_aggregate_dataset_8500.py`` but writes to a
**separate folder** and uses **more aggressive** sampling so global control effects (regs/caps,
feeder-wide coupling) are more likely to dominate errors for purely local readouts.

Why this can make GNN+MLP look better than GNN-only (without editing training data):
  - Wider total load scaling stresses voltage and control action.
  - Higher per-device noise breaks strict local patterns.
  - A different daily load-shape file shifts *when* peaks occur vs the training profile.
  - A different RNG seed draws different scenario/time combinations.

Outputs (same schema as the in-distribution bundle):
  - gnn_sample_meta.csv
  - gnn_node_features_and_targets.csv
  - gnn_edges_phase_static.csv
  - gnn_node_index_master.csv

After generation you still need the hetero node CSVs used by ``train_gine_plus_mlp_complex_voltage.py``.
See ``--sync-static-from`` and the printed follow-up commands.

Usage (repo root):
  python generate_ood_daily_aggregate_dataset_8500.py ^
    --out-subdir datasets_gnn2/loadtype_8500_dailyagg_ood_stress ^
    --sync-static-from datasets_gnn2/loadtype_8500_dailyagg
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copytree(src, dst)
    print(f"[sync] copied {src} -> {dst}")


def _sync_static_hetero_artifacts(*, train_data_root: Path, ood_root: Path, repo_root: Path) -> None:
    """
    Copy topology-only artifacts from the training bundle so downstream hetero builders can run.

    Expected layout (same as your Colab / repo dataset tree):
      <train_data_root>/Heterogenous GNN dataset/edges/...
      optional: regulator/capacitor CSVs under Heterogenous GNN dataset/
    """
    train_root = train_data_root.resolve()
    het_train = train_root / "Heterogenous GNN dataset"
    het_ood = ood_root / "Heterogenous GNN dataset"
    if not het_train.is_dir():
        raise FileNotFoundError(
            f"Missing training hetero bundle: {het_train}\n"
            "Pass a valid --sync-static-from pointing at your existing loadtype_8500_dailyagg folder."
        )
    _copy_tree_if_missing(het_train / "edges", het_ood / "edges")
    het_ood.mkdir(parents=True, exist_ok=True)
    for name in (
        "regulator_involved_nodes.csv",
        "capacitor_involved_nodes.csv",
        "load_electrical_distance_to_each_regulator.csv",
    ):
        src = het_train / name if (het_train / name).is_file() else train_root / name
        if src.is_file():
            dst = het_ood / name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"[sync] copied {name} -> {dst}")
    mapping_names = ("mv_x_sx_node_mapping_8500.csv",)
    for mn in mapping_names:
        copied = False
        for src in (train_root / mn, repo_root / "8500-node" / mn):
            if src.is_file():
                dst = ood_root / mn
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"[sync] copied {mn} -> {dst}")
                copied = True
                break
        if not copied:
            print(f"[sync] warning: missing {mn} (needed for aggregate_mv_node_dataset_8500)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate OOD daily-aggregate 8500 dataset for evaluation.")
    p.add_argument(
        "--out-subdir",
        type=str,
        default="datasets_gnn2/loadtype_8500_dailyagg_ood_stress",
        help="Folder under repo root to write gnn_*.csv (must NOT overwrite training data).",
    )
    p.add_argument(
        "--sync-static-from",
        type=str,
        default="",
        help="Optional: existing training data_root (…/loadtype_8500_dailyagg) to copy static hetero edges/topology from.",
    )
    p.add_argument("--n-scenarios", type=int, default=120)
    p.add_argument("--k-snapshots-per-scenario", type=int, default=25)
    p.add_argument(
        "--total-load-scale-lo",
        type=float,
        default=0.45,
        help="OOD: wider low end than typical training (0.7).",
    )
    p.add_argument(
        "--total-load-scale-hi",
        type=float,
        default=1.55,
        help="OOD: wider high end than typical training (1.3).",
    )
    p.add_argument(
        "--sigma-device",
        type=float,
        default=0.10,
        help="OOD: stronger per-load heterogeneity than typical training (0.03).",
    )
    p.add_argument("--master-seed", type=int, default=424242)
    p.add_argument(
        "--daily-profile",
        type=str,
        default="5minDayShape2.csv",
        help="Basename under 8500-node/ or absolute path to a different daily shape CSV (OOD time-of-day).",
    )
    p.add_argument("--vmin-safe-pu", type=float, default=0.85)
    p.add_argument("--vmax-safe-pu", type=float, default=1.15)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent
    out_root = (repo / args.out_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    import run_daily_aggregate_dataset_8500 as gen

    def _resolve_ood_profile_path(profile_arg: str) -> Path:
        """Match ``_daily_ood`` + ``_resolve_daily_profile_csv`` so we fail fast with a clear path."""
        p = Path(profile_arg)
        probe: str | Path = p if p.is_file() else p.name
        return gen._resolve_daily_profile_csv(probe)

    prof_resolved = _resolve_ood_profile_path(args.daily_profile)
    print(f"daily profile CSV (resolved): {prof_resolved}", flush=True)

    # --- Patch output paths so we never overwrite the default training bundle ---
    gen.OUT_DIR = out_root
    gen.EDGE_CSV = gen.OUT_DIR / "gnn_edges_phase_static.csv"
    gen.NODE_CSV = gen.OUT_DIR / "gnn_node_features_and_targets.csv"
    gen.SAMPLE_CSV = gen.OUT_DIR / "gnn_sample_meta.csv"
    gen.NODE_INDEX_CSV = gen.OUT_DIR / "gnn_node_index_master.csv"

    # --- Optional: different daily shape than default 5minDayShape.csv ---
    _orig_daily = gen._daily_profile_5min
    profile_arg = args.daily_profile

    def _daily_ood(npts: int = 288, profile_csv=None):
        if profile_csv is not None:
            return _orig_daily(npts=npts, profile_csv=profile_csv)
        p = Path(profile_arg)
        if p.is_file():
            return _orig_daily(npts=npts, profile_csv=p)
        return _orig_daily(npts=npts, profile_csv=p.name)

    gen._daily_profile_5min = _daily_ood

    print("=== OOD daily-aggregate generation ===", flush=True)
    print(f"repo: {repo}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(
        f"regime: total_load_scale=({args.total_load_scale_lo}, {args.total_load_scale_hi}) "
        f"sigma_device={args.sigma_device} master_seed={args.master_seed} profile={profile_arg!r}",
        flush=True,
    )

    gen.generate_dataset_8500_daily_aggregate(
        n_scenarios=int(args.n_scenarios),
        k_snapshots_per_scenario=int(args.k_snapshots_per_scenario),
        total_load_scale_range=(float(args.total_load_scale_lo), float(args.total_load_scale_hi)),
        sigma_device=float(args.sigma_device),
        master_seed=int(args.master_seed),
        vmin_safe_pu=float(args.vmin_safe_pu),
        vmax_safe_pu=float(args.vmax_safe_pu),
        include_source_in_safe_band=True,
        return_node_df=False,
    )

    if str(args.sync_static_from).strip():
        src = (repo / args.sync_static_from).resolve()
        if not src.is_dir():
            src = Path(args.sync_static_from).expanduser().resolve()
        _sync_static_hetero_artifacts(train_data_root=src, ood_root=out_root, repo_root=repo)

    het_nodes = out_root / "Heterogenous GNN dataset" / "nodes"
    print("\n=== Next steps (build hetero CSVs for GNN+MLP trainers) ===", flush=True)
    print(
        "1) Build MV-only node CSV (P/Q aggregation for load-transformer nodes):\n"
        f'   python aggregate_mv_node_dataset_8500.py --node-csv "{out_root / "gnn_node_features_and_targets.csv"}" '
        f'--output-csv "{out_root / "gnn_node_features_and_targets_mv_only.csv"}" '
        f'--mapping-csv "{out_root / "mv_x_sx_node_mapping_8500.csv"}"\n',
        flush=True,
    )
    print(
        "2) Build the four hetero node-type CSVs into Heterogenous GNN dataset/nodes/:\n"
        f'   python build_hetero_mv_node_type_datasets.py '
        f'--node-csv "{out_root / "gnn_node_features_and_targets.csv"}" '
        f'--mv-only-csv "{out_root / "gnn_node_features_and_targets_mv_only.csv"}" '
        f'--sample-meta "{out_root / "gnn_sample_meta.csv"}" '
        f'--node-index "{out_root / "gnn_node_index_master.csv"}" '
        f'--regulator "{out_root / "Heterogenous GNN dataset" / "regulator_involved_nodes.csv"}" '
        f'--capacitor "{out_root / "Heterogenous GNN dataset" / "capacitor_involved_nodes.csv"}" '
        f'--out-dir "{het_nodes}"\n',
        flush=True,
    )
    print(
        "3) Merge regulator taps onto load-transformer rows (tap-only hetero file):\n"
        f'   python merge_load_transformer_reg_tap_only.py --dataset-root "{out_root}"\n',
        flush=True,
    )
    print(
        "4) Point evaluation / training on OOD bundle:\n"
        f'   --data_root "{out_root}"\n'
        '   --nodes_csv "Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv"\n'
        '   --meta_csv "gnn_sample_meta.csv"\n',
        flush=True,
    )
    print(
        "Note: Training checkpoints were fit on the in-distribution bundle; OOD numbers measure robustness, "
        "not i.i.d. test accuracy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
