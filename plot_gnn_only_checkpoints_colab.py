"""
Colab-friendly: load GNN-only checkpoints (gine/sage/gcn) from train_gnn_only_compare_complex_voltage.py,
rebuild chunk streaming test split, run inference, save diagnostic plots to disk (e.g. Google Drive).

Does not require OpenDSS. For daily OpenDSS overlays, use compare_homo_mv_daily_global_localres.py instead.

Example (paths like your Colab training):
  python plot_gnn_only_checkpoints_colab.py \\
    --repo /content/GNN-Sandia \\
    --run-dir /content/drive/MyDrive/datasets_gnn2/runs/gnn_only_chunked_mvagg_YYYYMMDD_HHMMSS \\
    --plot-out-dir /content/drive/MyDrive/datasets_gnn2/runs/gnn_only_chunked_mvagg_YYYYMMDD_HHMMSS/plots_test \\
    --chunk-parent /content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_2000_40 \\
    --chunk-subdir-glob "run_*" \\
    --nodes-csv gnn_node_features_and_targets_mvagg.csv \\
    --edge-catalog-csv gnn_edges_phase_static.csv \\
    --edge-shared-csv /path/to/gnn_edges_phase_static.csv \\
    --cache-dir /content/drive/MyDrive/datasets_gnn2/cache/gnn_only_chunked_mvagg_full \\
    --node-feature-cols "p_load_kw,q_load_kvar,p_pv_kw,p_bess_kw,q_bess_kvar" \\
    --node-pe-csv /path/to/gnn_node_index_master.csv \\
    --node-pe-cols auto \\
    --train-frac 0.9 --val-frac 0.09 --sample-frac 1.0 --seed 42 \\
    --batch-size 32 --models gine,sage,gcn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def _collect_test_predictions(
    *,
    model: nn.Module,
    ctx,
    nodes_csv_name: str,
    node_feature_cols: list[str],
    node_pe_path: Path | None,
    node_pe_cols: str,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_ri, true_ri) flattened [Ntest, 2*n_nodes]."""
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    model.eval()
    y_mean_d = y_mean.to(device)
    y_std_d = y_std.to(device)
    with torch.no_grad():
        for ci, ch in enumerate(ctx.chunk_dirs):
            x, y_ri, _, _ntl = _ensure_chunk_tensor_cache_gnn(
                ch,
                nodes_name=nodes_csv_name,
                node_feature_cols=node_feature_cols,
                selected_sample_ids=ctx.selected_ids_list[ci],
                cache_pt=ctx.cache_pts[ci],
                ref_ntl=ctx.ref_ntl,
            )
            if node_pe_path is not None:
                x = _append_shared_pe_features(
                    x,
                    node_to_local=ctx.ref_ntl,
                    node_pe_csv=node_pe_path,
                    node_pe_cols=node_pe_cols,
                    verbose=False,
                )
            x_n = ((x - x_mean) / x_std).to(dtype=torch.float32)
            ds = GraphVoltageDataset(x_n, y_ri, ctx.edge_index, ctx.edge_attr)
            dl = DataLoader(
                Subset(ds, ctx.idx_test_list[ci].tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            for batch in dl:
                batch = batch.to(device)
                pred_n = model(batch)
                yp = pred_n * y_std_d + y_mean_d
                preds.append(yp.detach().cpu())
                tgts.append(batch.y.view(batch.num_graphs, -1).cpu())
            del x, y_ri, x_n, ds, dl
    return torch.cat(preds, dim=0), torch.cat(tgts, dim=0)


def _vmag_angle(pred: torch.Tensor, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = pred.view(-1, n_nodes, 2)
    re, im = p[..., 0], p[..., 1]
    mag = torch.sqrt(re * re + im * im + 1e-12)
    ang = torch.atan2(im, re)
    return mag, ang


def _angle_diff_deg(pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
    d = pred_rad - true_rad
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return torch.rad2deg(d)


def _plot_suite(
    *,
    model_type: str,
    pred: torch.Tensor,
    tgt: torch.Tensor,
    n_nodes: int,
    out_dir: Path,
    scatter_max_points: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pm, pa = _vmag_angle(pred, n_nodes)
    tm, ta = _vmag_angle(tgt, n_nodes)

    # --- Scatter |V| pred vs true (subsample) ---
    pv = pm.reshape(-1).numpy()
    tv = tm.reshape(-1).numpy()
    n = pv.size
    rng = np.random.default_rng(0)
    if n > scatter_max_points:
        idx = rng.choice(n, size=scatter_max_points, replace=False)
        pv, tv = pv[idx], tv[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(tv, pv, gridsize=80, mincnt=1, cmap="viridis")
    lim = (float(np.nanpercentile(np.concatenate([tv, pv]), 1)), float(np.nanpercentile(np.concatenate([tv, pv]), 99)))
    ax.plot(lim, lim, "r--", lw=1.0, label="y=x")
    ax.set_xlabel("True |V| (p.u.)")
    ax.set_ylabel("Pred |V| (p.u.)")
    ax.set_title(f"{model_type} |V| pred vs true (test)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_vmag_hexbin.png", dpi=150)
    plt.close(fig)

    # --- Histogram |V| error ---
    err = (pm - tm).reshape(-1).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=80, color="steelblue", alpha=0.85, edgecolor="none")
    ax.axvline(0.0, color="red", ls="--", lw=1)
    ax.set_xlabel("|V|_pred − |V|_true (p.u.)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_type} pooled |V| error (all test nodes × samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_vmag_error.png", dpi=150)
    plt.close(fig)

    # --- MAE per node ---
    mae_n = (pm - tm).abs().mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(mae_n, lw=0.8)
    ax.set_xlabel("Node index (dataset order)")
    ax.set_ylabel("Mean abs |V| err (p.u.)")
    ax.set_title(f"{model_type} test mean |V| error per node")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_per_node_index.png", dpi=150)
    plt.close(fig)

    # --- Angle error histogram ---
    ade = _angle_diff_deg(pa, ta).reshape(-1).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ade, bins=80, color="seagreen", alpha=0.85, edgecolor="none")
    ax.axvline(0.0, color="red", ls="--", lw=1)
    ax.set_xlabel("Angle error (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_type} voltage angle error (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_angle_error_deg.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GNN-only checkpoints on chunk test split (no OpenDSS).")
    p.add_argument("--repo", type=Path, required=True, help="Repo root containing train_gnn_only_compare_complex_voltage.py")
    p.add_argument("--run-dir", type=Path, required=True, help="Training out_dir with *_gnn_only_best.pt and x_mean.pt ...")
    p.add_argument("--plot-out-dir", type=Path, required=True, help="Where to write figure folders")
    p.add_argument("--chunk-parent", type=Path, required=True)
    p.add_argument("--chunk-subdir-glob", type=str, default="run_*")
    p.add_argument("--nodes-csv", type=str, default="gnn_node_features_and_targets_mvagg.csv")
    p.add_argument("--edge-catalog-csv", type=str, default="gnn_edges_phase_static.csv")
    p.add_argument("--edge-shared-csv", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--node-feature-cols", type=str, required=True)
    p.add_argument("--node-pe-csv", type=Path, default=None)
    p.add_argument("--node-pe-cols", type=str, default="auto")
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.09)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--models", type=str, default="gine,sage,gcn")
    p.add_argument("--scatter-max-points", type=int, default=200_000)
    p.add_argument("--hidden", type=int, default=None, help="Override if missing from checkpoint")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--node-emb-dim", type=int, default=None)
    p.add_argument("--edge-emb-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None, help="Inference dropout; default from checkpoint or 0.1")
    args = p.parse_args()

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    plot_root = args.plot_out_dir.resolve()
    plot_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from train_gnn_only_compare_complex_voltage import (
        GNNOnlyVoltageModel,
        GraphVoltageDataset,
        _append_shared_pe_features,
        _ensure_chunk_tensor_cache_gnn,
        _metrics_from_ri_flat,
        _parse_node_feature_cols,
        _prepare_chunk_streaming,
        _set_seed,
    )

    _set_seed(int(args.seed))
    node_feature_cols = _parse_node_feature_cols(args.node_feature_cols)
    node_pe_path = Path(args.node_pe_csv).resolve() if args.node_pe_csv else None

    def _load_norm_tensor(p: Path) -> torch.Tensor:
        z = torch.load(p, map_location="cpu", weights_only=False)
        return z.float() if isinstance(z, torch.Tensor) else torch.as_tensor(z, dtype=torch.float32)

    x_mean = _load_norm_tensor(run_dir / "x_mean.pt")
    x_std = _load_norm_tensor(run_dir / "x_std.pt")
    y_mean = _load_norm_tensor(run_dir / "y_mean.pt")
    y_std = _load_norm_tensor(run_dir / "y_std.pt")

    ctx = _prepare_chunk_streaming(
        chunk_parent=Path(args.chunk_parent).resolve(),
        chunk_subdir_glob=str(args.chunk_subdir_glob),
        nodes_csv_name=str(args.nodes_csv),
        edges_csv_name=str(args.edge_catalog_csv),
        node_feature_cols=node_feature_cols,
        edge_shared_csv=Path(args.edge_shared_csv).resolve() if args.edge_shared_csv else None,
        cache_dir=Path(args.cache_dir).resolve(),
        sample_frac=float(args.sample_frac),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        node_pe_path=node_pe_path,
        node_pe_cols=str(args.node_pe_cols),
    )

    if int(x_mean.numel()) != int(ctx.in_dim):
        raise RuntimeError(
            f"x_mean length {x_mean.numel()} != ctx.in_dim {ctx.in_dim}. "
            "Check --node-feature-cols / --node-pe-csv match the training run."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    summary: dict[str, object] = {"run_dir": str(run_dir), "plot_out_dir": str(plot_root), "device": str(device), "models": {}}

    for m in models:
        ckpt_path = run_dir / f"{m}_gnn_only_best.pt"
        if not ckpt_path.is_file():
            print(f"[skip] missing checkpoint: {ckpt_path}", flush=True)
            continue
        pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(pack, dict) or "model_state_dict" not in pack:
            raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

        hidden = int(args.hidden) if args.hidden is not None else int(pack["hidden"])
        layers = int(args.layers) if args.layers is not None else int(pack["layers"])
        node_emb_dim = int(args.node_emb_dim) if args.node_emb_dim is not None else int(pack.get("node_emb_dim", 0))
        edge_emb_dim = int(args.edge_emb_dim) if args.edge_emb_dim is not None else int(pack.get("edge_emb_dim", 0))
        dropout = float(args.dropout) if args.dropout is not None else float(pack.get("dropout", 0.1))

        model = GNNOnlyVoltageModel(
            model_type=m,
            in_dim=int(ctx.in_dim),
            hidden=hidden,
            layers=layers,
            n_nodes=ctx.n_nodes,
            num_edges=ctx.n_edges,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            gps_heads=4,
            gps_mlp_state_dim=4,
            gps_mlp_post_hidden=1024,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(pack["model_state_dict"], strict=True)
        print(f"[{m}] loaded {ckpt_path}", flush=True)

        pred, tgt = _collect_test_predictions(
            model=model,
            ctx=ctx,
            nodes_csv_name=str(args.nodes_csv),
            node_feature_cols=node_feature_cols,
            node_pe_path=node_pe_path,
            node_pe_cols=str(args.node_pe_cols),
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=int(args.batch_size),
            device=device,
        )
        met = _metrics_from_ri_flat(pred, tgt)
        sub = plot_root / m
        _plot_suite(
            model_type=m,
            pred=pred,
            tgt=tgt,
            n_nodes=ctx.n_nodes,
            out_dir=sub,
            scatter_max_points=int(args.scatter_max_points),
        )
        met_path = sub / "test_metrics.json"
        met_path.write_text(json.dumps(met, indent=2), encoding="utf-8")
        summary["models"][m] = {"checkpoint": str(ckpt_path), "n_test_graphs": int(pred.shape[0]), "metrics": met}
        print(f"[{m}] wrote plots under {sub}", flush=True)

    (plot_root / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Summary: {plot_root / 'plot_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
