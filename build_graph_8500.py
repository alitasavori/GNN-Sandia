"""
Step 4 — Build static graph tensors for the IEEE 8500 feeder (GNN2).

Reads the phase-edge CSV produced alongside the loadtype_8500 dataset
(`extract_static_phase_edges_to_csv` from run_injection_dataset) and writes:
  - edge_index.pt   (int64, shape [2, E])
  - edge_attr.pt    (float32, shape [E, F])  — R_full, X_full, length, phase (sin/cos optional later)
  - graph_meta.json — counts, paths, feature column names

Requires: pandas, numpy, torch (PyTorch). Install torch if missing.

Run after `run_loadtype_dataset_8500.py` so that these exist:
  datasets_gnn2/loadtype_8500/gnn_edges_phase_static.csv
  datasets_gnn2/loadtype_8500/gnn_node_index_master.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    raise SystemExit(
        "PyTorch is required for build_graph_8500.py. Install with: pip install torch"
    ) from e

try:
    _REPO = Path(__file__).resolve().parent
except NameError:
    _REPO = Path.cwd()

DEFAULT_DATASET_DIR = _REPO / "datasets_gnn2" / "loadtype_8500"
EDGE_CSV_NAME = "gnn_edges_phase_static.csv"
NODE_CSV_NAME = "gnn_node_index_master.csv"


def build_static_graph_8500(
    dataset_dir: str | os.PathLike | None = None,
    out_subdir: str = "graph_tensors",
    edge_attr_cols: tuple[str, ...] = ("R_full", "X_full", "length", "phase"),
) -> dict:
    """
    Build PyTorch tensors aligned with gnn_node_index_master.csv node indices.

    Returns a dict with paths and tensor shapes.
    """
    dset = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    edge_csv = dset / EDGE_CSV_NAME
    node_csv = dset / NODE_CSV_NAME
    if not edge_csv.is_file():
        raise FileNotFoundError(
            f"Missing {edge_csv}. Run run_loadtype_dataset_8500.py first to generate edges."
        )
    if not node_csv.is_file():
        raise FileNotFoundError(
            f"Missing {node_csv}. Run run_loadtype_dataset_8500.py first."
        )

    df_e = pd.read_csv(edge_csv)
    df_n = pd.read_csv(node_csv)

    n_nodes = len(df_n)
    if "u_idx" not in df_e.columns or "v_idx" not in df_e.columns:
        raise ValueError("Edge CSV must contain u_idx and v_idx columns.")

    u = df_e["u_idx"].to_numpy(dtype=np.int64)
    v = df_e["v_idx"].to_numpy(dtype=np.int64)
    if u.max() >= n_nodes or v.max() >= n_nodes:
        raise ValueError(
            f"Edge indices exceed node count: max_idx={max(u.max(), v.max())} n_nodes={n_nodes}"
        )

    edge_index = torch.from_numpy(np.stack([u, v], axis=0))

    feats = []
    for c in edge_attr_cols:
        if c not in df_e.columns:
            raise ValueError(f"Missing edge column {c!r} in {edge_csv}")
        feats.append(df_e[c].to_numpy(dtype=np.float32))
    edge_attr = torch.from_numpy(np.column_stack(feats))

    out_dir = dset / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path_ei = out_dir / "edge_index.pt"
    path_ea = out_dir / "edge_attr.pt"
    torch.save(edge_index, path_ei)
    torch.save(edge_attr, path_ea)

    node_to_idx = dict(zip(df_n["node"].astype(str), df_n["node_idx"].astype(int).tolist()))
    idx_to_node = {int(i): str(n) for n, i in node_to_idx.items()}

    meta = {
        "dataset_dir": str(dset.resolve()),
        "num_nodes": int(n_nodes),
        "num_edges": int(edge_index.shape[1]),
        "edge_index_path": str(path_ei.resolve()),
        "edge_attr_path": str(path_ea.resolve()),
        "edge_attr_columns": list(edge_attr_cols),
        "edge_attr_shape": list(edge_attr.shape),
        "source_csv": str(edge_csv.resolve()),
        "node_index_csv": str(node_csv.resolve()),
    }
    path_meta = out_dir / "graph_meta.json"
    path_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    path_map = out_dir / "node_index_map.json"
    path_map.write_text(
        json.dumps({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, indent=2),
        encoding="utf-8",
    )

    print(f"[build_graph_8500] Saved graph tensors to {out_dir}/")
    print(f"  edge_index: {tuple(edge_index.shape)} dtype={edge_index.dtype}")
    print(f"  edge_attr:  {tuple(edge_attr.shape)} dtype={edge_attr.dtype}")
    print(f"  num_nodes={n_nodes} num_edges={edge_index.shape[1]}")
    return meta


def main() -> None:
    build_static_graph_8500()


if __name__ == "__main__":
    main()
