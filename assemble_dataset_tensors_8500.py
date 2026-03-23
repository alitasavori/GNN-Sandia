"""
Step 5 — Stack node features / targets from the 8500 load-type CSV for GNN training.

Reads `gnn_node_features_and_targets.csv` (from `run_loadtype_dataset_8500.py`) and
optional `graph_tensors/graph_meta.json` (from `build_graph_8500.py`) to validate N.

Writes under `datasets_gnn2/loadtype_8500/dataset_tensors/`:
  - X.pt        float32, shape [S, N, F]  — per-sample node inputs
  - Y.pt        float32, shape [S, N]     — target column (default vmag_pu)
  - Y_angle.pt  float32, shape [S, N]     — optional `vang_deg` (for MLP baseline loss)
  - tensor_manifest.json                — shapes, feature column names, sample_ids order

Default features match `LOADTYPE_FEAT` in `run_gnn3_best7_train.py` (14 columns).

Run after Steps 3–4.
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
    raise SystemExit("PyTorch is required. Install with: pip install torch") from e

try:
    _REPO = Path(__file__).resolve().parent
except NameError:
    _REPO = Path.cwd()

DEFAULT_DATASET_DIR = _REPO / "datasets_gnn2" / "loadtype_8500"
NODE_CSV_NAME = "gnn_node_features_and_targets.csv"
GRAPH_META_REL = Path("graph_tensors") / "graph_meta.json"

# Same as run_gnn3_best7_train.LOADTYPE_FEAT — 8500 CSV uses identical names.
DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "electrical_distance_ohm",
    "m1_p_kw",
    "m1_q_kvar",
    "m2_p_kw",
    "m2_q_kvar",
    "m4_p_kw",
    "m4_q_kvar",
    "m5_p_kw",
    "m5_q_kvar",
    "q_cap_kvar",
    "p_pv_kw",
    "q_pv_kvar",
    "p_sys_balance_kw",
    "q_sys_balance_kvar",
)


def _load_num_nodes(dataset_dir: Path) -> int:
    meta_path = dataset_dir / GRAPH_META_REL
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["num_nodes"])
    master = dataset_dir / "gnn_node_index_master.csv"
    if not master.is_file():
        raise FileNotFoundError(
            f"Need {meta_path} or {master} to determine N. Run build_graph_8500 or dataset generation."
        )
    df_m = pd.read_csv(master)
    return int(df_m["node_idx"].max()) + 1


def assemble_dataset_tensors_8500(
    dataset_dir: str | os.PathLike | None = None,
    out_subdir: str = "dataset_tensors",
    feature_cols: tuple[str, ...] | None = None,
    target_col: str = "vmag_pu",
    angle_col: str | None = "vang_deg",
) -> dict:
    """
    Stack rows into [S, N, F] and [S, N]. Drops samples that do not have exactly N rows.
    """
    dset = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    node_csv = dset / NODE_CSV_NAME
    if not node_csv.is_file():
        raise FileNotFoundError(
            f"Missing {node_csv}. Run run_loadtype_dataset_8500.generate_gnn_snapshot_dataset_loadtype_8500 first."
        )

    feats = feature_cols if feature_cols is not None else DEFAULT_FEATURE_COLS
    N = _load_num_nodes(dset)

    df_n = pd.read_csv(node_csv)
    required = {"sample_id", "node_idx", target_col} | set(feats)
    missing = required - set(df_n.columns)
    if missing:
        raise ValueError(f"Node CSV missing columns: {missing}")
    if angle_col is not None and angle_col not in df_n.columns:
        angle_col = None

    df_n["sample_id"] = pd.to_numeric(df_n["sample_id"], errors="raise").astype(int)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    for c in feats:
        df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
    df_n[target_col] = pd.to_numeric(df_n[target_col], errors="coerce")
    if angle_col:
        df_n[angle_col] = pd.to_numeric(df_n[angle_col], errors="coerce")
        required_angle = set(required) | {angle_col}
    else:
        required_angle = required

    df_n = df_n.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_angle)).copy()
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    counts = df_n.groupby("sample_id")["node_idx"].count()
    good_ids = counts[counts == N].index.to_numpy()
    if len(good_ids) == 0:
        raise RuntimeError(
            f"No samples with exactly N={N} nodes. counts mode: {counts.value_counts().head()}"
        )
    good_ids = np.sort(good_ids)
    df_n = df_n[df_n["sample_id"].isin(good_ids)].copy()
    df_n = df_n.sort_values(["sample_id", "node_idx"]).reset_index(drop=True)

    S = len(good_ids)
    # Sanity: each block of N rows is one sample with node_idx 0..N-1
    for i, sid in enumerate(good_ids):
        sl = df_n.iloc[i * N : (i + 1) * N]
        if not (sl["sample_id"] == sid).all():
            raise RuntimeError(f"Internal sort error at sample_id={sid}")
        idxs = sl["node_idx"].to_numpy()
        if not np.array_equal(idxs, np.arange(N, dtype=int)):
            raise ValueError(
                f"sample_id={sid}: node_idx not 0..N-1 in order (got first few {idxs[:5]}...)"
            )

    X = df_n[list(feats)].to_numpy(dtype=np.float32).reshape(S, N, -1)
    Y = df_n[target_col].to_numpy(dtype=np.float32).reshape(S, N)

    out_dir = dset / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path_x = out_dir / "X.pt"
    path_y = out_dir / "Y.pt"
    torch.save(torch.from_numpy(X), path_x)
    torch.save(torch.from_numpy(Y), path_y)

    path_y_angle = None
    if angle_col:
        Ya = df_n[angle_col].to_numpy(dtype=np.float32).reshape(S, N)
        path_y_angle = out_dir / "Y_angle.pt"
        torch.save(torch.from_numpy(Ya), path_y_angle)

    manifest = {
        "dataset_dir": str(dset.resolve()),
        "num_samples": int(S),
        "num_nodes": int(N),
        "num_features": int(X.shape[2]),
        "feature_columns": list(feats),
        "target_column": target_col,
        "angle_column": angle_col,
        "X_path": str(path_x.resolve()),
        "Y_path": str(path_y.resolve()),
        "Y_angle_path": str(path_y_angle.resolve()) if path_y_angle else None,
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "sample_ids": [int(x) for x in good_ids.tolist()],
        "source_csv": str(node_csv.resolve()),
    }
    path_m = out_dir / "tensor_manifest.json"
    path_m.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[assemble_dataset_tensors_8500] Saved to {out_dir}/")
    extra = f"  Y_angle {angle_col!r}" if angle_col else ""
    print(f"  X {tuple(X.shape)}  Y {tuple(Y.shape)}  target={target_col!r}{extra}")
    print(f"  samples={S} (dropped incomplete samples vs N={N})")
    return manifest


def main() -> None:
    assemble_dataset_tensors_8500()


if __name__ == "__main__":
    main()
