"""
Create a NEW chunked dataset sibling with draft-consistent network-Y edges.

Never overwrites the source dataset unless --inplace is passed (discouraged).

Draft edge definition (same as run_ieee34_draft_gnn_dataset.py):
  - Network-only YPrim: Line / Transformer / Capacitor / Reactor
  - Loads / PV / Vsource / Storage / Generator excluded
  - Y at ControlMode=OFF with regulator taps forced to 1.0 when possible
  - CSV: directed j→i with R_full/X_full = Re(Y_ij)/Im(Y_ij) [siemens]

Copied as-is from source chunks (features / PE / meta unchanged):
  - gnn_node_features_and_targets*.csv
  - gnn_sample_meta.csv
  - gnn_node_index_master.csv  (keeps pe_*)
  - any other files except gnn_edges_phase_static.csv

Usage (default: write sibling *_yedges next to source):
  python stamp_network_y_edges.py --feeder 906 \\
    --chunk_parent "K:/My Drive/datasets_gnn2/original_906_lvtestcase_chunked"

  python stamp_network_y_edges.py --feeder 8500 \\
    --chunk_parent "K:/My Drive/datasets_gnn2/original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40"

Outputs e.g.:
  .../original_906_lvtestcase_chunked_yedges/
  .../original_8500_..._2000_40_yedges/
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

import run_ieee34_draft_gnn_dataset as draft
import run_original_style_dataset_8500_unbalanced as ds8500
import run_original_style_dataset_906_lvtestcase as ds906

# Files we replace (never copy from source when building a new sibling).
_EDGE_NAME = "gnn_edges_phase_static.csv"
_SKIP_COPY_NAMES = {
    _EDGE_NAME,
    f"{_EDGE_NAME}.ohm_bak",
    "_tmp_y_edges_stamp.csv",
}


def _sorted_run_dirs(chunk_parent: Path) -> list[Path]:
    return sorted(
        (p for p in chunk_parent.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
    )


def _default_out_parent(src: Path) -> Path:
    name = src.name
    if name.endswith("_yedges"):
        return src
    return src.with_name(f"{name}_yedges")


def _load_graph_node_order(chunk_dir: Path) -> list[str]:
    """Prefer gnn_node_index_master.csv; else unique nodes from existing edges by u/v idx."""
    master = chunk_dir / "gnn_node_index_master.csv"
    if master.is_file():
        df = pd.read_csv(master)
        if "node" not in df.columns:
            raise ValueError(f"{master} missing 'node'")
        if "node_idx" in df.columns:
            df = df.sort_values("node_idx")
        return [str(x).strip() for x in df["node"].tolist()]

    edge_csv = chunk_dir / _EDGE_NAME
    if not edge_csv.is_file():
        raise FileNotFoundError(
            f"Need gnn_node_index_master.csv or {_EDGE_NAME} in {chunk_dir}"
        )
    ed = pd.read_csv(edge_csv)
    need = {"from_node", "to_node", "u_idx", "v_idx"}
    miss = need - set(ed.columns)
    if miss:
        raise ValueError(f"{edge_csv} missing columns {sorted(miss)}")
    n = int(max(ed["u_idx"].max(), ed["v_idx"].max()) + 1)
    names = [""] * n
    for _, row in ed.iterrows():
        names[int(row["u_idx"])] = str(row["from_node"]).strip()
        names[int(row["v_idx"])] = str(row["to_node"]).strip()
    if any(not x for x in names):
        raise RuntimeError(f"Could not recover full node order from {edge_csv}")
    return names


def _force_taps_nominal() -> None:
    """Best-effort: OFF controls + transformer secondary taps = 1.0 (draft convention)."""
    try:
        dss.Text.Command("Set ControlMode=OFF")
    except Exception:
        pass
    try:
        dss.Text.Command("batchedit transformer..* wdg=2 tap=1.0")
    except Exception:
        pass
    try:
        dss.Solution.Solve()
    except Exception:
        pass


def _compile_feeder(feeder: str) -> None:
    f = feeder.strip().lower()
    if f == "906":
        ds906._compile_906_lvtestcase_snapshot_setup()
    elif f == "8500":
        ds8500._compile_8500_unbalanced_daily_setup()
    elif f in ("ieee34", "34"):
        if not draft.DSS_FILE.is_file():
            raise FileNotFoundError(draft.DSS_FILE)
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{draft.DSS_FILE}"')
        try:
            dss.Text.Command("Set ControlMode=OFF")
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown feeder {feeder!r}; use 906, 8500, or ieee34")


def _looks_like_google_drive(path: Path) -> bool:
    s = str(path).replace("/", "\\").lower()
    return (
        "\\my drive\\" in s
        or s.startswith("k:\\")
        or "google drive" in s
        or "shortcut-targets-by-id" in s
    )


def _chunked_copy(src: Path, dst: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> None:
    """Buffered copy that avoids Drive File Stream failures of shutil.copy2 on large CSVs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.partial_{os.getpid()}"
    try:
        with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
            while True:
                buf = fsrc.read(chunk_bytes)
                if not buf:
                    break
                fdst.write(buf)
            fdst.flush()
            os.fsync(fdst.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _link_or_copy(src: Path, dst: Path) -> str:
    """
    Mirror one file into the sibling dataset.

    Google Drive File Stream rejects hardlinks (WinError 1) and often fails
    shutil.copy2 on large CSVs (Errno 22). Use chunked copy there; try hardlink
    only on normal local disks.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Resume-friendly: identical size already present.
    if dst.is_file():
        try:
            if dst.stat().st_size == src.stat().st_size and dst.stat().st_size > 0:
                return "skip-exists"
        except OSError:
            pass
        try:
            dst.unlink()
        except OSError:
            pass

    on_drive = _looks_like_google_drive(src) or _looks_like_google_drive(dst)
    if not on_drive:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            _chunked_copy(src, dst)
            return "copy" if attempt == 0 else f"copy-retry{attempt}"
        except OSError as exc:
            last_err = exc
            # Brief backoff for Drive hydration / locks.
            import time

            time.sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise OSError(
        f"Failed to copy after retries:\n  src={src}\n  dst={dst}\n  last={last_err}"
    ) from last_err


def _mirror_chunk_non_edges(src_run: Path, dst_run: Path, *, dry_run: bool) -> int:
    """Copy/hardlink all files except edge CSVs into dst_run."""
    n = 0
    dst_run.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_run.iterdir(), key=lambda p: p.name):
        if not src_file.is_file():
            continue
        if src_file.name in _SKIP_COPY_NAMES:
            continue
        dest = dst_run / src_file.name
        if dry_run:
            print(f"[stamp-y] dry-run mirror {src_file.name} -> {dest}")
            n += 1
            continue
        mode = _link_or_copy(src_file, dest)
        n += 1
        if n <= 5 or src_file.name.startswith("gnn_") or mode.startswith("copy"):
            print(f"[stamp-y]   {mode}: {src_run.name}/{src_file.name}", flush=True)
    return n


def stamp_chunk_parent(
    *,
    feeder: str,
    chunk_parent: Path,
    out_chunk_parent: Path | None = None,
    inplace: bool = False,
    dry_run: bool = False,
    atol: float = 0.0,
) -> dict:
    src_parent = Path(chunk_parent)
    if not src_parent.is_dir():
        raise FileNotFoundError(src_parent)
    runs = _sorted_run_dirs(src_parent)
    if not runs:
        raise FileNotFoundError(f"No run_* under {src_parent}")

    if inplace:
        dst_parent = src_parent
        print(
            "[stamp-y] WARNING: --inplace will overwrite edges in the SOURCE dataset. "
            "Prefer a sibling *_yedges folder."
        )
    else:
        dst_parent = Path(out_chunk_parent) if out_chunk_parent is not None else _default_out_parent(src_parent)
        if dst_parent.resolve() == src_parent.resolve():
            raise ValueError(
                "out_chunk_parent equals source. Pass a different name or use --inplace."
            )

    # Node order from first usable source chunk.
    node_names: list[str] | None = None
    ref_chunk: Path | None = None
    for rd in runs:
        try:
            node_names = _load_graph_node_order(rd)
            ref_chunk = rd
            break
        except Exception:
            continue
    if node_names is None or ref_chunk is None:
        raise RuntimeError(f"Could not resolve graph node order under {src_parent}")

    print(f"[stamp-y] feeder={feeder}")
    print(f"[stamp-y] SRC={src_parent}")
    print(f"[stamp-y] DST={dst_parent}  (inplace={dst_parent.exists()})")
    print(f"[stamp-y] chunks={len(runs)} N={len(node_names)} ref={ref_chunk.name}")

    if not dry_run and not inplace:
        dst_parent.mkdir(parents=True, exist_ok=True)
        manifest = dst_parent / "STAMP_Y_EDGES_README.txt"
        manifest.write_text(
            "Sibling dataset with draft network-Y edges.\n"
            f"Source (unchanged): {src_parent}\n"
            "Node features / PE / meta: mirrored from source.\n"
            "Edges: R_full/X_full = Re(Y_ij)/Im(Y_ij) [siemens], taps≈1, network-only Y.\n",
            encoding="utf-8",
        )

    # Always reload draft so notebook sessions pick up NodeRef-skip fixes.
    importlib.reload(draft)

    _compile_feeder(feeder)
    _force_taps_nominal()
    Y, _ = draft.assemble_network_y_on_nodes(node_names)
    # Y is dense ndarray from assemble_network_y_on_nodes.
    diag = np.diag(Y)
    nnz_offdiag = int(np.count_nonzero(Y) - np.count_nonzero(diag))
    print(f"[stamp-y] Y shape={Y.shape} nnz_offdiag={nnz_offdiag}")

    edge_work_dir = src_parent if inplace else dst_parent
    edge_work_dir.mkdir(parents=True, exist_ok=True)
    tmp_edge = edge_work_dir / "_tmp_y_edges_stamp.csv"
    n_edges = draft.export_y_edges_csv(Y, node_names, tmp_edge, atol=float(atol))
    print(f"[stamp-y] directed Y-edges={n_edges} (R_full/X_full = Re/Im Y [S])")
    if dry_run:
        if tmp_edge.is_file():
            tmp_edge.unlink()
        print(f"[stamp-y] dry-run ok; would write {len(runs)} chunks under {dst_parent}")
        return {
            "feeder": feeder,
            "src_chunk_parent": str(src_parent),
            "dst_chunk_parent": str(dst_parent),
            "inplace": bool(inplace),
            "n_nodes": len(node_names),
            "n_edges": int(n_edges),
            "n_chunks_written": 0,
            "dry_run": True,
        }

    np.save(dst_parent / "Y_network_S_stamped.npy", Y)
    pd.DataFrame(
        {"matrix_index": np.arange(len(node_names)), "opendss_node": node_names}
    ).to_csv(dst_parent / "YNodeOrder_graph_stamped.csv", index=False)

    written = 0
    for src_run in runs:
        dst_run = src_run if inplace else (dst_parent / src_run.name)
        if not inplace:
            _mirror_chunk_non_edges(src_run, dst_run, dry_run=False)

        dest_edge = dst_run / _EDGE_NAME
        if inplace and dest_edge.is_file():
            bak = dst_run / f"{_EDGE_NAME}.ohm_bak"
            if not bak.is_file():
                shutil.copy2(dest_edge, bak)
        shutil.copy2(tmp_edge, dest_edge)
        written += 1
        print(f"[stamp-y] wrote Y-edges -> {dst_run.name}/{_EDGE_NAME}")

    if tmp_edge.is_file():
        tmp_edge.unlink()

    return {
        "feeder": feeder,
        "src_chunk_parent": str(src_parent),
        "dst_chunk_parent": str(dst_parent),
        "inplace": bool(inplace),
        "n_nodes": len(node_names),
        "n_edges": int(n_edges),
        "n_chunks_written": int(written),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--feeder", required=True, choices=("906", "8500", "ieee34", "34"))
    p.add_argument(
        "--chunk_parent",
        type=Path,
        required=True,
        help="SOURCE chunked dataset (left untouched unless --inplace).",
    )
    p.add_argument(
        "--out_chunk_parent",
        type=Path,
        default=None,
        help="NEW dataset root (default: <chunk_parent>_yedges).",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite edges inside --chunk_parent (NOT recommended).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--atol", type=float, default=0.0, help="Drop |Y_ij| <= atol")
    args = p.parse_args()
    info = stamp_chunk_parent(
        feeder=str(args.feeder),
        chunk_parent=Path(args.chunk_parent),
        out_chunk_parent=Path(args.out_chunk_parent) if args.out_chunk_parent else None,
        inplace=bool(args.inplace),
        dry_run=bool(args.dry_run),
        atol=float(args.atol),
    )
    print("[stamp-y] done:", info)
    if not info["inplace"]:
        print("[stamp-y] Train with CHUNK_PARENT =", info["dst_chunk_parent"])


if __name__ == "__main__":
    main()
