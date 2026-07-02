"""Resolve physics-loss topology CSVs under loadtype_8500_dailyagg."""

from __future__ import annotations

from pathlib import Path

PF_REG_CATALOG_REL = Path("Heterogenous GNN dataset") / "edges" / "hetero_mv_edge_catalog.csv"
PF_CAP_NODES_REL = Path("capacitor_involved_nodes.csv")
REPO_COLAB_PF_REL = Path("colab_pf_data")
REPO_PF_DATA_REL = Path("datasets_gnn2_from pc") / "loadtype_8500_dailyagg"
COLAB_DRIVE_PF_DATA = Path("/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg")
DATASETS_PC_REL = Path("datasets_gnn2_from pc")
NODES_PV_CSV_NAMES: tuple[str, ...] = (
    "gnn_node_features_and_targets.csv",
    "gnn_node_features_and_targets_mvagg.csv",
)
NODES_PV_REQUIRED_COLS: tuple[str, ...] = ("sample_id", "node", "p_pv_kw", "q_pv_kvar")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.expanduser().resolve()).lower()
        except OSError:
            key = str(p.expanduser()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p.expanduser())
    return out


def _repo_pf_roots(repo: Path) -> list[Path]:
    """Repo-bundled PF topology roots (minimal Colab bundle first, then full dailyagg)."""
    return [repo / REPO_COLAB_PF_REL, repo / REPO_PF_DATA_REL]


def candidate_pf_data_roots(
    *,
    repo: Path,
    preferred: Path | None = None,
    chunk_parent: Path | None = None,
) -> list[Path]:
    """Ordered search roots for mvagg PF topology CSVs."""
    roots: list[Path] = []
    if preferred is not None:
        roots.append(preferred)
    if chunk_parent is not None:
        roots.append(chunk_parent.parent / "loadtype_8500_dailyagg")
    roots.extend(_repo_pf_roots(repo))
    for clone in (Path("/content/GNN-Sandia"), Path("/content/GNN2"), Path("/content/GNN")):
        if clone.is_dir():
            roots.extend(_repo_pf_roots(clone))
    roots.append(COLAB_DRIVE_PF_DATA)
    roots.append(Path(r"D:\datasets\loadtype_8500_dailyagg"))
    return _dedupe_paths(roots)


def _pf_copy_instructions(drive_root: Path = COLAB_DRIVE_PF_DATA) -> str:
    reg_dst = drive_root / PF_REG_CATALOG_REL
    cap_dst = drive_root / PF_CAP_NODES_REL
    return (
        "After git pull, physics topology should resolve from repo "
        f"<repo>/{REPO_COLAB_PF_REL} (bundled ~1.3 MB).\n"
        "If missing, copy into Google Drive (create parent folders as needed):\n"
        f"  {reg_dst}\n"
        f"  {cap_dst}\n"
        "Source in a full GNN2 clone:\n"
        f"  <repo>/{REPO_COLAB_PF_REL / PF_REG_CATALOG_REL}\n"
        f"  <repo>/{REPO_COLAB_PF_REL / PF_CAP_NODES_REL}\n"
        f"  (or full dailyagg: <repo>/{REPO_PF_DATA_REL / PF_REG_CATALOG_REL})"
    )


def _nodes_pv_csv_usable(path: Path) -> bool:
    import pandas as pd

    try:
        cols = set(pd.read_csv(path, nrows=0).columns.astype(str).str.strip())
    except Exception:
        return False
    return set(NODES_PV_REQUIRED_COLS).issubset(cols)


def _repo_clone_roots(repo: Path) -> list[Path]:
    """Colab clone dirs that look like full GNN2 checkouts (not empty C:\\content stubs on Windows)."""
    roots = [repo]
    trainer = "train_da_gps_multitask_complex_voltage_gine.py"
    for clone in (Path("/content/GNN-Sandia"), Path("/content/GNN2"), Path("/content/GNN")):
        if clone.is_dir() and (clone / trainer).is_file():
            roots.append(clone)
    return _dedupe_paths(roots)


def candidate_nodes_pv_csv_paths(
    *,
    repo: Path,
    data_root: Path | None = None,
    chunk_parent: Path | None = None,
) -> list[Path]:
    """Ordered search paths for PV columns (p_pv_kw, q_pv_kvar) keyed by sample_id/node."""
    roots: list[Path] = []
    if data_root is not None:
        roots.append(data_root)
    if chunk_parent is not None:
        for run_dir in sorted(chunk_parent.glob("run_*")):
            roots.append(run_dir)
        roots.append(chunk_parent.parent / "loadtype_8500_dailyagg")
    for checkout in _repo_clone_roots(repo):
        roots.extend(
            [
                checkout / DATASETS_PC_REL / "loadtype_8500",
                checkout / DATASETS_PC_REL / "loadtype_8500_dailyagg",
                checkout / DATASETS_PC_REL / "original_plus_cap",
            ]
        )
    roots.append(Path(r"D:\datasets\loadtype_8500"))
    paths: list[Path] = []
    for root in roots:
        for name in NODES_PV_CSV_NAMES:
            paths.append(root / name)
    return _dedupe_paths(paths)


def resolve_nodes_pv_csv(
    *,
    repo: Path,
    data_root: Path | None = None,
    chunk_parent: Path | None = None,
) -> Path:
    """Return nodes CSV with PV injection columns for offline physics verification."""
    tried: list[str] = []
    for path in candidate_nodes_pv_csv_paths(
        repo=repo, data_root=data_root, chunk_parent=chunk_parent
    ):
        if not path.is_file():
            tried.append(f"{path}  (missing)")
            continue
        if not _nodes_pv_csv_usable(path):
            tried.append(f"{path}  (missing PV columns)")
            continue
        return path.resolve()
    tried_lines = "\n  ".join(tried) or "(none)"
    raise FileNotFoundError(
        "Physics verification requires a nodes CSV with "
        "sample_id, node, p_pv_kw, q_pv_kvar.\n"
        f"Searched:\n  {tried_lines}\n"
        "Typical locations:\n"
        f"  <repo>/{DATASETS_PC_REL / 'loadtype_8500' / NODES_PV_CSV_NAMES[0]}\n"
        f"  <chunk>/run_*/{NODES_PV_CSV_NAMES[1]}"
    )


def resolve_pf_catalog_paths(
    *,
    repo: Path,
    preferred_root: Path | None = None,
    chunk_parent: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return (reg_catalog, cap_nodes_csv, resolved_pf_data_root)."""
    tried: list[Path] = []
    for root in candidate_pf_data_roots(
        repo=repo, preferred=preferred_root, chunk_parent=chunk_parent
    ):
        tried.append(root)
        reg = root / PF_REG_CATALOG_REL
        cap = root / PF_CAP_NODES_REL
        if reg.is_file() and cap.is_file():
            return reg.resolve(), cap.resolve(), root.resolve()

    missing_names = [str(PF_REG_CATALOG_REL), str(PF_CAP_NODES_REL)]
    tried_lines = "\n  ".join(str(p) for p in tried) or "(none)"
    raise FileNotFoundError(
        "Physics loss requires mvagg topology CSVs:\n"
        f"  {missing_names[0]}\n"
        f"  {missing_names[1]}\n"
        f"Searched PF_DATA_ROOT candidates:\n  {tried_lines}\n"
        f"{_pf_copy_instructions()}\n"
        "Or set PF_DATA_ROOT to a folder that contains both files "
        "(e.g. repo datasets_gnn2_from pc/loadtype_8500_dailyagg)."
    )
