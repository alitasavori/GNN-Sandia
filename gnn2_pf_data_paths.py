"""Resolve physics-loss topology CSVs under loadtype_8500_dailyagg."""

from __future__ import annotations

from pathlib import Path

PF_REG_CATALOG_REL = Path("Heterogenous GNN dataset") / "edges" / "hetero_mv_edge_catalog.csv"
PF_CAP_NODES_REL = Path("capacitor_involved_nodes.csv")
REPO_PF_DATA_REL = Path("datasets_gnn2_from pc") / "loadtype_8500_dailyagg"
COLAB_DRIVE_PF_DATA = Path("/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg")


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
    return [repo / REPO_PF_DATA_REL]


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
        "Copy these files into Google Drive (create parent folders as needed):\n"
        f"  {reg_dst}\n"
        f"  {cap_dst}\n"
        "Source in a full GNN2 clone:\n"
        f"  <repo>/{REPO_PF_DATA_REL / PF_REG_CATALOG_REL}\n"
        f"  <repo>/{REPO_PF_DATA_REL / PF_CAP_NODES_REL}"
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
