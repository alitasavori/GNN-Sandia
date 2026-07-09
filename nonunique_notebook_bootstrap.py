"""Colab / local notebook bootstrap for nonunique warm-start and daily-compare cells."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_CHECKPOINT_SUBDIR = (
    "gnn2_architecture_search/attention checkpoints/"
    "da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE"
)
CACHE_CANDIDATES = (
    "datasets_gnn2_from pc/"
    "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt",
    "datasets_gnn2_from pc/run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt",
)
RELOAD_MODULES = (
    "nonunique_daily_experiment",
    "nonunique_da_gps_warmstart_band_daily",
    "nonunique_opendss_daily",
    "nonunique_da_gps_daily_compare",
    "run_da_gps_daily_opendss_compare",
    "compare_gnn_inference_utils",
    "nonunique_notebook_bootstrap",
)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return Path("/content").is_dir() and bool(os.environ.get("COLAB_RELEASE_TAG"))


def resolve_notebook_repo(explicit: Path | str | None = None) -> Path:
    """Find GNN2 repo root (marker: ``nonunique_opendss_daily.py``)."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if (p / "nonunique_opendss_daily.py").is_file():
            return p
        raise FileNotFoundError(f"GNN2 repo not found at {p}")

    candidates: list[Path] = []
    env = os.environ.get("GNN2_REPO_ROOT", "").strip()
    if env:
        candidates.append(Path(env))
    if is_colab():
        candidates.extend([Path("/content/GNN2"), Path("/content/GNN-Sandia")])
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    for raw in candidates:
        p = raw.expanduser().resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / "nonunique_opendss_daily.py").is_file():
            return p
    raise FileNotFoundError(
        "Could not locate GNN2 repo. Set GNN2_REPO_ROOT, clone to /content/GNN2 on Colab, "
        "or run from the repo directory."
    )


def resolve_cache_pt(repo: Path) -> Path:
    for rel in CACHE_CANDIDATES:
        p = (repo / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No DA-GPS cache .pt under {repo}. Expected one of: {list(CACHE_CANDIDATES)}"
    )


def resolve_inference_device(device: str | None = "auto") -> str:
    from nonunique_opendss_daily import resolve_da_gps_device

    if device is None or str(device).strip().lower() in ("", "auto", "default"):
        return resolve_da_gps_device(None)
    return resolve_da_gps_device(str(device))


def configure_gnn_inference_env(*, on_colab: bool) -> None:
    """Match Colab daily-compare defaults (CUDA graphs + deferred D2H)."""
    os.environ.setdefault("GNN_CUDA_GRAPHS", "1")
    os.environ.setdefault("GNN_DEFER_D2H", "1")
    if on_colab:
        os.environ.setdefault("GNN_TORCH_COMPILE", "0")


@dataclass(frozen=True)
class NotebookBootstrap:
    repo: Path
    on_colab: bool
    device: str
    day1: Path
    run_dir: Path
    cache_pt: Path
    checkpoint: Path
    load_profile: Path
    irr_profile: Path
    der_profile: Path
    out_dir: Path


def bootstrap_warmstart_notebook(
    *,
    repo: Path | str | None = None,
    device: str | None = "auto",
    out_parent: Path | str | None = None,
    run_tag: str | None = None,
    reload_modules: bool = True,
    checkpoint_subdir: str = DEFAULT_CHECKPOINT_SUBDIR,
) -> NotebookBootstrap:
    """``chdir`` + ``sys.path`` + env for Colab or local; return standard warm-start paths."""
    on_colab = is_colab()
    repo_path = resolve_notebook_repo(repo)
    os.chdir(repo_path)
    repo_s = str(repo_path)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    os.environ["GNN2_REPO_ROOT"] = repo_s
    configure_gnn_inference_env(on_colab=on_colab)

    if reload_modules:
        for name in RELOAD_MODULES:
            sys.modules.pop(name, None)

    dev = resolve_inference_device(device)
    day1 = repo_path / "a representativ days"
    run_dir = (repo_path / checkpoint_subdir).resolve()
    cache_pt = resolve_cache_pt(repo_path)
    checkpoint = run_dir / "training_last.pt"
    tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_parent is not None:
        out_dir = Path(out_parent).expanduser().resolve() / tag
    else:
        out_dir = repo_path / "warmstart_band_runs" / tag

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    for label, p in (
        ("load_day_004", day1 / "load_day_004.csv"),
        ("irr_day_004", day1 / "irr_day_004.csv"),
        ("der profile", day1 / "battery_arbitrage_der_injection.csv"),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {label}: {p}")

    boot = NotebookBootstrap(
        repo=repo_path,
        on_colab=on_colab,
        device=dev,
        day1=day1,
        run_dir=run_dir,
        cache_pt=cache_pt,
        checkpoint=checkpoint,
        load_profile=day1 / "load_day_004.csv",
        irr_profile=day1 / "irr_day_004.csv",
        der_profile=day1 / "battery_arbitrage_der_injection.csv",
        out_dir=out_dir,
    )
    print(
        f"[bootstrap] env={'Colab' if on_colab else 'local'}  repo={boot.repo}\n"
        f"[bootstrap] device={boot.device}  cache={boot.cache_pt.name}\n"
        f"[bootstrap] checkpoint={boot.checkpoint.parent.name}/training_last.pt\n"
        f"[bootstrap] out_dir={boot.out_dir}",
        flush=True,
    )
    return boot


__all__ = [
    "NotebookBootstrap",
    "bootstrap_warmstart_notebook",
    "configure_gnn_inference_env",
    "is_colab",
    "resolve_cache_pt",
    "resolve_inference_device",
    "resolve_notebook_repo",
]
