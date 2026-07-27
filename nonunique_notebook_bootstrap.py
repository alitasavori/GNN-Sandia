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
    # Prefer full chunk caches (many samples) over 1-row slim packs used for Method A timing.
    "datasets_gnn2_from pc/"
    "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt",
    "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt",
    "datasets_gnn2_from pc/run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt",
)

# Feeder-specific tensor caches (prefer Drive paths; fall back to repo / K:).
FEEDER_CACHE_PT_NAMES = {
    "8500": (
        "run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt",
        "run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt",
    ),
    "ieee34": (
        "run_001_scen_0000_0049_seed_3520233__full__nobess__maux80f03146.pt",
    ),
    "906": (
        "run_001_scen_0000_0049_seed_90720233__full__nobess__mauxa6f0b9b7.pt",
    ),
}
FEEDER_CACHE_DIRS = {
    "8500": (
        "cache/gnn_only_chunked_mvagg_full_gine",
        "cache/da_gps_chunked_mvagg_full_gine",
    ),
    "ieee34": (
        "cache/da_gps_chunked_ieee34_full_gine",
        "cache/gnn_only_chunked_ieee34_full_gine",
    ),
    "906": (
        "cache/da_gps_chunked_906_full_gine",
        "cache/gnn_only_chunked_906_full_gine",
    ),
}
FEEDER_RUN_NAMES = {
    "8500": "da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE",
    "ieee34": "da_gps_ieee34_l2_h64_gine_gridPQ_20260717_041907",
    "906": "da_gps_906_l2_h64_gine_gridPQ_20260716_175536",
}

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


def normalize_feeder_key(feeder: str | None) -> str:
    key = str(feeder or "8500").strip().lower()
    aliases = {
        "8500": "8500",
        "ieee8500": "8500",
        "ieee34": "ieee34",
        "34": "ieee34",
        "ieee34_mirzaei": "ieee34",
        "906": "906",
        "lvtestcase": "906",
        "906_lvtestcase": "906",
    }
    if key not in aliases:
        raise ValueError(f"Unknown feeder={feeder!r}; expected 8500, ieee34, or 906")
    return aliases[key]


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


def _datasets_gnn2_roots(repo: Path) -> list[Path]:
    roots: list[Path] = []
    for raw in (
        Path("/content/drive/MyDrive/datasets_gnn2"),
        Path(r"K:\My Drive\datasets_gnn2"),
        repo / "datasets_gnn2",
        repo / "datasets_gnn2_from pc",
    ):
        p = raw.expanduser()
        if p.is_dir():
            roots.append(p.resolve())
    return roots


def resolve_cache_pt(repo: Path, feeder: str | None = "8500") -> Path:
    """Resolve a DA-GPS tensor cache ``.pt``.

    Prefers non-``slim`` packs (many samples) over 1-row Method A timing packs.
    """
    key = normalize_feeder_key(feeder)
    found: list[Path] = []

    def _consider(p: Path) -> None:
        try:
            rp = p.expanduser().resolve()
        except OSError:
            return
        if rp.is_file() and rp not in found:
            found.append(rp)

    if key == "8500":
        for rel in CACHE_CANDIDATES:
            _consider(repo / rel)
    names = FEEDER_CACHE_PT_NAMES.get(key, ())
    dirs = FEEDER_CACHE_DIRS.get(key, ())
    for root in _datasets_gnn2_roots(repo):
        for drel in dirs:
            for name in names:
                _consider(root / drel / name)
        for name in names:
            _consider(root / name)
            for drel in dirs:
                folder = root / drel
                if folder.is_dir():
                    for hit in sorted(folder.glob("run_001*.pt")):
                        _consider(hit)

    if not found:
        raise FileNotFoundError(
            f"No DA-GPS cache .pt for feeder={key} under {repo} / Drive. "
            f"Expected names {list(names)} in {list(dirs)}."
        )

    def _rank(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        slim = 1 if "slim" in name else 0
        # Prefer nobess+regce+maux packs aligned with CCE training when present.
        pref = 0
        if "regce" in name:
            pref -= 2
        if "maux" in name:
            pref -= 1
        if "nobess" in name:
            pref -= 1
        return (slim, pref, str(p))

    found.sort(key=_rank)
    return found[0]


def resolve_feeder_run_dir(repo: Path, feeder: str, run_name: str | None = None) -> Path:
    """Resolve training run dir (prefers Drive ``datasets_gnn2/runs/<name>``)."""
    key = normalize_feeder_key(feeder)
    name = (run_name or FEEDER_RUN_NAMES.get(key, "")).strip()
    if not name:
        raise FileNotFoundError(f"No default run name for feeder={key}")
    cands: list[Path] = []
    for root in _datasets_gnn2_roots(repo):
        cands.append(root / "runs" / name)
    cands.append(repo / "gnn2_architecture_search" / "attention checkpoints" / name)
    cands.append(repo / "datasets_gnn2" / "runs" / name)
    for p in cands:
        p = p.expanduser().resolve()
        if p.is_dir() and (
            (p / "da_gps_multitask_best.pt").is_file() or (p / "training_last.pt").is_file()
        ):
            return p
    raise FileNotFoundError(
        f"Checkpoint run folder not found for feeder={key} name={name}. Tried:\n  "
        + "\n  ".join(str(c) for c in cands)
    )


def resolve_feeder_checkpoint(run_dir: Path) -> Path:
    """Prefer ``da_gps_multitask_best.pt``, then ``training_last.pt``."""
    best = run_dir / "da_gps_multitask_best.pt"
    last = run_dir / "training_last.pt"
    if best.is_file():
        return best.resolve()
    if last.is_file():
        return last.resolve()
    raise FileNotFoundError(f"No checkpoint in {run_dir} (need da_gps_multitask_best.pt or training_last.pt)")


def resolve_inference_device(device: str | None = "auto") -> str:
    from nonunique_opendss_daily import resolve_da_gps_device

    if device is None or str(device).strip().lower() in ("", "auto", "default"):
        return resolve_da_gps_device(None)
    return resolve_da_gps_device(str(device))


def configure_gnn_inference_env(*, on_colab: bool, device: str | None = None) -> None:
    """Match Colab daily-compare defaults (CUDA graphs + deferred D2H); CPU disables compile."""
    dev = str(device or "").strip().lower()
    if dev == "cpu":
        os.environ["GNN_TORCH_COMPILE"] = "0"
        os.environ.pop("GNN_CUDA_GRAPHS", None)
        os.environ.pop("GNN_DEFER_D2H", None)
        return
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
    feeder: str = "8500"


def bootstrap_warmstart_notebook(
    *,
    repo: Path | str | None = None,
    device: str | None = "auto",
    out_parent: Path | str | None = None,
    run_tag: str | None = None,
    reload_modules: bool = True,
    checkpoint_subdir: str = DEFAULT_CHECKPOINT_SUBDIR,
    feeder: str = "8500",
    run_name: str | None = None,
) -> NotebookBootstrap:
    """``chdir`` + ``sys.path`` + env for Colab or local; return standard warm-start paths."""
    on_colab = is_colab()
    repo_path = resolve_notebook_repo(repo)
    os.chdir(repo_path)
    repo_s = str(repo_path)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    os.environ["GNN2_REPO_ROOT"] = repo_s

    if reload_modules:
        for name in RELOAD_MODULES:
            sys.modules.pop(name, None)

    feeder_key = normalize_feeder_key(feeder)
    dev = resolve_inference_device(device)
    configure_gnn_inference_env(on_colab=on_colab, device=dev)
    day1 = repo_path / "a representativ days"

    if feeder_key == "8500" and run_name is None:
        run_dir = (repo_path / checkpoint_subdir).resolve()
        cache_pt = resolve_cache_pt(repo_path, feeder="8500")
        checkpoint = run_dir / "training_last.pt"
        if not checkpoint.is_file():
            checkpoint = resolve_feeder_checkpoint(run_dir)
        load_profile = day1 / "load_day_004.csv"
        irr_profile = day1 / "irr_day_004.csv"
    else:
        run_dir = resolve_feeder_run_dir(repo_path, feeder_key, run_name=run_name)
        cache_pt = resolve_cache_pt(repo_path, feeder=feeder_key)
        checkpoint = resolve_feeder_checkpoint(run_dir)
        if feeder_key == "ieee34":
            mir = repo_path / "new dss from dr mirzaei"
            load_profile = mir / "5minDayShape.csv"
            irr_profile = mir / "5MinuteIrradiance.csv"
            if not load_profile.is_file():
                load_profile = day1 / "load_day_004.csv"
            if not irr_profile.is_file():
                irr_profile = day1 / "irr_day_004.csv"
        else:
            load_profile = day1 / "load_day_004.csv"
            irr_profile = day1 / "irr_day_004.csv"

    tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_parent is not None:
        out_dir = Path(out_parent).expanduser().resolve() / tag
    else:
        out_dir = repo_path / "warmstart_band_runs" / tag

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    for label, p in (
        ("load profile", load_profile),
        ("irr profile", irr_profile),
        ("der profile", day1 / "battery_arbitrage_der_injection.csv"),
    ):
        if label == "der profile" and feeder_key != "8500":
            # DER off by default for ieee34/906 Method A cells; file optional.
            continue
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
        load_profile=load_profile,
        irr_profile=irr_profile,
        der_profile=day1 / "battery_arbitrage_der_injection.csv",
        out_dir=out_dir,
        feeder=feeder_key,
    )
    print(
        f"[bootstrap] env={'Colab' if on_colab else 'local'}  repo={boot.repo}\n"
        f"[bootstrap] feeder={boot.feeder}  device={boot.device}  cache={boot.cache_pt.name}\n"
        f"[bootstrap] checkpoint={boot.checkpoint}\n"
        f"[bootstrap] out_dir={boot.out_dir}",
        flush=True,
    )
    return boot


__all__ = [
    "FEEDER_CACHE_DIRS",
    "FEEDER_CACHE_PT_NAMES",
    "FEEDER_RUN_NAMES",
    "NotebookBootstrap",
    "bootstrap_warmstart_notebook",
    "configure_gnn_inference_env",
    "is_colab",
    "normalize_feeder_key",
    "resolve_cache_pt",
    "resolve_feeder_checkpoint",
    "resolve_feeder_run_dir",
    "resolve_inference_device",
    "resolve_notebook_repo",
]
