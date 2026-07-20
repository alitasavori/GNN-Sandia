"""Colab/local launcher for PowerFlowMultiNet — oracle device states (GENConv).

Mirrors ``da_gps_colab_mlp_train.py`` UX: Drive mount, chunk selection (span>=50),
preflight print, subprocess to ``train_powerflowmultinet.py``.

Paper-faithful defaults (arXiv:2403.00892v3): epochs=1000, lambda_sub=1.0 (joint
V/φ + substation P/Q), hidden=128, L=12 for ieee34/906/8500, effective batch≈128.

Speed defaults: feeder-aware ``batch_size`` / ``grad_accum``, ``interactive_pause=False``,
and on Colab ``cache_local=True`` (copy Drive caches → ``/content/pfmn_cache`` then train).

Artifacts: ``pfmn_oracle_best.pt``, ``training_last.pt``, ``pfmn_report.json``,
``run_manifest.json``. Cache schema ``__pfmn_oracle_v2.pt`` (delete old v1 caches).
"""

from __future__ import annotations

import datetime
import fnmatch
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from nonunique_notebook_bootstrap import is_colab, normalize_feeder_key, resolve_notebook_repo

DRIVE_ROOT = Path("/content/drive")
MYDRIVE_DATA = DRIVE_ROOT / "MyDrive/datasets_gnn2"
_FULL_CHUNK_MIN_SCEN = 50
_CACHE_SUFFIX = "__pfmn_oracle_v2.pt"
_COLAB_LOCAL_CACHE_ROOT = Path("/content/pfmn_cache")


@dataclass(frozen=True)
class FeederPfmnTrainConfig:
    feeder: str
    chunk_parent_colab: Path
    chunk_parent_win: Path
    chunk_parent_repo_rel: str
    cache_name: str
    runs_parent_colab: Path
    runs_parent_win: Path
    run_name_prefix: str
    # Implementation choices (paper silent on exact L/hidden; unified across feeders).
    hidden: int
    layers: int
    # Feeder-aware microbatch; keep batch×accum ≈128.
    batch_size: int
    grad_accum: int
    use_full_span_glob: bool = True


# Unified architecture across feeders (paper silent on exact L/hidden).
_PFMN_HIDDEN = 128
_PFMN_LAYERS = 12

FEEDER_PFMN_CONFIGS: dict[str, FeederPfmnTrainConfig] = {
    "8500": FeederPfmnTrainConfig(
        feeder="8500",
        chunk_parent_colab=MYDRIVE_DATA / "original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40",
        cache_name="pfmn_chunked_8500_oracle_v2",
        runs_parent_colab=Path("/content/GNN-Sandia/gnn2_architecture_search/attention checkpoints"),
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="pfmn_oracle_8500_l{layers}_h{hidden}",
        hidden=_PFMN_HIDDEN,
        layers=_PFMN_LAYERS,
        batch_size=16,
        grad_accum=8,
        use_full_span_glob=False,
    ),
    "ieee34": FeederPfmnTrainConfig(
        feeder="ieee34",
        chunk_parent_colab=MYDRIVE_DATA / "original_ieee34_mirzaei_chunked",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_ieee34_mirzaei_chunked"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_ieee34_mirzaei_chunked",
        cache_name="pfmn_chunked_ieee34_oracle_v2",
        runs_parent_colab=MYDRIVE_DATA / "runs",
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="pfmn_oracle_ieee34_l{layers}_h{hidden}",
        hidden=_PFMN_HIDDEN,
        layers=_PFMN_LAYERS,
        batch_size=64,
        grad_accum=2,
        use_full_span_glob=True,
    ),
    "906": FeederPfmnTrainConfig(
        feeder="906",
        chunk_parent_colab=MYDRIVE_DATA / "original_906_lvtestcase_chunked",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_906_lvtestcase_chunked"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_906_lvtestcase_chunked",
        cache_name="pfmn_chunked_906_oracle_v2",
        runs_parent_colab=MYDRIVE_DATA / "runs",
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="pfmn_oracle_906_l{layers}_h{hidden}",
        hidden=_PFMN_HIDDEN,
        layers=_PFMN_LAYERS,
        batch_size=32,
        grad_accum=4,
        use_full_span_glob=True,
    ),
}


def _is_windows_drive_path(path) -> bool:
    s = str(path).strip().replace("/", "\\")
    return len(s) >= 2 and s[1] == ":" and s[0].isalpha()


def _drive_mounted() -> bool:
    return DRIVE_ROOT.is_dir() and (DRIVE_ROOT / "MyDrive").is_dir()


def _resolve_data_path(path: Path, *, label: str, colab_fallback: Path | None = None) -> Path:
    raw = str(path)
    if _is_windows_drive_path(raw):
        if is_colab() and colab_fallback is not None:
            warnings.warn(
                f"{label}={raw!r} is a Windows path on Linux/Colab; using {colab_fallback} instead.",
                UserWarning,
                stacklevel=2,
            )
            return colab_fallback.expanduser().resolve()
        if is_colab():
            raise ValueError(
                f"{label}={raw!r} is a Windows absolute path and invalid on Colab. "
                f"Mount Drive and use a /content/drive/MyDrive/... path."
            )
        return Path(raw).expanduser().resolve()
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def _sorted_run_chunk_dirs(chunk_parent: Path) -> list[Path]:
    return sorted(
        (p for p in chunk_parent.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
    )


def _parse_run_scen_span(name: str) -> int | None:
    m = re.search(r"_scen_(\d+)_(\d+)(?:_|$)", name)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if end < start:
        return None
    return end - start + 1


def _classify_run_chunks(chunk_parent: Path, *, min_full_span: int = _FULL_CHUNK_MIN_SCEN):
    all_runs = _sorted_run_chunk_dirs(chunk_parent)
    full, other = [], []
    for p in all_runs:
        span = _parse_run_scen_span(p.name)
        if span is not None and span >= min_full_span:
            full.append(p)
        else:
            other.append((p, span))
    return all_runs, full, other


def select_training_chunk_glob(
    chunk_parent: Path,
    *,
    smoke: bool,
    smoke_count: int,
    min_full_span: int = _FULL_CHUNK_MIN_SCEN,
) -> str:
    all_runs, full, other = _classify_run_chunks(chunk_parent, min_full_span=min_full_span)
    if not all_runs:
        raise FileNotFoundError(f"No run_* under {chunk_parent}")
    print(
        f"Chunk selection (SMOKE_TEST={smoke}, min_full_scen_span={min_full_span}): "
        f"{len(all_runs)} run_* found; {len(full)} full-size, {len(other)} other/smoke"
    )
    if smoke:
        pool = full if full else all_runs
        source = "full-size" if full else "any run_* (no full-size chunks found)"
        if len(pool) < smoke_count:
            raise ValueError(
                f"SMOKE_CHUNK_COUNT={smoke_count} but only {len(pool)} {source} under {chunk_parent}"
            )
        kept = pool[:smoke_count]
        return ",".join(p.name for p in kept)
    if not full:
        raise RuntimeError(
            f"SMOKE_TEST=False but no full-size chunks (scen span >= {min_full_span}) under {chunk_parent}."
        )
    return ",".join(p.name for p in full)


def _chunks_from_subdir_glob(chunk_parent: Path, glob_pat: str) -> list[Path]:
    glob_pat = str(glob_pat).strip()
    if "," in glob_pat:
        allowed = {s.strip() for s in glob_pat.split(",") if s.strip()}
        chunks = sorted(
            (p for p in chunk_parent.iterdir() if p.is_dir() and p.name in allowed),
            key=lambda p: p.name,
        )
        missing = allowed - {p.name for p in chunks}
        if missing:
            raise FileNotFoundError(f"Missing smoke chunk folders: {sorted(missing)}")
        return chunks
    return sorted(
        (p for p in chunk_parent.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob_pat)),
        key=lambda p: p.name,
    )


def _configure_device(device: str) -> str:
    dev = str(device or "auto").strip().lower()
    if dev not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda, or cpu; got {device!r}")
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif dev == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda but CUDA is not available")
    return dev


def _estimate_n_samples(chunks: list[Path]) -> int:
    """Estimate sample count from run_* scen spans (fallback 50 per chunk)."""
    total = 0
    for p in chunks:
        span = _parse_run_scen_span(p.name)
        total += span if span is not None else 50
    return total


def _sync_cache_files(src_dir: Path, dst_dir: Path, *, label: str) -> int:
    """Copy ``*__pfmn_oracle_v2.pt`` from src→dst when missing or newer. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if not src_dir.is_dir():
        return 0
    for src in sorted(src_dir.glob(f"*{_CACHE_SUFFIX}")):
        dst = dst_dir / src.name
        if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime + 1e-6:
            shutil.copy2(src, dst)
            n += 1
    if n:
        print(f"[pfmn cache_local] {label}: copied {n} file(s) {src_dir} -> {dst_dir}", flush=True)
    return n


def _prepare_train_cache(
    drive_cache: Path,
    *,
    cache_name: str,
    cache_local: bool,
    on_colab: bool,
) -> tuple[Path, Path | None]:
    """Return (train_cache_dir, drive_dir_to_sync_back_or_None).

    On Colab with ``cache_local=True``: copy existing Drive caches to
    ``/content/pfmn_cache/<name>`` once, train from local SSD, sync new caches back after.
    """
    drive_cache.mkdir(parents=True, exist_ok=True)
    if not (cache_local and on_colab):
        return drive_cache, None
    local = _COLAB_LOCAL_CACHE_ROOT / cache_name
    local.mkdir(parents=True, exist_ok=True)
    n = _sync_cache_files(drive_cache, local, label="Drive→local")
    print(
        f"[pfmn cache_local] training I/O from {local} "
        f"(Drive mirror={drive_cache}; pre-copied={n})",
        flush=True,
    )
    return local, drive_cache


@dataclass
class PfmnTrainLaunchResult:
    feeder: str
    chunk_parent: Path
    cache_root: Path
    out_dir: Path
    cmd: list[str]
    device: str


def launch_pfmn_training(
    feeder: str,
    *,
    repo: Path | str | None = None,
    device: str = "auto",
    smoke_test: bool = False,
    smoke_chunk_count: int = 3,
    smoke_epochs: int = 15,
    smoke_patience: int = 5,
    full_epochs: int = 1000,
    full_patience: int = 80,
    seed: int = 42,
    hidden: int | None = None,
    layers: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    lambda_sub: float = 1.0,
    mount_drive: bool = True,
    interactive_pause: bool = False,
    cache_local: bool | None = None,
) -> PfmnTrainLaunchResult:
    """Preflight + subprocess train for one feeder (oracle tap/cap PFMN baseline).

    Unified defaults for ieee34 / 906 / 8500: L=12, hidden=128, epochs=1000,
    lambda_sub=1.0 (joint V/φ + substation P/Q), effective batch ≈128.
    Feeder-aware microbatch (ieee34 64×2, 906 32×4, 8500 16×8).
    OUT_DIR: ``pfmn_oracle_{feeder}_l{L}_h{H}_{timestamp}``.

    ``interactive_pause`` (default False): if True, after each eval_every=10 checkpoint
    pause for continue/stop. Colab subprocesses are non-TTY — create ``CONTINUE`` or
    ``STOP`` under OUT_DIR.

    ``cache_local`` (default True on Colab): copy Drive ``*__pfmn_oracle_v2.pt`` caches
    to ``/content/pfmn_cache/...``, train from local disk, sync new caches back to Drive.
    """
    key = normalize_feeder_key(feeder)
    cfg = FEEDER_PFMN_CONFIGS[key]
    if hidden is None:
        hidden = cfg.hidden
    if layers is None:
        layers = cfg.layers
    if batch_size is None:
        batch_size = cfg.batch_size
    if grad_accum is None:
        grad_accum = cfg.grad_accum
    on_colab = is_colab()
    if cache_local is None:
        cache_local = on_colab
    if on_colab and mount_drive and not _drive_mounted():
        from google.colab import drive

        drive.mount("/content/drive")

    repo_path = resolve_notebook_repo(repo)
    os.chdir(repo_path)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("GNN2_REPO_ROOT", str(repo_path))

    # Never inherit TRAIN_INTERACTIVE=1 from a prior cell unless user asks to pause.
    if not interactive_pause:
        os.environ.pop("TRAIN_INTERACTIVE", None)

    dev = _configure_device(device)

    if on_colab:
        chunk_parent_raw = cfg.chunk_parent_colab
        drive_cache_raw = MYDRIVE_DATA / f"cache/{cfg.cache_name}"
        runs_parent_raw = cfg.runs_parent_colab
        colab_fb = cfg.chunk_parent_colab
    elif os.name == "nt":
        chunk_parent_raw = cfg.chunk_parent_win
        drive_cache_raw = cfg.chunk_parent_win.parent / "cache" / cfg.cache_name
        if key == "8500":
            runs_parent_raw = repo_path / "gnn2_architecture_search/attention checkpoints"
        else:
            runs_parent_raw = cfg.runs_parent_win
        colab_fb = None
    else:
        chunk_parent_raw = repo_path / cfg.chunk_parent_repo_rel
        drive_cache_raw = repo_path / "datasets_gnn2_from pc/cache" / cfg.cache_name
        if key == "8500":
            runs_parent_raw = repo_path / "gnn2_architecture_search/attention checkpoints"
        else:
            runs_parent_raw = repo_path / "datasets_gnn2_from pc/runs"
        colab_fb = None

    chunk_parent = _resolve_data_path(chunk_parent_raw, label="CHUNK_PARENT", colab_fallback=colab_fb)
    drive_cache = _resolve_data_path(
        drive_cache_raw,
        label="PFMN_CACHE_ROOT",
        colab_fallback=MYDRIVE_DATA / f"cache/{cfg.cache_name}" if on_colab else None,
    )
    runs_parent = _resolve_data_path(
        runs_parent_raw,
        label="RUNS_PARENT",
        colab_fallback=MYDRIVE_DATA / "runs" if on_colab else None,
    )
    cache_root, cache_sync_back = _prepare_train_cache(
        drive_cache,
        cache_name=cfg.cache_name,
        cache_local=bool(cache_local),
        on_colab=on_colab,
    )

    if smoke_test:
        epochs = smoke_epochs
        patience = smoke_patience
        if cfg.use_full_span_glob:
            chunk_glob = select_training_chunk_glob(
                chunk_parent, smoke=True, smoke_count=smoke_chunk_count
            )
        else:
            names = [p.name for p in _sorted_run_chunk_dirs(chunk_parent)]
            if len(names) < smoke_chunk_count:
                raise ValueError(f"SMOKE_CHUNK_COUNT={smoke_chunk_count} but only {len(names)} run_*")
            chunk_glob = ",".join(names[:smoke_chunk_count])
    else:
        epochs = full_epochs
        patience = full_patience
        chunk_glob = (
            select_training_chunk_glob(chunk_parent, smoke=False, smoke_count=smoke_chunk_count)
            if cfg.use_full_span_glob
            else "run_*"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    runs_parent.mkdir(parents=True, exist_ok=True)

    chunks = _chunks_from_subdir_glob(chunk_parent, chunk_glob)
    if not chunks:
        raise RuntimeError(f"No folders for CHUNK_GLOB={chunk_glob!r} under {chunk_parent}")

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_suffix = "_smoke" if smoke_test else ""
    out_dir = runs_parent / (
        cfg.run_name_prefix.format(layers=layers, hidden=hidden) + f"{smoke_suffix}_{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_est = _estimate_n_samples(chunks)
    n_train_est = max(1, int(round(n_est * 0.80)))
    micro_per_epoch = max(1, math.ceil(n_train_est / max(1, batch_size)))
    opt_steps_per_epoch = max(1, math.ceil(micro_per_epoch / max(1, grad_accum)))

    num_workers = 0 if os.name == "nt" else 4
    cmd = [
        sys.executable,
        "-u",
        "train_powerflowmultinet.py",
        "--device",
        dev,
        "--chunk_parent",
        str(chunk_parent),
        "--chunk_subdir_glob",
        chunk_glob,
        "--nodes_csv",
        "gnn_node_features_and_targets_mvagg.csv",
        "--edge_catalog_csv",
        "gnn_edges_phase_static.csv",
        "--meta_csv",
        "gnn_sample_meta.csv",
        "--out_dir",
        str(out_dir),
        "--cache_dir",
        str(cache_root),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--grad_accum",
        str(grad_accum),
        "--hidden",
        str(hidden),
        "--layers",
        str(layers),
        "--lr",
        "1e-3",
        "--weight_decay",
        "1e-5",
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--train_frac",
        "0.80",
        "--val_frac",
        "0.10",
        "--sample_frac",
        "1.0",
        "--num_workers",
        str(num_workers),
        "--eval_every",
        "10",
        "--checkpoint_every",
        "10",
        "--dropout",
        "0.1",
        "--lambda_sub",
        str(lambda_sub),
    ]
    if interactive_pause:
        cmd.append("--interactive_pause")
        os.environ.setdefault("TRAIN_INTERACTIVE", "1")
    else:
        cmd.append("--no_interactive_pause")

    print(f"=== Preflight (PowerFlowMultiNet oracle {key}) ===")
    print(f"REPO:           {repo_path}")
    print(f"DEVICE:         {dev}")
    print(f"SMOKE_TEST:     {smoke_test}")
    print(f"HIDDEN/LAYERS:  {hidden}/{layers}  (unified; paper silent on exact sizes)")
    print(f"EPOCHS:         {epochs}  (paper=1000; smoke uses {smoke_epochs})")
    print(f"LAMBDA_SUB:     {lambda_sub}  (1=joint V/φ+sub P/Q; 0=volt-only)")
    print(
        f"EFF_BATCH:      ~{batch_size * grad_accum}  "
        f"(feeder default batch={batch_size} × accum={grad_accum})"
    )
    print(f"EST_SAMPLES:    ~{n_est}  (~{n_train_est} train @ train_frac=0.80)")
    print(
        f"STEPS/EPOCH:    ~{micro_per_epoch} microbatches  "
        f"(~{opt_steps_per_epoch} optimizer steps)"
    )
    print(f"CHUNK_PARENT:   {chunk_parent}")
    print(f"CHUNK_GLOB:     {chunk_glob}")
    print(f"PFMN_CACHE:     {cache_root}  (schema {_CACHE_SUFFIX})")
    if cache_sync_back is not None:
        print(f"CACHE_LOCAL:    True  (Drive mirror={cache_sync_back})")
    else:
        print(f"CACHE_LOCAL:    False")
    print(f"RUNS_PARENT:    {runs_parent}")
    print(f"OUT_DIR:        {out_dir}")
    print(f"INTERACTIVE_PAUSE: {interactive_pause}")
    if interactive_pause:
        print(
            "  After each eval_every=10: create empty CONTINUE or STOP under OUT_DIR "
            "(Colab subprocess has no TTY). Or type c/s if running trainer in a terminal."
        )
        print(f"  e.g.  !touch '{out_dir / 'STOP'}'   /   !touch '{out_dir / 'CONTINUE'}'")
    else:
        print(
            "  Pause off (default). Pass interactive_pause=True to stop every eval_every=10."
        )
    print(f"Found {len(chunks)} chunk(s)")
    print("Artifacts: pfmn_oracle_best.pt, training_last.pt, pfmn_report.json, run_manifest.json")
    print("=================")
    print("\nRunning:\n ", " ".join(cmd), "\n", flush=True)

    with subprocess.Popen(
        cmd,
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        rc = proc.wait()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    if cache_sync_back is not None:
        _sync_cache_files(cache_root, cache_sync_back, label="local→Drive")

    print("\nTraining completed.")
    print("Run dir:", out_dir.resolve())
    print("Checkpoint (best):", (out_dir / "pfmn_oracle_best.pt").resolve())
    print("Checkpoint (last):", (out_dir / "training_last.pt").resolve())
    print("Report:", (out_dir / "pfmn_report.json").resolve())
    print("Manifest:", (out_dir / "run_manifest.json").resolve())

    return PfmnTrainLaunchResult(
        feeder=key,
        chunk_parent=chunk_parent,
        cache_root=cache_root,
        out_dir=out_dir,
        cmd=cmd,
        device=dev,
    )


__all__ = [
    "FEEDER_PFMN_CONFIGS",
    "FeederPfmnTrainConfig",
    "PfmnTrainLaunchResult",
    "launch_pfmn_training",
    "select_training_chunk_glob",
]
