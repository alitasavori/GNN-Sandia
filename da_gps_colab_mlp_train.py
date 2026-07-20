"""Colab/local launcher for plain MLP volt-only training on chunk_parent datasets.

Reuses ``train_da_gps_multitask_complex_voltage_gine.py`` chunk_parent loop, metrics,
checkpointing (``da_gps_multitask_best.pt``, ``training_last.pt``), and ``da_gps_report.json``.
"""

from __future__ import annotations

import datetime
import fnmatch
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from nonunique_notebook_bootstrap import is_colab, normalize_feeder_key, resolve_notebook_repo

DRIVE_ROOT = Path("/content/drive")
MYDRIVE_DATA = DRIVE_ROOT / "MyDrive/datasets_gnn2"
_FULL_CHUNK_MIN_SCEN = 50


@dataclass(frozen=True)
class FeederMlpTrainConfig:
    feeder: str
    chunk_parent_colab: Path
    chunk_parent_win: Path
    chunk_parent_repo_rel: str
    da_cache_name: str
    gnn_cache_name: str
    runs_parent_colab: Path
    runs_parent_win: Path
    run_name_prefix: str
    # Default MLP width/depth for launch_mlp_training(feeder) when hidden/layers omitted.
    # OUT_DIR tags use run_name_prefix -> e.g. mlp_8500_l4_h256_<timestamp>.
    hidden: int
    layers: int
    use_full_span_glob: bool = True


FEEDER_MLP_CONFIGS: dict[str, FeederMlpTrainConfig] = {
    # Largest feeder: wider/deeper MLP than the shared l2_h64 baseline.
    "8500": FeederMlpTrainConfig(
        feeder="8500",
        chunk_parent_colab=MYDRIVE_DATA / "original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40",
        da_cache_name="da_gps_chunked_mvagg_full_mlp",
        gnn_cache_name="gnn_only_chunked_mvagg_full_gine",
        runs_parent_colab=Path("/content/GNN-Sandia/gnn2_architecture_search/attention checkpoints"),
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="mlp_8500_l{layers}_h{hidden}",
        hidden=256,
        layers=4,
        use_full_span_glob=False,
    ),
    # Small feeder: keep original small MLP baseline (l2_h64).
    "ieee34": FeederMlpTrainConfig(
        feeder="ieee34",
        chunk_parent_colab=MYDRIVE_DATA / "original_ieee34_mirzaei_chunked",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_ieee34_mirzaei_chunked"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_ieee34_mirzaei_chunked",
        da_cache_name="da_gps_chunked_ieee34_full_mlp",
        gnn_cache_name="gnn_only_chunked_ieee34_full_gine",
        runs_parent_colab=MYDRIVE_DATA / "runs",
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="mlp_ieee34_l{layers}_h{hidden}",
        hidden=64,
        layers=2,
        use_full_span_glob=True,
    ),
    # Medium feeder: mid-size MLP (between ieee34 baseline and 8500).
    "906": FeederMlpTrainConfig(
        feeder="906",
        chunk_parent_colab=MYDRIVE_DATA / "original_906_lvtestcase_chunked",
        chunk_parent_win=Path(r"K:\My Drive\datasets_gnn2\original_906_lvtestcase_chunked"),
        chunk_parent_repo_rel="datasets_gnn2_from pc/original_906_lvtestcase_chunked",
        da_cache_name="da_gps_chunked_906_full_mlp",
        gnn_cache_name="gnn_only_chunked_906_full_gine",
        runs_parent_colab=MYDRIVE_DATA / "runs",
        runs_parent_win=Path(r"K:\My Drive\datasets_gnn2\runs"),
        run_name_prefix="mlp_906_l{layers}_h{hidden}",
        hidden=128,
        layers=3,
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


def _find_node_pe_csv(chunk_parent: Path) -> Path:
    hits = sorted(chunk_parent.glob("run_*/gnn_node_index_master.csv"))
    if not hits:
        raise FileNotFoundError(f"No run_*/gnn_node_index_master.csv under {chunk_parent}")
    return hits[0]


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


@dataclass
class MlpTrainLaunchResult:
    feeder: str
    chunk_parent: Path
    da_cache_root: Path
    gnn_cache_root: Path
    out_dir: Path
    cmd: list[str]
    device: str


def launch_mlp_training(
    feeder: str,
    *,
    repo: Path | str | None = None,
    device: str = "auto",
    smoke_test: bool = False,
    smoke_chunk_count: int = 3,
    smoke_epochs: int = 15,
    smoke_patience: int = 5,  # calendar epochs since meaningful best (min_delta)
    full_epochs: int = 200,
    # Aggressive MLP-only early stop: flat baselines should die by ~epoch 25–40, not crawl to 200.
    # DA-GPS/GINE keep trainer defaults (patience=30, min_delta=1e-4) unless they pass these flags.
    full_patience: int = 15,
    min_delta: float = 1e-3,  # ignore val_tot noise like 0.9963→0.9956 for patience
    seed: int = 42,
    hidden: int | None = None,
    layers: int | None = None,
    node_emb_dim: int = 2,
    batch_size: int = 64,
    mount_drive: bool = True,
    interactive_pause: bool = True,
) -> MlpTrainLaunchResult:
    """Preflight + subprocess train for one feeder. Mount Drive on Colab when ``mount_drive=True``.

    ``hidden`` / ``layers`` default to ``FEEDER_MLP_CONFIGS[feeder]`` (8500: 256/4, 906: 128/3,
    ieee34: 64/2). OUT_DIR embeds size via ``run_name_prefix`` (e.g. ``mlp_8500_l4_h256_...``).

    ``interactive_pause`` (default True): after each eval_every=10 checkpoint, pause for
    continue/stop. Colab subprocesses are non-TTY — create ``CONTINUE`` or ``STOP`` under OUT_DIR.
    """
    key = normalize_feeder_key(feeder)
    cfg = FEEDER_MLP_CONFIGS[key]
    if hidden is None:
        hidden = cfg.hidden
    if layers is None:
        layers = cfg.layers
    on_colab = is_colab()
    if on_colab and mount_drive and not _drive_mounted():
        from google.colab import drive

        drive.mount("/content/drive")

    repo_path = resolve_notebook_repo(repo)
    os.chdir(repo_path)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("GNN2_REPO_ROOT", str(repo_path))

    dev = _configure_device(device)

    if on_colab:
        chunk_parent_raw = cfg.chunk_parent_colab
        da_cache_raw = MYDRIVE_DATA / f"cache/{cfg.da_cache_name}"
        gnn_cache_raw = MYDRIVE_DATA / f"cache/{cfg.gnn_cache_name}"
        runs_parent_raw = cfg.runs_parent_colab
        colab_fb = cfg.chunk_parent_colab
    elif os.name == "nt":
        chunk_parent_raw = cfg.chunk_parent_win
        da_cache_raw = cfg.chunk_parent_win.parent / "cache" / cfg.da_cache_name
        gnn_cache_raw = cfg.chunk_parent_win.parent / "cache" / cfg.gnn_cache_name
        if key == "8500":
            runs_parent_raw = repo_path / "gnn2_architecture_search/attention checkpoints"
        else:
            runs_parent_raw = cfg.runs_parent_win
        colab_fb = None
    else:
        chunk_parent_raw = repo_path / cfg.chunk_parent_repo_rel
        da_cache_raw = repo_path / "datasets_gnn2_from pc/cache" / cfg.da_cache_name
        gnn_cache_raw = repo_path / "datasets_gnn2_from pc/cache" / cfg.gnn_cache_name
        if key == "8500":
            runs_parent_raw = repo_path / "gnn2_architecture_search/attention checkpoints"
        else:
            runs_parent_raw = repo_path / "datasets_gnn2_from pc/runs"
        colab_fb = None

    chunk_parent = _resolve_data_path(chunk_parent_raw, label="CHUNK_PARENT", colab_fallback=colab_fb)
    da_cache_root = _resolve_data_path(
        da_cache_raw,
        label="DA_CACHE_ROOT",
        colab_fallback=MYDRIVE_DATA / f"cache/{cfg.da_cache_name}" if on_colab else None,
    )
    gnn_cache_root = _resolve_data_path(
        gnn_cache_raw,
        label="GNN_CACHE_ROOT",
        colab_fallback=MYDRIVE_DATA / f"cache/{cfg.gnn_cache_name}" if on_colab else None,
    )
    runs_parent = _resolve_data_path(
        runs_parent_raw,
        label="RUNS_PARENT",
        colab_fallback=MYDRIVE_DATA / "runs" if on_colab else None,
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

    node_pe = _find_node_pe_csv(chunk_parent)
    da_cache_root.mkdir(parents=True, exist_ok=True)
    gnn_cache_root.mkdir(parents=True, exist_ok=True)
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

    num_workers = 0 if os.name == "nt" else 4
    cmd = [
        sys.executable,
        "-u",
        "train_da_gps_multitask_complex_voltage_gine.py",
        "--model",
        "mlp",
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
        "--node_feature_cols",
        "p_load_kw,q_load_kvar,p_pv_kw",
        "--exclude_bess_features",
        "--node_pe_csv",
        str(node_pe),
        "--node_pe_cols",
        "auto",
        "--out_dir",
        str(out_dir),
        "--cache_dir",
        str(da_cache_root),
        "--bootstrap_gnn_cache_dir",
        str(gnn_cache_root),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--hidden",
        str(hidden),
        "--layers",
        str(layers),
        "--node_emb_dim",
        str(node_emb_dim),
        "--lr",
        "5e-4",
        "--weight_decay",
        "1e-5",
        "--patience",
        str(patience),
        "--min_delta",
        str(min_delta),
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
        "--log_every",
        "0",
        "--checkpoint_every",
        "10",
        "--early_stop_on",
        "total",
        "--dropout",
        "0.1",
    ]
    if interactive_pause:
        cmd.append("--interactive_pause")
        os.environ.setdefault("TRAIN_INTERACTIVE", "1")
    else:
        cmd.append("--no_interactive_pause")

    print(f"=== Preflight (MLP volt-only {key}) ===")
    print(f"REPO:           {repo_path}")
    print(f"DEVICE:         {dev}")
    print(f"SMOKE_TEST:     {smoke_test}")
    print(f"HIDDEN/LAYERS:  {hidden}/{layers}")
    print(f"EPOCHS:         {epochs}")
    print(
        f"PATIENCE:       {patience}  (calendar epochs since meaningful best; "
        f"early_stop_on=total≈val_volt for mlp)"
    )
    print(
        f"MIN_DELTA:      {min_delta:g}  (patience resets only if best-val >= min_delta; "
        f"ckpt still saves on any improve)"
    )
    print(f"CHUNK_PARENT:   {chunk_parent}")
    print(f"CHUNK_GLOB:     {chunk_glob}")
    print(f"DA_CACHE_ROOT:  {da_cache_root}")
    print(f"GNN_CACHE_ROOT: {gnn_cache_root}")
    print(f"RUNS_PARENT:    {runs_parent}")
    print(f"OUT_DIR:        {out_dir}")
    print(f"INTERACTIVE_PAUSE: {interactive_pause}")
    if interactive_pause:
        print(
            "  After each eval_every=10: create empty CONTINUE or STOP under OUT_DIR "
            "(Colab subprocess has no TTY). Or type c/s if running trainer in a terminal."
        )
        print(f"  e.g.  !touch '{out_dir / 'STOP'}'   /   !touch '{out_dir / 'CONTINUE'}'")
    print(f"Found {len(chunks)} chunk(s)")
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

    print("\nTraining completed.")
    print("Run dir:", out_dir.resolve())
    print("Checkpoint (best):", (out_dir / "da_gps_multitask_best.pt").resolve())
    print("Checkpoint (last):", (out_dir / "training_last.pt").resolve())
    print("Report:", (out_dir / "da_gps_report.json").resolve())

    return MlpTrainLaunchResult(
        feeder=key,
        chunk_parent=chunk_parent,
        da_cache_root=da_cache_root,
        gnn_cache_root=gnn_cache_root,
        out_dir=out_dir,
        cmd=cmd,
        device=dev,
    )


__all__ = [
    "FEEDER_MLP_CONFIGS",
    "FeederMlpTrainConfig",
    "MlpTrainLaunchResult",
    "launch_mlp_training",
    "select_training_chunk_glob",
]
