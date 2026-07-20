"""
Daily OpenDSS vs **DA-GPS** checkpoint: |V| / angle profiles + timing buckets.

Supports:

- **GINE** checkpoints from ``train_da_gps_multitask_complex_voltage_gine.py`` (``mpnn.conv.*``).
- **Legacy EdgeAttn** checkpoints from ``train_da_gps_multitask_complex_voltage.py`` (``mpnn.msg.*`` / gated MPNN),
  selected automatically from the state dict.

What counts as a “daily run” here
----------------------------------
We mirror ``compare_homo_mv_daily`` / ``compare_hetero_mv_daily`` timing style. Default feeder is
**8500** solar-unbalanced (``Master-PV2MW-inv.dss``). Pass ``--feeder ieee34`` or ``--feeder 906`` to
compile Mirzaei ``IEEE34_PV.dss`` or LVTestCase ``Master.dss`` instead. Profile CLI args scale loads /
PV in Python; they do **not** switch the DSS master (use ``--feeder`` for that).

For **8500**, irradiance also rebinds ``Loadshape.IrradDay001`` then neutralizes it for snapshot
solves; each step sets ``Pmpp = Pmpp0 × m_irr[i]`` explicitly. ``Load..Daily`` shapes are detached
so per-step ``kW`` / ``kvar`` are not scaled twice. ieee34 / 906 skip IrradDay001 when absent.
  - Read OpenDSS |V| and voltage angle (deg) for all ``*.[123]`` buses; build DA-GPS node inputs, run the
    checkpoint once, denormalize the complex voltage head, scatter predicted |V| and angle (``atan2``).
    Node ``p_pv_kw`` is recomputed each step as ``Pmpp0 × m_irr[i]`` on bus phases — same as dataset
    ``p_nominal = pmpp_set[i] * m_pv_t`` in ``_apply_snapshot_with_pv`` when ``pmpp_set ≈ Pmpp0``.
  - **Voltage PNGs** (|V| + angle, two stacked panels): path from ``--voltage-png-subdir`` when set;
    with ``--plot-all-cache-nodes`` and empty subdir, default ``daily_voltage/`` under ``--out-dir``.
    With explicit ``--plot-node`` only, default is ``--out-dir`` (pass ``--voltage-png-subdir daily_voltage``
    to mirror the bulk layout). Cap/reg/meta figures always use ``--out-dir``.
    By default, |V| y-axis is **auto**
    (min/max of both curves ± padding). Pass ``--v-ylim-fixed``
    with ``--ymin``/``ymax`` to pin the **|V|** panel only; the angle panel stays auto-scaled.
    Per-node PNG filenames use a zero-padded ``rNNNN_oMMMM`` segment (rank / count) ordered **worst → best**
    by per-node |V| MAE (pu), plus compact ``pu…`` / ``ang…`` tags and ``bus_phase`` so folder sort matches rank;
    basenames stay short for Windows ``MAX_PATH``, with an extended-length path prefix when the full path is long.
    ``--plot-all-max-nodes N`` (with ``--plot-all-cache-nodes``) keeps only the **worst N** by |V| MAE; ``0`` plots **every** cache∩circuit node.
- **Cap bank + regulator taps + meta aux:** after each converged solve, reads OpenDSS
    fields aligned with ``run_original_style_dataset_8500_unbalanced`` / ``run_daily_aggregate_dataset_8500``:
    capacitor steps, regulator taps, per-``PVSystem`` post-solve P/Q (``TotalPowers``),
    and ``Circuit.Losses()`` for ``P_loss_total_post_kw`` / ``Q_loss_total_post_kvar``.
    Meta-aux heads are denormalized with ``pv_mean.pt`` / ``pv_std.pt`` when present.
    Cap-bank figures use one y-axis (DSS bank-on 0/1 vs model sigmoid); regulator taps and
    meta-aux scalars use a **single shared** y-axis for OpenDSS and DA-GPS (same units).
    Regulator **DSS** taps resolve ``reg_target_cols`` with the same ``_read_reg_control_state`` keys as
    dataset generation, plus **case-insensitive** and **stem** matching so training’s lowercased
    ``gnn_sample_meta`` column names still align with OpenDSS ``RegControls.Name()`` casing.

**OOD / aggressiveness knob**

- ``--daily-stress``: amplifies deviations of the profile from 1.0:
    ``m_eff = clip(1 + (m_raw - 1) * (1 + stress), lo, hi)``.
  ``0`` reproduces the nominal profile shape; larger values exaggerate peaks/valleys
  (more aggressive vs typical training-time snapshots).
- ``--scenario-scale``: uniform extra multiplier on top of the stressed profile
  (stress the whole day’s loading).

**Feature-time MVP (with vs without BESS)**

- ``p_load_kw`` / ``q_load_kvar`` are filled from **live** OpenDSS bus-phase aggregation
  of scaled loads (same construction as ``compare_homo_mv_daily``), optionally using
  ``--mv-sx-mapping`` when your ``node`` strings are MV keys that aggregate two LV phases.
  On **non-BESS** checkpoints (no ``p_bess_kw`` / ``q_bess_kvar``), active DER is **added** here on each
  cached ``bus.phase`` row for that bus (``_add_der_to_pq_load_columns``); if the cache only has e.g.
  ``l2917359.1``, that bus's share of P/Q is split across **cached** phases only.
- ``p_pv_kw``: if present in ``node_feature_cols``, each step uses **``Pmpp0 × m_irr[i]``** split on PV
  bus phases (``_collect_pv_to_busph_weights``), matching ``run_original_style_dataset_8500_unbalanced``
  ``p_nominal = pmpp_set[i] * m_pv_t`` when ``pmpp_set`` is the nameplate read after compile. Other ``x``
  columns still come from ``--ref-sample-index`` on the tensor cache.
- ``p_bess_kw`` / ``q_bess_kvar``: from the **reference** cache row unless DER is on for a **BESS**
  checkpoint, in which case they are **zeroed** then filled from the DER schedule
  (``_fill_der_as_bess_node_features``). **Non-BESS** checkpoints keep all cache tails except load/PV
  and encode DER only via added load P/Q. OpenDSS ``New Generator`` DER is applied for both when DER
  CLI args are valid.

Inputs
-------
- ``--run-dir``: training output folder with norm tensors (``x_mean.pt``, …) and any of:
  ``da_gps_report.json`` (end of training), ``da_gps_run_manifest.json`` (written before the loop),
  or ``training_last.pt`` / ``da_gps_multitask_best.pt`` with chunk metadata (mid-train / snapshot folders).
- ``--cache-pt``: one chunk tensor cache ``*.pt`` with ``x``, ``node_to_local`` (same graph
  as training) — used for reference features + node order.
- ``--edge-csv`` (optional): defaults from ``da_gps_report.json`` (first chunk or single-run
  ``edges_csv``).

Timing buckets match the **OpenDSS vs GNN pipeline** diagram used elsewhere in this repo:
OpenDSS total ≈ **apply** + **Solve() only** + **collect** (``collect`` = Python read of |V| and angle after solve for MAE/plots);
GNN total ≈ **feature generation** + **forward** (see printed ``dss_*_ms`` / ``gnn_*_ms`` block and ``da_gps_daily_run_summary.json``).

GNN deployment inference opts (env, default off unless noted):
``GNN_CUDA_GRAPHS`` (default on CUDA), ``GNN_DEFER_D2H`` (default on CUDA),
``GNN_BATCH_STEPS`` / ``gnn_batch_steps`` kwarg, ``GNN_TORCH_COMPILE=1``, ``GNN_TF32=0``.
Timing: ``gnn_setup_once_s``, ``gnn_per_step_s``, ``gnn_total_wall_s`` in summary JSON and console.

For meta-aux mismatches (DSS vs GNN curves), pass ``--meta-debug`` or set ``GNN_DAILY_META_DEBUG=1`` to log
normalized vs denormalized heads, DSS meta rows, clock, and ``x_n`` means at a few timesteps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opendssdirect as dss
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_daily_aggregate_dataset_8500 as rd8500
from compare_gnn_inference_utils import (
    DailyGnnInferenceRunner,
    build_scatter_indices,
    configure_cuda_inference,
    maybe_torch_compile,
    read_gnn_batch_steps,
)
from compare_mv_daily_timing import (
    compute_mv_daily_timing_metrics,
    print_mv_daily_timing_summary,
    sync_inference_device,
)
from nonunique_opendss_daily import log_da_gps_device, resolve_da_gps_device
from compare_opendss_snapshot_helpers import (
    apply_explicit_loads_and_pv_pmpp,
    discover_pv_system_names,
    prepare_parity_profiles,
    read_pv_base_pmpp_kw,
    reassert_snapshot_and_set_clock,
    setup_da_gps_snapshot_opendss,
    snapshot_step_hr_sec,
    step_irradiance_multiplier,
    step_load_multiplier,
)
from train_da_gps_multitask_complex_voltage import DAGPSModel as DAGPSModelEdgeAttn
from train_da_gps_multitask_complex_voltage_gine import (
    DAGPSModel as DAGPSModelGine,
    _reg_indices_to_tap_pu,
)
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges


REPO_ROOT = Path(__file__).resolve().parent

_FEEDER_ALIASES = {
    "8500": "8500",
    "ieee8500": "8500",
    "ieee34": "ieee34",
    "34": "ieee34",
    "ieee34_mirzaei": "ieee34",
    "906": "906",
    "lvtestcase": "906",
    "906_lvtestcase": "906",
}


def normalize_feeder(feeder: str | None) -> str:
    key = str(feeder or "8500").strip().lower()
    if key not in _FEEDER_ALIASES:
        raise ValueError(
            f"Unknown feeder={feeder!r}. Expected one of: 8500, ieee34, 906 "
            f"(aliases: {sorted(set(_FEEDER_ALIASES) - {'8500', 'ieee34', '906'})})."
        )
    return _FEEDER_ALIASES[key]


def _ieee34_dss_path() -> Path:
    return (REPO_ROOT / "new dss from dr mirzaei" / "IEEE34_PV.dss").resolve()


def _compile_feeder_master(feeder: str) -> dict[str, str]:
    """Compile the OpenDSS master for ``feeder`` ∈ {8500, ieee34, 906}. Returns path metadata."""
    key = normalize_feeder(feeder)
    if key == "8500":
        rd8500._compile_8500_solar_unbalanced_pv_daily_setup()
        return {
            "feeder": key,
            "master_dss": str(rd8500.MASTER_PV2_INV_DSS.resolve()),
            "model_dir": str(rd8500.SOLAR_UNBAL_8500_DIR.resolve()),
        }
    if key == "ieee34":
        dss_path = _ieee34_dss_path()
        if not dss_path.is_file():
            raise FileNotFoundError(f"Missing IEEE34 Mirzaei DSS: {dss_path}")
        inj.compile_once()
        try:
            dss.Text.Command("Set ControlMode=Static")
            dss.Text.Command(f"Set MaxControlIter={int(getattr(inj, 'MAX_CONTROL_ITER', 200))}")
        except Exception:
            pass
        return {
            "feeder": key,
            "master_dss": str(dss_path),
            "model_dir": str(dss_path.parent),
        }
    # 906 LVTestCase
    import run_original_style_dataset_906_lvtestcase as ds906

    ds906._compile_906_lvtestcase_snapshot_setup()
    try:
        ds906._detach_yearly_daily_from_loads()
    except Exception:
        pass
    return {
        "feeder": key,
        "master_dss": str(Path(ds906.MASTER_DSS).resolve()),
        "model_dir": str(Path(ds906.MODEL_DIR).resolve()),
    }


def _default_feeder_profiles(feeder: str, *, out_dir: Path, npts: int, step_min: float) -> tuple[Path, Path]:
    """Default (load_csv, irr_csv) when CLI profile paths are empty."""
    key = normalize_feeder(feeder)
    day1 = REPO_ROOT / "a representativ days"
    if key == "ieee34":
        mir = REPO_ROOT / "new dss from dr mirzaei"
        load_p = mir / "5minDayShape.csv"
        irr_p = mir / "5MinuteIrradiance.csv"
        if load_p.is_file() and irr_p.is_file():
            return load_p.resolve(), irr_p.resolve()
    if key == "906":
        load_p = day1 / "load_day_004.csv"
        if not load_p.is_file():
            load_p = REPO_ROOT / "8500-node" / "5minDayShape.csv"
        # No PV on LVTestCase — unity irradiance multiplier CSV for prepare_parity_profiles.
        irr_p = out_dir / f"_unity_irr_npts{int(npts)}.csv"
        t = np.arange(int(npts), dtype=np.float64) * (float(step_min) / 60.0)
        ones = np.ones(int(npts), dtype=np.float64)
        irr_p.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(irr_p, np.column_stack([t, ones]), delimiter=",", fmt="%.8g")
        if load_p.is_file():
            return load_p.resolve(), irr_p.resolve()
    # 8500 (and fallbacks)
    load_p = rd8500._resolve_daily_profile_csv(None)
    irr_p = (rd8500.SOLAR_UNBAL_8500_DIR / "irr_day_001.csv").resolve()
    if not irr_p.is_file():
        irr_p = (day1 / "irr_day_004.csv").resolve()
    return Path(load_p).resolve(), Path(irr_p).resolve()


def _state_dict_is_legacy_edgeattn(state_dict: dict[str, torch.Tensor]) -> bool:
    """``train_da_gps_multitask_complex_voltage.py`` uses ``EdgeAttnMPNN`` (msg/gate/node_mlp); GINE uses ``conv.*``."""
    return any(".mpnn.msg." in k for k in state_dict)


def _state_dict_uses_reg_ce_heads(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("reg_ce_heads.") for k in state_dict)


def _state_dict_per_device_cap(state_dict: dict[str, torch.Tensor]) -> bool:
    return "cap_W" in state_dict and state_dict["cap_W"] is not None


def _state_dict_per_device_reg(state_dict: dict[str, torch.Tensor]) -> bool:
    return _state_dict_uses_reg_ce_heads(state_dict) or (
        "reg_W" in state_dict and state_dict["reg_W"] is not None
    )


def _infer_reg_nclasses_from_state_dict(state_dict: dict[str, torch.Tensor], n_reg: int) -> list[int] | None:
    """``reg_ce_heads.{j}.weight`` shape ``[n_classes_j, hidden]``."""
    if not _state_dict_uses_reg_ce_heads(state_dict):
        return None
    out: list[int] = []
    for j in range(int(n_reg)):
        w = state_dict.get(f"reg_ce_heads.{j}.weight")
        if w is None:
            return None
        out.append(int(w.shape[0]))
    return out


def _resolve_reg_loss_mode(bundle: dict[str, object], state_dict: dict[str, torch.Tensor]) -> str:
    hp = bundle.get("hyperparameters") or {}
    rl = str(bundle.get("reg_loss", "") or hp.get("reg_loss", "mse")).strip().lower()
    if rl in ("ce", "cce", "cross_entropy"):
        return "ce"
    if _state_dict_uses_reg_ce_heads(state_dict):
        return "ce"
    if rl in ("mae",):
        return "mae"
    return "mse"


def _parse_cols(spec: str) -> list[str]:
    return [c.strip() for c in str(spec).split(",") if c.strip()]


def _load_mv_sx_mapping(path: Path) -> list[dict[str, str]]:
    rules: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mv = (row.get("mv_node") or "").strip()
            lv1 = (row.get("lv_x_node_1") or "").strip()
            lv2 = (row.get("lv_x_node_2") or "").strip()
            sx1 = (row.get("sx_node_1") or "").strip()
            sx2 = (row.get("sx_node_2") or "").strip()
            if not mv or not lv1 or not lv2:
                continue
            la, lb = (sx1, sx2) if sx1 and sx2 else (lv1, lv2)
            rules.append({"mv_key": mv.lower(), "load_a": la.lower(), "load_b": lb.lower()})
    return rules


def _safe_stem(s: str) -> str:
    t = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s).strip())[:96]
    return t or "da_gps"


def _path_str_for_png_write(p: Path) -> str:
    """Return a path string safe for ``matplotlib`` / PIL on Windows.

    Very long ``--out-dir`` trees hit ``MAX_PATH`` (~260); PIL then raises ``FileNotFoundError``.
    Prefix ``\\\\?\\`` enables extended-length paths when needed.
    """
    rp = p.resolve()
    s = str(rp)
    if os.name == "nt" and len(s) >= 220 and not s.startswith("\\\\?\\"):
        if not s.startswith("\\\\"):
            s = "\\\\?\\" + s
    return s


def _voltage_daily_png_basename(
    *,
    stem_safe: str,
    rk: str,
    tot: str,
    n_mae: float,
    n_mae_ang: float,
    node_fn: str,
) -> str:
    """Short basename: deep output dirs + long ``daily_*`` names exceed Windows ``MAX_PATH``."""
    st = stem_safe[:24] if stem_safe else "ckpt"
    mv = f"{n_mae:.4f}" if np.isfinite(n_mae) else "nan"
    ma = f"{n_mae_ang:.3f}" if np.isfinite(n_mae_ang) else "nan"
    return f"v_{st}_r{rk}_o{tot}_pu{mv}_ang{ma}_{node_fn}.png"


def _resolve_da_gps_checkpoint(
    ckpt_path: Path, run_dir: Path, report_dir: Path | None = None
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Load weights + architecture metadata for ``DAGPSModel``.

    - **da_gps_multitask_best.pt**: full bundle; weights = ``model_state_dict``.
    - **training_last.pt**: weights prefer ``best_model_state_dict`` (best val during training), else
      ``model_state_dict``; ``n_cap`` / ``n_reg`` / … are taken from ``da_gps_multitask_best.pt`` in
      ``run_dir``, ``run_dir.parent``, ``report_dir``, or ``ckpt_path.parent`` if the resume file omits them.
    """
    rd = report_dir if report_dir is not None else run_dir
    raw: dict[str, object] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in raw:
        raise KeyError(f"{ckpt_path} missing 'model_state_dict'")

    best_sd = raw.get("best_model_state_dict")
    if best_sd is not None and isinstance(best_sd, dict) and len(best_sd) > 0:
        state = best_sd
        print(
            f"[da_gps_daily] checkpoint={ckpt_path.name}: loading **best_model_state_dict**",
            flush=True,
        )
    else:
        state = raw["model_state_dict"]
        print(
            f"[da_gps_daily] checkpoint={ckpt_path.name}: loading **model_state_dict** (no best snapshot in file)",
            flush=True,
        )

    if "n_cap" in raw and "n_reg" in raw:
        return raw, state

    meta_candidates = [
        run_dir / "da_gps_multitask_best.pt",
        run_dir.parent / "da_gps_multitask_best.pt",
        rd / "da_gps_multitask_best.pt",
        ckpt_path.parent / "da_gps_multitask_best.pt",
    ]
    meta_pt: Path | None = None
    for p in meta_candidates:
        if p.is_file() and p.resolve() != ckpt_path.resolve():
            meta_pt = p
            break
    if meta_pt is None:
        raise FileNotFoundError(
            f"{ckpt_path} has no n_cap/n_reg architecture fields (typical of training_last.pt). "
            f"Place **da_gps_multitask_best.pt** next to it or in the parent folder (metadata only), "
            f"or pass a full **da_gps_multitask_best.pt** as --checkpoint."
        )
    meta_raw: dict[str, object] = torch.load(meta_pt, map_location="cpu", weights_only=False)
    print(f"[da_gps_daily] architecture metadata from {meta_pt}", flush=True)
    return meta_raw, state


def _chunk_name_from_cache_stem(cache_pt: Path) -> str | None:
    """``run_001_...__full__....pt`` → ``run_001_...`` (chunk folder basename)."""
    stem = cache_pt.stem
    if "__" not in stem:
        return None
    return stem.split("__", 1)[0]


_DA_GPS_REPORT_JSON = "da_gps_report.json"
_DA_GPS_MANIFEST_JSON = "da_gps_run_manifest.json"


def _find_upward_file(start: Path, filename: str, *, max_levels: int = 3) -> Path | None:
    cur = Path(start).resolve()
    for _ in range(int(max_levels) + 1):
        p = cur / filename
        if p.is_file():
            return p.resolve()
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _resolve_norm_dir(run_dir: Path, *, ckpt_hint: Path | None = None) -> Path:
    """Directory containing ``x_mean.pt`` (run_dir, checkpoint folder, or up to 3 parents each)."""
    for start in (run_dir, ckpt_hint.parent if ckpt_hint is not None else None):
        if start is None:
            continue
        hit = _find_upward_file(start, "x_mean.pt", max_levels=3)
        if hit is not None:
            return hit.parent.resolve()
    raise FileNotFoundError(
        f"Missing x_mean.pt near --run-dir={run_dir}"
        + (f" or checkpoint parent {ckpt_hint.parent}" if ckpt_hint is not None else "")
        + " (searched up to 3 parent levels).\n"
        "Copy x_mean.pt, x_std.pt, y_mean.pt, y_std.pt from the Colab training OUT_DIR "
        "(written before epoch 1), or set --run-dir to that OUT_DIR."
    )


def _resolve_default_checkpoint(run_dir: Path, norm_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        ckpt_path = Path(checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"{ckpt_path}\n"
                "Pass an existing ``da_gps_multitask_best.pt`` or ``training_last.pt``."
            )
        return ckpt_path
    for cand in (
        run_dir / "da_gps_multitask_best.pt",
        run_dir / "training_last.pt",
        norm_dir / "da_gps_multitask_best.pt",
        norm_dir / "training_last.pt",
    ):
        if cand.is_file():
            print(f"[da_gps_daily] resolved default checkpoint -> {cand.resolve()}", flush=True)
            return cand.resolve()
    raise FileNotFoundError(
        f"No checkpoint in run_dir={run_dir} or norm_dir={norm_dir}.\n"
        "Pass ``--checkpoint`` to ``training_last.pt`` or ``da_gps_multitask_best.pt``."
    )


def _synthesize_report_from_checkpoint(ckpt_path: Path, cache_pt: Path) -> dict[str, object]:
    """Build minimal report dict from ``training_last.pt`` / ``da_gps_multitask_best.pt`` metadata."""
    raw: dict[str, object] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in raw and "best_model_state_dict" not in raw:
        raise KeyError(f"{ckpt_path} is not a DA-GPS checkpoint (missing model weights).")
    if "n_cap" not in raw or "n_reg" not in raw:
        raise KeyError(
            f"{ckpt_path} lacks architecture metadata (n_cap/n_reg). "
            "Use a checkpoint from train_da_gps_multitask_complex_voltage_gine.py after the metadata update, "
            "or copy ``da_gps_run_manifest.json`` from the training OUT_DIR into --run-dir."
        )

    chunk_parent = str(raw.get("chunk_parent", "") or "").strip()
    chunk_folders = [str(p) for p in (raw.get("chunk_folders") or [])]
    cn = _chunk_name_from_cache_stem(cache_pt)
    chunks = list(chunk_folders)
    if not chunks and chunk_parent and cn:
        chunks = [str(Path(chunk_parent) / cn)]

    meta_cols = [str(c) for c in (raw.get("meta_aux_target_cols") or raw.get("pv_target_cols") or [])]
    stem_l = cache_pt.stem.lower()
    feat_cols = "p_load_kw,q_load_kvar,p_pv_kw"
    hp: dict[str, object] = {
        "node_feature_cols": feat_cols,
        "edge_catalog_csv": "gnn_edges_phase_static.csv",
        "nodes_csv": "gnn_node_features_and_targets_mvagg.csv",
        "meta_csv": "gnn_sample_meta.csv",
        "chunk_parent": chunk_parent,
        "exclude_bess_features": "nobess" in stem_l,
        "aux_meta_cols": ",".join(meta_cols),
        "reg_loss": str(raw.get("reg_loss", "mse")),
        "dropout": 0.1,
    }

    return {
        "task": "DA-GPS (synthesized from checkpoint for daily compare)",
        "chunk_parent": chunk_parent,
        "chunks": chunks,
        "hyperparameters": hp,
        "cap_target_cols": list(raw.get("cap_target_cols") or []),
        "reg_target_cols": list(raw.get("reg_target_cols") or []),
        "meta_aux_target_cols": meta_cols,
        "reg_loss": str(raw.get("reg_loss", "mse")),
        "synthesized_from_checkpoint": str(ckpt_path.resolve()),
    }


def _load_da_gps_report_bundle(
    run_dir: Path,
    norm_dir: Path,
    ckpt_path: Path,
    cache_pt: Path,
) -> tuple[dict[str, object], Path]:
    """``da_gps_report.json`` > ``da_gps_run_manifest.json`` > checkpoint synthesis."""
    report_path = _find_upward_file(run_dir, _DA_GPS_REPORT_JSON, max_levels=3)
    if report_path is not None:
        print(f"[da_gps_daily] using {_DA_GPS_REPORT_JSON} -> {report_path}", flush=True)
        return json.loads(report_path.read_text(encoding="utf-8")), report_path.parent.resolve()

    manifest_path = _find_upward_file(run_dir, _DA_GPS_MANIFEST_JSON, max_levels=3)
    if manifest_path is not None:
        print(f"[da_gps_daily] using {_DA_GPS_MANIFEST_JSON} -> {manifest_path}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path.parent.resolve()

    print(
        f"[da_gps_daily] no {_DA_GPS_REPORT_JSON} or {_DA_GPS_MANIFEST_JSON}; "
        f"synthesizing recipe from checkpoint {ckpt_path.name}",
        flush=True,
    )
    return _synthesize_report_from_checkpoint(ckpt_path, cache_pt), norm_dir


def _resolve_default_edge_csv(report: dict, hp: dict, cache_pt: Path) -> Path:
    """Locate ``gnn_edges_phase_static.csv`` (or ``edge_catalog_csv``) for compacted edges.

    Training reports often store **absolute** ``chunks[0]`` paths from another machine; if the
    primary path is missing, we fall back to:

    - ``hyperparameters.chunk_parent`` / ``<chunk_name>`` / edge CSV, with ``chunk_name`` parsed
      from the tensor-cache filename (``<chunk>__full__....pt`` → ``<chunk>``).
    - ``<parent of cache_pt>`` / edge CSV (when caches live inside the chunk folder).
    """
    edge_name = str(hp.get("edge_catalog_csv", "gnn_edges_phase_static.csv"))
    trials: list[Path] = []

    chunks = report.get("chunks") or []
    if chunks:
        ch0 = Path(str(chunks[0]))
        trials.append((ch0 / edge_name).resolve())

    es = report.get("edges_csv")
    if es:
        trials.append(Path(str(es)).resolve())

    cp = str(hp.get("chunk_parent", "")).strip()
    cn = _chunk_name_from_cache_stem(cache_pt)
    if cp and cn:
        trials.append((Path(cp) / cn / edge_name).resolve())

    trials.append((cache_pt.parent / edge_name).resolve())

    seen: set[str] = set()
    uniq: list[Path] = []
    for t in trials:
        k = str(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)

    first_tried = uniq[0] if uniq else None
    for t in uniq:
        if t.is_file():
            if first_tried is not None and t.resolve() != first_tried.resolve():
                print(f"[da_gps_daily] resolved edge CSV -> {t}", flush=True)
            return t

    raise FileNotFoundError(
        "Could not find edge catalog CSV for compacted edges. Tried:\n  "
        + "\n  ".join(str(x) for x in uniq)
        + "\nFix ``chunks`` paths in da_gps_report.json / da_gps_run_manifest.json, set hyperparameters.chunk_parent, or pass --edge-csv."
    )


def _discover_pv_system_names() -> list[str]:
    """Backward-compatible alias; see ``compare_opendss_snapshot_helpers.discover_pv_system_names``."""
    return discover_pv_system_names()


def _read_pv_base_pmpp_kw(pv_names: list[str]) -> dict[str, float]:
    """Backward-compatible alias; see ``compare_opendss_snapshot_helpers.read_pv_base_pmpp_kw``."""
    return read_pv_base_pmpp_kw(pv_names)


def _collect_pv_to_busph_weights() -> dict[str, list[tuple[str, int, float]]]:
    """Bus + phase + equal weight per PVSystem (same as ``run_original_style_dataset_8500_unbalanced._collect_pv_maps``)."""
    pv_to_busph: dict[str, list[tuple[str, int, float]]] = {}
    try:
        if not dss.PVsystems.First():
            return pv_to_busph
        while True:
            name = str(dss.PVsystems.Name())
            dss.PVsystems.Name(name)
            buses = dss.CktElement.BusNames()
            bus1 = str(buses[0]).split(".")[0] if buses else ""
            nph = int(dss.CktElement.NumPhases())
            phases = list(range(1, max(1, nph) + 1))
            w = 1.0 / float(len(phases))
            pv_to_busph[name] = [(bus1, ph, w) for ph in phases]
            if not dss.PVsystems.Next():
                break
    except Exception:
        pass
    return pv_to_busph


def _fill_p_pv_kw_from_pmpp_and_irr(
    x_step: np.ndarray,
    col_pv: int,
    node_order: list[str],
    pv_names: list[str],
    pv_base_pmpp_kw: dict[str, float],
    pv_to_busph: dict[str, list[tuple[str, int, float]]],
    m_irr_t: float,
) -> None:
    """Presolve PV kW per phase node: ``pmpp_on_device * m_irr[t]`` split on bus phases.

    Same as ``run_original_style_dataset_8500_unbalanced._apply_snapshot_with_pv``:
    ``p_nominal = float(pmpp_set[i]) * float(m_pv_t)`` with ``m_pv_t`` ← ``m_irr_t`` here. Use the same
    ``pmpp`` values you set on ``PVsystem.Pmpp`` before solve (typically nameplate ``Pmpp0``).
    """
    bus_ph_p: dict[tuple[str, int], float] = {}
    mv = float(m_irr_t)
    for nm in pv_names:
        raw = str(nm).strip()
        b0 = float(pv_base_pmpp_kw.get(raw, 0.0))
        if b0 <= 0.0 or not np.isfinite(b0):
            continue
        p_tot = b0 * mv
        for bus, ph, w in pv_to_busph.get(raw, []):
            bk = str(bus).strip().lower()
            key = (bk, int(ph))
            bus_ph_p[key] = bus_ph_p.get(key, 0.0) + p_tot * float(w)
    x_step[:, col_pv] = 0.0
    for li in range(int(x_step.shape[0])):
        nk0 = str(node_order[li]).strip().lower()
        parts = nk0.rsplit(".", 1)
        if len(parts) != 2:
            continue
        bus_k, phs = parts[0], parts[1]
        try:
            ph = int(phs)
        except ValueError:
            continue
        x_step[li, col_pv] = float(bus_ph_p.get((bus_k, ph), 0.0))


def _circuit_losses_kw_kvar() -> tuple[float, float]:
    """Same scaling convention as ``run_original_style_dataset_8500_unbalanced``."""
    loss = dss.Circuit.Losses()
    # OpenDSS Circuit.Losses() is W/var; other power APIs are kW/kvar.
    return float(loss[0]) / 1000.0, float(loss[1]) / 1000.0


def _pq_from_ckt_total_powers(pwr) -> tuple[float, float] | None:
    """Map ``CktElement.TotalPowers()`` to (+P,+Q) injection kW/kvar (same sign rule as dataset script)."""
    if pwr is None or len(pwr) < 2:
        return None
    if len(pwr) == 2:
        return (-float(pwr[0]), -float(pwr[1]))
    p_sum, q_sum = 0.0, 0.0
    n_pair = (len(pwr) // 2) * 2
    for i in range(0, n_pair, 2):
        p_sum += float(pwr[i])
        q_sum += float(pwr[i + 1])
    return (-p_sum, -q_sum)


def _read_pv_totals_post_solve_kw_kvar(pv_names: list[str]) -> dict[str, tuple[float, float]]:
    """Per PVSystem name: (+P,+Q) kW/kvar **injected** into the feeder (post-solve).

    ``PVsystems.kW()`` is often **not** the converged inverter P (can stay ~0) while ``kvar()`` may
    reflect Volt-VAR. When ``abs(kW())`` is negligible, map ``CktElement.TotalPowers()`` after
    ``SetActiveElement("PVSystem.<name>")`` using the same sign convention as the dataset scripts.
    """
    out: dict[str, tuple[float, float]] = {}
    for raw in pv_names:
        name = str(raw).strip()
        if not name:
            continue
        p_inj, q_inj = 0.0, 0.0
        try:
            dss.PVsystems.Name(name)
            p_inj = float(dss.PVsystems.kW())
            q_inj = float(dss.PVsystems.kvar())
        except Exception:
            p_inj, q_inj = 0.0, 0.0
        # ``PVsystems.kW()`` is often **not** refreshed to the solved inverter P (stays ~0) while
        # ``kvar()`` can track Volt-VAR — use ``TotalPowers`` whenever real P from properties is ~0.
        if abs(p_inj) > 1e-3:
            out[name] = (float(p_inj), float(q_inj))
            continue
        got: tuple[float, float] | None = None
        try:
            dss.PVsystems.Name(name)
        except Exception:
            pass
        for elem_prefix in ("PVSystem", "PVsystem"):
            try:
                dss.Circuit.SetActiveElement(f"{elem_prefix}.{name}")
                got = _pq_from_ckt_total_powers(dss.CktElement.TotalPowers())
            except Exception:
                got = None
            if got is not None and (abs(got[0]) + abs(got[1]) > 1e-12):
                p_inj, q_inj = got
                break
        if got is not None and (abs(got[0]) + abs(got[1]) <= 1e-12):
            p_inj, q_inj = got
        out[name] = (float(p_inj), float(q_inj))
    return out


def _dss_pv_scalar_from_meta_column(col: str, pv_totals: dict[str, tuple[float, float]]) -> float | None:
    """Match meta PV columns to DSS using the same naming rule as training: ``pv_{dss_name.lower()}_...``."""
    cl = str(col).strip().lower()
    for dss_nm, pq in pv_totals.items():
        nm = str(dss_nm).strip().lower()
        if cl == f"pv_{nm}_p_post_kw":
            return float(pq[0])
        if cl == f"pv_{nm}_q_post_kvar":
            return float(pq[1])
    return None


def _pv_lookup_lower(pv_totals: dict[str, tuple[float, float]], stem: str) -> tuple[float, float] | None:
    st = str(stem).strip().lower()
    for dss_nm, pq in pv_totals.items():
        if str(dss_nm).strip().lower() == st:
            return pq
    return None


def _stem_from_pv_p_post_meta_col(col: str) -> str | None:
    """Meta column ``pv_<dssname>_p_post_kw`` → ``<dssname>`` stem (case preserved)."""
    m = re.match(r"^pv_(.+)_p_post_kw$", str(col).strip(), flags=re.IGNORECASE)
    return str(m.group(1)).strip() if m else None


def _nameplate_pmpp_kw_for_pv_stem(stem: str, pv_base_pmpp_kw: dict[str, float]) -> float | None:
    """``Pmpp0`` (kW) for PVSystem whose DSS name matches ``stem`` (case-insensitive)."""
    sl = str(stem).strip().lower()
    if not sl:
        return None
    for k, v in pv_base_pmpp_kw.items():
        if str(k).strip().lower() == sl:
            fv = float(v)
            if np.isfinite(fv) and fv > 0.0:
                return fv
            return None
    return None


def _dss_scalar_for_meta_aux_col(
    col: str,
    *,
    pv_totals: dict[str, tuple[float, float]],
    p_loss_kw: float,
    q_loss_kvar: float,
) -> float | None:
    """Map ``gnn_sample_meta``-style column name to a post-solve OpenDSS scalar (kW or kvar)."""
    cl = str(col).strip().lower()
    if cl == "p_loss_total_post_kw":
        return float(p_loss_kw)
    if cl == "q_loss_total_post_kvar":
        return float(q_loss_kvar)
    v = _dss_pv_scalar_from_meta_column(cl, pv_totals)
    if v is not None:
        return v
    # Fallback: regex stem when meta column used a short alias (underscore mismatch vs DSS Name)
    m = re.match(r"^pv_(.+)_p_post_kw$", cl)
    if m:
        pq = _pv_lookup_lower(pv_totals, m.group(1))
        return float(pq[0]) if pq is not None else None
    m = re.match(r"^pv_(.+)_q_post_kvar$", cl)
    if m:
        pq = _pv_lookup_lower(pv_totals, m.group(1))
        return float(pq[1]) if pq is not None else None
    return None


def _norm_reg_tap_stem(s: str) -> str | None:
    """``reg_<anything>_tap_pu`` → alphanumeric stem (lowercase) for fuzzy name match."""
    m = re.match(r"^reg_(.+)_tap_pu$", str(s).strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return re.sub(r"[^a-z0-9]+", "", m.group(1).lower())


def _norm_cap_stem(s: str) -> str | None:
    """``cap_<anything>_n_steps_on`` → alphanumeric stem (lowercase) for fuzzy name match."""
    m = re.match(r"^cap_(.+)_n_steps_on$", str(s).strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return re.sub(r"[^a-z0-9]+", "", m.group(1).lower())


def _norm_device_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _da_gps_col_stem(col: str) -> str | None:
    """Training column name → stem for matching OpenDSS ``RegControls`` / ``Capacitors`` names."""
    return _norm_reg_tap_stem(col) or _norm_cap_stem(col)


def align_da_gps_trajectory_to_opendss_names(
    opendss_names: list[str],
    da_gps_cols: list[str],
    da_gps_traj: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map ``(npts, n_col)`` DA-GPS trajectories onto OpenDSS device names by stem (case-insensitive).

    ``da_gps_cols`` entries are training targets like ``reg_feeder_rega_tap_pu`` or
    ``cap_capbank2a_n_steps_on``. Unmatched OpenDSS names are omitted from the returned dict.
    """
    traj = np.asarray(da_gps_traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"da_gps_traj must be 2-D (npts, n_col); got shape {traj.shape}")
    col_by_stem: dict[str, int] = {}
    for j, col in enumerate(da_gps_cols):
        stem = _da_gps_col_stem(str(col))
        if stem and stem not in col_by_stem:
            col_by_stem[stem] = int(j)
    out: dict[str, np.ndarray] = {}
    for nm in opendss_names:
        stem = _norm_device_stem(nm)
        j = col_by_stem.get(stem)
        if j is not None and j < traj.shape[1]:
            out[str(nm)] = traj[:, j].astype(np.float64, copy=True)
    return out


def _float_or_none(v: object) -> float | None:
    try:
        fv = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def _lookup_reg_tap_pu(reg_col: str, tap_raw: dict[str, float | int]) -> float | None:
    """Resolve ``reg_*_tap_pu`` column to OpenDSS tap (pu), same semantics as ``_read_reg_control_state`` keys.

    Training loads ``gnn_sample_meta`` with columns lowercased; DSS dict keys use ``reg_{RegControls.Name()}_tap_pu``
    with **original** element-name casing. Match: exact key, case-insensitive key, then normalized stem
    (ignore non-alphanumeric separators) so ``reg_feeder_rega_tap_pu`` ↔ ``reg_FeederRegA_tap_pu`` when stems align.
    """
    col = str(reg_col).strip()
    if col in tap_raw:
        return _float_or_none(tap_raw[col])
    cl = col.lower()
    for k, v in tap_raw.items():
        if str(k).strip().lower() == cl:
            return _float_or_none(v)
    want = _norm_reg_tap_stem(col)
    if not want:
        return None
    matches: list[tuple[str, float]] = []
    for k, v in tap_raw.items():
        nk = _norm_reg_tap_stem(str(k))
        if nk == want:
            fv = _float_or_none(v)
            if fv is not None:
                matches.append((str(k), fv))
    if not matches:
        return None
    matches.sort(key=lambda t: t[0])
    return matches[0][1]


def _resolve_profile_csv_path(
    spec: str | Path | None,
    *,
    default_if_dir: str,
    fallback_file: Path,
) -> Path:
    """``spec`` is a CSV file, or a **directory** containing ``default_if_dir``, or empty → ``fallback_file``."""
    if spec is None or str(spec).strip() == "":
        if not fallback_file.is_file():
            raise FileNotFoundError(fallback_file)
        return fallback_file.resolve()
    p = Path(spec).expanduser().resolve()
    if p.is_dir():
        out = p / default_if_dir
        if not out.is_file():
            raise FileNotFoundError(
                f"Profile directory {p} has no {default_if_dir!r} (two-column time + mult CSV)."
            )
        return out
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _rebind_irradiance_loadshape_irradday001(
    irr_csv: Path,
    *,
    npts: int,
    step_min: float,
) -> None:
    """Backward-compatible wrapper around ``compare_opendss_snapshot_helpers``."""
    from compare_opendss_snapshot_helpers import rebind_irradiance_loadshape_irradday001

    rebind_irradiance_loadshape_irradday001(irr_csv, npts=npts, step_min=step_min)


def _parse_der_bus_list(spec: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[,;\s]+", str(spec).strip()):
        b = part.strip().lower()
        if b and b not in out:
            out.append(b)
    return out


def _install_der_three_phase_generators(der_buses: list[str]) -> dict[str, str]:
    """``bus`` (no phase, lower) → DSS ``Generator.DADGDER_*`` name. ``phases=3`` at ``bus1=bus``."""
    gen_by_bus: dict[str, str] = {}
    for i, bus_raw in enumerate(der_buses):
        bus = str(bus_raw).strip().lower()
        gname = f"DADGDER_{i:03d}"
        kv_ll = 12.47
        try:
            dss.Circuit.SetActiveBus(bus)
            kv_ln = float(dss.Bus.kVBase())
            if np.isfinite(kv_ln) and kv_ln > 0:
                kv_ll = float(kv_ln) * math.sqrt(3.0)
        except Exception:
            pass
        cmd = (
            f"New Generator.{gname} phases=3 bus1={bus} conn=wye model=1 "
            f"kV={kv_ll:.6f} kW=0 kvar=0 vminpu=0.01 vmaxpu=10"
        )
        try:
            dss.Text.Command(cmd)
            gen_by_bus[bus] = gname
        except Exception:
            pass
    return gen_by_bus


def _set_der_generators_kw(
    gen_by_bus: dict[str, str],
    der_buses: list[str],
    *,
    p_profile_scale: float,
    der_max_kw: float,
    der_q_frac: float,
) -> None:
    """Total ``P = p_profile_scale * der_max_kw`` split across buses; each 3ph gen gets ``Q = der_q_frac * P_bus``."""
    n_bus = max(1, len(der_buses))
    p_total = float(p_profile_scale) * float(der_max_kw)
    p_bus = p_total / float(n_bus)
    q_bus = float(der_q_frac) * p_bus
    for bus in der_buses:
        bk = str(bus).strip().lower()
        gname = gen_by_bus.get(bk)
        if not gname:
            continue
        try:
            dss.Generators.Name(gname)
            dss.Generators.kW(float(p_bus))
            dss.Generators.kvar(float(q_bus))
        except Exception:
            pass


def _der_phases_per_bus_in_cache(der_buses: list[str], node_to_local: dict[str, int]) -> dict[str, list[int]]:
    """``bus`` (lower) → phases ``[1,2,3]`` that exist as ``bus.phase`` keys in ``node_to_local``."""
    out: dict[str, list[int]] = {}
    for bus_raw in der_buses:
        bk = str(bus_raw).strip().lower()
        present = [ph for ph in (1, 2, 3) if f"{bk}.{ph}" in node_to_local]
        if present:
            out[bk] = present
    return out


def _add_der_to_pq_load_columns(
    x_step: np.ndarray,
    der_buses: list[str],
    der_bus_phases: dict[str, list[int]],
    node_to_local: dict[str, int],
    col_p: int | None,
    col_q: int | None,
    *,
    p_profile_scale: float,
    der_max_kw: float,
    der_q_frac: float,
) -> None:
    """Add DER P/Q into ``p_load_kw`` / ``q_load_kvar`` at tensor-cache nodes for each DER bus.

    ``P_total = p_profile_scale * der_max_kw`` split equally across ``der_buses``; each bus then splits
    across **only** ``bus.phase`` rows listed in ``der_bus_phases`` (equal P and Q per cached phase).
    """
    if col_p is None and col_q is None:
        return
    n_bus = max(1, len(der_buses))
    p_total = float(p_profile_scale) * float(der_max_kw)
    p_per_bus = p_total / float(n_bus)
    for bus_raw in der_buses:
        bk = str(bus_raw).strip().lower()
        phases = der_bus_phases.get(bk) or []
        if not phases:
            continue
        q_bus = float(der_q_frac) * p_per_bus
        nph = max(1, len(phases))
        p_ph = p_per_bus / float(nph)
        q_ph = q_bus / float(nph)
        for ph in phases:
            nk = f"{bk}.{int(ph)}"
            li = node_to_local.get(nk)
            if li is None:
                continue
            li = int(li)
            if col_p is not None:
                x_step[li, int(col_p)] = float(x_step[li, int(col_p)]) + float(p_ph)
            if col_q is not None:
                x_step[li, int(col_q)] = float(x_step[li, int(col_q)]) + float(q_ph)


def _zero_bess_columns(x_step: np.ndarray, col_pb: int | None, col_qb: int | None) -> None:
    if col_pb is not None:
        x_step[:, int(col_pb)] = 0.0
    if col_qb is not None:
        x_step[:, int(col_qb)] = 0.0


def _fill_der_as_bess_node_features(
    x_step: np.ndarray,
    der_buses: list[str],
    der_bus_phases: dict[str, list[int]],
    node_to_local: dict[str, int],
    col_pb: int | None,
    col_qb: int | None,
    *,
    p_profile_scale: float,
    der_max_kw: float,
    der_q_frac: float,
) -> None:
    """Set ``p_bess_kw`` / ``q_bess_kvar`` from DER on each cached ``bus.phase`` row (split per bus, then per phase)."""
    if col_pb is None and col_qb is None:
        return
    n_bus = max(1, len(der_buses))
    p_total = float(p_profile_scale) * float(der_max_kw)
    p_per_bus = p_total / float(n_bus)
    for bus_raw in der_buses:
        bk = str(bus_raw).strip().lower()
        phases = der_bus_phases.get(bk) or []
        if not phases:
            continue
        q_bus = float(der_q_frac) * p_per_bus
        nph = max(1, len(phases))
        p_ph = p_per_bus / float(nph)
        q_ph = q_bus / float(nph)
        for ph in phases:
            nk = f"{bk}.{int(ph)}"
            li = node_to_local.get(nk)
            if li is None:
                continue
            li = int(li)
            if col_pb is not None:
                x_step[li, int(col_pb)] = float(p_ph)
            if col_qb is not None:
                x_step[li, int(col_qb)] = float(q_ph)


def _fill_pq_columns(
    x_step: np.ndarray,
    col_p: int | None,
    col_q: int | None,
    node_order: list[str],
    node_P: dict[str, float],
    node_Q: dict[str, float],
    mv_rules: list[dict[str, str]],
) -> None:
    if col_p is None and col_q is None:
        return
    if mv_rules:
        for rec in mv_rules:
            mv = rec["mv_key"]
            try:
                li = node_order.index(mv)
            except ValueError:
                continue
            pa = float(node_P.get(rec["load_a"], 0.0) + node_P.get(rec["load_b"], 0.0))
            qa = float(node_Q.get(rec["load_a"], 0.0) + node_Q.get(rec["load_b"], 0.0))
            if col_p is not None:
                x_step[li, col_p] = pa
            if col_q is not None:
                x_step[li, col_q] = qa
    else:
        for li, nk in enumerate(node_order):
            if col_p is not None:
                x_step[li, col_p] = float(node_P.get(nk, 0.0))
            if col_q is not None:
                x_step[li, col_q] = float(node_Q.get(nk, 0.0))


def _busph_key_from_node_key(key_str: str) -> tuple[str, int] | None:
    parts = str(key_str).strip().lower().rsplit(".", 1)
    if len(parts) != 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def _precompute_daily_feature_tables(
    *,
    ref_x: np.ndarray,
    node_order: list[str],
    load_to_busph: dict[str, list[tuple[str, int, float]]],
    base_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    col_p: int | None,
    col_q: int | None,
    col_pv: int | None,
    col_bess_p: int | None,
    col_bess_q: int | None,
    mv_rules: list[dict[str, str]],
    pv_names: list[str],
    pv_base_pmpp_kw: dict[str, float],
    pv_to_busph: dict[str, list[tuple[str, int, float]]],
    der_effective_buses: list[str],
    der_bus_phases: dict[str, list[int]],
    node_to_local: dict[str, int],
    der_use_bess_columns: bool,
    der_max_kw: float,
    der_q_frac: float,
) -> dict[str, object]:
    """Static tables for per-step node feature assembly (load P/Q, PV presolve, DER).

    Deployment split: topology-bound columns live in ``x_static``; only load P/Q, ``p_pv_kw``
    (irradiance scalar), and DER scalars are written each step via :func:`_apply_daily_feature_tables`.
    """
    N = int(ref_x.shape[0])
    key_to_idx: dict[tuple[str, int], int] = {}
    load_j_list: list[int] = []
    w_list: list[float] = []
    idx_list: list[int] = []
    for j, name in enumerate(base_names):
        for bus, ph, w in load_to_busph.get(name, []):
            bk = str(bus).strip().lower()
            key = (bk, int(ph))
            if key not in key_to_idx:
                key_to_idx[key] = len(key_to_idx)
            idx_list.append(key_to_idx[key])
            load_j_list.append(j)
            w_list.append(float(w))
    # PV buses (e.g. pv1/pv2 on l3235256, p850080) often carry no loads, so they are absent
    # from load_to_busph. Register their bus.phase keys so `pv_node_coeff` / `p_pv_kw`
    # match `_fill_p_pv_kw_from_pmpp_and_irr` (Pmpp0×m_irr split on element phases).
    for nm in pv_names:
        raw = str(nm).strip()
        for bus, ph, w in pv_to_busph.get(raw, []):
            bk = str(bus).strip().lower()
            key = (bk, int(ph))
            if key not in key_to_idx:
                key_to_idx[key] = len(key_to_idx)
    n_busph = len(key_to_idx)
    busph_load_j = np.asarray(load_j_list, dtype=np.int32)
    busph_idx = np.asarray(idx_list, dtype=np.int32)
    busph_w = np.asarray(w_list, dtype=np.float64)

    node_busph_idx = np.full(N, -1, dtype=np.int32)
    for li, nk in enumerate(node_order):
        key = _busph_key_from_node_key(nk)
        if key is not None:
            bi = key_to_idx.get(key)
            if bi is not None:
                node_busph_idx[li] = int(bi)

    mv_li: np.ndarray | None = None
    mv_a_idx: np.ndarray | None = None
    mv_b_idx: np.ndarray | None = None
    if mv_rules and (col_p is not None or col_q is not None):
        nk_to_li = {str(nk).strip().lower(): li for li, nk in enumerate(node_order)}
        mvl: list[int] = []
        mva: list[int] = []
        mvb: list[int] = []
        for rec in mv_rules:
            mv = str(rec["mv_key"]).strip().lower()
            li = nk_to_li.get(mv)
            if li is None:
                continue
            ka = _busph_key_from_node_key(rec["load_a"])
            kb = _busph_key_from_node_key(rec["load_b"])
            if ka is None or kb is None:
                continue
            a_idx = key_to_idx.get(ka)
            b_idx = key_to_idx.get(kb)
            if a_idx is None or b_idx is None:
                continue
            mvl.append(int(li))
            mva.append(int(a_idx))
            mvb.append(int(b_idx))
        if mvl:
            mv_li = np.asarray(mvl, dtype=np.int32)
            mv_a_idx = np.asarray(mva, dtype=np.int32)
            mv_b_idx = np.asarray(mvb, dtype=np.int32)

    pv_node_coeff: np.ndarray | None = None
    if col_pv is not None:
        pv_busph_coeff = np.zeros(max(1, n_busph), dtype=np.float32)
        for nm in pv_names:
            raw = str(nm).strip()
            b0 = float(pv_base_pmpp_kw.get(raw, 0.0))
            if b0 <= 0.0 or not np.isfinite(b0):
                continue
            for bus, ph, w in pv_to_busph.get(raw, []):
                bk = str(bus).strip().lower()
                key = (bk, int(ph))
                bi = key_to_idx.get(key)
                if bi is not None:
                    pv_busph_coeff[bi] += float(b0) * float(w)
        pv_node_coeff = np.zeros(N, dtype=np.float32)
        valid = node_busph_idx >= 0
        pv_node_coeff[valid] = pv_busph_coeff[node_busph_idx[valid]]

    der_p_unit = np.zeros(N, dtype=np.float32)
    der_q_unit = np.zeros(N, dtype=np.float32)
    if der_effective_buses:
        n_bus = max(1, len(der_effective_buses))
        for bus_raw in der_effective_buses:
            bk = str(bus_raw).strip().lower()
            phases = der_bus_phases.get(bk) or []
            if not phases:
                continue
            nph = max(1, len(phases))
            p_unit = 1.0 / float(n_bus * nph)
            q_unit = float(der_q_frac) * p_unit
            for ph in phases:
                nk = f"{bk}.{int(ph)}"
                li = node_to_local.get(nk)
                if li is None:
                    continue
                der_p_unit[int(li)] = float(p_unit)
                der_q_unit[int(li)] = float(q_unit)

    return {
        "x_static": np.ascontiguousarray(ref_x, dtype=np.float32),
        "base_kw": base_kw,
        "base_kvar": base_kvar,
        "busph_load_j": busph_load_j,
        "busph_idx": busph_idx,
        "busph_w": busph_w,
        "n_busph": n_busph,
        "node_busph_idx": node_busph_idx,
        "mv_li": mv_li,
        "mv_a_idx": mv_a_idx,
        "mv_b_idx": mv_b_idx,
        "pv_node_coeff": pv_node_coeff,
        "der_p_unit": der_p_unit,
        "der_q_unit": der_q_unit,
        "der_use_bess_columns": bool(der_use_bess_columns),
        "der_max_kw": float(der_max_kw),
        "der_q_frac": float(der_q_frac),
        "col_p": col_p,
        "col_q": col_q,
        "col_pv": col_pv,
        "col_bess_p": col_bess_p,
        "col_bess_q": col_bess_q,
    }


def _apply_daily_feature_tables(
    x_step: np.ndarray,
    ft: dict[str, object],
    *,
    m_t: float,
    ir_t: float,
    m_der_t: float,
    want_der: bool,
) -> None:
    """Fill dynamic node-feature columns in ``x_step`` (reuses static tail from prior copy)."""
    kw_set = ft["base_kw"] * float(m_t)
    kvar_set = ft["base_kvar"] * float(m_t)
    w = ft["busph_w"]
    lj = ft["busph_load_j"]
    bi = ft["busph_idx"]
    nb = int(ft["n_busph"])
    busph_P = np.bincount(bi, weights=kw_set[lj] * w, minlength=nb)
    busph_Q = np.bincount(bi, weights=kvar_set[lj] * w, minlength=nb)

    col_p = ft["col_p"]
    col_q = ft["col_q"]
    mv_li = ft["mv_li"]
    if mv_li is not None:
        if col_p is not None:
            x_step[mv_li, int(col_p)] = (busph_P[ft["mv_a_idx"]] + busph_P[ft["mv_b_idx"]]).astype(np.float32)
        if col_q is not None:
            x_step[mv_li, int(col_q)] = (busph_Q[ft["mv_a_idx"]] + busph_Q[ft["mv_b_idx"]]).astype(np.float32)
    else:
        nbi = ft["node_busph_idx"]
        valid = nbi >= 0
        if col_p is not None:
            x_step[:, int(col_p)] = 0.0
            x_step[valid, int(col_p)] = busph_P[nbi[valid]].astype(np.float32)
        if col_q is not None:
            x_step[:, int(col_q)] = 0.0
            x_step[valid, int(col_q)] = busph_Q[nbi[valid]].astype(np.float32)

    col_pv = ft["col_pv"]
    pv_node_coeff = ft["pv_node_coeff"]
    if col_pv is not None and pv_node_coeff is not None:
        x_step[:, int(col_pv)] = pv_node_coeff * float(ir_t)

    if want_der:
        der_scale = float(m_der_t) * float(ft["der_max_kw"])
        if ft["der_use_bess_columns"]:
            col_pb = ft["col_bess_p"]
            col_qb = ft["col_bess_q"]
            if col_pb is not None:
                x_step[:, int(col_pb)] = ft["der_p_unit"] * der_scale
            if col_qb is not None:
                x_step[:, int(col_qb)] = ft["der_q_unit"] * der_scale
        else:
            if col_p is not None:
                x_step[:, int(col_p)] += ft["der_p_unit"] * der_scale
            if col_q is not None:
                x_step[:, int(col_q)] += ft["der_q_unit"] * der_scale


def _auto_ylim_padded(*series: np.ndarray, pad_frac: float = 0.22, pad_floor: float = 0.0) -> tuple[float, float] | None:
    """Union of finite values across 1+ arrays → (lo, hi) with padding (for shared-scale plots)."""
    chunks: list[np.ndarray] = []
    for s in series:
        a = np.asarray(s, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            chunks.append(a)
    if not chunks:
        return None
    ys = np.concatenate(chunks)
    lo, hi = float(np.min(ys)), float(np.max(ys))
    span = hi - lo
    mag = max(abs(lo), abs(hi), 1.0)
    # Extra breathing room so traces do not sit on the axis edges; flat series get a visible band.
    if not np.isfinite(span) or span <= 1e-15 * max(mag, 1.0):
        bump = max(float(pad_floor), 0.04 * mag, 1e-3)
        return lo - bump, hi + bump
    span_e = max(span, 1e-12)
    pad = max(float(pad_floor), span_e * float(pad_frac))
    pad += 0.04 * span_e
    pad += 0.02 * mag
    return lo - pad, hi + pad


def _plot_cap_bank_daily_compare(
    *,
    t_hours: np.ndarray,
    dss_n_steps: np.ndarray,
    gnn_sigmoid: np.ndarray,
    col_name: str,
    out_path: Path,
    show_plots: bool,
) -> None:
    """Single y-axis: DSS bank-on indicator (0/1 from ``n_steps_on > 0``) vs ``sigmoid(cap_logit)`` in [0, 1].

    Raw ``n_steps_on`` is not drawn on the same scale as probability; the training target is binary bank-on,
    so DSS is shown as 0/1 for a direct overlay with model P(bank on).
    """
    steps = np.asarray(dss_n_steps, dtype=np.float64)
    gnn = np.asarray(gnn_sigmoid, dtype=np.float64)
    dss_on = (steps > 0.5).astype(np.float32)
    m = np.isfinite(dss_on) & np.isfinite(gnn)
    mae_bin = float(np.mean(np.abs(dss_on[m] - gnn[m]))) if m.any() else float("nan")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t_hours, dss_on, color="C0", linewidth=2.0, drawstyle="steps-post", label="OpenDSS bank on (1 if n_steps_on>0)")
    ax.plot(t_hours, gnn, color="C1", linestyle="--", linewidth=1.6, label="DA-GPS P(bank on) = sigmoid(logit)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Bank on (0–1)")
    ax.grid(True, alpha=0.3)
    lim = _auto_ylim_padded(dss_on.astype(np.float64), gnn, pad_frac=0.12, pad_floor=0.02)
    if lim is not None:
        lo, hi = float(lim[0]), float(lim[1])
        ax.set_ylim(max(-0.05, lo), min(1.05, hi))
    else:
        ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"24h cap status: {col_name}  (MAE vs binary DSS: {mae_bin:.4f})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_regulator_tap_daily_compare(
    *,
    t_hours: np.ndarray,
    dss_tap_pu: np.ndarray,
    gnn_tap_pu: np.ndarray,
    col_name: str,
    out_path: Path,
    show_plots: bool,
) -> None:
    m = np.isfinite(dss_tap_pu) & np.isfinite(gnn_tap_pu)
    mae = float(np.mean(np.abs(dss_tap_pu[m] - gnn_tap_pu[m]))) if m.any() else float("nan")
    fig = plt.figure(figsize=(10, 4.2))
    plt.plot(t_hours, dss_tap_pu, linewidth=2.0, label="OpenDSS tap (pu)")
    plt.plot(t_hours, gnn_tap_pu, linestyle="--", linewidth=1.6, label=f"DA-GPS tap (pu) MAE={mae:.5f}")
    plt.xlabel("Hour of day")
    plt.ylabel("Tap (pu)")
    plt.title(f"24h regulator tap: {col_name}")
    lim = _auto_ylim_padded(dss_tap_pu, gnn_tap_pu, pad_frac=0.22, pad_floor=1e-5)
    if lim is not None:
        plt.ylim(lim[0], lim[1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_meta_aux_scalar_daily_compare(
    *,
    t_hours: np.ndarray,
    dss_y: np.ndarray,
    gnn_y: np.ndarray,
    col_name: str,
    y_label: str,
    out_path: Path,
    show_plots: bool,
    presolve_y: np.ndarray | None = None,
    presolve_label: str = "Presolve P (Pmpp×m_irr, kW)",
) -> None:
    """Single y-axis: OpenDSS and DA-GPS share scale; optional third trace for scheduled presolve P."""
    m = np.isfinite(dss_y) & np.isfinite(gnn_y)
    mae = float(np.mean(np.abs(dss_y[m] - gnn_y[m]))) if m.any() else float("nan")
    n_dss = int(np.sum(np.isfinite(dss_y)))
    n_gnn = int(np.sum(np.isfinite(gnn_y)))
    n_both = int(np.sum(m))
    pres = np.asarray(presolve_y, dtype=np.float64) if presolve_y is not None else None
    has_pre = pres is not None and np.any(np.isfinite(pres))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t_hours, dss_y, color="C0", linewidth=2.0, label="OpenDSS (post-solve)")
    ax.plot(t_hours, gnn_y, color="C1", linestyle="--", linewidth=1.6, label="DA-GPS (post-solve)")
    if has_pre:
        ax.plot(t_hours, pres, color="C2", linestyle=":", linewidth=2.0, label=presolve_label)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    series = [dss_y, gnn_y]
    if has_pre:
        series.append(pres.astype(np.float64))
    lim = _auto_ylim_padded(*series, pad_frac=0.22, pad_floor=0.0)
    if lim is not None:
        ax.set_ylim(lim[0], lim[1])
    ttl = f"24h meta aux: {col_name}  MAE(DSS vs GNN post)={mae:.5g}"
    if has_pre:
        ttl += "  (+ presolve schedule)"
    if n_dss < len(dss_y) * 0.5 or n_gnn < len(gnn_y) * 0.5:
        ttl += f"  (finite: DSS {n_dss}/{len(dss_y)}, GNN {n_gnn}/{len(gnn_y)}, both {n_both})"
    ax.set_title(ttl)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _gnn_voltages_for_nodes(
    node_names: list[str],
    v_gnn: np.ndarray,
    node_to_idx: dict[str, int],
    npts: int,
) -> dict[str, np.ndarray]:
    """Map full-circuit ``v_gnn`` columns to requested ``bus.phase`` monitor names."""
    out: dict[str, np.ndarray] = {}
    for nk in node_names:
        key = str(nk).strip().lower()
        j = node_to_idx.get(key)
        if j is None:
            print(
                f"[da_gps_daily] WARNING: node {nk!r} not in circuit index; returning NaN series",
                flush=True,
            )
            out[str(nk)] = np.full(int(npts), np.nan, dtype=np.float64)
        else:
            out[str(nk)] = v_gnn[:, j].astype(np.float64, copy=True)
    return out


def run_da_gps_daily_voltages(
    *,
    run_dir: Path | str,
    cache_pt: Path | str,
    node_names: list[str],
    checkpoint: Path | str | None = None,
    edge_csv: Path | str | None = None,
    load_profile_path: str | None = None,
    load_profile_filename: str = "5minDayShape.csv",
    pv_irradiance_profile_path: str | None = None,
    pv_irradiance_filename: str = "irr_day_001.csv",
    der_profile_path: str | None = None,
    der_profile_filename: str = "der_5min.csv",
    der_max_kw: float = 0.0,
    der_buses: str = "",
    der_q_frac_p: float = 0.1,
    npts: int = 288,
    step_min: int = 5,
    daily_stress: float = 0.0,
    scenario_scale: float = 1.0,
    stress_clip_lo: float = 0.1,
    stress_clip_hi: float = 3.0,
    ref_sample_index: int = 0,
    mv_sx_mapping: Path | str | None = None,
    device: str | None = None,
    skip_opendss: bool = True,
    return_device_states: bool = True,
    gnn_batch_steps: int | None = None,
) -> dict[str, np.ndarray] | dict[str, object]:
    """Run DA-GPS daily GNN inference without plots or CSV exports.

    By default (``return_device_states=True``) returns::

        {
            "voltages": {node_name: (npts,) |V| pu},
            "reg_tap_pu": (npts, n_reg) winding tap pu from reg head (CE → class tap pu, else denorm),
            "cap_sigmoid": (npts, n_cap) P(bank on) = sigmoid(cap logit),
            "reg_cols": [...],  # training ``reg_*_tap_pu`` column names (device order)
            "cap_cols": [...],  # training ``cap_*_n_steps_on`` column names
        }

    Set ``return_device_states=False`` for the legacy flat ``{node: |V| array}`` mapping only.

    By default (``skip_opendss=True``) runs **only** the GNN forward pass over profile-scaled node
    features — no OpenDSS ``Solve()`` loop. Reg/cap heads use the same per-step ``x`` as voltages.
    """
    run_dir_p = Path(run_dir).expanduser().resolve()
    cache_pt_p = Path(cache_pt).expanduser().resolve()
    ckpt = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
    ec = Path(edge_csv).expanduser().resolve() if edge_csv is not None else None
    mv = Path(mv_sx_mapping).expanduser().resolve() if mv_sx_mapping is not None else None
    out_dir = run_dir_p / "_da_gps_overlay_scratch"
    result = run(
        run_dir=run_dir_p,
        cache_pt=cache_pt_p,
        out_dir=out_dir,
        checkpoint=ckpt,
        edge_csv=ec,
        daily_profile_csv=None,
        plot_nodes=[str(n) for n in node_names],
        plot_all_cache_nodes=False,
        npts=int(npts),
        step_min=int(step_min),
        daily_stress=float(daily_stress),
        scenario_scale=float(scenario_scale),
        stress_clip_lo=float(stress_clip_lo),
        stress_clip_hi=float(stress_clip_hi),
        ref_sample_index=int(ref_sample_index),
        mv_sx_mapping=mv,
        ymin=0.92,
        ymax=1.08,
        v_ylim_fixed=False,
        show_plots=False,
        device=device,
        load_profile_path=load_profile_path,
        load_profile_filename=str(load_profile_filename),
        pv_irradiance_profile_path=pv_irradiance_profile_path,
        pv_irradiance_filename=str(pv_irradiance_filename),
        der_profile_path=der_profile_path,
        der_profile_filename=str(der_profile_filename),
        der_max_kw=float(der_max_kw),
        der_buses=str(der_buses),
        der_q_frac_p=float(der_q_frac_p),
        voltages_only=True,
        skip_opendss_solve=bool(skip_opendss),
        gnn_batch_steps=gnn_batch_steps,
    )
    if result is None:
        raise RuntimeError("run_da_gps_daily_voltages: internal run() returned None")
    if return_device_states:
        if isinstance(result, dict) and "voltages" in result:
            return result
        raise RuntimeError("run_da_gps_daily_voltages: expected device-state bundle from run()")
    if isinstance(result, dict) and "voltages" in result:
        return result["voltages"]  # type: ignore[return-value]
    return result  # type: ignore[return-value]


def run_da_gps_gnn_only_daily_voltages(
    *,
    run_dir: Path | str,
    cache_pt: Path | str,
    node_names: list[str],
    **kwargs: object,
) -> dict[str, np.ndarray]:
    """Alias for :func:`run_da_gps_daily_voltages` with ``skip_opendss=True`` (GNN surrogate only)."""
    return run_da_gps_daily_voltages(
        run_dir=run_dir,
        cache_pt=cache_pt,
        node_names=node_names,
        skip_opendss=True,
        **kwargs,  # type: ignore[arg-type]
    )


def _resolve_voltage_png_dir(
    out_dir: Path,
    *,
    plot_all_cache_nodes: bool,
    voltage_png_subdir: str,
) -> Path:
    """Where to write daily |V|+angle compare PNGs. ``out_dir/daily_voltage`` when ``--plot-all-cache-nodes`` and subdir unset; ``out_dir`` otherwise unless ``--voltage-png-subdir`` is set."""
    s = str(voltage_png_subdir).strip()
    if s == ".":
        return out_dir
    if s and s not in ("..",):
        p = (out_dir / s).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if plot_all_cache_nodes:
        p = (out_dir / "daily_voltage").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return out_dir


def run(
    *,
    run_dir: Path,
    cache_pt: Path,
    out_dir: Path,
    checkpoint: Path | None,
    edge_csv: Path | None,
    daily_profile_csv: str | Path | None,
    plot_nodes: list[str],
    plot_all_cache_nodes: bool = False,
    plot_all_max_nodes: int = 0,
    voltage_png_subdir: str = "",
    voltage_plot_dpi: int = 0,
    voltage_plot_fig_w: float = 0.0,
    voltage_plot_fig_h: float = 0.0,
    npts: int,
    step_min: int,
    daily_stress: float,
    scenario_scale: float,
    stress_clip_lo: float,
    stress_clip_hi: float,
    ref_sample_index: int,
    mv_sx_mapping: Path | None,
    ymin: float,
    ymax: float,
    v_ylim_fixed: bool,
    show_plots: bool,
    device: str | None,
    meta_debug: bool = False,
    load_profile_path: str | None = None,
    load_profile_filename: str = "5minDayShape.csv",
    pv_irradiance_profile_path: str | None = None,
    pv_irradiance_filename: str = "irr_day_001.csv",
    der_profile_path: str | None = None,
    der_profile_filename: str = "der_5min.csv",
    der_max_kw: float = 0.0,
    der_buses: str = "",
    der_q_frac_p: float = 0.1,
    voltages_only: bool = False,
    skip_opendss_solve: bool = False,
    gnn_batch_steps: int | None = None,
    feeder: str = "8500",
) -> dict[str, np.ndarray] | None:
    feeder_key = normalize_feeder(feeder)
    run_dir = Path(run_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"--run-dir is not an existing directory:\n  {run_dir}\n"
            "Use your real DA-GPS training output folder (where ``da_gps_report.json`` / ``x_mean.pt`` live, "
            "or an epoch subfolder whose parent has them). Notebook placeholders like ``...\\\\your_run_dir...`` "
            "are not valid paths."
        )
    cache_pt = Path(cache_pt).expanduser().resolve()
    if not cache_pt.is_file():
        raise FileNotFoundError(
            f"--cache-pt is not an existing file:\n  {cache_pt}\n"
            "Pass the real chunk tensor ``.pt`` (``x`` + ``node_to_local``). A literal ``run_*__*.pt`` string is "
            "not a path — resolve it first, e.g. ``sorted(REPO.glob('**/run_*__*.pt'))[0]``."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_hint = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
    if ckpt_hint is not None and not ckpt_hint.is_file():
        ckpt_hint = None
    norm_dir = _resolve_norm_dir(run_dir, ckpt_hint=ckpt_hint)
    ckpt_path = _resolve_default_checkpoint(run_dir, norm_dir, checkpoint)
    report, report_dir = _load_da_gps_report_bundle(run_dir, norm_dir, ckpt_path, Path(cache_pt))
    hp = report.get("hyperparameters") or {}

    print(
        f"[da_gps_daily] report_dir={report_dir}\n"
        f"[da_gps_daily] norm_dir={norm_dir}\n"
        f"[da_gps_daily] run_dir={run_dir}\n"
        f"[da_gps_daily] checkpoint={ckpt_path}",
        flush=True,
    )

    x_mean_path = norm_dir / "x_mean.pt"
    x_std_path = norm_dir / "x_std.pt"
    y_mean_path = norm_dir / "y_mean.pt"
    y_std_path = norm_dir / "y_std.pt"
    for p in (x_mean_path, x_std_path, y_mean_path, y_std_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing normalization tensor: {p}")

    node_feature_cols = _parse_cols(str(hp.get("node_feature_cols", "p_load_kw,q_load_kvar")))
    col_index = {c: j for j, c in enumerate(node_feature_cols)}
    col_p = col_index.get("p_load_kw")
    col_q = col_index.get("q_load_kvar")
    col_pv = col_index.get("p_pv_kw")
    col_bess_p = col_index.get("p_bess_kw")
    col_bess_q = col_index.get("q_bess_kvar")
    der_use_bess_columns = col_bess_p is not None or col_bess_q is not None
    if col_p is None or col_q is None:
        raise ValueError(
            f"This daily driver expects 'p_load_kw' and 'q_load_kvar' in node_feature_cols; got {node_feature_cols!r}"
        )

    z = torch.load(Path(cache_pt).resolve(), map_location="cpu", weights_only=False)
    if "x" not in z or "node_to_local" not in z:
        keys = sorted(str(k) for k in z.keys())
        hint = ""
        if "optimizer_state_dict" in z or "best_model_state_dict" in z:
            hint = (
                " This file looks like training_last.pt (resume bundle), not a chunk tensor cache. "
                "Use a cache .pt from your training --cache_dir (names like run_*__*.pt) that contains "
                "precomputed tensors x, y_ri, sample_ids, node_to_local."
            )
        elif "model_state_dict" in z and "n_nodes" in z:
            hint = " This file looks like da_gps_multitask_best.pt — pass that as --checkpoint / inside --run-dir, not as --cache-pt."
        raise KeyError(
            f"{cache_pt} must be a chunk **tensor cache** with keys 'x' and 'node_to_local'. "
            f"Found keys: {keys[:25]}{'...' if len(keys) > 25 else ''}.{hint}"
        )
    x_cache: torch.Tensor = z["x"]
    ntl_raw = z["node_to_local"]
    node_to_local: dict[str, int] = {str(k).strip().lower(): int(v) for k, v in ntl_raw.items()}
    node_order = sorted(node_to_local.keys(), key=lambda k: node_to_local[k])
    N = len(node_order)
    if int(x_cache.shape[1]) != N:
        raise RuntimeError(f"cache x N mismatch: x.shape[1]={x_cache.shape[1]} vs |node_order|={N}")
    ref_i = int(ref_sample_index)
    if ref_i < 0 or ref_i >= int(x_cache.shape[0]):
        raise IndexError(f"ref_sample_index={ref_i} out of range for x with shape[0]={x_cache.shape[0]}")
    ref_x = x_cache[ref_i].numpy().astype(np.float32, copy=False)
    if ref_x.shape[0] != N or ref_x.ndim != 2:
        raise RuntimeError(f"Unexpected ref_x shape: {ref_x.shape}")

    der_bus_list = _parse_der_bus_list(str(der_buses or ""))
    der_effective_buses: list[str] = []
    der_bus_phases: dict[str, list[int]] = {}
    der_m_series = np.zeros(int(npts), dtype=np.float64)
    der_csv_resolved: Path | None = None
    want_der = float(der_max_kw) > 0.0 and bool(der_bus_list)
    if want_der:
        if not str(der_profile_path or "").strip():
            raise ValueError(
                "der_max_kw>0 with non-empty der_buses requires ``der_profile_path`` "
                "(two-column CSV file, or directory containing ``der_profile_filename``)."
            )
        fb_dummy = REPO_ROOT / "8500-node" / "5minDayShape.csv"
        der_csv_resolved = _resolve_profile_csv_path(
            der_profile_path,
            default_if_dir=str(der_profile_filename),
            fallback_file=fb_dummy,
        )
        der_m_series = np.asarray(
            inj.read_profile_csv_two_col_noheader(str(der_csv_resolved), npts=npts, debug=False),
            dtype=np.float64,
        )
        der_m_series = np.where(np.isfinite(der_m_series), der_m_series, 0.0)
        der_bus_phases = _der_phases_per_bus_in_cache(der_bus_list, node_to_local)
        miss_entirely = [
            str(b).strip().lower() for b in der_bus_list if str(b).strip().lower() not in der_bus_phases
        ]
        if miss_entirely:
            raise ValueError(
                "Each DER bus needs at least one ``bus.phase`` row in the tensor-cache ``node_to_local``. "
                "No matching phases for: "
                + ", ".join(miss_entirely[:24])
                + (" ..." if len(miss_entirely) > 24 else "")
            )
        thin = [bk for bk, phs in der_bus_phases.items() if len(phs) < 3]
        if thin:
            print(
                f"[da_gps_daily] DER: tensor cache has <3 phases for bus(es) {thin}; "
                f"that bus's share of P/Q is split only across cached phases ({der_bus_phases}).",
                flush=True,
            )
        der_effective_buses = list(der_bus_list)
        if der_use_bess_columns:
            _der_feat = "GNN: DER → ``p_bess_kw`` / ``q_bess_kvar`` at injection buses (cache BESS tails cleared)"
        else:
            _der_feat = "GNN: DER → added to ``p_load_kw`` / ``q_load_kvar`` at injection buses (no BESS columns)"
        print(
            f"[da_gps_daily] DER schedule: {der_csv_resolved}  max|ΣP|={float(der_max_kw):.6g} kW  "
            f"Q={float(der_q_frac_p):.6g}×P  buses={der_effective_buses}  ({_der_feat}; OpenDSS: ``New Generator``)",
            flush=True,
        )

    if edge_csv is not None:
        es = str(edge_csv).strip()
        if es in (".", "..") or (len(es) <= 2 and Path(es).name == "."):
            raise ValueError(
                f"Invalid --edge-csv {edge_csv!r} (often from EDGE_CSV=Path('') in a notebook → '.'). "
                "Omit --edge-csv to auto-resolve, or pass a real gnn_edges_phase_static.csv path."
            )
        edge_path = Path(edge_csv).resolve()
    else:
        edge_path = _resolve_default_edge_csv(report, hp, Path(cache_pt))
    print(f"[da_gps_daily] using edge CSV: {edge_path}", flush=True)
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)
    edge_index, edge_attr = _load_compacted_edges(edge_path, node_to_local)

    x_mean = torch.load(x_mean_path, map_location="cpu", weights_only=False).float()
    x_std = torch.load(x_std_path, map_location="cpu", weights_only=False).float()
    y_mean = torch.load(y_mean_path, map_location="cpu", weights_only=False).float()
    y_std = torch.load(y_std_path, map_location="cpu", weights_only=False).float()
    if int(x_mean.shape[-1]) != int(ref_x.shape[-1]):
        raise RuntimeError(
            f"x width mismatch: x_mean has {int(x_mean.shape[-1])} features but cache x has {int(ref_x.shape[-1])}"
        )
    n_feat = int(ref_x.shape[-1])
    if int(y_mean.numel()) != 2 * N:
        raise RuntimeError(f"y_mean length {int(y_mean.numel())} != 2*N ({2 * N})")

    bundle, state_dict = _resolve_da_gps_checkpoint(ckpt_path, run_dir, report_dir)
    n_cap = int(bundle["n_cap"])
    n_reg = int(bundle["n_reg"])
    cap_cols = [str(c) for c in (bundle.get("cap_target_cols") or [])]
    reg_cols = [str(c) for c in (bundle.get("reg_target_cols") or [])]
    if len(cap_cols) != n_cap:
        print(
            f"[da_gps_daily] WARNING: len(cap_target_cols)={len(cap_cols)} vs n_cap={n_cap}; using min length for aux plots.",
            flush=True,
        )
    if len(reg_cols) != n_reg:
        print(
            f"[da_gps_daily] WARNING: len(reg_target_cols)={len(reg_cols)} vs n_reg={n_reg}; using min length for aux plots.",
            flush=True,
        )
    n_cap_plot = min(n_cap, len(cap_cols))
    n_reg_plot = min(n_reg, len(reg_cols))

    reg_mean_path = norm_dir / "reg_mean.pt"
    reg_std_path = norm_dir / "reg_std.pt"
    reg_mean: torch.Tensor | None = None
    reg_std: torch.Tensor | None = None
    if n_reg_plot > 0 and reg_mean_path.is_file() and reg_std_path.is_file():
        reg_mean = torch.load(reg_mean_path, map_location="cpu", weights_only=False).float().reshape(1, -1)
        reg_std = torch.load(reg_std_path, map_location="cpu", weights_only=False).float().reshape(1, -1)
        if int(reg_mean.numel()) != n_reg or int(reg_std.numel()) != n_reg:
            print(
                f"[da_gps_daily] WARNING: reg_mean/std numel vs n_reg={n_reg}; disabling reg tap plots.",
                flush=True,
            )
            reg_mean = reg_std = None
    elif n_reg_plot > 0:
        print(
            f"[da_gps_daily] WARNING: missing {reg_mean_path.name} / {reg_std_path.name} — regulator tap denorm disabled.",
            flush=True,
        )

    n_sys = int(bundle["n_system_tokens"])
    hidden = int(bundle["hidden"])
    n_layers = int(bundle["layers"])
    heads = int(bundle["heads"])
    node_emb_dim = int(bundle.get("node_emb_dim", 0))
    edge_emb_dim = int(bundle.get("edge_emb_dim", 0))
    per_node_heads = bool(bundle.get("per_node_heads", False)) or (
        "volt_W" in state_dict and state_dict["volt_W"] is not None
    )
    per_device_cap_head = bool(bundle.get("per_device_cap_head", False)) or _state_dict_per_device_cap(
        state_dict
    )
    per_device_reg_head = bool(bundle.get("per_device_reg_head", False)) or _state_dict_per_device_reg(
        state_dict
    )
    reg_loss_mode = _resolve_reg_loss_mode(bundle, state_dict)
    reg_nclasses_raw = bundle.get("reg_nclasses")
    reg_nclasses: list[int] | None = None
    if isinstance(reg_nclasses_raw, (list, tuple)) and len(reg_nclasses_raw) == n_reg:
        reg_nclasses = [int(c) for c in reg_nclasses_raw]
    if reg_nclasses is None:
        reg_nclasses = _infer_reg_nclasses_from_state_dict(state_dict, n_reg)
    reg_class_values: torch.Tensor | None = None
    if reg_loss_mode == "ce":
        rcv_path = norm_dir / "reg_class_values.pt"
        if not rcv_path.is_file():
            raise FileNotFoundError(
                f"reg_loss=ce checkpoint requires {rcv_path} (from training OUT_DIR)."
            )
        reg_class_values = torch.load(rcv_path, map_location="cpu", weights_only=False).float()
        if reg_nclasses is None:
            reg_nclasses = [int(reg_class_values.shape[1])] * n_reg if reg_class_values.dim() == 2 else None
        print(
            f"[da_gps_daily] regulator head: CE ({n_reg} per-device classifiers); "
            f"tap plots use reg_class_values.pt, not reg_mean/std denorm.",
            flush=True,
        )
    n_pv_aux = int(bundle.get("n_pv_aux", 0))
    bundle_meta_cols = [str(c).strip().lower() for c in (bundle.get("meta_aux_target_cols") or bundle.get("pv_target_cols") or [])]
    pv_aux_cols = list(bundle_meta_cols)
    if n_pv_aux > 0 and len(pv_aux_cols) != n_pv_aux:
        print(
            f"[da_gps_daily] WARNING: len(meta_aux cols)={len(pv_aux_cols)} vs n_pv_aux={n_pv_aux}; using min length for meta plots.",
            flush=True,
        )

    # Row ``j`` of ``pv_W`` / ``pv_mean`` / ``pv_std`` matches **tensor-cache** ``meta_aux_cols[j]``, not
    # necessarily the string order inside ``da_gps_multitask_best.pt`` if that file was regenerated. Using
    # the wrong order denormalizes each head with another column's mean/std (tiny nonsensical GNN curves).
    if n_pv_aux > 0:
        cache_meta = z.get("meta_aux_cols")
        if meta_debug:
            print(
                f"[da_gps_daily][meta_debug] checkpoint meta_aux_target_cols ({len(bundle_meta_cols)}): {bundle_meta_cols}",
                flush=True,
            )
            print(
                f"[da_gps_daily][meta_debug] tensor cache meta_aux_cols: {cache_meta!r}",
                flush=True,
            )
        if isinstance(cache_meta, (list, tuple)) and len(cache_meta) == n_pv_aux:
            cm = [str(c).strip().lower() for c in cache_meta]
            bset = set(pv_aux_cols)
            cset = set(cm)
            if bset == cset and pv_aux_cols != cm:
                print(
                    f"[da_gps_daily] meta_aux column order: using tensor-cache list (matches pv_mean rows): {cm}",
                    flush=True,
                )
                pv_aux_cols = cm
            elif bset != cset:
                print(
                    f"[da_gps_daily] WARNING: tensor cache meta_aux_cols {cm} differs from checkpoint "
                    f"meta_aux_target_cols {pv_aux_cols} (same length but not same set). Meta denorm/plots may be wrong.",
                    flush=True,
                )
        elif isinstance(cache_meta, (list, tuple)) and cache_meta:
            print(
                f"[da_gps_daily] WARNING: tensor cache meta_aux_cols len={len(cache_meta)} vs n_pv_aux={n_pv_aux}; "
                f"using checkpoint order for plots.",
                flush=True,
            )

    n_pv_plot = min(n_pv_aux, len(pv_aux_cols)) if n_pv_aux > 0 else 0
    pv_mean_path = norm_dir / "pv_mean.pt"
    pv_std_path = norm_dir / "pv_std.pt"
    pv_mean: torch.Tensor | None = None
    pv_std: torch.Tensor | None = None
    if n_pv_plot > 0 and pv_mean_path.is_file() and pv_std_path.is_file():
        pv_mean = torch.load(pv_mean_path, map_location="cpu", weights_only=False).float().reshape(1, -1)
        pv_std = torch.load(pv_std_path, map_location="cpu", weights_only=False).float().reshape(1, -1)
        pv_std = torch.clamp(pv_std, min=1e-6)
        if int(pv_mean.numel()) != n_pv_aux or int(pv_std.numel()) != n_pv_aux:
            print(
                f"[da_gps_daily] WARNING: pv_mean/std numel vs n_pv_aux={n_pv_aux}; disabling meta-aux denorm.",
                flush=True,
            )
            pv_mean = pv_std = None
            n_pv_plot = 0
        else:
            print(
                "[da_gps_daily] meta_aux denorm stats (mean | std) per head — GNN curves use pred*std+mean:",
                flush=True,
            )
            for jm in range(min(n_pv_plot, int(pv_mean.numel()))):
                cn = pv_aux_cols[jm] if jm < len(pv_aux_cols) else f"col{jm}"
                print(
                    f"  [{jm}] {cn}: mean={float(pv_mean[0, jm]):.6g}  std={float(pv_std[0, jm]):.6g}",
                    flush=True,
                )
    elif n_pv_plot > 0:
        print(
            f"[da_gps_daily] WARNING: missing {pv_mean_path.name} / {pv_std_path.name} — meta-aux model denorm disabled.",
            flush=True,
        )
        n_pv_plot = 0

    use_legacy_edgeattn = _state_dict_is_legacy_edgeattn(state_dict)
    if use_legacy_edgeattn:
        print("[da_gps_daily] checkpoint backbone: legacy EdgeAttnMPNN (train_da_gps_multitask_complex_voltage.py)", flush=True)
    else:
        print("[da_gps_daily] checkpoint backbone: GINE (train_da_gps_multitask_complex_voltage_gine.py)", flush=True)

    dev = torch.device(resolve_da_gps_device(device))
    log_da_gps_device(str(dev))
    configure_cuda_inference(dev)
    t_gnn_setup0 = time.perf_counter()

    dropout = float(hp.get("dropout", 0.1))
    if bool(hp.get("disable_dropout", False)):
        dropout = 0.0

    if use_legacy_edgeattn:
        base_model = DAGPSModelEdgeAttn(
            n_nodes=N,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            heads=heads,
            n_layers=n_layers,
            n_cap=n_cap,
            n_reg=n_reg,
            n_system=n_sys,
            node_in_dim=n_feat,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            edge_dim=int(edge_attr.shape[1]),
            dropout=dropout,
            gradient_checkpointing=False,
            per_node_heads=per_node_heads,
            per_device_cap_head=per_device_cap_head,
            per_device_reg_head=per_device_reg_head,
        )
    else:
        base_model = DAGPSModelGine(
            n_nodes=N,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            heads=heads,
            n_layers=n_layers,
            n_cap=n_cap,
            n_reg=n_reg,
            n_system=n_sys,
            node_in_dim=n_feat,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            edge_dim=int(edge_attr.shape[1]),
            dropout=dropout,
            gradient_checkpointing=False,
            per_node_heads=per_node_heads,
            per_device_cap_head=per_device_cap_head,
            per_device_reg_head=per_device_reg_head,
            n_pv_aux=n_pv_aux,
            reg_nclasses=reg_nclasses if reg_loss_mode == "ce" else None,
        )
    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"load_state_dict failed: {e}\n"
            "Common causes: (1) checkpoint vs graph/cache mismatch; (2) GINE vs legacy EdgeAttn mismatch "
            "(this script auto-picks EdgeAttn if weights contain '.mpnn.msg.'); "
            "(3) mixed training_last vs best.pt from different runs."
        ) from e
    base_model.eval()
    model = maybe_torch_compile(base_model, label="da_gps_daily", device=dev)
    model.to(dev)

    x_mean_d = x_mean.to(dev)
    x_std_d = x_std.to(dev)
    y_mean_d = y_mean.to(dev).view(1, 2 * N)
    y_std_d = y_std.to(dev).view(1, 2 * N)
    reg_mean_d: torch.Tensor | None = None
    reg_std_d: torch.Tensor | None = None
    if reg_mean is not None and reg_std is not None:
        reg_mean_d = reg_mean.to(dev)
        reg_std_d = reg_std.to(dev)
    pv_mean_d: torch.Tensor | None = None
    pv_std_d: torch.Tensor | None = None
    if pv_mean is not None and pv_std is not None:
        pv_mean_d = pv_mean.to(dev)
        pv_std_d = pv_std.to(dev)

    mpath = mv_sx_mapping if mv_sx_mapping is not None else (REPO_ROOT / "8500-node" / "mv_x_sx_node_mapping_8500.csv")
    mv_rules: list[dict[str, str]] = []
    if feeder_key == "8500" and mpath.is_file():
        mv_rules = _load_mv_sx_mapping(mpath)
    if mv_rules:
        print(f"[da_gps_daily] mv↔sx mapping: {len(mv_rules)} rules from {mpath}", flush=True)
    elif feeder_key == "8500":
        print(f"[da_gps_daily] WARNING: no MV↔sx mapping at {mpath} (using direct bus.phase P/Q)", flush=True)
    else:
        print(f"[da_gps_daily] feeder={feeder_key}: skipping 8500 MV↔sx mapping (direct bus.phase P/Q)", flush=True)

    compile_meta = _compile_feeder_master(feeder_key)
    if skip_opendss_solve:
        print(
            f"[da_gps_daily] OpenDSS compile (static maps only): {compile_meta['master_dss']} "
            f"(cwd {compile_meta['model_dir']}); **no** per-step Solve()",
            flush=True,
        )
    else:
        print(
            f"[da_gps_daily] OpenDSS compile feeder={feeder_key}: {compile_meta['master_dss']} "
            f"(cwd {compile_meta['model_dir']})",
            flush=True,
        )
    print(
        f"[da_gps_daily] Feeder master selected via --feeder={feeder_key}. "
        "Load / irradiance / DER profile CLI args scale injections; they do not redirect another DSS master.",
        flush=True,
    )
    fb_load_default, fb_irr_default = _default_feeder_profiles(
        feeder_key, out_dir=out_dir, npts=int(npts), step_min=float(step_min)
    )
    if str(pv_irradiance_profile_path or "").strip():
        irr_csv = _resolve_profile_csv_path(
            pv_irradiance_profile_path,
            default_if_dir=str(pv_irradiance_filename),
            fallback_file=fb_irr_default,
        )
    else:
        irr_csv = fb_irr_default
    if str(load_profile_path or "").strip():
        prof_resolved = _resolve_profile_csv_path(
            load_profile_path,
            default_if_dir=str(load_profile_filename),
            fallback_file=fb_load_default,
        )
        print(f"[da_gps_daily] daily load profile (override): {prof_resolved}", flush=True)
    elif str(daily_profile_csv or "").strip() and feeder_key == "8500":
        prof_resolved = rd8500._resolve_daily_profile_csv(daily_profile_csv)
        print(f"[da_gps_daily] daily load profile: {prof_resolved}", flush=True)
    else:
        prof_resolved = fb_load_default
        print(f"[da_gps_daily] daily load profile (feeder default): {prof_resolved}", flush=True)
    parity_profiles = prepare_parity_profiles(
        prof_resolved,
        irr_csv,
        npts=int(npts),
        step_min=float(step_min),
        daily_stress=float(daily_stress),
        stress_clip_lo=float(stress_clip_lo),
        stress_clip_hi=float(stress_clip_hi),
    )
    m_raw, m_eff, m_irr = parity_profiles.m_raw, parity_profiles.m_eff, parity_profiles.m_irr
    m_irr_ref_mean = float(np.mean(m_irr)) if len(m_irr) else 1.0
    if not np.isfinite(m_irr_ref_mean) or m_irr_ref_mean < 1e-6:
        m_irr_ref_mean = 1.0
    print(
        f"[da_gps_daily] PV irradiance mult m_irr (col 2, training ``m_pv_t``): {irr_csv}  "
        f"span=[{float(np.min(m_irr)):.4g},{float(np.max(m_irr)):.4g}]  "
        f"mean(m_irr)={m_irr_ref_mean:.4g} "
        f"(``p_pv_kw`` = Pmpp0×m_irr[i]; DSS ``Pmpp`` = Pmpp0×m_irr[i] under snapshot)",
        flush=True,
    )
    if not skip_opendss_solve:
        setup_da_gps_snapshot_opendss(npts=int(npts), step_min=float(step_min))
        print(
            "[da_gps_daily] OpenDSS: IrradDay001 → unity for snapshot solves; "
            "PV Pmpp = Pmpp0×m_irr[i] each step (shared setup with compare snapshot mode).",
            flush=True,
        )
    pv_names_dss = _discover_pv_system_names()
    pv_base_pmpp_kw = _read_pv_base_pmpp_kw(pv_names_dss)
    pv_to_busph_w = _collect_pv_to_busph_weights()
    if col_pv is not None and pv_names_dss:
        n_alloc = sum(len(pv_to_busph_w.get(str(nm).strip(), [])) for nm in pv_names_dss)
        print(
            f"[da_gps_daily] p_pv_kw (GNN x): ``pmpp_set×m_pv_t`` style = Pmpp0×m_irr[i] per PV, "
            f"equal split over element phases ({len(pv_names_dss)} PVsystems, {n_alloc} bus-phase terms) — "
            f"``_apply_snapshot_with_pv`` / ``_collect_pv_maps``; DSS ``Pmpp`` = Pmpp0×m_irr[i] "
            f"(unity IrradDay001 under snapshot mode).",
            flush=True,
        )
        if n_alloc == 0:
            print(
                "[da_gps_daily] WARNING: ``_collect_pv_to_busph_weights`` returned no allocations — "
                "``p_pv_kw`` will stay zero.",
                flush=True,
            )
    if not skip_opendss_solve:
        print("[da_gps_daily] OpenDSS: snapshot mode per Solve() (shared setup with compare snapshot mode).", flush=True)

    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]

    print(
        f"[da_gps_daily] stress: daily_stress={daily_stress:g} scenario_scale={scenario_scale:g} "
        f"clip=[{stress_clip_lo:g},{stress_clip_hi:g}]  m_raw∈[{float(np.min(m_raw)):.4f},{float(np.max(m_raw)):.4f}] "
        f"m_eff∈[{float(np.min(m_eff)):.4f},{float(np.max(m_eff)):.4f}]",
        flush=True,
    )

    if skip_opendss_solve:
        all_nodes = list(node_order)
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        print(
            f"[da_gps_daily] GNN-only path: node index from tensor cache ({len(all_nodes)} bus.phase rows); "
            "skipping OpenDSS AllNodeNames + daily Solve loop.",
            flush=True,
        )
    else:
        all_nodes: list[str] = []
        for n in dss.Circuit.AllNodeNames():
            s = str(n).strip().lower()
            if "." not in s:
                continue
            phs = s.rsplit(".", 1)[1]
            try:
                ph = int(phs)
            except ValueError:
                continue
            if ph in (1, 2, 3):
                all_nodes.append(s)
        all_nodes = list(dict.fromkeys(all_nodes))
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    der_gen_by_bus: dict[str, str] = {}
    if der_effective_buses and not skip_opendss_solve:
        der_gen_by_bus = _install_der_three_phase_generators(der_effective_buses)
        if not der_gen_by_bus:
            print(
                "[da_gps_daily] WARNING: DER OpenDSS ``New Generator`` failed for all buses — check bus names vs circuit.",
                flush=True,
            )

    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)
    v_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_gnn = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    va_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    va_gnn = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)

    cap_names_dss = rd8500._discover_capacitors()
    reg_names_dss = rd8500._discover_reg_controls()
    print(
        f"[da_gps_daily] multitask daily series: cap={n_cap_plot} reg={n_reg_plot} meta_aux={n_pv_plot} "
        f"(OpenDSS: {len(cap_names_dss)} caps, {len(reg_names_dss)} regs, {len(pv_names_dss)} PVsystems)",
        flush=True,
    )
    if n_pv_plot > 0 and use_legacy_edgeattn:
        print(
            "[da_gps_daily] WARNING: legacy EdgeAttn has no meta-aux output; meta_aux GNN curves will be NaN.",
            flush=True,
        )
    if n_pv_plot > 0:
        if pv_names_dss:
            print(f"[da_gps_daily] DSS PVSystem names (for pv_*_p_post_kw meta match): {pv_names_dss}", flush=True)
        else:
            print(
                "[da_gps_daily] WARNING: meta_aux needs PV post-solve P/Q but **no PVSystem** elements were found "
                "in the compiled daily circuit — DSS PV meta curves will be NaN.",
                flush=True,
            )
    cap_dss_steps = (
        None
        if skip_opendss_solve or n_cap_plot <= 0
        else np.full((npts, n_cap_plot), np.nan, dtype=np.float32)
    )
    cap_gnn_prob = np.full((npts, n_cap_plot), np.nan, dtype=np.float32) if n_cap_plot > 0 else None
    reg_dss_tap = (
        None
        if skip_opendss_solve or n_reg_plot <= 0
        else np.full((npts, n_reg_plot), np.nan, dtype=np.float32)
    )
    reg_gnn_tap = np.full((npts, n_reg_plot), np.nan, dtype=np.float32) if n_reg_plot > 0 else None
    meta_dss = (
        None
        if skip_opendss_solve or n_pv_plot <= 0
        else np.full((npts, n_pv_plot), np.nan, dtype=np.float32)
    )
    meta_gnn = np.full((npts, n_pv_plot), np.nan, dtype=np.float32) if n_pv_plot > 0 else None
    meta_presolve_p = np.full((npts, n_pv_plot), np.nan, dtype=np.float32) if n_pv_plot > 0 else None

    n_nonconv = 0
    open_apply_s_total = 0.0
    open_reassert_s_total = 0.0
    open_solve_only_s_total = 0.0
    open_get_s_total = 0.0
    feature_build_s_total = 0.0
    gnn_infer_s_total = 0.0
    gnn_forward_only_s_total = 0.0
    gnn_setup_once_s = 0.0
    first_diag = True
    pv_read_diag = False
    reg_tap_align_printed = False

    ei = edge_index.to(dev)
    ea = edge_attr.to(dev)

    feat_tables = _precompute_daily_feature_tables(
        ref_x=ref_x,
        node_order=node_order,
        load_to_busph=load_to_busph,
        base_names=base_names,
        base_kw=base_kw,
        base_kvar=base_kvar,
        col_p=col_p,
        col_q=col_q,
        col_pv=col_pv,
        col_bess_p=col_bess_p,
        col_bess_q=col_bess_q,
        mv_rules=mv_rules,
        pv_names=pv_names_dss,
        pv_base_pmpp_kw=pv_base_pmpp_kw,
        pv_to_busph=pv_to_busph_w,
        der_effective_buses=der_effective_buses,
        der_bus_phases=der_bus_phases,
        node_to_local=node_to_local,
        der_use_bess_columns=der_use_bess_columns,
        der_max_kw=float(der_max_kw),
        der_q_frac=float(der_q_frac_p),
    )
    batch_k = read_gnn_batch_steps(gnn_batch_steps)
    if batch_k > 1:
        print(
            f"[da_gps_daily] GNN batched inference: {batch_k} steps/forward "
            f"(GNN_BATCH_STEPS or gnn_batch_steps; per-step P/Q still applied on host)",
            flush=True,
        )
    x_static_np = feat_tables["x_static"]
    x_ring_bufs: list[np.ndarray] = []
    x_ring_torch: list[torch.Tensor] = []
    for _ in range(max(1, batch_k)):
        xt = torch.from_numpy(np.ascontiguousarray(x_static_np, dtype=np.float32))
        if dev.type == "cuda":
            xt = xt.pin_memory()
        x_ring_torch.append(xt)
        x_ring_bufs.append(xt.numpy())
    scatter_li_t, scatter_j_np, scatter_li_np = build_scatter_indices(node_order, node_to_idx, dev)
    gnn_runner = DailyGnnInferenceRunner(
        model=model,
        device=dev,
        n_nodes=N,
        n_feat=n_feat,
        edge_index=ei,
        edge_attr=ea,
        x_mean_d=x_mean_d,
        x_std_d=x_std_d,
        y_mean_d=y_mean_d,
        y_std_d=y_std_d,
        reg_mean_d=reg_mean_d,
        reg_std_d=reg_std_d,
        pv_mean_d=pv_mean_d,
        pv_std_d=pv_std_d,
        reg_loss_mode=reg_loss_mode,
        reg_class_values=reg_class_values.to(dev) if reg_class_values is not None else None,
        n_cap=n_cap_plot,
        n_reg=n_reg_plot,
        n_pv=n_pv_plot,
        batch_steps=batch_k,
        scatter_li=scatter_li_t,
        scatter_j=torch.tensor(scatter_j_np, dtype=torch.long, device=dev) if scatter_j_np.size else None,
        n_scatter_cols=int(scatter_j_np.size),
    )
    gnn_runner.alloc_deferred_buffers(npts)
    gnn_setup_graph_s = gnn_runner.setup()
    gnn_setup_once_s = time.perf_counter() - t_gnn_setup0
    print(
        f"[da_gps_daily] GNN setup once: {gnn_setup_once_s:.4f}s "
        f"(model+norm tensors+static feature tables+cuda-graph/warmup={gnn_setup_graph_s:.4f}s)  "
        f"defer_d2h={gnn_runner.defer_d2h} cuda_graphs={gnn_runner.use_cuda_graphs}",
        flush=True,
    )

    pending_step_i: list[int] = []
    pending_x_hosts: list[torch.Tensor] = []
    ring_slot = 0

    _meta_dbg_steps: set[int] = set()
    if meta_debug:
        _meta_dbg_steps = {
            0,
            max(0, npts // 4),
            max(0, npts // 2),
            min(max(0, npts - 1), max(0, (3 * npts) // 4)),
            min(143, max(0, npts - 1)),
            max(0, npts - 1),
        }
        print(
            f"[da_gps_daily][meta_debug] extra logs at converged steps {_meta_dbg_steps} "
            f"(norm + denorm meta heads, DSS meta row, clock, m_eff, m_irr, x_n stats)",
            flush=True,
        )
        if col_pv is not None:
            print(
                "[da_gps_daily][meta_debug] p_pv_kw: each step = (Pmpp on device)×m_irr[i] split on PV bus phases "
                "(``_collect_pv_to_busph_weights``); DSS Pmpp = Pmpp0×m_irr[i] under snapshot.",
                flush=True,
            )

    def _flush_gnn_pending(*, force: bool = False) -> None:
        nonlocal gnn_infer_s_total, gnn_forward_only_s_total
        if not pending_step_i:
            return
        if not force and len(pending_step_i) < batch_k:
            return
        t_gnn0 = time.perf_counter()
        t_fwd0 = time.perf_counter()
        if len(pending_step_i) == 1:
            gnn_runner.copy_host_features(pending_x_hosts[0])
            outs = [gnn_runner.forward_single(sync_forward=True)]
        else:
            outs = gnn_runner.forward_batch(pending_x_hosts, sync_forward=True)
        t_fwd1 = time.perf_counter()
        gnn_forward_only_s_total += t_fwd1 - t_fwd0
        for si, out in zip(pending_step_i, outs):
            if gnn_runner.defer_d2h:
                gnn_runner.store_step(si, out)
            else:
                gnn_runner.write_step_host(
                    si,
                    out,
                    v_gnn=v_gnn,
                    va_gnn=va_gnn,
                    cap_gnn_prob=cap_gnn_prob,
                    reg_gnn_tap=reg_gnn_tap,
                    meta_gnn=meta_gnn,
                    scatter_j_np=scatter_j_np,
                    scatter_li_np=scatter_li_np,
                    n_cap_plot=n_cap_plot,
                    n_reg_plot=n_reg_plot,
                    n_pv_plot=n_pv_plot,
                )
            if (
                meta_debug
                and meta_gnn is not None
                and n_pv_plot > 0
                and (si in _meta_dbg_steps)
                and out.pv_dn is not None
            ):
                hr, sec = snapshot_step_hr_sec(si, step_min=float(step_min))
                try:
                    dbl_h = float(dss.Solution.DblHour())
                except Exception:
                    dbl_h = float("nan")
                pv_dn_np = out.pv_dn.detach().cpu().numpy().reshape(-1)[:n_pv_plot]
                print(
                    f"[da_gps_daily][meta_debug] step={si} clock hour={hr} sec={sec} DblHour={dbl_h:.6g} | "
                    f"GNN pv_pred denorm={np.array2string(pv_dn_np, precision=4, separator=', ')}",
                    flush=True,
                )
        if not gnn_runner.defer_d2h:
            sync_inference_device(dev)
        gnn_infer_s_total += time.perf_counter() - t_gnn0
        pending_step_i.clear()
        pending_x_hosts.clear()

    for i in range(npts):
        hr, sec = snapshot_step_hr_sec(i, step_min=float(step_min))
        m_t = step_load_multiplier(m_eff, i, scenario_scale)
        ir_t = step_irradiance_multiplier(m_irr, i)

        if not skip_opendss_solve:
            t_apply0 = time.perf_counter()
            apply_explicit_loads_and_pv_pmpp(
                base_names=base_names,
                base_kw=base_kw,
                base_kvar=base_kvar,
                m_t=m_t,
                pv_names=pv_names_dss,
                pv_base_pmpp_kw=pv_base_pmpp_kw,
                ir_t=ir_t,
            )
            m_der_t = (
                float(der_m_series[i])
                if want_der and i < int(der_m_series.shape[0])
                else 0.0
            )
            if der_gen_by_bus and der_effective_buses:
                _set_der_generators_kw(
                    der_gen_by_bus,
                    der_effective_buses,
                    p_profile_scale=m_der_t,
                    der_max_kw=float(der_max_kw),
                    der_q_frac=float(der_q_frac_p),
                )
            open_apply_s_total += time.perf_counter() - t_apply0

            t_reassert0 = time.perf_counter()
            reassert_snapshot_and_set_clock(i, step_min=float(step_min))
            open_reassert_s_total += time.perf_counter() - t_reassert0

            t_solve0 = time.perf_counter()
            dss.Solution.Solve()
            open_solve_only_s_total += time.perf_counter() - t_solve0

            if not dss.Solution.Converged():
                n_nonconv += 1
                continue

            t_get0 = time.perf_counter()
            vmag, vang = inj.get_all_node_voltage_pu_and_angle_filtered(all_nodes)
            v_dss[i, :] = np.asarray(vmag, dtype=np.float32)
            va_dss[i, :] = np.asarray(vang, dtype=np.float32)
            open_get_s_total += time.perf_counter() - t_get0

            if cap_dss_steps is not None:
                cap_fields = rd8500._read_capacitor_sample_fields(cap_names_dss)
                for jc in range(n_cap_plot):
                    col = cap_cols[jc]
                    vv = cap_fields.get(col)
                    if vv is None:
                        cap_dss_steps[i, jc] = np.nan
                    else:
                        try:
                            fv = float(vv)
                            cap_dss_steps[i, jc] = fv if np.isfinite(fv) else np.nan
                        except (TypeError, ValueError):
                            cap_dss_steps[i, jc] = np.nan
            if reg_dss_tap is not None:
                tap_raw = rd8500._read_reg_control_state(reg_names_dss)
                if not reg_tap_align_printed and n_reg_plot > 0:
                    reg_tap_align_printed = True
                    bits: list[str] = []
                    miss_cols: list[str] = []
                    for jr in range(n_reg_plot):
                        colr = reg_cols[jr]
                        vv = _lookup_reg_tap_pu(colr, tap_raw)
                        if vv is None:
                            miss_cols.append(str(colr))
                            bits.append(f"{colr}=NA")
                        else:
                            bits.append(f"{colr}={vv:.5g}")
                    print(
                        "[da_gps_daily] regulator taps: ``reg_target_cols`` ↔ ``_read_reg_control_state`` "
                        f"(first converged step). Resolved: " + "; ".join(bits),
                        flush=True,
                    )
                    if miss_cols:
                        print(
                            f"[da_gps_daily] WARNING: no DSS tap for columns {miss_cols}. "
                            f"OpenDSS ``reg_*`` keys (sample): {sorted(tap_raw.keys())[:12]}",
                            flush=True,
                        )
                for jr in range(n_reg_plot):
                    colr = reg_cols[jr]
                    vv = _lookup_reg_tap_pu(colr, tap_raw)
                    if vv is None:
                        reg_dss_tap[i, jr] = np.nan
                    else:
                        reg_dss_tap[i, jr] = float(vv)
            if meta_dss is not None and n_pv_plot > 0:
                p_loss_kw, q_loss_kvar = _circuit_losses_kw_kvar()
                pv_tot = _read_pv_totals_post_solve_kw_kvar(pv_names_dss)
                if not pv_read_diag:
                    pv_read_diag = True
                    mx = max((abs(a) + abs(b) for (a, b) in pv_tot.values()), default=0.0)
                    nz = int(sum(1 for a, b in pv_tot.values() if abs(a) + abs(b) > 1e-6))
                    det = ", ".join(f"{k}:(P={a:.5g},Q={b:.5g})" for k, (a, b) in sorted(pv_tot.items()))
                    print(
                        f"[da_gps_daily] PV post-solve (first converged step): {det}  |  "
                        f"{len(pv_tot)} PVSystem element(s), {nz} with |P|+|Q|>1e-6, max |P|+|Q|={mx:.4g}  "
                        f"(meta keys: pv_<dssname>_p_post_kw / _q_post_kvar)",
                        flush=True,
                    )
                for jm in range(n_pv_plot):
                    colm = pv_aux_cols[jm]
                    vmeta = _dss_scalar_for_meta_aux_col(
                        colm, pv_totals=pv_tot, p_loss_kw=p_loss_kw, q_loss_kvar=q_loss_kvar
                    )
                    if vmeta is None:
                        meta_dss[i, jm] = np.nan
                    else:
                        fv = float(vmeta)
                        meta_dss[i, jm] = fv if np.isfinite(fv) else np.nan
                if meta_presolve_p is not None:
                    for jm in range(n_pv_plot):
                        colm = pv_aux_cols[jm]
                        stem = _stem_from_pv_p_post_meta_col(colm)
                        if stem is None:
                            continue
                        b0n = _nameplate_pmpp_kw_for_pv_stem(stem, pv_base_pmpp_kw)
                        if b0n is not None:
                            meta_presolve_p[i, jm] = float(b0n * ir_t)
        else:
            m_der_t = (
                float(der_m_series[i])
                if want_der and i < int(der_m_series.shape[0])
                else 0.0
            )

        step_buf = x_ring_bufs[ring_slot]
        t_fb0 = time.perf_counter()
        _apply_daily_feature_tables(
            step_buf,
            feat_tables,
            m_t=float(m_t),
            ir_t=float(ir_t),
            m_der_t=float(m_der_t),
            want_der=bool(want_der),
        )

        if first_diag:
            first_diag = False
            nz = int(np.sum(np.abs(step_buf[:, col_p]) + np.abs(step_buf[:, col_q]) > 1e-3))
            print(f"[da_gps_daily] feature diag (first step): nodes with |P|+|Q|>1e-3: {nz}/{N}", flush=True)

        feature_build_s_total += time.perf_counter() - t_fb0

        if meta_debug and (i in _meta_dbg_steps):
            try:
                dbl_h = float(dss.Solution.DblHour())
            except Exception:
                dbl_h = float("nan")
            bits = [
                f"step={i} clock hour={hr} sec={sec} DblHour={dbl_h:.6g}",
                f"m_eff[i]={float(m_eff[i]):.6g}",
                f"ir_t(m_irr)={ir_t:.6g}",
            ]
            if want_der:
                bits.append(f"m_der[i]={m_der_t:.6g}")
            if col_pv is not None:
                bits.append(f"x_step[p_pv_kw]_mean={float(np.mean(step_buf[:, col_pv])):.6g}")
            if col_p is not None:
                bits.append(f"x_step[p_load_kw]_mean={float(np.mean(step_buf[:, col_p])):.6g}")
            print("[da_gps_daily][meta_debug] " + " | ".join(bits), flush=True)
            if meta_dss is not None and n_pv_plot > 0:
                print(
                    f"[da_gps_daily][meta_debug]   DSS meta_dss[{i}]={np.array2string(np.asarray(meta_dss[i, :n_pv_plot]), precision=4, separator=', ')}",
                    flush=True,
                )

        pending_step_i.append(i)
        pending_x_hosts.append(x_ring_torch[ring_slot])
        ring_slot = (ring_slot + 1) % max(1, batch_k)
        at_last = (i + 1) == npts
        if len(pending_step_i) >= batch_k or at_last:
            _flush_gnn_pending(force=at_last)

        if (i + 1) % max(1, npts // 12) == 0 or (voltages_only and (i + 1) == npts):
            if skip_opendss_solve:
                print(
                    f"[{i + 1}/{npts}] GNN-only: feat={feature_build_s_total:.2f}s gnn={gnn_infer_s_total:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[{i + 1}/{npts}] apply={open_apply_s_total:.2f}s reassert={open_reassert_s_total:.2f}s "
                    f"solve={open_solve_only_s_total:.2f}s getV={open_get_s_total:.2f}s "
                    f"feat={feature_build_s_total:.2f}s gnn={gnn_infer_s_total:.2f}s",
                    flush=True,
                )

    _flush_gnn_pending(force=True)
    if gnn_runner.defer_d2h:
        t_fin0 = time.perf_counter()
        with torch.no_grad():
            gnn_runner.finalize_deferred(
                v_gnn,
                va_gnn,
                cap_gnn_prob,
                reg_gnn_tap,
                meta_gnn,
                scatter_j_np,
            )
        gnn_infer_s_total += time.perf_counter() - t_fin0

    if scatter_j_np.size > 0 and npts >= 2:
        j0 = int(scatter_j_np[0])
        v_trace = v_gnn[: min(npts, 12), j0]
        fin = v_trace[np.isfinite(v_trace)]
        if fin.size >= 2 and np.allclose(fin, fin[0], rtol=0, atol=1e-7):
            print(
                "[da_gps_daily] WARNING: GNN |V| appears flat across timesteps at first mapped node "
                f"({node_order[int(scatter_li_np[0])] if scatter_li_np.size else '?'}) — check pin_memory / feature apply.",
                flush=True,
            )

    n_ok = int(npts - n_nonconv)
    gnn_per_step_s = (feature_build_s_total + gnn_infer_s_total) / max(n_ok, 1)
    gnn_total_wall_s = gnn_setup_once_s + feature_build_s_total + gnn_infer_s_total

    if voltages_only:
        targets = plot_nodes if plot_nodes else list(node_order)
        out_v = _gnn_voltages_for_nodes(targets, v_gnn, node_to_idx, npts)
        n_finite = sum(int(np.isfinite(v).any()) for v in out_v.values())
        reg_out = (
            reg_gnn_tap.astype(np.float64, copy=True)
            if reg_gnn_tap is not None and n_reg_plot > 0
            else np.zeros((npts, 0), dtype=np.float64)
        )
        cap_out = (
            cap_gnn_prob.astype(np.float64, copy=True)
            if cap_gnn_prob is not None and n_cap_plot > 0
            else np.zeros((npts, 0), dtype=np.float64)
        )
        n_reg_fin = int(np.sum(np.isfinite(reg_out))) if reg_out.size else 0
        n_cap_fin = int(np.sum(np.isfinite(cap_out))) if cap_out.size else 0
        print(
            f"[da_gps_daily] voltages_only: returning {len(out_v)} node |V| series "
            f"({n_finite} with finite values), reg_tap_pu {reg_out.shape} ({n_reg_fin} finite), "
            f"cap_sigmoid {cap_out.shape} ({n_cap_fin} finite); skipped plots/CSV exports.",
            flush=True,
        )
        meta_out = (
            meta_gnn.astype(np.float64, copy=True)
            if meta_gnn is not None and n_pv_plot > 0
            else np.zeros((npts, 0), dtype=np.float64)
        )
        return {
            "voltages": out_v,
            "reg_tap_pu": reg_out,
            "cap_sigmoid": cap_out,
            "reg_cols": [str(c) for c in reg_cols[:n_reg_plot]],
            "cap_cols": [str(c) for c in cap_cols[:n_cap_plot]],
            "meta_aux_gnn": meta_out,
            "meta_aux_cols": [str(c) for c in pv_aux_cols[:n_pv_plot]],
            "gnn_setup_once_s": gnn_setup_once_s,
            "gnn_per_step_s": gnn_per_step_s,
            "gnn_total_wall_s": gnn_total_wall_s,
            "feature_build_s_total": feature_build_s_total,
            "gnn_forward_only_s_total": gnn_forward_only_s_total,
            "gnn_bucket_s_total": gnn_infer_s_total,
            "n_ok": n_ok,
            "npts": npts,
        }

    cfg_stem = _safe_stem(ckpt_path.stem)
    _backbone = "EdgeAttn (legacy)" if use_legacy_edgeattn else "GINE"
    _timing_title = f"Daily Timing Summary (DA-GPS {_backbone} vs OpenDSS)"

    print_mv_daily_timing_summary(
        n_ok=n_ok,
        npts=npts,
        n_nonconv=n_nonconv,
        open_apply_s_total=open_apply_s_total,
        open_reassert_s_total=open_reassert_s_total,
        open_solve_only_s_total=open_solve_only_s_total,
        open_get_s_total=open_get_s_total,
        feature_build_s_total=feature_build_s_total,
        gnn_forward_only_s_total=gnn_forward_only_s_total,
        gnn_bucket_s_total=gnn_infer_s_total,
        device=str(dev),
        title=_timing_title,
        feature_label="DA-GPS feature build (bus P/Q + PV presolve p_pv_kw)",
        log_prefix="[da_gps_daily]",
        gnn_setup_once_s=gnn_setup_once_s,
        gnn_per_step_s=gnn_per_step_s,
        gnn_total_wall_s=gnn_total_wall_s,
    )

    pipeline_metrics = compute_mv_daily_timing_metrics(
        n_ok=n_ok,
        open_apply_s_total=open_apply_s_total,
        open_solve_only_s_total=open_solve_only_s_total,
        open_get_s_total=open_get_s_total,
        feature_build_s_total=feature_build_s_total,
        gnn_forward_only_s_total=gnn_forward_only_s_total,
    )
    dss_apply_ms = pipeline_metrics["dss_apply_ms"]
    dss_solve_ms = pipeline_metrics["dss_solve_ms"]
    dss_collect_ms = pipeline_metrics["dss_collect_ms"]
    gnn_feature_gen_ms = pipeline_metrics["gnn_feature_ms"]
    gnn_forward_ms = pipeline_metrics["gnn_forward_ms"]
    dss_pipeline_ms = pipeline_metrics["dss_total_ms"]
    gnn_pipeline_ms = pipeline_metrics["gnn_total_ms"]

    mask = np.isfinite(v_dss) & np.isfinite(v_gnn)
    if mask.any():
        y_t = v_dss[mask].astype(np.float64, copy=False)
        y_p = v_gnn[mask].astype(np.float64, copy=False)
        mae = float(np.mean(np.abs(y_t - y_p)))
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        ss_res = float(np.sum((y_t - y_p) ** 2))
        y_mean = float(np.mean(y_t))
        ss_tot = float(np.sum((y_t - y_mean) ** 2))
        if ss_tot > 1e-30 and np.isfinite(ss_tot):
            r2_vmag = float(1.0 - ss_res / ss_tot)
        else:
            r2_vmag = float("nan")
        n_v_overlap = int(mask.sum())
        print(
            f"\nOverall |V|: MAE={mae:.6f} pu  RMSE={rmse:.6f} pu  R²={r2_vmag:.6f}  "
            f"n_points={n_v_overlap} nonconv={n_nonconv}",
            flush=True,
        )
    else:
        mae = rmse = r2_vmag = float("nan")
        n_v_overlap = 0
        print("\nOverall: no finite overlapping points (check convergence / mapping).", flush=True)

    mask_ang = np.isfinite(va_dss) & np.isfinite(va_gnn)
    if mask_ang.any():
        d_ang = va_dss[mask_ang] - va_gnn[mask_ang]
        wrap_ang = np.mod(d_ang.astype(np.float64) + 180.0, 360.0) - 180.0
        mae_ang_deg = float(np.mean(np.abs(wrap_ang)))
        n_ang_overlap = int(mask_ang.sum())
        a_t = va_dss[mask_ang].astype(np.float64, copy=False)
        ss_res_a = float(np.sum(wrap_ang.astype(np.float64) ** 2))
        a_mean = float(np.mean(a_t))
        ss_tot_a = float(np.sum((a_t - a_mean) ** 2))
        if ss_tot_a > 1e-30 and np.isfinite(ss_tot_a):
            r2_vang = float(1.0 - ss_res_a / ss_tot_a)
        else:
            r2_vang = float("nan")
        print(
            f"Overall angle: MAE={mae_ang_deg:.4f} deg (circular)  R²(naive linear DSS)={r2_vang:.6f}  "
            f"n_points={n_ang_overlap}",
            flush=True,
        )
    else:
        mae_ang_deg = r2_vang = float("nan")
        n_ang_overlap = 0
        print("\nOverall angle: no finite overlapping points.", flush=True)

    node_rows = []
    for j, n in enumerate(all_nodes):
        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_gnn[:, j])
        ma_deg = float("nan")
        if not m.any():
            continue
        ma = float(np.mean(np.abs(v_dss[m, j] - v_gnn[m, j])))
        ma_ang_mask = np.isfinite(va_dss[:, j]) & np.isfinite(va_gnn[:, j]) & m
        if ma_ang_mask.any():
            dd = va_dss[ma_ang_mask, j] - va_gnn[ma_ang_mask, j]
            ma_deg = float(np.mean(np.abs(np.mod(dd.astype(np.float64) + 180.0, 360.0) - 180.0)))
        node_rows.append((n, ma, ma_deg))
    df_mae = pd.DataFrame(node_rows, columns=["node", "mae", "mae_angle_deg"]).sort_values("mae", ascending=False)
    df_mae.to_csv(out_dir / f"daily_mae_per_node_{cfg_stem}.csv", index=False)

    aux_outputs: dict[str, str] = {}
    if cap_dss_steps is not None and cap_gnn_prob is not None and n_cap_plot > 0:
        cap_rows: dict[str, object] = {"step_idx": list(range(npts)), "hour": t_hours.astype(np.float64).tolist()}
        for jc in range(n_cap_plot):
            col = cap_cols[jc]
            cap_rows[f"{col}__dss_n_steps_on"] = cap_dss_steps[:, jc].astype(np.float64).tolist()
            cap_rows[f"{col}__gnn_sigmoid"] = cap_gnn_prob[:, jc].astype(np.float64).tolist()
        cap_csv = out_dir / f"daily_cap_bank_status_{cfg_stem}.csv"
        pd.DataFrame(cap_rows).to_csv(cap_csv, index=False)
        aux_outputs["daily_cap_bank_status_csv"] = str(cap_csv)
        print(f"[da_gps_daily] wrote {cap_csv}", flush=True)
        for jc in range(n_cap_plot):
            col = cap_cols[jc]
            ppath = out_dir / f"daily_cap_bank_{cfg_stem}_{_safe_stem(col)}.png"
            _plot_cap_bank_daily_compare(
                t_hours=t_hours,
                dss_n_steps=cap_dss_steps[:, jc],
                gnn_sigmoid=cap_gnn_prob[:, jc],
                col_name=col,
                out_path=ppath,
                show_plots=show_plots,
            )

    if reg_dss_tap is not None and reg_gnn_tap is not None and reg_mean_d is not None and n_reg_plot > 0:
        reg_rows: dict[str, object] = {"step_idx": list(range(npts)), "hour": t_hours.astype(np.float64).tolist()}
        for jr in range(n_reg_plot):
            col = reg_cols[jr]
            reg_rows[f"{col}__dss_tap_pu"] = reg_dss_tap[:, jr].astype(np.float64).tolist()
            reg_rows[f"{col}__gnn_tap_pu"] = reg_gnn_tap[:, jr].astype(np.float64).tolist()
        reg_csv = out_dir / f"daily_regulator_tap_{cfg_stem}.csv"
        pd.DataFrame(reg_rows).to_csv(reg_csv, index=False)
        aux_outputs["daily_regulator_tap_csv"] = str(reg_csv)
        print(f"[da_gps_daily] wrote {reg_csv}", flush=True)
        for jr in range(n_reg_plot):
            col = reg_cols[jr]
            ppath = out_dir / f"daily_regulator_tap_{cfg_stem}_{_safe_stem(col)}.png"
            _plot_regulator_tap_daily_compare(
                t_hours=t_hours,
                dss_tap_pu=reg_dss_tap[:, jr],
                gnn_tap_pu=reg_gnn_tap[:, jr],
                col_name=col,
                out_path=ppath,
                show_plots=show_plots,
            )
    elif n_reg_plot > 0 and reg_mean_d is None:
        print("[da_gps_daily] skip regulator tap series (missing reg_mean.pt / reg_std.pt in norm_dir).", flush=True)

    if meta_dss is not None and meta_gnn is not None and n_pv_plot > 0:
        meta_rows: dict[str, object] = {"step_idx": list(range(npts)), "hour": t_hours.astype(np.float64).tolist()}
        for jm in range(n_pv_plot):
            col = pv_aux_cols[jm]
            meta_rows[f"{col}__dss"] = meta_dss[:, jm].astype(np.float64).tolist()
            meta_rows[f"{col}__gnn_denorm"] = meta_gnn[:, jm].astype(np.float64).tolist()
            if meta_presolve_p is not None and _stem_from_pv_p_post_meta_col(col) is not None:
                meta_rows[f"{col}__presolve_p_sched_kw"] = meta_presolve_p[:, jm].astype(np.float64).tolist()
        meta_csv = out_dir / f"daily_meta_aux_{cfg_stem}.csv"
        pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
        aux_outputs["daily_meta_aux_csv"] = str(meta_csv)
        print(f"[da_gps_daily] wrote {meta_csv}", flush=True)
        for jm in range(n_pv_plot):
            col = pv_aux_cols[jm]
            cl = str(col).lower()
            if "kvar" in cl:
                ylab = "kvar"
            elif "kw" in cl or "_p_" in cl or "loss" in cl:
                ylab = "kW"
            else:
                ylab = "value (training units)"
            ppath = out_dir / f"daily_meta_aux_{cfg_stem}_{_safe_stem(col)}.png"
            pre_y = None
            if meta_presolve_p is not None and _stem_from_pv_p_post_meta_col(col) is not None:
                pre_y = meta_presolve_p[:, jm]
            _plot_meta_aux_scalar_daily_compare(
                t_hours=t_hours,
                dss_y=meta_dss[:, jm],
                gnn_y=meta_gnn[:, jm],
                col_name=col,
                y_label=ylab,
                out_path=ppath,
                show_plots=show_plots,
                presolve_y=pre_y,
            )
        bad_meta = [pv_aux_cols[jm] for jm in range(n_pv_plot) if bool(np.all(~np.isfinite(meta_dss[:, jm])))]
        if bad_meta:
            print(
                f"[da_gps_daily] WARNING: meta_aux columns had no finite OpenDSS mapping (check names vs circuit): {bad_meta}",
                flush=True,
            )
    elif int(bundle.get("n_pv_aux", 0) or 0) > 0 and n_pv_plot == 0:
        print(
            "[da_gps_daily] skip meta_aux series (missing pv_mean.pt / pv_std.pt or column list mismatch).",
            flush=True,
        )

    plot_list: list[str]
    if plot_all_cache_nodes:
        if plot_nodes:
            print(
                "[da_gps_daily] --plot-all-cache-nodes: ignoring explicit --plot-node list "
                f"({len(plot_nodes)} entries).",
                flush=True,
            )
        plot_candidates = sorted(
            {str(nk).strip().lower() for nk in node_order if str(nk).strip().lower() in node_to_idx}
        )
        n_tot = len(plot_candidates)
        plot_list = list(plot_candidates)
        print(
            f"[da_gps_daily] --plot-all-cache-nodes: {n_tot} bus.phase nodes (tensor-cache ∩ OpenDSS); "
            "PNG order = worst→best by per-node |V| MAE after the daily loop.",
            flush=True,
        )
        if int(plot_all_max_nodes) > 0:
            print(
                f"[da_gps_daily] --plot-all-max-nodes={int(plot_all_max_nodes)}: after that sort, "
                "only the first N (worst MAE) receive PNGs.",
                flush=True,
            )
        if len(plot_list) > 500 and int(plot_all_max_nodes) <= 0:
            print(
                "[da_gps_daily] NOTE: many |V|+angle PNGs (2-panel); they go under ``daily_voltage/`` "
                "(lower dpi by default). Cap/reg/meta PNGs stay in ``--out-dir``. "
                "Use ``--plot-all-max-nodes N`` to keep only the **worst-N** by |V| MAE after ranking, "
                "``--voltage-plot-dpi 72`` for smaller files, "
                "or omit ``--plot-all-cache-nodes`` and pass ``--plot-node`` for a subset.",
                flush=True,
            )
    else:
        plot_list = [str(x).strip().lower() for x in plot_nodes if str(x).strip()]
    volt_png_dir = _resolve_voltage_png_dir(
        out_dir,
        plot_all_cache_nodes=plot_all_cache_nodes,
        voltage_png_subdir=voltage_png_subdir,
    )
    vdpi = int(voltage_plot_dpi) if int(voltage_plot_dpi) > 0 else (96 if plot_all_cache_nodes else 160)
    vf_w = float(voltage_plot_fig_w) if float(voltage_plot_fig_w) > 0 else (7.5 if plot_all_cache_nodes else 10.0)
    vf_h = float(voltage_plot_fig_h) if float(voltage_plot_fig_h) > 0 else (3.2 if plot_all_cache_nodes else 4.2)
    vf_h_panel = vf_h
    vf_h_fig = vf_h * 2.0
    print(
        f"[da_gps_daily] daily |V| + angle (°) PNG directory: {volt_png_dir}  "
        f"(dpi={vdpi}, figsize=({vf_w:.3g} x {vf_h_fig:.3g}) in, 2×{vf_h_panel:.3g}-in panels)",
        flush=True,
    )

    def _per_node_voltage_maes(nk: str) -> tuple[float, float]:
        jj = node_to_idx.get(nk)
        if jj is None:
            return (float("nan"), float("nan"))
        m_v = np.isfinite(v_dss[:, jj]) & np.isfinite(v_gnn[:, jj])
        mae_v = (
            float(np.mean(np.abs(v_dss[m_v, jj] - v_gnn[m_v, jj]))) if m_v.any() else float("nan")
        )
        m_a = np.isfinite(va_dss[:, jj]) & np.isfinite(va_gnn[:, jj])
        if m_a.any():
            dd = va_dss[m_a, jj] - va_gnn[m_a, jj]
            mae_a = float(np.mean(np.abs(np.mod(dd.astype(np.float64) + 180.0, 360.0) - 180.0)))
        else:
            mae_a = float("nan")
        return (mae_v, mae_a)

    plot_rows: list[tuple[str, float, float]] = []
    for n in plot_list:
        if n not in node_to_idx:
            print(f"[da_gps_daily] skip plot: node {n!r} not in OpenDSS node list", flush=True)
            continue
        mv, ma = _per_node_voltage_maes(n)
        plot_rows.append((n, mv, ma))

    def _rank_sort_key(row: tuple[str, float, float]) -> tuple[int, float, str]:
        _nk, mv, _ma = row
        if np.isfinite(mv):
            return (0, -float(mv), _nk)
        return (1, 0.0, _nk)

    plot_rows.sort(key=_rank_sort_key)
    n_all_for_rank = len(plot_rows)
    if plot_all_cache_nodes and int(plot_all_max_nodes) > 0 and n_all_for_rank > int(plot_all_max_nodes):
        cap_m = int(plot_all_max_nodes)
        plot_rows = plot_rows[:cap_m]
        print(
            f"[da_gps_daily] --plot-all-max-nodes={cap_m}: wrote PNGs for worst {len(plot_rows)}/{n_all_for_rank} "
            "nodes by |V| MAE.",
            flush=True,
        )
    n_plot_ranked = len(plot_rows)
    rank_w = max(4, len(str(max(1, n_plot_ranked))))
    stem_safe = _safe_stem(cfg_stem)
    print(
        f"[da_gps_daily] voltage PNG filenames: rank 1..{n_plot_ranked} = worst→best by per-node |V| MAE (pu); "
        f"zero-padded rank prefix so folder sort matches rank.",
        flush=True,
    )
    volt_png_dir.mkdir(parents=True, exist_ok=True)

    for rank_idx, (n, n_mae, n_mae_ang) in enumerate(plot_rows):
        j = node_to_idx[n]
        m = np.isfinite(v_dss[:, j]) & np.isfinite(v_gnn[:, j])
        m_ang = np.isfinite(va_dss[:, j]) & np.isfinite(va_gnn[:, j])

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(vf_w, vf_h_fig))
        ax0.plot(t_hours, v_dss[:, j], linewidth=1.4, label="OpenDSS")
        ax0.plot(
            t_hours,
            v_gnn[:, j],
            "--",
            linewidth=1.2,
            label=f"DA-GPS ({cfg_stem}) MAE={n_mae:.4f} pu",
        )
        ax0.set_ylabel("|V| (pu)")
        r_human = rank_idx + 1
        ax0.set_title(f"24h @ {n}  |  |V| rank {r_human}/{n_plot_ranked} (worst→best by MAE)")
        if v_ylim_fixed:
            ax0.set_ylim(ymin, ymax)
        else:
            ys = np.concatenate([v_dss[m, j], v_gnn[m, j]]).astype(np.float64, copy=False)
            ys = ys[np.isfinite(ys)]
            if ys.size > 0:
                lo, hi = float(np.min(ys)), float(np.max(ys))
                span = hi - lo
                mag = max(abs(lo), abs(hi), 0.5)
                if not np.isfinite(span) or span <= 1e-9 * max(mag, 1.0):
                    bump = max(0.004, 0.02 * mag)
                    ax0.set_ylim(lo - bump, hi + bump)
                else:
                    span_e = max(span, 1e-9)
                    pad = max(0.003, span_e * 0.14)
                    pad += 0.04 * span_e + 0.01 * mag
                    ax0.set_ylim(lo - pad, hi + pad)
            else:
                ax0.set_ylim(ymin, ymax)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best")

        ax1.plot(t_hours, va_dss[:, j], linewidth=1.4, label="OpenDSS")
        lab_ang = f"DA-GPS ({cfg_stem}) MAE={n_mae_ang:.3f}°" if np.isfinite(n_mae_ang) else f"DA-GPS ({cfg_stem})"
        ax1.plot(t_hours, va_gnn[:, j], "--", linewidth=1.2, label=lab_ang)
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("V angle (deg)")
        if m_ang.any():
            ys_a = np.concatenate([va_dss[m_ang, j], va_gnn[m_ang, j]]).astype(np.float64, copy=False)
            ys_a = ys_a[np.isfinite(ys_a)]
            if ys_a.size > 0:
                lo_a, hi_a = float(np.min(ys_a)), float(np.max(ys_a))
                span_a = hi_a - lo_a
                mag_a = max(abs(lo_a), abs(hi_a), 10.0)
                if not np.isfinite(span_a) or span_a <= 1e-9 * max(mag_a, 1.0):
                    bump_a = max(0.5, 0.02 * mag_a)
                    ax1.set_ylim(lo_a - bump_a, hi_a + bump_a)
                else:
                    span_ea = max(span_a, 1e-9)
                    pad_a = max(0.5, span_ea * 0.14)
                    pad_a += 0.04 * span_ea + 0.01 * mag_a
                    ax1.set_ylim(lo_a - pad_a, hi_a + pad_a)
            else:
                ax1.set_ylim(-180.0, 180.0)
        else:
            ax1.set_ylim(-180.0, 180.0)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        plt.tight_layout()
        node_fn = n.replace(".", "_")
        rk = str(r_human).zfill(rank_w)
        tot = str(n_plot_ranked).zfill(rank_w)
        png_path = volt_png_dir / _voltage_daily_png_basename(
            stem_safe=stem_safe,
            rk=rk,
            tot=tot,
            n_mae=n_mae,
            n_mae_ang=n_mae_ang,
            node_fn=node_fn,
        )
        plt.savefig(_path_str_for_png_write(png_path), dpi=int(vdpi))
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "cache_pt": str(Path(cache_pt).resolve()),
        "edge_csv": str(edge_path),
        "daily_profile_csv": str(prof_resolved),
        "load_profile_path_arg": str(load_profile_path or ""),
        "load_profile_filename": str(load_profile_filename),
        "pv_irradiance_profile_csv": str(irr_csv.resolve()),
        "pv_irradiance_profile_path_arg": str(pv_irradiance_profile_path or ""),
        "pv_irradiance_filename": str(pv_irradiance_filename),
        "der": {
            "enabled_features": bool(want_der),
            "der_profile_csv": str(der_csv_resolved) if der_csv_resolved is not None else None,
            "der_profile_path_arg": str(der_profile_path or ""),
            "der_profile_filename": str(der_profile_filename),
            "der_max_kw": float(der_max_kw),
            "der_buses": list(der_effective_buses),
            "der_bus_phases_in_cache": {str(k): [int(p) for p in v] for k, v in der_bus_phases.items()},
            "der_q_frac_p": float(der_q_frac_p),
            "dss_generators_by_bus": {str(k): str(v) for k, v in der_gen_by_bus.items()},
            "gnn_der_feature_mode": (
                "into_bess_columns"
                if want_der and der_use_bess_columns
                else ("add_to_p_load_q_load_at_injection_buses" if want_der else "off")
            ),
        },
        "npts": npts,
        "step_min": step_min,
        "daily_stress": daily_stress,
        "scenario_scale": scenario_scale,
        "stress_clip": [stress_clip_lo, stress_clip_hi],
        "ref_sample_index": ref_i,
        "mae_global": mae,
        "rmse_global": rmse,
        "r2_global_vmag_pu": r2_vmag,
        "n_points_vmag_finite_overlap": n_v_overlap,
        "daily_voltage_regression_pu": {
            "reference": "OpenDSS |V| (pu)",
            "predicted": "DA-GPS |V| (pu), denormalized voltage head magnitude",
            "mae": mae,
            "rmse": rmse,
            "r2": r2_vmag,
            "n_points": n_v_overlap,
        },
        "mae_angle_global_deg_circular": mae_ang_deg,
        "r2_global_vang_deg_naive": r2_vang,
        "n_points_vang_finite_overlap": n_ang_overlap,
        "daily_voltage_angle_regression_deg": {
            "reference": "OpenDSS voltage angle (deg)",
            "predicted": "DA-GPS angle from denormalized complex head (deg)",
            "mae_circular_deg": mae_ang_deg,
            "r2_naive": r2_vang,
            "n_points": n_ang_overlap,
            "r2_note": "R² uses circular residual for SS_res and linear DSS variance for SS_tot; interpret cautiously near ±180° wraps.",
        },
        "n_nonconv": n_nonconv,
        "n_ok": n_ok,
        "node_feature_cols": node_feature_cols,
        "timing_s_totals": {
            "dss_apply": open_apply_s_total,
            "dss_solve_only": open_solve_only_s_total,
            "dss_collect_v": open_get_s_total,
            "dss_reassert": open_reassert_s_total,
            "gnn_setup_once": gnn_setup_once_s,
            "gnn_feature_generation": feature_build_s_total,
            "gnn_forward_only": gnn_forward_only_s_total,
            "gnn_bucket_including_prep": gnn_infer_s_total,
            "gnn_per_step_amortized": gnn_per_step_s,
            "gnn_total_wall": gnn_total_wall_s,
            "da_gps_deployment_wall_line": (
                f"{gnn_setup_once_s:.6g}s + {n_ok} × {gnn_per_step_s:.6g}s = {gnn_total_wall_s:.6g}s"
            ),
            "opendss_solve_wall_line": (
                f"{n_ok} × {open_solve_only_s_total / max(n_ok, 1):.6g}s = {open_solve_only_s_total:.6g}s"
            ),
        },
        "timing_ms_per_ok_step": {
            "dss_apply_ms": dss_apply_ms,
            "dss_solve_ms": dss_solve_ms,
            "dss_collect_ms": dss_collect_ms,
            "dss_apply_plus_solve_ms": pipeline_metrics["dss_true_ms"],
            "dss_apply_plus_solve_plus_collect_ms": dss_pipeline_ms,
            "gnn_feature_generation_ms": gnn_feature_gen_ms,
            "gnn_forward_ms": gnn_forward_ms,
            "gnn_feature_plus_forward_ms": gnn_pipeline_ms,
        },
        "speedup": {
            "true_speedup": pipeline_metrics["true_speedup"],
            "true_speedup_basis": "dss_apply_plus_solve_plus_collect_ms / gnn_feature_plus_forward_ms",
            "full_dss_speedup": pipeline_metrics["full_dss_speedup"],
            "full_dss_speedup_basis": "dss_apply_plus_solve_plus_collect_ms / gnn_feature_plus_forward_ms",
            "apply_solve_speedup": pipeline_metrics["apply_solve_speedup"],
            "apply_solve_speedup_basis": "dss_apply_plus_solve_ms / gnn_feature_plus_forward_ms",
            "deploy_speedup": pipeline_metrics["deploy_speedup"],
            "deploy_speedup_basis": "dss_apply_plus_solve_ms / (dss_apply_ms + gnn_feature_plus_forward_ms)",
            "net_speedup": pipeline_metrics["net_speedup"],
            "net_speedup_basis": "dss_solve_ms / gnn_feature_plus_forward_ms",
        },
        "cap_reg_aux_outputs": aux_outputs,
        "voltage_plot_mode": "all_cache_nodes" if plot_all_cache_nodes else "explicit_plot_node",
        "n_voltage_profile_pngs": int(n_plot_ranked),
        "voltage_png_dir": str(volt_png_dir.resolve()),
        "voltage_plot_dpi": int(vdpi),
        "voltage_plot_figsize_in": [float(vf_w), float(vf_h_fig)],
        "voltage_plot_panel_height_in": float(vf_h_panel),
        "voltage_png_layout": "|V|_pu_and_angle_deg_stacked",
        "voltage_png_filename_rank_sort": "mae_pu_desc_worst_first",
        "voltage_png_filename_pattern": "v_<ckpt_stem_up_to_24>_rNNNN_oMMMM_pu<4f>_ang<3f>_<bus_phase>.png (short basename for Windows MAX_PATH; extended-length path prefix if len>=220)",
        "voltage_ylim": "fixed" if v_ylim_fixed else "auto_from_data",
        "voltage_ylim_fixed_bounds_pu": [float(ymin), float(ymax)] if v_ylim_fixed else None,
        "opendss_compile": {
            "feeder": feeder_key,
            "master_dss": str(compile_meta["master_dss"]),
            "model_dir": str(compile_meta["model_dir"]),
            "pv_irradiance_profile_csv": str(Path(irr_csv).resolve()),
        },
    }
    (out_dir / "da_gps_daily_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved under {out_dir}", flush=True)
    return None


def main() -> None:
    print("[da_gps_daily] starting...", flush=True)
    p = argparse.ArgumentParser(description="DA-GPS daily OpenDSS |V|+angle compare + timing (GINE multitask checkpoint).")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training OUT_DIR or snapshot folder: x_mean.pt + training_last.pt and/or da_gps_report.json / da_gps_run_manifest.json",
    )
    p.add_argument("--cache-pt", type=str, required=True, help="Chunk tensor cache .pt (x + node_to_local)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Weights: da_gps_multitask_best.pt or training_last.pt (best_model_state_dict). "
        "Default: run-dir/da_gps_multitask_best.pt. For training_last.pt, da_gps_multitask_best.pt "
        "must exist alongside (or parent) for n_cap/n_reg metadata.",
    )
    p.add_argument("--edge-csv", type=str, default="", help="Override compacted edge CSV (from_node,to_node,R_full,X_full)")
    p.add_argument("--out-dir", type=str, default="", help="Output folder (default: run-dir/da_gps_daily_compare_<tag>)")
    p.add_argument("--daily-profile", type=str, default="", help="Profile CSV basename or path (see run_daily_aggregate_dataset_8500)")
    p.add_argument(
        "--load-profile-path",
        type=str,
        default="",
        help="Daily **load** multiplier CSV (two columns) or a directory containing --load-profile-filename (default 5minDayShape.csv). "
        "Empty: resolve from --daily-profile / repo default.",
    )
    p.add_argument(
        "--load-profile-filename",
        type=str,
        default="5minDayShape.csv",
        help="When --load-profile-path is a directory, read this file inside it.",
    )
    p.add_argument(
        "--pv-irradiance-profile-path",
        type=str,
        default="",
        help="PV irradiance multiplier CSV or directory containing --pv-irradiance-filename (default irr_day_001.csv). "
        "Empty: use solar-unbalanced model default.",
    )
    p.add_argument(
        "--pv-irradiance-filename",
        type=str,
        default="irr_day_001.csv",
        help="Filename inside directory when --pv-irradiance-profile-path is a directory.",
    )
    p.add_argument(
        "--der-profile-path",
        type=str,
        default="",
        help="DER schedule CSV (col 2 multiplier) or directory with --der-profile-filename. Required when --der-max-kw>0 and --der-buses set.",
    )
    p.add_argument("--der-profile-filename", type=str, default="der_5min.csv")
    p.add_argument(
        "--der-max-kw",
        type=float,
        default=0.0,
        help="Total DER active power cap (kW); per step P_total = m_der[i]×this, split across buses and OpenDSS 3φ generators.",
    )
    p.add_argument(
        "--der-buses",
        type=str,
        default="",
        help="Comma/space-separated bus names (no phase suffix). Each bus must appear as at least one "
        "``bus.phase`` row in the tensor-cache ``node_to_local``; that bus's P/Q is split across **cached** "
        "phases only (1, 2, or 3 of them). OpenDSS still uses a 3φ ``Generator`` at the bus.",
    )
    p.add_argument(
        "--der-q-frac-p",
        type=float,
        default=0.1,
        help="OpenDSS generator kvar per bus = this × kW per bus (after P split). "
        "BESS checkpoints: same ratio on phased ``p_bess_kw`` / ``q_bess_kvar``. "
        "Non-BESS: same ratio added with P into ``p_load_kw`` / ``q_load_kvar``.",
    )
    p.add_argument(
        "--plot-node",
        action="append",
        default=None,
        help="Repeat for each node to plot (bus.phase, lower case). Example: --plot-node l2673319.1",
    )
    p.add_argument(
        "--plot-all-cache-nodes",
        action="store_true",
        help="Emit one daily |V|+angle PNG per tensor-cache bus.phase that exists in the compiled OpenDSS node list. "
        "Filenames are sorted worst→best by per-node |V| MAE (``rank####_of####`` prefix). "
        "When set, explicit --plot-node arguments are ignored.",
    )
    p.add_argument(
        "--plot-all-max-nodes",
        type=int,
        default=0,
        help="With --plot-all-cache-nodes, after ranking all nodes by |V| MAE (worst first), emit at most this many PNGs (0 = no cap; plot every node).",
    )
    p.add_argument(
        "--voltage-png-subdir",
        type=str,
        default="",
        help="Subfolder under --out-dir for daily |V| PNGs (created if needed). "
        "Default: with --plot-all-cache-nodes use ``daily_voltage``; otherwise write in out_dir root. "
        "Pass ``.`` to force out_dir root even when plotting all cache nodes.",
    )
    p.add_argument(
        "--voltage-plot-dpi",
        type=int,
        default=0,
        help="DPI for daily |V| PNGs only. 0=auto (96 with --plot-all-cache-nodes, else 160). Use 72–96 for smaller files.",
    )
    p.add_argument(
        "--voltage-plot-fig-w",
        type=float,
        default=0.0,
        help="|V| figure width (inches). 0=auto (7.5 with --plot-all-cache-nodes, else 10).",
    )
    p.add_argument(
        "--voltage-plot-fig-h",
        type=float,
        default=0.0,
        help="|V| figure height (inches). 0=auto (3.2 with --plot-all-cache-nodes, else 4.2).",
    )
    p.add_argument("--npts", type=int, default=288)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument(
        "--daily-stress",
        type=float,
        default=0.0,
        help="Amplify profile deviations from 1.0: m_eff=clip(1+(m-1)*(1+stress),...). 0=nominal shape.",
    )
    p.add_argument(
        "--scenario-scale",
        type=float,
        default=1.0,
        help="Uniform multiplier on top of stressed per-step profile (extra OOD lever).",
    )
    p.add_argument("--stress-clip-lo", type=float, default=0.1)
    p.add_argument("--stress-clip-hi", type=float, default=3.0)
    p.add_argument("--ref-sample-index", type=int, default=0, help="Which row of cache x to use for static feature tails")
    p.add_argument("--mv-sx-mapping", type=str, default="", help="Optional mv_x_sx_node_mapping_8500.csv")
    p.add_argument("--ymin", type=float, default=0.92, help="|V| plot lower bound when using --v-ylim-fixed (or fallback when no finite data).")
    p.add_argument("--ymax", type=float, default=1.08, help="|V| plot upper bound when using --v-ylim-fixed (or fallback when no finite data).")
    p.add_argument(
        "--v-ylim-fixed",
        action="store_true",
        help="Pin daily |V| PNG y-axis to --ymin/--ymax. Default: auto y-limits per plot from OpenDSS + DA-GPS curves with padding so both lines stay visible.",
    )
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--feeder",
        type=str,
        default="8500",
        help="OpenDSS master: 8500 (solar-unbalanced PV), ieee34 (Mirzaei IEEE34_PV.dss), or 906 (LVTestCase).",
    )
    p.add_argument(
        "--meta-debug",
        action="store_true",
        help="Log meta-aux: DSS row, GNN pred in normalized + denormalized space, clock, m_eff/m_irr, x_n means "
        "at a few timesteps. Or set env GNN_DAILY_META_DEBUG=1.",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (run_dir / f"da_gps_daily_compare_{tag}")

    prof = str(args.daily_profile).strip() or None
    load_pp = str(args.load_profile_path).strip() or None
    pv_irr_pp = str(args.pv_irradiance_profile_path).strip() or None
    der_pp = str(args.der_profile_path).strip() or None
    ck = Path(args.checkpoint).resolve() if str(args.checkpoint).strip() else None
    _es = str(args.edge_csv).strip()
    ec = None if not _es or _es in (".", "..") else Path(_es).resolve()
    mv = Path(args.mv_sx_mapping).resolve() if str(args.mv_sx_mapping).strip() else None

    meta_dbg = bool(args.meta_debug)
    if str(os.environ.get("GNN_DAILY_META_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        meta_dbg = True
    if meta_dbg:
        print("[da_gps_daily] meta-debug enabled (--meta-debug or GNN_DAILY_META_DEBUG=1)", flush=True)

    run(
        run_dir=run_dir,
        cache_pt=Path(args.cache_pt),
        out_dir=out_dir,
        checkpoint=ck,
        edge_csv=ec,
        daily_profile_csv=prof,
        load_profile_path=load_pp,
        load_profile_filename=str(args.load_profile_filename),
        pv_irradiance_profile_path=pv_irr_pp,
        pv_irradiance_filename=str(args.pv_irradiance_filename),
        der_profile_path=der_pp,
        der_profile_filename=str(args.der_profile_filename),
        der_max_kw=float(args.der_max_kw),
        der_buses=str(args.der_buses),
        der_q_frac_p=float(args.der_q_frac_p),
        plot_nodes=list(args.plot_node) if args.plot_node else [],
        plot_all_cache_nodes=bool(args.plot_all_cache_nodes),
        plot_all_max_nodes=int(args.plot_all_max_nodes),
        voltage_png_subdir=str(args.voltage_png_subdir),
        voltage_plot_dpi=int(args.voltage_plot_dpi),
        voltage_plot_fig_w=float(args.voltage_plot_fig_w),
        voltage_plot_fig_h=float(args.voltage_plot_fig_h),
        npts=int(args.npts),
        step_min=int(args.step_min),
        daily_stress=float(args.daily_stress),
        scenario_scale=float(args.scenario_scale),
        stress_clip_lo=float(args.stress_clip_lo),
        stress_clip_hi=float(args.stress_clip_hi),
        ref_sample_index=int(args.ref_sample_index),
        mv_sx_mapping=mv,
        ymin=float(args.ymin),
        ymax=float(args.ymax),
        v_ylim_fixed=bool(args.v_ylim_fixed),
        show_plots=bool(args.show_plots),
        device=args.device,
        meta_debug=meta_dbg,
        feeder=str(args.feeder),
    )


if __name__ == "__main__":
    import sys
    import traceback

    _err_log = Path(__file__).resolve().parent / "da_gps_daily_last_error.log"
    try:
        main()
    except BrokenPipeError:
        raise SystemExit(0) from None
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            _err_log.write_text(tb, encoding="utf-8")
            print(f"\n[da_gps_daily] Traceback saved to: {_err_log}", flush=True)
        except OSError as w:
            print(f"\n[da_gps_daily] Could not write {_err_log}: {w}", flush=True)
        try:
            sys.stderr.write(tb)
        except OSError:
            pass
        raise SystemExit(1) from None
