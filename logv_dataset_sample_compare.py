"""Replay a GNN dataset sample on live OpenDSS + Log(v), score vs saved labels.

Mirrors the corrected IEEE-34 profile-cell path:
  exact node-feature P/Q, approximate ZIP shares (when present), meta taps/caps,
  refresh Log(v) loads + create_load_model, then synced/static/off Log(v) solve.

Primary accuracy is vs dataset OpenDSS labels (cache ``y_ri`` / node targets).
Live OpenDSS is reported as a secondary diagnostic (ZIP IDs are approximate).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


CHUNK_ROOTS = {
    "ieee34": (
        Path(r"K:\My Drive\datasets_gnn2\original_ieee34_mirzaei_chunked"),
        Path("/content/drive/MyDrive/datasets_gnn2/original_ieee34_mirzaei_chunked"),
        Path("datasets_gnn2") / "original_ieee34_mirzaei_chunked",
    ),
    "906": (
        # Prefer PV Volt-Var fixedctrlinit chunk; fall back to stock no-PV.
        Path(r"K:\My Drive\datasets_gnn2\original_906_lvtestcase_pv_voltvar_fixedctrlinit"),
        Path("/content/drive/MyDrive/datasets_gnn2/original_906_lvtestcase_pv_voltvar_fixedctrlinit"),
        Path("datasets_gnn2") / "original_906_lvtestcase_pv_voltvar_fixedctrlinit",
        Path(r"K:\My Drive\datasets_gnn2\original_906_lvtestcase_chunked"),
        Path("/content/drive/MyDrive/datasets_gnn2/original_906_lvtestcase_chunked"),
        Path("datasets_gnn2") / "original_906_lvtestcase_chunked",
    ),
    "8500": (
        Path(r"K:\My Drive\datasets_gnn2\original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40"),
        Path(r"K:\My Drive\datasets_gnn2\original_8500_unbalanced_chunked"),
        Path("/content/drive/MyDrive/datasets_gnn2/original_8500_unbalanced_chunked_no_bess_new_diverse_2000_40"),
        Path("datasets_gnn2") / "original_8500_unbalanced_chunked",
    ),
}

FEEDER_REGS = {
    "ieee34": ("rega1", "rega2", "rega3", "regb1", "regb2", "regb3"),
    "906": (),  # use meta / RegControls.AllNames discovery
    "8500": (),
}
FEEDER_CAPS = {
    "ieee34": ("c844", "c848"),
    "906": (),
    "8500": (),
}


@dataclass
class DatasetReplayResult:
    feeder: str
    control_mode: str
    sample_id: int
    row_i: int
    logv_label: str
    n_nodes: int
    mae_logv_vs_dataset: float
    rmse_logv_vs_dataset: float
    mae_live_od_vs_dataset: float
    mae_logv_vs_live_od: float
    bias_logv_minus_dataset: float
    frac_over_vs_dataset: float
    frac_under_vs_dataset: float
    p_load_kw: float
    q_load_kvar: float
    forced_taps: dict[str, float]
    n_caps_set: int
    n_remap_warning: str
    node_mae: pd.DataFrame | None = None
    # Solve-only wall times (compile / feature IO excluded).
    # Fair paper-style wall-clock baseline = OpenDSS cold InitSnap+Solve.
    t_opendss_solve_s: float = float("nan")  # warm Solve() used for metrics
    t_opendss_cold_s: float = float("nan")  # InitSnap+Solve (fair OD baseline)
    t_logv_setup_s: float = float("nan")  # load refresh / A rebuild (amortized)
    t_logv_solve_s: float = float("nan")  # online linear PF only (fair Log(v))


def resolve_chunk_dir(repo: Path, cache_pt: Path, feeder: str) -> Path:
    chunk_name = cache_pt.stem.split("__", 1)[0]
    roots = list(CHUNK_ROOTS.get(feeder, ()))
    roots.append(repo / "datasets_gnn2")
    for root in roots:
        root = Path(root)
        if not root.is_absolute():
            root = repo / root
        cand = root / chunk_name
        if cand.is_dir() and (cand / "gnn_node_index_master.csv").is_file():
            return cand
        # some layouts store chunk folders directly under root with matching prefix
        if root.is_dir():
            hits = sorted(root.glob(chunk_name)) + sorted(root.glob(f"{chunk_name}*"))
            for h in hits:
                if h.is_dir() and (h / "gnn_node_index_master.csv").is_file():
                    return h
    raise FileNotFoundError(
        f"Chunk dir for cache stem {chunk_name!r} (feeder={feeder}) not found"
    )


def _flatten_case_vm(case, which: str) -> dict[str, float]:
    vm_map: dict[str, float] = {}
    res = (case.results.get(which) or {}).get("vm") or {}
    phases = case.bus_phases or {}
    for bus, phs in phases.items():
        arr = res.get(bus)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).ravel()
        for i, ph in enumerate(list(phs)):
            if i >= arr.size or not np.isfinite(arr[i]):
                continue
            vm_map[f"{str(bus).strip().lower()}.{int(ph)}"] = float(arr[i])
    return vm_map


def _load_mv_sx_rules(repo: Path | None = None) -> list[dict[str, str]]:
    """IEEE 8500 MV↔SX split-phase rules for reversing mvagg P/Q onto DSS secondary buses."""
    cands: list[Path] = []
    if repo is not None:
        r = Path(repo)
        cands.extend(
            [
                r / "8500-node" / "mv_x_sx_node_mapping_8500.csv",
                r / "8500 nodes with solar unbalanced" / "mv_x_sx_node_mapping_8500.csv",
            ]
        )
    cands.append(Path("8500-node") / "mv_x_sx_node_mapping_8500.csv")
    path = next((p for p in cands if p.is_file()), None)
    if path is None:
        return []
    rules: list[dict[str, str]] = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        mv = str(row.get("mv_node") or "").strip().lower()
        lv1 = str(row.get("lv_x_node_1") or "").strip().lower()
        lv2 = str(row.get("lv_x_node_2") or "").strip().lower()
        sx1 = str(row.get("sx_node_1") or "").strip().lower()
        sx2 = str(row.get("sx_node_2") or "").strip().lower()
        if not mv or not lv1 or not lv2:
            continue
        la, lb = (sx1, sx2) if sx1 and sx2 else (lv1, lv2)
        rules.append({"mv_key": mv, "load_a": la, "load_b": lb})
    return rules


def _expand_mvagg_pq_to_sx(
    feat_pq: dict[str, tuple[float, float, float]],
    rules: list[dict[str, str]],
) -> dict[str, tuple[float, float, float]]:
    """Split each MV-aggregated (P,Q) equally onto the two SX/LV leaf nodes."""
    if not rules:
        return feat_pq
    out = dict(feat_pq)
    n_exp = 0
    for rec in rules:
        mv = rec["mv_key"]
        if mv not in feat_pq:
            continue
        p, q, pv = feat_pq[mv]
        if abs(p) < 1e-15 and abs(q) < 1e-15 and abs(pv) < 1e-15:
            continue
        # Dataset aggregation is sum(load_a, load_b); reverse with equal split.
        # PV on MV is uncommon for 8500 loads; keep on MV key only.
        ha, hb = 0.5 * p, 0.5 * q
        for leaf in (rec["load_a"], rec["load_b"]):
            op, oq, opv = out.get(leaf, (0.0, 0.0, 0.0))
            out[leaf] = (op + ha, oq + hb, opv)
        n_exp += 1
    if n_exp:
        print(
            f"[logv_dataset_replay] expanded mvagg→SX/LV for {n_exp} MV nodes "
            f"(equal split onto leaf pair)",
            flush=True,
        )
    return out


def _load_feature_pq(
    chunk_dir: Path,
    sample_id: int,
    *,
    repo: Path | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Load per-node P/Q(/PV) for DSS injection.

    Prefer the full ``gnn_node_features_and_targets.csv`` when present: on IEEE 8500
    that file keeps secondary ``sx*`` load buses that match OpenDSS. The training
    ``*_mvagg.csv`` collapses those onto MV primaries and cannot be applied via a
    live load→bus map without reverse expansion.
    """
    full_csv = chunk_dir / "gnn_node_features_and_targets.csv"
    mvagg_csv = chunk_dir / "gnn_node_features_and_targets_mvagg.csv"
    if full_csv.is_file():
        nodes_csv = full_csv
        used_mvagg = False
    elif mvagg_csv.is_file():
        nodes_csv = mvagg_csv
        used_mvagg = True
    else:
        raise FileNotFoundError(
            f"No node-feature CSV in {chunk_dir} "
            "(expected gnn_node_features_and_targets.csv or *_mvagg.csv)"
        )

    df = pd.read_csv(nodes_csv)
    df = df.loc[df["sample_id"].astype(int) == int(sample_id)].copy()
    if df.empty:
        raise KeyError(f"sample_id={sample_id} missing from {nodes_csv}")
    df["node"] = df["node"].astype(str).str.strip().str.lower()
    out: dict[str, tuple[float, float, float]] = {}
    for _, r in df.iterrows():
        out[str(r["node"])] = (
            float(r["p_load_kw"]),
            float(r["q_load_kvar"]),
            float(r["p_pv_kw"]) if "p_pv_kw" in df.columns else 0.0,
        )
    print(
        f"[logv_dataset_replay] feature P/Q from {nodes_csv.name}  "
        f"n_nodes={len(out)}  P_sum={sum(p for p,_,_ in out.values()):.1f} kW",
        flush=True,
    )
    if used_mvagg:
        # Only needed when raw SX rows were deleted after aggregation.
        out = _expand_mvagg_pq_to_sx(out, _load_mv_sx_rules(repo))
    return out


def _dataset_mag_from_cache(
    pack: dict[str, Any],
    row_i: int,
    node_order: list[str],
) -> np.ndarray:
    y_ri = pack["y_ri"].float()
    ri = y_ri[int(row_i)]
    return torch.sqrt(ri[:, 0] ** 2 + ri[:, 1] ** 2 + 1e-12).numpy()


def _force_taps_from_meta(row: pd.Series, reg_names: tuple[str, ...] | list[str]) -> dict[str, float]:
    import opendssdirect as dss

    # Prefer explicit list; else discover from meta columns + live RegControls
    names = list(reg_names)
    if not names:
        names = []
        for c in row.index:
            cs = str(c)
            if cs.startswith("reg_") and cs.endswith("_tap_pu"):
                # reg_<name>_tap_pu
                mid = cs[len("reg_") : -len("_tap_pu")]
                if mid and not mid.startswith("feeder_"):
                    names.append(mid)
        try:
            names = sorted(set(names) | set(dss.RegControls.AllNames() or []))
        except Exception:
            names = sorted(set(names))

    out: dict[str, float] = {}
    for nm in names:
        key = f"reg_{nm}_tap_pu"
        if key not in row or not np.isfinite(float(row[key])):
            continue
        tap = float(row[key])
        out[str(nm)] = tap
        try:
            dss.RegControls.Name(nm)
            dss.RegControls.TapNumber(int(round((tap - 1.0) / 0.00625)))
        except Exception:
            pass
        try:
            dss.Transformers.Name(nm)
            nwind = int(dss.Transformers.NumWindings())
            dss.Transformers.Wdg(min(2, nwind))
            dss.Transformers.Tap(tap)
        except Exception:
            pass
    return out


def _force_caps_from_meta(row: pd.Series, cap_names: tuple[str, ...] | list[str]) -> int:
    import opendssdirect as dss

    names = list(cap_names)
    if not names:
        for c in row.index:
            cs = str(c)
            if cs.startswith("cap_") and cs.endswith("_q_post_kvar"):
                names.append(cs[len("cap_") : -len("_q_post_kvar")])
        try:
            names = sorted(set(names) | set(n for n in (dss.Capacitors.AllNames() or []) if n and n != "NONE"))
        except Exception:
            names = sorted(set(names))

    n = 0
    for nm in names:
        qkey = f"cap_{nm}_q_post_kvar"
        onkey = f"cap_{nm}_n_steps_on"
        if qkey not in row and onkey not in row:
            continue
        on = True
        if onkey in row and np.isfinite(float(row[onkey])):
            on = float(row[onkey]) > 0.5
        elif qkey in row:
            on = abs(float(row[qkey])) > 1e-6
        try:
            dss.Capacitors.Name(nm)
            dss.Capacitors.States([1 if on else 0])
            n += 1
        except Exception:
            pass
    return n


def _claims_from_ieee34_device_maps(
    dev_to_dss_load: dict,
    dev_to_busph_load: dict,
) -> dict[tuple[str, int], list[tuple[str, float]]]:
    claims: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for dev, busphs in dev_to_busph_load.items():
        ln = dev_to_dss_load.get(dev)
        if ln is None:
            continue
        for bus, ph, w in busphs:
            claims.setdefault((str(bus).lower(), int(ph)), []).append((str(ln), float(w)))
    return claims


def _claims_from_live_dss_loads() -> dict[tuple[str, int], list[tuple[str, float]]]:
    """Generic map: each OpenDSS Load → its connected bus.phases (equal weight)."""
    import opendssdirect as dss

    claims: dict[tuple[str, int], list[tuple[str, float]]] = {}
    try:
        names = [n for n in dss.Loads.AllNames() if n and n != "NONE"]
    except Exception:
        names = []
    for ln in names:
        try:
            dss.Loads.Name(ln)
            busstr = str(dss.CktElement.BusNames()[0]).split(".")
        except Exception:
            continue
        bus = str(busstr[0]).strip().lower()
        if len(busstr) <= 1:
            phases = [1, 2, 3]
        else:
            phases = [int(x) for x in busstr[1:] if str(x).isdigit() and int(x) != 0] or [1, 2, 3]
        for ph in phases:
            claims.setdefault((bus, int(ph)), []).append((str(ln), 1.0))
    return claims


def _apply_feature_loads_and_zip(
    *,
    feat_pq: dict[str, tuple[float, float, float]],
    dev_to_dss_load: dict,
    dev_to_busph_load: dict,
    row: pd.Series,
) -> dict[str, float]:
    import opendssdirect as dss

    # Prefer ieee34 DEVICE_* maps when they actually match the compiled DSS.
    # Otherwise (906/8500) build claims from live DSS load bus connections.
    claims = _claims_from_ieee34_device_maps(dev_to_dss_load, dev_to_busph_load)
    n_mapped = len({ln for owners in claims.values() for ln, _ in owners})
    if n_mapped == 0:
        claims = _claims_from_live_dss_loads()
        n_mapped = len({ln for owners in claims.values() for ln, _ in owners})
        print(f"[logv_dataset_replay] using live-DSS load map  n_loads={n_mapped}")
    else:
        print(f"[logv_dataset_replay] using DEVICE_P_SHARE load map  n_loads={n_mapped}")

    load_pq: dict[str, list[float]] = {}
    for (bus, ph), owners in claims.items():
        p, q, _pv = feat_pq.get(f"{bus}.{int(ph)}", (0.0, 0.0, 0.0))
        wsum = sum(max(w, 0.0) for _, w in owners) + 1e-12
        for ln, w in owners:
            frac = max(w, 0.0) / wsum
            cur = load_pq.setdefault(ln, [0.0, 0.0])
            cur[0] += p * frac
            cur[1] += q * frac

    for ln, (p, q) in load_pq.items():
        dss.Loads.Name(ln)
        dss.Loads.kW(float(p))
        dss.Loads.kvar(float(q))

    feat_p_sum = float(sum(p for p, _, _ in feat_pq.values()))
    applied_p = float(sum(p for p, _ in load_pq.values()))
    if feat_p_sum > 1.0 and applied_p < 0.01 * feat_p_sum:
        print(
            f"[logv_dataset_replay] WARNING: applied P_load={applied_p:.1f} kW "
            f"<< feature P_sum={feat_p_sum:.1f} kW — bus-name mismatch "
            f"(e.g. mvagg MV keys vs DSS sx* loads)?",
            flush=True,
        )

    targets = {
        1: float(row.get("share_m1_p", 0.0) or 0.0),
        2: float(row.get("share_m2_p", 0.0) or 0.0),
        4: float(row.get("share_m4_p", 0.0) or 0.0),
        5: float(row.get("share_m5_p", 0.0) or 0.0),
    }
    if sum(targets.values()) > 1e-8:
        items = sorted(load_pq.items(), key=lambda kv: kv[1][0], reverse=True)
        tot = sum(max(p, 0.0) for _, (p, _) in items) + 1e-12
        assigned = {1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}
        for ln, (p, _) in items:
            deficits = {m: targets[m] * tot - assigned[m] for m in targets}
            m = max(deficits, key=deficits.get)
            dss.Loads.Name(ln)
            dss.Loads.Model(int(m))
            assigned[m] += max(p, 0.0)

    # PV: zero when irradiance multiplier is ~0; else use meta post-solve columns when present
    try:
        pv_names = [n for n in dss.PVsystems.AllNames() if n and n != "NONE"]
    except Exception:
        pv_names = []
    m_irr = float(row.get("m_irradshape", 0.0) or 0.0)
    for pv in pv_names:
        pkey = f"pv_{pv}_p_post_kw"
        # also try ieee34 naming pv_pv850_...
        alts = [pkey, f"pv_pv{pv}_p_post_kw", f"pv_{pv.lower()}_p_post_kw"]
        p_post = None
        for k in alts:
            if k in row and np.isfinite(float(row[k])):
                p_post = float(row[k])
                break
        try:
            dss.PVsystems.Name(pv)
            if abs(m_irr) < 1e-12:
                dss.PVsystems.Pmpp(0.0)
            elif p_post is not None:
                dss.PVsystems.Pmpp(max(abs(p_post), 0.0))
        except Exception:
            pass

    return {
        "p_load_kw": float(sum(p for p, _ in load_pq.values())),
        "q_load_kvar": float(sum(q for _, q in load_pq.values())),
    }


def _align_vm(node_order: list[str], vm: dict[str, float]) -> np.ndarray:
    out = np.full(len(node_order), np.nan, dtype=float)
    for i, node in enumerate(node_order):
        if node in vm:
            out[i] = vm[node]
    return out


def _err_stats(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    m = np.isfinite(pred) & np.isfinite(ref)
    if not m.any():
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "frac_over": float("nan"),
            "frac_under": float("nan"),
            "n": 0.0,
        }
    e = pred[m] - ref[m]
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "frac_over": float(np.mean(e > 1e-4)),
        "frac_under": float(np.mean(e < -1e-4)),
        "n": float(m.sum()),
    }



def _dataset_mag_from_csv(
    chunk_dir: Path,
    sample_id: int,
) -> tuple[list[str], np.ndarray]:
    """Load OpenDSS |V| labels from chunk feature CSV (no GNN .pt cache needed)."""
    full_csv = chunk_dir / "gnn_node_features_and_targets.csv"
    mvagg_csv = chunk_dir / "gnn_node_features_and_targets_mvagg.csv"
    nodes_csv = full_csv if full_csv.is_file() else mvagg_csv
    if not nodes_csv.is_file():
        raise FileNotFoundError(f"No node-feature CSV in {chunk_dir}")
    df = pd.read_csv(nodes_csv, usecols=lambda c: c in ("sample_id", "node", "vmag_pu"))
    df = df.loc[df["sample_id"].astype(int) == int(sample_id)].copy()
    if df.empty:
        raise KeyError(f"sample_id={sample_id} missing from {nodes_csv}")
    if "vmag_pu" not in df.columns:
        raise KeyError(f"{nodes_csv.name} missing vmag_pu column")
    df["node"] = df["node"].astype(str).str.strip().str.lower()
    node_order = [str(n) for n in df["node"].tolist()]
    mag = df["vmag_pu"].astype(float).to_numpy()
    return node_order, mag


def _resolve_906_pv_chunk_dir(repo: Path) -> Path:
    """Pick the first existing 906 PV fixedctrlinit chunk run folder."""
    roots = list(CHUNK_ROOTS.get("906", ()))
    roots.append(repo / "datasets_gnn2")
    for root in roots:
        root = Path(root)
        if not root.is_absolute():
            root = repo / root
        if not root.is_dir():
            continue
        name = root.name.lower()
        parents = [root]
        if "pv_voltvar" not in name and "fixedctrlinit" not in name:
            parents.append(root / "original_906_lvtestcase_pv_voltvar_fixedctrlinit")
        for parent in parents:
            if not parent.is_dir():
                continue
            runs = sorted(parent.glob("run_*"))
            for r in runs:
                if (r / "gnn_sample_meta.csv").is_file() and (
                    (r / "gnn_node_features_and_targets_mvagg.csv").is_file()
                    or (r / "gnn_node_features_and_targets.csv").is_file()
                ):
                    return r
    raise FileNotFoundError(
        "No original_906_lvtestcase_pv_voltvar_fixedctrlinit run_* chunk found"
    )


def replay_dataset_sample_logv(
    repo: Path,
    *,
    feeder: str,
    cache_pt: Path | None = None,
    chunk_dir: Path | None = None,
    sample_id: int | None = None,
    row_i: int | None = None,
    control_mode: str = "synced",
    fixedctrlinit: bool | None = None,
) -> DatasetReplayResult:
    """Run corrected live OD + Log(v) on one dataset sample; score vs labels.

    ``cache_pt`` may be omitted when ``chunk_dir`` + ``sample_id`` are given: |V|
    labels are then taken from the chunk feature CSV ``vmag_pu`` column (needed for
    906 PV fixedctrlinit before a GNN .pt cache exists).
    """
    import time
    import warnings

    import opendssdirect as dss
    import run_injection_dataset as inj
    from logv3lpf_daily_demo import (
        FEEDERS,
        _apply_opendss_cap_pv_to_logv,
        _apply_opendss_regtaps_to_logv,
        _quiet,
        _rebuild_logv_A_from_case_taps,
        _store_logv_regtaps,
        compile_cmd,
        ensure_logv3lpf,
        reset_invcontrol_pv_to_pf1,
        resolve_feeder,
        run_logv_autonomous,
        _reset_reg_taps_opendss,
    )

    repo = Path(repo).resolve()
    # Reload logv3lpf from disk BEFORE binding linpf/utils so notebook kernels
    # pick up transformer Yprim fixes without a restart. Log(v) is always
    # re-solved here (never read from a cached vm_logv CSV).
    ensure_logv3lpf(repo, reload=True)
    import logv3lpf  # noqa: F401
    import logv3lpf.linpf as linpf
    import logv3lpf.utils as _logv_utils
    from logv3lpf.DSSParser import DSScase

    _utils_path = Path(getattr(_logv_utils, "__file__", "") or "").resolve()
    _yprim_src = Path(_utils_path).read_text(encoding="utf-8") if _utils_path.is_file() else ""
    if "_opendss_transformer_yprim_pu" in _yprim_src:
        # Stale kernels used to keep ``v_ln = kVBase * 1000`` in memory.
        _vb_line = next(
            (
                ln.strip()
                for ln in _yprim_src.splitlines()
                if "v_ln" in ln and "kVBase" in ln and not ln.strip().startswith("#")
            ),
            "",
        )
        if "* 1000" in _vb_line or "*1000" in _vb_line:
            raise RuntimeError(
                f"Stale/buggy transformer Yprim Vbase scaling still present in {_utils_path}: {_vb_line}"
            )
    key = resolve_feeder(feeder)
    mode = str(control_mode).strip().lower()
    if mode not in ("synced", "static", "off"):
        raise ValueError("control_mode must be synced|static|off")

    use_csv_labels = cache_pt is None
    if use_csv_labels:
        if chunk_dir is None:
            if key == "906":
                chunk_dir = _resolve_906_pv_chunk_dir(repo)
            else:
                raise ValueError("chunk_dir required when cache_pt is None")
        chunk_dir = Path(chunk_dir)
        if sample_id is None:
            meta0 = pd.read_csv(chunk_dir / "gnn_sample_meta.csv", usecols=["sample_id"])
            sample_id = int(meta0["sample_id"].iloc[0])
        sample_id = int(sample_id)
        row_i = 0
        node_order, dataset_mag = _dataset_mag_from_csv(chunk_dir, sample_id)
    else:
        cache_pt = Path(cache_pt)
        chunk_dir = Path(chunk_dir) if chunk_dir is not None else resolve_chunk_dir(repo, cache_pt, key)
        pack = torch.load(cache_pt, map_location="cpu", weights_only=False)
        node_to_local = {str(k).strip().lower(): int(v) for k, v in pack["node_to_local"].items()}
        sample_ids = [int(s) for s in pack["sample_ids"]]
        local_to_node = {int(v): str(k) for k, v in node_to_local.items()}
        node_order = [local_to_node[i] for i in range(int(pack["x"].shape[1]))]

        if row_i is None:
            if sample_id is None:
                raise ValueError("Provide sample_id or row_i")
            sid_to_row = {int(s): i for i, s in enumerate(sample_ids)}
            if int(sample_id) not in sid_to_row:
                raise KeyError(f"sample_id={sample_id} not in cache")
            row_i = int(sid_to_row[int(sample_id)])
        sample_id = int(sample_ids[int(row_i)])
        dataset_mag = _dataset_mag_from_cache(pack, int(row_i), node_order)

    meta_df = pd.read_csv(chunk_dir / "gnn_sample_meta.csv")
    meta_df["sample_id"] = meta_df["sample_id"].astype(int)
    hit = meta_df.loc[meta_df["sample_id"] == sample_id]
    if hit.empty:
        raise KeyError(f"sample_id={sample_id} missing from gnn_sample_meta.csv")
    meta = hit.iloc[0]
    feat_pq = _load_feature_pq(chunk_dir, sample_id, repo=repo)

    dss_path = Path(FEEDERS[key]["dss"](repo)).resolve()
    do_fixedctrl = fixedctrlinit
    if do_fixedctrl is None:
        do_fixedctrl = key in ("906",) and "PV_voltvar" in str(dss_path)

    with _quiet():
        case = logv3lpf.case(
            compile_cmd(dss_path),
            FEEDERS[key]["sourcebus"],
            float(FEEDERS[key]["refvm"]),
            0,
        )

    inj.setup_daily()
    try:
        inj.set_time_index(int(meta["t_index"]))
    except Exception:
        pass
    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    # DEVICE_P_SHARE maps are IEEE-34-only. Other feeders use live DSS load→bus maps.
    if key == "ieee34":
        _loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(
            bus_to_phases
        )
    else:
        dev_to_dss_load, dev_to_busph_load = {}, {}
    inj_stats = _apply_feature_loads_and_zip(
        feat_pq=feat_pq,
        dev_to_dss_load=dev_to_dss_load,
        dev_to_busph_load=dev_to_busph_load,
        row=meta,
    )

    forced = _force_taps_from_meta(meta, FEEDER_REGS.get(key, ()))
    n_caps = _force_caps_from_meta(meta, FEEDER_CAPS.get(key, ()))

    remap_msg = ""
    t_opendss_solve_s = float("nan")
    t_opendss_cold_s = float("nan")
    t_logv_setup_s = float("nan")
    t_logv_solve_s = float("nan")
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        if mode == "static":
            try:
                dss.Text.Command("Set Mode=Snapshot")
                dss.Text.Command("Set ControlMode=Static")
                dss.Text.Command("Set MaxControlIter=100")
            except Exception:
                pass
            with _quiet():
                if do_fixedctrl:
                    reset_invcontrol_pv_to_pf1()
                t0 = time.perf_counter()
                dss.Solution.Solve()
                t_opendss_solve_s = float(time.perf_counter() - t0)
                t_setup0 = time.perf_counter()
                DSScase.process_openDSS_solution(case)
                DSScase.get_loads_from_dss(case)
                DSScase.append_pvsystems_as_negative_loads(case)
                DSScase.get_constants(case)
                DSScase.create_load_model(case)
                _apply_opendss_cap_pv_to_logv(case)
                t_logv_setup_s = float(time.perf_counter() - t_setup0)
                t_lv0 = time.perf_counter()
                run_logv_autonomous(case, max_iter=40)
                _store_logv_regtaps(case)
                t_logv_solve_s = float(time.perf_counter() - t_lv0)
                try:
                    _reset_reg_taps_opendss()
                    if do_fixedctrl:
                        reset_invcontrol_pv_to_pf1()
                    dss.Text.Command("Set ControlMode=Static")
                    t_c0 = time.perf_counter()
                    dss.Solution.InitSnap()
                    dss.Solution.Solve()
                    t_opendss_cold_s = float(time.perf_counter() - t_c0)
                except Exception:
                    t_opendss_cold_s = float("nan")
            logv_label = "Log(v), static states"
        elif mode == "synced":
            # Bugfix: synced previously used ControlMode=OFF so InvControl never settled.
            try:
                dss.Text.Command("Set Mode=Snapshot")
                dss.Text.Command("Set ControlMode=Static")
                dss.Text.Command("Set MaxControlIter=100")
            except Exception:
                pass
            with _quiet():
                if do_fixedctrl:
                    reset_invcontrol_pv_to_pf1()
                t0 = time.perf_counter()
                dss.Solution.Solve()
                t_opendss_solve_s = float(time.perf_counter() - t0)
                if not bool(dss.Solution.Converged()):
                    raise RuntimeError(f"OpenDSS did not converge on {key} sample_id={sample_id}")
                t_setup0 = time.perf_counter()
                DSScase.process_openDSS_solution(case)
                DSScase.get_loads_from_dss(case)
                DSScase.append_pvsystems_as_negative_loads(case)
                DSScase.get_constants(case)
                DSScase.create_load_model(case)
                _apply_opendss_cap_pv_to_logv(case)
                _apply_opendss_regtaps_to_logv(case)
                _rebuild_logv_A_from_case_taps(case)
                t_logv_setup_s = float(time.perf_counter() - t_setup0)
                t_lv0 = time.perf_counter()
                linpf.rank_k_correction_solve(case, True)
                _store_logv_regtaps(case)
                t_logv_solve_s = float(time.perf_counter() - t_lv0)
                try:
                    if do_fixedctrl:
                        reset_invcontrol_pv_to_pf1()
                    dss.Text.Command("Set ControlMode=Static")
                    t_c0 = time.perf_counter()
                    dss.Solution.InitSnap()
                    dss.Solution.Solve()
                    t_opendss_cold_s = float(time.perf_counter() - t_c0)
                except Exception:
                    t_opendss_cold_s = float("nan")
            logv_label = "Log(v), oracle states"
        else:
            try:
                dss.Text.Command("Set Mode=Snapshot")
                dss.Text.Command("Set ControlMode=OFF")
            except Exception:
                pass
            with _quiet():
                t0 = time.perf_counter()
                dss.Solution.Solve()
                t_opendss_solve_s = float(time.perf_counter() - t0)
                if not bool(dss.Solution.Converged()):
                    raise RuntimeError(f"OpenDSS did not converge on {key} sample_id={sample_id}")
                t_setup0 = time.perf_counter()
                DSScase.process_openDSS_solution(case)
                DSScase.get_loads_from_dss(case)
                DSScase.append_pvsystems_as_negative_loads(case)
                DSScase.get_constants(case)
                DSScase.create_load_model(case)
                linpf.calculate_base_matrices(case)
                t_logv_setup_s = float(time.perf_counter() - t_setup0)
                t_lv0 = time.perf_counter()
                linpf.rank_k_correction_solve(case, True)
                _store_logv_regtaps(case)
                t_logv_solve_s = float(time.perf_counter() - t_lv0)
                try:
                    dss.Text.Command("Set ControlMode=OFF")
                    t_c0 = time.perf_counter()
                    dss.Solution.InitSnap()
                    dss.Solution.Solve()
                    t_opendss_cold_s = float(time.perf_counter() - t_c0)
                except Exception:
                    t_opendss_cold_s = float("nan")
            logv_label = "Log(v), fixed states"
        for w in wlist:
            if "remapped" in str(w.message).lower():
                remap_msg = str(w.message)
                break

    od_mag = _align_vm(node_order, _flatten_case_vm(case, "openDSS"))
    logv_mag = _align_vm(node_order, _flatten_case_vm(case, "logv3lpf"))

    s_lv_ds = _err_stats(logv_mag, dataset_mag)
    s_od_ds = _err_stats(od_mag, dataset_mag)
    s_lv_od = _err_stats(logv_mag, od_mag)

    rows = []
    for i, node in enumerate(node_order):
        rows.append(
            {
                "node": node,
                "vm_dataset": float(dataset_mag[i]),
                "vm_opendss_live": float(od_mag[i]) if np.isfinite(od_mag[i]) else np.nan,
                "vm_logv": float(logv_mag[i]) if np.isfinite(logv_mag[i]) else np.nan,
                "abs_err_logv_vs_dataset": (
                    abs(float(logv_mag[i]) - float(dataset_mag[i]))
                    if np.isfinite(logv_mag[i]) and np.isfinite(dataset_mag[i])
                    else np.nan
                ),
            }
        )

    return DatasetReplayResult(
        feeder=key,
        control_mode=mode,
        sample_id=sample_id,
        row_i=int(row_i),
        logv_label=logv_label,
        n_nodes=len(node_order),
        mae_logv_vs_dataset=s_lv_ds["mae"],
        rmse_logv_vs_dataset=s_lv_ds["rmse"],
        mae_live_od_vs_dataset=s_od_ds["mae"],
        mae_logv_vs_live_od=s_lv_od["mae"],
        bias_logv_minus_dataset=s_lv_ds["bias"],
        frac_over_vs_dataset=s_lv_ds["frac_over"],
        frac_under_vs_dataset=s_lv_ds["frac_under"],
        p_load_kw=float(inj_stats["p_load_kw"]),
        q_load_kvar=float(inj_stats["q_load_kvar"]),
        forced_taps=forced,
        n_caps_set=int(n_caps),
        n_remap_warning=remap_msg,
        node_mae=pd.DataFrame(rows),
        t_opendss_solve_s=float(t_opendss_solve_s),
        t_opendss_cold_s=float(t_opendss_cold_s),
        t_logv_setup_s=float(t_logv_setup_s),
        t_logv_solve_s=float(t_logv_solve_s),
    )


def _meta_fields_for_sample(chunk_dir: Path, sample_id: int) -> dict[str, Any]:
    meta_df = pd.read_csv(Path(chunk_dir) / "gnn_sample_meta.csv")
    meta_df["sample_id"] = meta_df["sample_id"].astype(int)
    hit = meta_df.loc[meta_df["sample_id"] == int(sample_id)]
    if hit.empty:
        return {}
    row = hit.iloc[0]
    out: dict[str, Any] = {}
    key_map = {
        "scenario_id": "scenario_id",
        "scenario": "scenario_id",
        "t_index": "t_index",
        "t_minutes": "t_minutes",
        "m_loadshape": "m_loadshape",
        "m_irradshape": "m_irradshape",
        "P_load_total_kw": "P_load_total_kw",
    }
    for src, dst in key_map.items():
        if src in row.index and pd.notna(row[src]) and dst not in out:
            try:
                out[dst] = int(row[src]) if dst in ("scenario_id", "t_index") else float(row[src])
            except Exception:
                out[dst] = row[src]
    return out


def _mean_finite(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v)
    return float(np.mean(v[m])) if m.any() else float("nan")


def evaluate_dataset_samples(
    repo: Path,
    *,
    feeder: str,
    cache_pt: Path,
    control_mode: str = "synced",
    sample_ids: list[int] | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Evaluate multiple dataset samples; return per-sample metrics vs labels."""
    pack = torch.load(Path(cache_pt), map_location="cpu", weights_only=False)
    all_ids = [int(s) for s in pack["sample_ids"]]
    if sample_ids is None:
        sample_ids = all_ids
    if max_samples is not None:
        sample_ids = list(sample_ids)[: int(max_samples)]

    chunk_dir = resolve_chunk_dir(Path(repo), Path(cache_pt), resolve_feeder_safe(feeder))
    rows = []
    for sid in sample_ids:
        meta_bits = _meta_fields_for_sample(chunk_dir, int(sid))
        try:
            r = replay_dataset_sample_logv(
                Path(repo),
                feeder=feeder,
                cache_pt=Path(cache_pt),
                chunk_dir=chunk_dir,
                sample_id=int(sid),
                control_mode=control_mode,
            )
            nm = r.node_mae
            rows.append(
                {
                    "sample_id": r.sample_id,
                    **meta_bits,
                    "mae_logv_vs_dataset": r.mae_logv_vs_dataset,
                    "rmse_logv_vs_dataset": r.rmse_logv_vs_dataset,
                    "mae_live_od_vs_dataset": r.mae_live_od_vs_dataset,
                    "mae_logv_vs_live_od": r.mae_logv_vs_live_od,
                    "bias_logv_minus_dataset": r.bias_logv_minus_dataset,
                    "mean_vm_dataset": _mean_finite(None if nm is None else nm["vm_dataset"]),
                    "mean_vm_logv": _mean_finite(None if nm is None else nm["vm_logv"]),
                    "mean_vm_live_od": _mean_finite(
                        None if nm is None else nm["vm_opendss_live"]
                    ),
                    "p_load_kw": r.p_load_kw,
                    "t_opendss_solve_s": r.t_opendss_solve_s,
                    "t_opendss_cold_s": r.t_opendss_cold_s,
                    "t_logv_setup_s": r.t_logv_setup_s,
                    "t_logv_solve_s": r.t_logv_solve_s,
                    "ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": int(sid),
                    **meta_bits,
                    "mae_logv_vs_dataset": np.nan,
                    "rmse_logv_vs_dataset": np.nan,
                    "mae_live_od_vs_dataset": np.nan,
                    "mae_logv_vs_live_od": np.nan,
                    "bias_logv_minus_dataset": np.nan,
                    "mean_vm_dataset": np.nan,
                    "mean_vm_logv": np.nan,
                    "mean_vm_live_od": np.nan,
                    "p_load_kw": np.nan,
                    "t_opendss_solve_s": np.nan,
                    "t_opendss_cold_s": np.nan,
                    "t_logv_setup_s": np.nan,
                    "t_logv_solve_s": np.nan,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset_latency(daily_df: pd.DataFrame) -> dict[str, float]:
    """Print/return OpenDSS vs Log(v) solve times over ok samples.

    Fair primary comparison (closest to paper wall-clock intent):
      OpenDSS cold InitSnap+Solve  vs  Log(v) online linear solve
    Paper Table II itself is FLOPs for Ybus inverse update (Woodbury vs LU),
    not wall-clock ms — reported here separately as context only.
    """
    ok = daily_df.loc[daily_df.get("ok", True)].copy() if "ok" in daily_df.columns else daily_df.copy()
    if ok.empty or "t_logv_solve_s" not in ok.columns:
        print("[logv_dataset_replay] no latency rows to summarize")
        return {}

    def _arr(col: str) -> np.ndarray:
        if col not in ok.columns:
            return np.full(len(ok), np.nan, dtype=float)
        return pd.to_numeric(ok[col], errors="coerce").to_numpy(dtype=float)

    od_warm = _arr("t_opendss_solve_s")
    od_cold = _arr("t_opendss_cold_s")
    lv_setup = _arr("t_logv_setup_s")
    lv_on = _arr("t_logv_solve_s")
    m = np.isfinite(lv_on) & (np.isfinite(od_cold) | np.isfinite(od_warm))
    n = int(m.sum())
    if n == 0:
        print("[logv_dataset_replay] latency columns present but all non-finite")
        return {}

    def _stats(a: np.ndarray) -> tuple[float, float, float, float]:
        aa = a[m & np.isfinite(a)]
        if aa.size == 0:
            return (float("nan"),) * 4
        return (
            float(np.mean(aa)),
            float(np.median(aa)),
            float(np.sum(aa)),
            float(aa.size),
        )

    mean_warm_s, med_warm_s, tot_warm_s, n_warm = _stats(od_warm)
    mean_cold_s, med_cold_s, tot_cold_s, n_cold = _stats(od_cold)
    mean_setup_s, med_setup_s, tot_setup_s, _ = _stats(lv_setup)
    mean_lv_s, med_lv_s, tot_lv_s, _ = _stats(lv_on)
    fair_od_s = mean_cold_s if np.isfinite(mean_cold_s) else mean_warm_s
    speedup_fair = (fair_od_s / mean_lv_s) if (np.isfinite(fair_od_s) and mean_lv_s > 0) else float("nan")

    print("\n" + "-" * 72)
    print(f"LATENCY  n_ok={n}  (compile / feature-CSV IO excluded)")
    print("  --- OpenDSS ---")
    print(
        f"  warm Solve()            mean={mean_warm_s*1e3:.3f} ms  "
        f"median={med_warm_s*1e3:.3f} ms  total={tot_warm_s:.3f} s  "
        f"raw_mean={mean_warm_s:.6f} s"
    )
    print(
        f"  COLD InitSnap+Solve     mean={mean_cold_s*1e3:.3f} ms  "
        f"median={med_cold_s*1e3:.3f} ms  total={tot_cold_s:.3f} s  "
        f"raw_mean={mean_cold_s:.6f} s  (fair OD baseline; n={int(n_cold)})"
    )
    print("  --- Log(v) ---")
    print(
        f"  setup (load/A refresh)  mean={mean_setup_s*1e3:.3f} ms  "
        f"median={med_setup_s*1e3:.3f} ms  total={tot_setup_s:.3f} s  "
        f"(amortized / not online PF)"
    )
    print(
        f"  ONLINE linear solve     mean={mean_lv_s*1e3:.3f} ms  "
        f"median={med_lv_s*1e3:.3f} ms  total={tot_lv_s:.3f} s  "
        f"raw_mean={mean_lv_s:.6f} s  (fair Log(v) baseline)"
    )
    print("  --- Fair speedup ---")
    print(
        f"  OpenDSS_cold / Log(v)_online  = {speedup_fair:.3f}x   "
        f"(raw means s: OD_cold={fair_od_s:.6f}  Log(v)={mean_lv_s:.6f})"
    )
    print(
        "  note: paper Table II is FLOPs for sparse Ybus inverse update "
        "(Woodbury vs LU), not wall-clock; wall-clock here is InitSnap+Solve vs online Log(v)."
    )
    print("-" * 72)
    return {
        "n_ok": float(n),
        "mean_opendss_warm_s": mean_warm_s,
        "mean_opendss_cold_s": mean_cold_s,
        "mean_logv_setup_s": mean_setup_s,
        "mean_logv_online_s": mean_lv_s,
        "median_opendss_cold_s": med_cold_s,
        "median_logv_online_s": med_lv_s,
        "total_opendss_cold_s": tot_cold_s,
        "total_logv_online_s": tot_lv_s,
        "speedup_cold_od_over_logv_online": float(speedup_fair),
    }
def plot_dataset_daily_compare(
    daily_df: pd.DataFrame,
    *,
    feeder: str,
    control_mode: str,
    out_path: Path | None = None,
    show: bool = True,
    scenario_id: int | None = None,
) -> None:
    """Inline/daily figure: dataset OpenDSS labels vs Log(v) (+ live OD diagnostic).

    If ``scenario_id`` is set (or multiple scenarios exist), only that one scenario
    is visualized even when metrics were computed over many scenarios.
    """
    import matplotlib.pyplot as plt

    ok = daily_df.loc[daily_df.get("ok", True)].copy() if "ok" in daily_df.columns else daily_df.copy()
    if ok.empty:
        print("[logv_dataset_replay] no successful samples to plot")
        return
    if "scenario_id" in ok.columns and ok["scenario_id"].notna().any():
        scen_vals = sorted({int(s) for s in ok["scenario_id"].dropna().tolist()})
        if scenario_id is None and len(scen_vals) > 1:
            scenario_id = int(scen_vals[0])
            print(
                f"[logv_dataset_replay] plot uses scenario_id={scenario_id} only "
                f"(metrics cover {len(scen_vals)} scenarios)"
            )
        if scenario_id is not None:
            ok = ok.loc[ok["scenario_id"].astype(int) == int(scenario_id)].copy()
    if ok.empty:
        print("[logv_dataset_replay] no rows left after scenario filter")
        return
    if "t_index" in ok.columns and ok["t_index"].notna().any():
        sort_cols = [c for c in ("scenario_id", "t_index", "sample_id") if c in ok.columns]
        ok = ok.sort_values(sort_cols, na_position="last")
        x = ok["t_index"].to_numpy(dtype=float)
        xlab = "t_index"
    else:
        ok = ok.reset_index(drop=True)
        x = np.arange(len(ok), dtype=float)
        xlab = "sample order"

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    ax0, ax1 = axes
    ax0.plot(x, ok["mean_vm_dataset"], "kx-", ms=4, lw=1.0, label="Dataset OpenDSS labels")
    ax0.plot(x, ok["mean_vm_logv"], "^-", color="#d95f02", ms=4, lw=1.0, label="Log(v)")
    if "mean_vm_live_od" in ok.columns:
        ax0.plot(
            x,
            ok["mean_vm_live_od"],
            "o-",
            color="#1f78b4",
            ms=3.5,
            lw=0.9,
            alpha=0.85,
            label="OpenDSS live (diagnostic)",
        )
    ax0.set_ylabel("Network-mean |V| (p.u.)")
    title = f"{feeder}  |  {control_mode}  |  dataset labels vs Log(v)"
    if scenario_id is not None:
        title += f"  (plot scen={scenario_id})"
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(x, ok["mae_logv_vs_dataset"], "s-", color="#d95f02", ms=4, lw=1.0, label="Log(v) vs labels")
    if "mae_live_od_vs_dataset" in ok.columns:
        ax1.plot(
            x,
            ok["mae_live_od_vs_dataset"],
            "o-",
            color="#1f78b4",
            ms=3.5,
            lw=0.9,
            alpha=0.85,
            label="Live OD vs labels",
        )
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("MAE (p.u.)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[logv_dataset_replay] wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def resolve_feeder_safe(feeder: str) -> str:
    from logv3lpf_daily_demo import resolve_feeder

    return resolve_feeder(feeder)
