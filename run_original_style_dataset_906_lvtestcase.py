"""
Original-style node feature/target dataset generation on IEEE European LV Test Case (906 buses).

Mirrors run_original_style_dataset_8500_unbalanced.py enough for DA-GPS training:
  - scenario sampling (P_load / Q_load + noise); P_pv fixed at 0 (no PV on this feeder)
  - time selection via three profiles (load / pv-proxy / net); pv-proxy reuses load shape
  - per-node features: p_load_kw, q_load_kvar, p_pv_kw (=0), optional zero BESS cols
  - targets: vmag_pu, vang_deg
  - artifacts per run_*:
      gnn_node_index_master.csv
      gnn_edges_phase_static.csv
      gnn_sample_meta.csv
      gnn_node_features_and_targets.csv
      gnn_node_features_and_targets_mvagg.csv  (compat filename; no real MV aggregation)

Cap/reg meta columns use the hardcoded 8500 TARGET_* names with constant dummies so the
existing trainer can load sample_meta without code changes. Physics loss should stay off
for 906 (no regs/caps / no 8500 Y-catalog mapping).
"""
from __future__ import annotations

import csv
import importlib
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_injection_dataset as inj
import run_loadtype_dataset as lt_dist
import run_loadtype_dataset_8500 as lt8500
import run_original_style_dataset_8500_unbalanced as ds8500

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
lt8500 = importlib.reload(lt8500)
ds8500 = importlib.reload(ds8500)

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

MODEL_DIR = (
    REPO_ROOT
    / "906 bus system"
    / "OpenDSS-master"
    / "OpenDSS-master"
    / "Distrib"
    / "IEEETestCases"
    / "LVTestCase"
)
MASTER_DSS = MODEL_DIR / "Master.dss"
PROFILE_DIR = MODEL_DIR / "Daily_1min_100profiles"

# Default OUT_DIR when run as a script; notebook chunked driver overrides ns["OUT_DIR"].
_K_GNN2 = Path(r"K:\My Drive\datasets_gnn2")
_K_MYDRIVE = Path(r"K:\My Drive")
try:
    import google.colab  # noqa: F401
    OUT_DIR = Path("/content/drive/MyDrive/datasets_gnn2/original_906_lvtestcase")
except ImportError:
    if _K_GNN2.exists() or _K_MYDRIVE.exists():
        OUT_DIR = _K_GNN2 / "original_906_lvtestcase"
    elif Path(r"D:\datasets").exists():
        OUT_DIR = Path(r"D:\datasets\original_906_lvtestcase")
    else:
        OUT_DIR = REPO_ROOT / "datasets_gnn2" / "original_906_lvtestcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CSV = OUT_DIR / "gnn_edges_phase_static.csv"
NODE_CSV = OUT_DIR / "gnn_node_features_and_targets.csv"
SAMPLE_CSV = OUT_DIR / "gnn_sample_meta.csv"
NODE_INDEX_CSV = OUT_DIR / "gnn_node_index_master.csv"
MVAGG_CSV = OUT_DIR / "gnn_node_features_and_targets_mvagg.csv"

NPTS = 1440
STEP_MIN = 1
N_LOADS_NOMINAL = 55
P_LOAD_MEAN_DEFAULT_KW = float(N_LOADS_NOMINAL)  # each load rated 1 kW
Q_LOAD_MEAN_DEFAULT_KVAR = float(P_LOAD_MEAN_DEFAULT_KW * math.tan(math.acos(0.95)))

# Hardcoded 8500 trainer TARGET_* column names (dummy constants for 906).
DUMMY_REG_COLS: tuple[str, ...] = (
    "reg_feeder_rega_tap_pu",
    "reg_feeder_regb_tap_pu",
    "reg_feeder_regc_tap_pu",
    "reg_vreg2_a_tap_pu",
    "reg_vreg2_b_tap_pu",
    "reg_vreg2_c_tap_pu",
    "reg_vreg3_a_tap_pu",
    "reg_vreg3_b_tap_pu",
    "reg_vreg3_c_tap_pu",
    "reg_vreg4_a_tap_pu",
    "reg_vreg4_b_tap_pu",
    "reg_vreg4_c_tap_pu",
)
DUMMY_CAP_COLS: tuple[str, ...] = (
    "cap_capbank0a_n_steps_on",
    "cap_capbank0b_n_steps_on",
    "cap_capbank0c_n_steps_on",
    "cap_capbank1a_n_steps_on",
    "cap_capbank1b_n_steps_on",
    "cap_capbank1c_n_steps_on",
    "cap_capbank2a_n_steps_on",
    "cap_capbank2b_n_steps_on",
    "cap_capbank2c_n_steps_on",
    "cap_capbank3_n_steps_on",
)


def _is_source_bus(bus_name: str) -> bool:
    return str(bus_name).strip().lower().startswith("sourcebus")


def _filter_graph_nodes(node_names: list[str]) -> list[str]:
    out: list[str] = []
    for n in node_names:
        bus = str(n).split(".")[0]
        if _is_source_bus(bus):
            continue
        out.append(str(n))
    return out


def _dummy_cap_reg_meta() -> dict[str, float]:
    out: dict[str, float] = {c: 1.0 for c in DUMMY_REG_COLS}
    out.update({c: 0.0 for c in DUMMY_CAP_COLS})
    return out


def _read_single_col_profile(path: Path, npts: int = NPTS) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing load profile: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 1:
        raise RuntimeError(f"{path} has no columns")
    y = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < npts:
        raise RuntimeError(f"{path} has only {len(y)} points < required {npts}")
    return y[:npts].astype(float)


def _compile_906_lvtestcase_snapshot_setup() -> None:
    """Compile LVTestCase network without Master.dss yearly demand-interval solve."""
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Missing LVTestCase model dir: {MODEL_DIR}")
    if not MASTER_DSS.is_file():
        raise FileNotFoundError(f"Missing master DSS: {MASTER_DSS}")

    dss.Basic.ClearAll()
    dss.Text.Command(f'cd "{os.path.abspath(str(MODEL_DIR))}"')
    dss.Text.Command("Set DefaultBaseFrequency=50")
    dss.Text.Command("New circuit.LVTest")
    dss.Text.Command("Edit Vsource.Source BasekV=11 pu=1.05 ISC3=3000 ISC1=5")
    dss.Text.Command("Redirect LineCode.txt")
    dss.Text.Command("Redirect LoadShapes.txt")
    dss.Text.Command("batchedit loadshape..* useactual=no")
    dss.Text.Command("Redirect Lines.txt")
    dss.Text.Command("Redirect Transformers.txt")
    dss.Text.Command("Redirect Loads.txt")
    dss.Text.Command("Set voltagebases=[11 .416]")
    dss.Text.Command("Calcvoltagebases")
    dss.Text.Command("set mode=snapshot")
    dss.Text.Command("set controlmode=off")
    dss.Text.Command("set maxiterations=100")
    dss.Text.Command("set maxcontroliter=10")


def _detach_yearly_daily_from_loads() -> None:
    if not dss.Loads.First():
        return
    while True:
        nm = dss.Loads.Name()
        dss.Loads.Name(nm)
        try:
            dss.Loads.Yearly("")
        except Exception:
            pass
        try:
            dss.Loads.Daily("")
        except Exception:
            pass
        if not dss.Loads.Next():
            break


def _collect_loads_and_maps() -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[tuple[str, int, float]]]]:
    names: list[str] = []
    kw0: list[float] = []
    kvar0: list[float] = []
    load_to_busph: dict[str, list[tuple[str, int, float]]] = {}
    if not dss.Loads.First():
        return names, np.array([], dtype=float), np.array([], dtype=float), load_to_busph
    while True:
        name = str(dss.Loads.Name())
        dss.Loads.Name(name)
        names.append(name)
        kw0.append(float(dss.Loads.kW()))
        kvar0.append(float(dss.Loads.kvar()))
        load_to_busph[name] = lt8500._busph_fracs_load(name)
        if not dss.Loads.Next():
            break
    return names, np.asarray(kw0, dtype=float), np.asarray(kvar0, dtype=float), load_to_busph


def _load_per_load_profiles(load_names: list[str]) -> np.ndarray:
    """Return (n_loads, NPTS) multipliers from Daily_1min_100profiles."""
    rows: list[np.ndarray] = []
    for i, _name in enumerate(load_names, start=1):
        # Loads.txt assigns Yearly=Shape_i for LOAD i; profiles are load_profile_i.txt.
        path = PROFILE_DIR / f"load_profile_{i}.txt"
        if not path.is_file():
            # Case-insensitive fallback used by some Windows/OpenDSS installs.
            alt = PROFILE_DIR / f"Load_profile_{i}.txt"
            path = alt if alt.is_file() else path
        rows.append(_read_single_col_profile(path, npts=NPTS))
    return np.vstack(rows)


def _mean_system_load_profile(per_load: np.ndarray) -> np.ndarray:
    return np.mean(per_load, axis=0)


def _apply_snapshot_loads_only(
    *,
    load_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    load_to_busph: dict[str, list[tuple[str, int, float]]],
    per_load_profiles: np.ndarray,
    p_load_total_kw: float,
    q_load_total_kvar: float,
    t_index: int,
    m_load_t: float,
    sigma_load: float,
    rng: np.random.Generator,
) -> tuple[dict, dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    """Scale scenario totals by mean profile, then allocate by base*per-load profile at t."""
    p_load_t = float(p_load_total_kw) * float(m_load_t)
    q_load_t = float(q_load_total_kvar) * float(m_load_t)

    w_p = base_kw * np.maximum(per_load_profiles[:, int(t_index)], 0.0)
    w_q = base_kvar * np.maximum(per_load_profiles[:, int(t_index)], 0.0)
    sum_wp = float(np.sum(w_p))
    sum_wq = float(np.sum(w_q))
    if sum_wp <= 0.0:
        w_p = np.maximum(base_kw, 0.0)
        sum_wp = float(np.sum(w_p)) or 1.0
    if sum_wq <= 0.0:
        w_q = np.maximum(base_kvar, 0.0)
        sum_wq = float(np.sum(w_q)) or 1.0

    noise_p = np.maximum(0.0, 1.0 + rng.normal(0.0, float(sigma_load), size=len(load_names)))
    noise_q = np.maximum(0.0, 1.0 + rng.normal(0.0, float(sigma_load), size=len(load_names)))
    kw_set = (p_load_t * w_p / sum_wp) * noise_p
    kvar_set = (q_load_t * w_q / sum_wq) * noise_q

    busphP_load: dict[tuple[str, int], float] = {}
    busphQ_load: dict[tuple[str, int], float] = {}
    for i, name in enumerate(load_names):
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw_set[i]))
        dss.Loads.kvar(float(kvar_set[i]))
        for (bus, ph, w) in load_to_busph.get(name, []):
            busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + float(kw_set[i]) * float(w)
            busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + float(kvar_set[i]) * float(w)

    totals = {
        "P_load_time_kw": float(p_load_t),
        "Q_load_time_kvar": float(q_load_t),
        "P_pv_time_kw": 0.0,
        "p_load_kw_set_total": float(np.sum(kw_set)),
        "q_load_kvar_set_total": float(np.sum(kvar_set)),
        "p_pv_pmpp_kw_set_total": 0.0,
    }
    return totals, busphP_load, busphQ_load


def _write_mvagg_compat_from_raw(node_csv: Path, mvagg_csv: Path) -> None:
    """Copy raw node CSV to *_mvagg.csv, dropping BESS cols (no real MV aggregation on 906)."""
    df = pd.read_csv(node_csv)
    drop_cols = [c for c in ("p_bess_kw", "q_bess_kvar") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # Ensure p_pv_kw exists and is zero-filled.
    if "p_pv_kw" not in df.columns:
        df["p_pv_kw"] = 0.0
    else:
        df["p_pv_kw"] = df["p_pv_kw"].fillna(0.0).astype(float)
    preferred = [
        "sample_id",
        "node",
        "node_idx",
        "bus",
        "phase",
        "p_load_kw",
        "q_load_kvar",
        "p_pv_kw",
        "vmag_pu",
        "vang_deg",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df[cols].to_csv(mvagg_csv, index=False)


def generate_original_style_dataset_906_lvtestcase(
    *,
    n_scenarios: int = 50,
    k_snapshots_per_scenario_total: int = 40,
    bins_by_profile: dict | None = None,
    include_anchors: bool = True,
    master_seed: int = 90620230,
    sigma_load: float = 0.5,
    sigma_pv: float = 0.0,
    node_pe_k: int = 8,
    node_pe_seed: int = 42,
    node_pe_zero_eig_tol: float = 1e-8,
    node_pe_from_csv: str | None = None,
    node_pe_save_csv: str | None = None,
    p_load_mean_kw: float = P_LOAD_MEAN_DEFAULT_KW,
    q_load_mean_kvar: float = Q_LOAD_MEAN_DEFAULT_KVAR,
    p_load_scale_range: tuple[float, float] = (0.3, 1.8),
    q_load_scale_range: tuple[float, float] = (0.3, 1.8),
    p_pv_scale_range: tuple[float, float] = (0.0, 0.0),
    vmin_safe_pu: float = 0.55,
    vmax_safe_pu: float = 1.45,
    include_source_in_safe_band: bool = False,
    return_node_df: bool = False,
    write_mvagg_compat: bool = True,
    delete_raw_node_csv_after_mvagg: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del sigma_pv, p_pv_scale_range  # no PV; kept for notebook API parity with 8500 cell

    if bins_by_profile is None:
        bins_by_profile = {"load": 3, "pv": 3, "net": 3}
    if k_snapshots_per_scenario_total < 1:
        raise ValueError("k_snapshots_per_scenario_total must be >= 1")
    if not (0.0 < float(vmin_safe_pu) < float(vmax_safe_pu)):
        raise ValueError(f"Invalid safe voltage band: [{vmin_safe_pu}, {vmax_safe_pu}]")
    if float(sigma_load) < 0.0:
        raise ValueError("sigma_load must be non-negative.")
    if int(node_pe_k) < 0:
        raise ValueError("node_pe_k must be >= 0")
    if float(p_load_mean_kw) <= 0.0 or float(q_load_mean_kvar) <= 0.0:
        raise ValueError("p_load_mean_kw and q_load_mean_kvar must be positive.")

    _compile_906_lvtestcase_snapshot_setup()
    _detach_yearly_daily_from_loads()

    node_names_all, _, _, _ = inj.get_all_bus_phase_nodes()
    node_names_graph = _filter_graph_nodes(node_names_all)
    if not node_names_graph:
        raise RuntimeError("No nodes left after filtering SourceBus for graph artifacts.")
    node_to_idx_all = {n: i for i, n in enumerate(node_names_all)}
    print(f"[diag] nodes_all={len(node_names_all)} nodes_graph={len(node_names_graph)} (SourceBus filtered)")
    print("[diag] no PV / no BESS / no regs / no caps — dummy 8500 cap/reg meta columns will be written.")

    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_graph,
        edge_csv_path=str(EDGE_CSV),
        excluded_buses=(),
    )
    ds8500._enrich_edges_with_basekv_and_length_km(EDGE_CSV, node_names_graph)

    node_to_dist = lt_dist._compute_electrical_distance_from_source(node_names_graph, str(EDGE_CSV))
    node_to_base_kv = ds8500._node_base_kv_map(node_names_graph)
    node_index_df = pd.DataFrame(
        {
            "node": node_names_graph,
            "node_idx": np.arange(len(node_names_graph), dtype=int),
            "base_kv": [float(node_to_base_kv.get(n, np.nan)) for n in node_names_graph],
            "electrical_distance_ohm": [float(node_to_dist.get(n, np.nan)) for n in node_names_graph],
        }
    )

    pe_src = str(node_pe_from_csv).strip() if node_pe_from_csv is not None else ""
    if pe_src:
        pe_path = Path(pe_src)
        if not pe_path.is_file():
            raise FileNotFoundError(f"node_pe_from_csv not found: {pe_path}")
        pe_df = pd.read_csv(pe_path)
        if "node" not in pe_df.columns:
            raise ValueError(f"{pe_path} must contain a 'node' column.")
        pe_cols = sorted([c for c in pe_df.columns if str(c).lower().startswith("pe_")])
        if not pe_cols:
            raise ValueError(f"{pe_path} contains no pe_* columns.")
        pe_df["node"] = pe_df["node"].astype(str).str.strip().str.lower()
        node_index_df["node"] = node_index_df["node"].astype(str).str.strip().str.lower()
        pe_map = pe_df.set_index("node")[pe_cols]
        aligned = pe_map.reindex(node_index_df["node"].tolist())
        miss_nodes = aligned.index[aligned.isna().any(axis=1)].tolist()
        if miss_nodes:
            raise ValueError(
                f"{pe_path}: missing PE rows for {len(miss_nodes)} nodes (showing up to 5): {miss_nodes[:5]}"
            )
        for c in pe_cols:
            node_index_df[c] = aligned[c].to_numpy(dtype=float)
        print(f"[diag] loaded node PE from CSV: {pe_path} (k={len(pe_cols)})")
    elif int(node_pe_k) > 0:
        pe = ds8500._compute_laplacian_pe_from_edges(
            node_names=node_names_graph,
            edge_csv_path=EDGE_CSV,
            k=int(node_pe_k),
            seed=int(node_pe_seed),
            zero_eig_tol=float(node_pe_zero_eig_tol),
        )
        for j in range(int(node_pe_k)):
            node_index_df[f"pe_{j+1}"] = pe[:, j]
        print(f"[diag] computed node PE: k={int(node_pe_k)} for {len(node_names_graph)} graph nodes")
        pe_save = str(node_pe_save_csv).strip() if node_pe_save_csv is not None else ""
        if pe_save:
            save_path = Path(pe_save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            node_index_df[["node", *[f"pe_{j+1}" for j in range(int(node_pe_k))]]].to_csv(save_path, index=False)
            print(f"[diag] saved node PE CSV: {save_path}")
    node_index_df.to_csv(NODE_INDEX_CSV, index=False)

    safe_band_eval_indices = []
    for i, n in enumerate(node_names_all):
        b = n.split(".")[0].strip().lower()
        if (not include_source_in_safe_band) and b.startswith("sourcebus"):
            continue
        safe_band_eval_indices.append(i)
    if not safe_band_eval_indices:
        raise RuntimeError("No nodes available for safe-band evaluation.")

    load_names, base_kw, base_kvar, load_to_busph = _collect_loads_and_maps()
    if len(load_names) == 0:
        raise RuntimeError("No loads found in LVTestCase model.")
    per_load_profiles = _load_per_load_profiles(load_names)
    mL = _mean_system_load_profile(per_load_profiles)
    # No PV: reuse load shape for time-selection diversity; applied PV power stays 0.
    mPV = mL.copy()

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[dict] = []
    sample_id = 0
    skipped_nonconv = 0
    skipped_bad_v = 0
    total_v_outside_band = 0
    n_node_rows_written = 0
    dummy_meta = _dummy_cap_reg_meta()

    node_fieldnames = [
        "sample_id",
        "node",
        "node_idx",
        "bus",
        "phase",
        "p_load_kw",
        "q_load_kvar",
        "p_pv_kw",
        "p_bess_kw",
        "q_bess_kvar",
        "vmag_pu",
        "vang_deg",
    ]
    with open(NODE_CSV, "w", newline="", encoding="utf-8") as f_node:
        node_writer = csv.DictWriter(f_node, fieldnames=node_fieldnames)
        node_writer.writeheader()

        for s in range(n_scenarios):
            t0_s = time.time()
            _compile_906_lvtestcase_snapshot_setup()
            _detach_yearly_daily_from_loads()
            load_names, base_kw, base_kvar, load_to_busph = _collect_loads_and_maps()
            per_load_profiles = _load_per_load_profiles(load_names)
            mL = _mean_system_load_profile(per_load_profiles)
            mPV = mL.copy()

            p_load = float(p_load_mean_kw) * float(rng_master.uniform(*p_load_scale_range))
            q_load = float(q_load_mean_kvar) * float(rng_master.uniform(*q_load_scale_range))
            p_pv = 0.0

            prof_load, prof_pv = mL, mPV
            prof_net = (p_load * mL) - (p_pv * mPV)
            rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            times = inj.select_times_three_profiles(
                prof_load=prof_load,
                prof_pv=prof_pv,
                prof_net=prof_net,
                K_total=k_snapshots_per_scenario_total,
                bins_by_profile=bins_by_profile,
                include_anchors=include_anchors,
                rng=rng_times,
            )

            rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            outside_band_this_scenario = 0
            below_band_this_scenario = 0
            above_band_this_scenario = 0
            finite_v_count_this_scenario = 0
            nonconv_this_scenario = 0
            badv_this_scenario = 0
            offender_counts: dict[str, int] = {}

            times_int = [int(x) for x in times]
            total_times_this_s = len(times_int)
            for k_t, t in enumerate(times_int, start=1):
                # 1-minute steps: hour + seconds within hour.
                hr = int(t // 60)
                sec = int((t % 60) * 60)
                dss.Text.Command(f"set hour={hr} sec={sec}")

                totals, busphP_load, busphQ_load = _apply_snapshot_loads_only(
                    load_names=load_names,
                    base_kw=base_kw,
                    base_kvar=base_kvar,
                    load_to_busph=load_to_busph,
                    per_load_profiles=per_load_profiles,
                    p_load_total_kw=p_load,
                    q_load_total_kvar=q_load,
                    t_index=t,
                    m_load_t=float(mL[t]),
                    sigma_load=sigma_load,
                    rng=rng_solve,
                )

                try:
                    dss.Solution.Solve()
                except Exception:
                    pass
                if not dss.Solution.Converged():
                    skipped_nonconv += 1
                    nonconv_this_scenario += 1
                    continue

                vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_all)
                vmag_arr = np.asarray(vmag_m, dtype=float)
                if not np.isfinite(vmag_arr).all():
                    skipped_bad_v += 1
                    badv_this_scenario += 1
                    continue

                vmag_eval = vmag_arr[safe_band_eval_indices]
                mask_below = vmag_eval < float(vmin_safe_pu)
                mask_above = vmag_eval > float(vmax_safe_pu)
                mask_out = mask_below | mask_above
                n_below = int(np.sum(mask_below))
                n_above = int(np.sum(mask_above))
                n_outside = int(np.sum(mask_out))
                n_finite = int(np.sum(np.isfinite(vmag_eval)))
                outside_band_this_scenario += n_outside
                below_band_this_scenario += n_below
                above_band_this_scenario += n_above
                finite_v_count_this_scenario += n_finite
                total_v_outside_band += n_outside

                if n_outside > 0:
                    eval_idx = np.asarray(safe_band_eval_indices, dtype=int)
                    for local_idx in np.where(mask_out)[0].tolist():
                        nm = str(node_names_all[int(eval_idx[local_idx])]).lower()
                        offender_counts[nm] = offender_counts.get(nm, 0) + 1

                p_load_post_kw, q_load_post_kvar = ds8500._sum_loads_post_solve_kw_kvar()
                p_loss_post_kw, q_loss_post_kvar = ds8500._circuit_losses_kw_kvar()
                p_grid_post_kw, q_grid_post_kvar = ds8500._grid_upstream_post_kw_kvar()
                vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_all, vmag_m, vang_m)}

                rows_sample.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": s,
                        "t_index": t,
                        "t_minutes": int(t * STEP_MIN),
                        "P_load_total_kw": float(p_load),
                        "Q_load_total_kvar": float(q_load),
                        "P_pv_total_kw": float(p_pv),
                        "sigma_load": float(sigma_load),
                        "sigma_pv": 0.0,
                        "bess_total_mva_mean": 0.0,
                        "bess_total_mva_sigma": 0.0,
                        "bess_total_mva_scenario": 0.0,
                        "bess_num_nodes": 0,
                        "bess_num_3ph_buses": 0,
                        "bess_total_nodes_1ph_equiv": 0,
                        "bess_s_rated_kva_per_bus": 0.0,
                        "bess_s_rated_kva_per_node": 0.0,
                        "bess_q_frac_max": 0.0,
                        "bess_candidate_count": 0,
                        "bess_candidate_count_nodes": 0,
                        "bess_buses_csv": "",
                        "bess_nodes_csv": "",
                        "P_bess_set_total_kw": 0.0,
                        "Q_bess_set_total_kvar": 0.0,
                        "m_loadshape": float(mL[t]),
                        "m_irradshape": 0.0,
                        "P_load_time_kw": float(totals["P_load_time_kw"]),
                        "Q_load_time_kvar": float(totals["Q_load_time_kvar"]),
                        "P_pv_time_kw": 0.0,
                        "p_load_kw_set_total": float(totals["p_load_kw_set_total"]),
                        "q_load_kvar_set_total": float(totals["q_load_kvar_set_total"]),
                        "p_pv_pmpp_kw_set_total": 0.0,
                        "prof_load": float(prof_load[t]),
                        "prof_net": float(prof_net[t]),
                        "P_load_sum_post_kw": float(p_load_post_kw),
                        "Q_load_sum_post_kvar": float(q_load_post_kvar),
                        "P_loss_total_post_kw": float(p_loss_post_kw),
                        "Q_loss_total_post_kvar": float(q_loss_post_kvar),
                        "P_grid_upstream_post_kw": float(p_grid_post_kw),
                        "Q_grid_upstream_post_kvar": float(q_grid_post_kvar),
                        "safe_vmin_pu": float(vmin_safe_pu),
                        "safe_vmax_pu": float(vmax_safe_pu),
                        "n_v_outside_safe_band": int(n_outside),
                        "n_v_below_safe_band": int(n_below),
                        "n_v_above_safe_band": int(n_above),
                        **dummy_meta,
                    }
                )

                rows_node_this_sample: list[dict] = []
                for n in node_names_all:
                    bus, phs = n.split(".")
                    ph = int(phs)
                    vm, va = vdict_m.get(n, (np.nan, np.nan))
                    rows_node_this_sample.append(
                        {
                            "sample_id": sample_id,
                            "node": n,
                            "node_idx": int(node_to_idx_all[n]),
                            "bus": bus,
                            "phase": int(ph),
                            "p_load_kw": float(busphP_load.get((bus, ph), 0.0)),
                            "q_load_kvar": float(busphQ_load.get((bus, ph), 0.0)),
                            "p_pv_kw": 0.0,
                            "p_bess_kw": 0.0,
                            "q_bess_kvar": 0.0,
                            "vmag_pu": float(vm),
                            "vang_deg": float(va),
                        }
                    )
                node_writer.writerows(rows_node_this_sample)
                n_node_rows_written += len(rows_node_this_sample)
                sample_id += 1

                if (k_t % 5 == 0) or (k_t == total_times_this_s):
                    print(
                        f"[scenario {s+1}/{n_scenarios}] progress {k_t}/{total_times_this_s} "
                        f"(global kept_samples={sample_id})",
                        flush=True,
                    )

            pct_out = 100.0 * outside_band_this_scenario / max(finite_v_count_this_scenario, 1)
            top_off = sorted(offender_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_off_str = ", ".join([f"{k}:{v}" for k, v in top_off]) if top_off else "none"
            print(
                f"[scenario {s+1}/{n_scenarios}] kept_samples={sample_id} "
                f"nonconv_this_s={nonconv_this_scenario} badV_this_s={badv_this_scenario} "
                f"v_outside_band_this_s={outside_band_this_scenario} "
                f"(below={below_band_this_scenario}, above={above_band_this_scenario}, pct={pct_out:.2f}%) "
                f"N_nodes={len(node_names_all)} top_offenders=[{top_off_str}] "
                f"skip_nonconv_total={skipped_nonconv} skip_badV_total={skipped_bad_v} "
                f"elapsed_s={time.time()-t0_s:.1f}"
            )

    df_sample = pd.DataFrame(rows_sample)
    df_sample.to_csv(SAMPLE_CSV, index=False)

    if write_mvagg_compat:
        _write_mvagg_compat_from_raw(NODE_CSV, MVAGG_CSV)
        print(f"[diag] wrote compat mvagg (identity, no aggregation): {MVAGG_CSV}")
        if delete_raw_node_csv_after_mvagg and NODE_CSV.is_file():
            NODE_CSV.unlink()
            print(f"[diag] deleted raw node CSV: {NODE_CSV.name}")

    df_node = pd.read_csv(NODE_CSV) if (return_node_df and NODE_CSV.is_file()) else pd.DataFrame()

    print("\n[ORIGINAL-STYLE 906 LVTESTCASE DATASET] saved.")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  sample_meta: {SAMPLE_CSV}")
    print(f"  node_features_targets: {NODE_CSV if NODE_CSV.is_file() else '(deleted after mvagg)'}")
    print(f"  mvagg: {MVAGG_CSV if write_mvagg_compat else '(skipped)'}")
    print(f"  kept samples: {df_sample['sample_id'].nunique() if len(df_sample) else 0}")
    print(f"  skipped_nonconv={skipped_nonconv} skipped_badV={skipped_bad_v}")
    print(
        f"  safe_band=[{float(vmin_safe_pu):.3f}, {float(vmax_safe_pu):.3f}] "
        f"total_v_outside_safe_band={int(total_v_outside_band)}"
    )
    print(f"  node_rows_written={int(n_node_rows_written)}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_original_style_dataset_906_lvtestcase(
        n_scenarios=2,
        k_snapshots_per_scenario_total=8,
        bins_by_profile={"load": 3, "pv": 3, "net": 3},
        include_anchors=True,
        master_seed=90620230,
        sigma_load=0.5,
        p_load_mean_kw=P_LOAD_MEAN_DEFAULT_KW,
        q_load_mean_kvar=Q_LOAD_MEAN_DEFAULT_KVAR,
        p_load_scale_range=(0.3, 1.8),
        q_load_scale_range=(0.3, 1.8),
        vmin_safe_pu=0.55,
        vmax_safe_pu=1.45,
        include_source_in_safe_band=False,
        return_node_df=False,
        write_mvagg_compat=True,
        delete_raw_node_csv_after_mvagg=False,
    )
