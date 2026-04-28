"""
Original-style node feature/target dataset generation on IEEE 8500 unbalanced PV model.

Goal:
  - Keep the core behavior from run_original_dataset.py:
      * scenario sampling (P_load/Q_load/P_pv + noise)
      * time selection using 3 profiles (load, pv, net)
      * per-node features: p_load_kw, q_load_kvar, p_pv_kw, q_pv_kvar
      * targets: vmag_pu, vang_deg
  - Save artifacts in the same structure/style as run_daily_aggregate_dataset_8500.py:
      datasets_gnn2/<name>/
        - gnn_node_index_master.csv
        - gnn_edges_phase_static.csv
        - gnn_sample_meta.csv
        - gnn_node_features_and_targets.csv
"""
from __future__ import annotations

import csv
import importlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

import run_injection_dataset as inj
import run_loadtype_dataset as lt_dist
import run_loadtype_dataset_8500 as lt8500

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
lt8500 = importlib.reload(lt8500)

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

MODEL_DIR = REPO_ROOT / "8500 nodes with solar unbalanced"
MASTER_DSS = MODEL_DIR / "Master-PV2MW-inv.dss"
OUT_DIR = REPO_ROOT / "datasets_gnn2" / "original_8500_unbalanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CSV = OUT_DIR / "gnn_edges_phase_static.csv"
NODE_CSV = OUT_DIR / "gnn_node_features_and_targets.csv"
SAMPLE_CSV = OUT_DIR / "gnn_sample_meta.csv"
NODE_INDEX_CSV = OUT_DIR / "gnn_node_index_master.csv"

NPTS = 288
STEP_MIN = 5


def _is_x_or_sx_bus(bus_name: str) -> bool:
    b = str(bus_name).strip().lower()
    return b.startswith("x") or b.startswith("sx")


def _is_source_bus(bus_name: str) -> bool:
    return str(bus_name).strip().lower().startswith("sourcebus")


def _is_hv_side_sub_bus(bus_name: str) -> bool:
    return str(bus_name).strip().lower().startswith("hvmv_sub_hsb")


def _filter_out_x_sx_nodes(node_names: list[str]) -> list[str]:
    out: list[str] = []
    for n in node_names:
        bus = str(n).split(".")[0]
        if _is_x_or_sx_bus(bus) or _is_source_bus(bus) or _is_hv_side_sub_bus(bus):
            continue
        out.append(str(n))
    return out


def _normalize_bus_name(bus_spec: str) -> str:
    return str(bus_spec).split(".")[0].strip()


def _compile_8500_unbalanced_daily_setup() -> None:
    if not MASTER_DSS.is_file():
        raise FileNotFoundError(f"Missing master DSS entrypoint: {MASTER_DSS}")
    dss.Basic.ClearAll()
    # Ensure nested Redirect paths resolve inside model folder.
    dss.Text.Command(f'cd "{os.path.abspath(str(MODEL_DIR))}"')
    dss.Text.Command(f'redirect "{os.path.abspath(str(MASTER_DSS))}"')
    # Use explicit 5-minute daily profile for loads.
    dayshape_csv = MODEL_DIR / "5minDayShape.csv"
    if not dayshape_csv.is_file():
        raise FileNotFoundError(f"Missing loadshape CSV: {dayshape_csv}")
    dss.Text.Command(
        f'New Loadshape.Day5min npts=288 interval=0.0833333333333333 mult=(file="{os.path.abspath(str(dayshape_csv))}", col=2, header=no)'
    )
    dss.Text.Command("BatchEdit Load..* Daily=Day5min")
    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=5m")
    dss.Text.Command("set number=1")
    dss.Text.Command("set maxiterations=30")
    # Keep control-loop budget very modest so hard points fail fast and are skipped.
    dss.Text.Command("set maxcontroliter=30")


def _detach_daily_loadshape_from_loads() -> None:
    # We set kW/kvar explicitly per timestamp; keeping Daily attached would double scale.
    if not dss.Loads.First():
        return
    while True:
        nm = dss.Loads.Name()
        dss.Loads.Name(nm)
        dss.Loads.Daily("")
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


def _collect_pv_maps() -> tuple[list[str], np.ndarray, dict[str, list[tuple[str, int, float]]]]:
    names: list[str] = []
    pmpp0: list[float] = []
    pv_to_busph: dict[str, list[tuple[str, int, float]]] = {}
    if not dss.PVsystems.First():
        return names, np.array([], dtype=float), pv_to_busph
    while True:
        name = str(dss.PVsystems.Name())
        dss.PVsystems.Name(name)
        names.append(name)
        pmpp0.append(float(dss.PVsystems.Pmpp()))

        buses = dss.CktElement.BusNames()
        bus1 = str(buses[0]).split(".")[0] if buses else ""
        nph = int(dss.CktElement.NumPhases())
        phases = list(range(1, max(1, nph) + 1))
        w = 1.0 / float(len(phases))
        pv_to_busph[name] = [(bus1, ph, w) for ph in phases]
        if not dss.PVsystems.Next():
            break
    return names, np.asarray(pmpp0, dtype=float), pv_to_busph


def _collect_bess_candidate_nodes_from_mv_load_transformers(node_names_all: list[str]) -> list[str]:
    """
    Candidate BESS nodes:
      - phase nodes in MV feeder
      - electrically on MV side buses of load transformers
    Heuristic for load transformer: at least one LV winding (<=1.0 kV) and one MV winding (>1.0 and <=40 kV).
    """
    # 1) Build node lookup by bus.
    bus_to_nodes: dict[str, list[str]] = {}
    for n in node_names_all:
        bus = _normalize_bus_name(n).lower()
        bus_to_nodes.setdefault(bus, []).append(str(n).lower())

    # 2) MV load buses from active load element terminal buses (more reliable than Loads.Bus1 API).
    mv_load_buses: set[str] = set()
    try:
        for nm in dss.Loads.AllNames():
            name = str(nm)
            dss.Circuit.SetActiveElement(f"Load.{name}")
            bnames = list(dss.CktElement.BusNames())
            if not bnames:
                continue
            bus = _normalize_bus_name(str(bnames[0])).lower()
            kvb = np.nan
            try:
                dss.Circuit.SetActiveBus(bus)
                kvb = float(dss.Bus.kVBase())
            except Exception:
                kvb = np.nan
            if np.isfinite(kvb) and (kvb > 1.0) and (kvb <= 40.0):
                mv_load_buses.add(bus)
    except Exception:
        mv_load_buses = set()

    # 3) Transformer-adjacent buses from transformer terminal buses.
    tx_buses: set[str] = set()
    try:
        for tx_name in dss.Transformers.AllNames():
            name = str(tx_name)
            if not name or name.lower() == "none":
                continue
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            for b in list(dss.CktElement.BusNames()):
                tx_buses.add(_normalize_bus_name(str(b)).lower())
    except Exception:
        tx_buses = set()

    # 4) Candidate buses = MV load buses that are transformer-adjacent.
    cand_buses = mv_load_buses.intersection(tx_buses)

    # 5) Expand to phase nodes (1/2/3) available on those buses.
    candidates: list[str] = []
    for bus in sorted(cand_buses):
        for n in sorted(bus_to_nodes.get(bus, [])):
            try:
                ph = int(str(n).rsplit(".", 1)[1])
            except Exception:
                continue
            if ph in (1, 2, 3):
                candidates.append(str(n))
    return sorted(list(dict.fromkeys(candidates)))


def _install_bess_generators(selected_nodes: list[str]) -> dict[str, str]:
    """
    Install one single-phase generator element per selected node.
    Returns map node -> generator DSS name.
    """
    out: dict[str, str] = {}
    for i, node in enumerate(selected_nodes):
        bus, phs = str(node).split(".")
        ph = int(phs)
        try:
            dss.Circuit.SetActiveBus(bus)
            kvbase = float(dss.Bus.kVBase())
        except Exception:
            kvbase = np.nan
        if not np.isfinite(kvbase) or kvbase <= 0:
            kvbase = 7.2
        gname = f"BESS_{i+1:03d}"
        dss.Text.Command(
            f"New Generator.{gname} phases=1 bus1={bus}.{ph} conn=wye model=1 kV={kvbase:.6f} kW=0 kvar=0"
        )
        out[str(node)] = gname
    return out


def _candidate_three_phase_buses_from_nodes(candidate_nodes: list[str]) -> dict[str, list[str]]:
    """
    Build {bus: [bus.1, bus.2, bus.3]} from candidate node list.
    Only buses with all three phases available are kept.
    """
    bus_to_phase_nodes: dict[str, dict[int, str]] = {}
    for node in candidate_nodes:
        s = str(node).strip().lower()
        if "." not in s:
            continue
        bus, phs = s.rsplit(".", 1)
        try:
            ph = int(phs)
        except Exception:
            continue
        if ph not in (1, 2, 3):
            continue
        if bus not in bus_to_phase_nodes:
            bus_to_phase_nodes[bus] = {}
        bus_to_phase_nodes[bus][ph] = f"{bus}.{ph}"

    out: dict[str, list[str]] = {}
    for bus in sorted(bus_to_phase_nodes.keys()):
        ph_map = bus_to_phase_nodes[bus]
        if all(p in ph_map for p in (1, 2, 3)):
            out[bus] = [ph_map[1], ph_map[2], ph_map[3]]
    return out


def _discover_reg_controls() -> list[str]:
    reg_names: list[str] = []
    try:
        if dss.RegControls.First():
            while True:
                reg_names.append(str(dss.RegControls.Name()))
                if not dss.RegControls.Next():
                    break
    except Exception:
        reg_names = []
    return sorted(reg_names)


def _discover_capacitors() -> list[str]:
    cap_names: list[str] = []
    try:
        if dss.Capacitors.First():
            while True:
                cap_names.append(str(dss.Capacitors.Name()))
                if not dss.Capacitors.Next():
                    break
    except Exception:
        cap_names = []
    return sorted(cap_names)


def _read_reg_control_state(reg_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for nm in reg_names:
        tap_val = np.nan
        try:
            dss.RegControls.Name(nm)
            xfmr = str(dss.RegControls.Transformer())
            wdg = int(dss.RegControls.Winding())
            if xfmr:
                dss.Transformers.Name(xfmr)
                dss.Transformers.Wdg(wdg)
                tap_val = float(dss.Transformers.Tap())
        except Exception:
            tap_val = np.nan
        out[f"reg_{nm}_tap_pu"] = float(tap_val) if np.isfinite(tap_val) else np.nan
    return out


def _read_capacitor_sample_fields(cap_names: list[str]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for nm in cap_names:
        steps: list[int] = []
        try:
            dss.Capacitors.Name(nm)
            st = dss.Capacitors.States()
            if st is None:
                steps = []
            elif isinstance(st, (list, tuple, np.ndarray)):
                steps = [int(x) for x in st]
            else:
                steps = [int(st)]
        except Exception:
            steps = []

        n_on = int(sum(1 for x in steps if int(x) > 0))
        q_nom = np.nan
        try:
            dss.Capacitors.Name(nm)
            q_nom = float(dss.Capacitors.kvar())
        except Exception:
            pass

        q_post = np.nan
        try:
            dss.Circuit.SetActiveElement(f"Capacitor.{nm}")
            pwr = dss.CktElement.TotalPowers()
            if pwr is not None and len(pwr) >= 2:
                q_post = float(-float(pwr[1]))
        except Exception:
            pass

        out[f"cap_{nm}_n_steps_on"] = n_on
        out[f"cap_{nm}_q_nominal_kvar"] = float(q_nom) if np.isfinite(q_nom) else np.nan
        out[f"cap_{nm}_q_post_kvar"] = float(q_post) if np.isfinite(q_post) else np.nan
    return out


def _sum_loads_post_solve_kw_kvar() -> tuple[float, float]:
    p_sum, q_sum = 0.0, 0.0
    if not dss.Loads.First():
        return p_sum, q_sum
    while True:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        pwr = dss.CktElement.TotalPowers()
        if pwr is not None and len(pwr) >= 2:
            p_sum += float(pwr[0])
            q_sum += float(pwr[1])
        if not dss.Loads.Next():
            break
    return p_sum, q_sum


def _circuit_losses_kw_kvar() -> tuple[float, float]:
    loss = dss.Circuit.Losses()
    p_l, q_l = float(loss[0]), float(loss[1])
    if abs(p_l) > 1000.0 or abs(q_l) > 1000.0:
        p_l /= 1000.0
        q_l /= 1000.0
    return p_l, q_l


def _grid_upstream_post_kw_kvar() -> tuple[float, float]:
    pwr = dss.Circuit.TotalPower()
    return -float(pwr[0]), -float(pwr[1])


def _read_pv_totals_post_solve_kw_kvar(pv_names: list[str]) -> dict[str, tuple[float, float]]:
    """Read post-solve actual PV injections per PVSystem name: (+P,+Q)=into feeder."""
    out: dict[str, tuple[float, float]] = {}
    for name in pv_names:
        p_inj, q_inj = 0.0, 0.0
        try:
            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            pwr = dss.CktElement.TotalPowers()
            if pwr is not None and len(pwr) >= 2:
                p_inj = -float(pwr[0])  # OpenDSS generation sign -> injection sign
                q_inj = -float(pwr[1])
        except Exception:
            pass
        out[str(name)] = (float(p_inj), float(q_inj))
    return out


def _node_base_kv_map(node_names: list[str]) -> dict[str, float]:
    """Return node -> bus base kV (line-to-neutral base as reported by OpenDSS Bus.kVBase)."""
    out: dict[str, float] = {}
    for n in node_names:
        bus = str(n).split(".")[0]
        kvb = np.nan
        try:
            dss.Circuit.SetActiveBus(bus)
            kvb = float(dss.Bus.kVBase())
        except Exception:
            kvb = np.nan
        out[str(n)] = float(kvb) if np.isfinite(kvb) else np.nan
    return out


def _line_length_to_km(length_value: float, units_code: int) -> float:
    # OpenDSS length-units enum (common):
    # 0=none, 1=mi, 2=kft, 3=km, 4=m, 5=ft, 6=in, 7=cm, 8=mm
    conv = {
        1: 1.609344,     # mile -> km
        2: 0.3048,       # kft -> km
        3: 1.0,          # km -> km
        4: 0.001,        # m -> km
        5: 0.0003048,    # ft -> km
        6: 2.54e-5,      # in -> km
        7: 1.0e-5,       # cm -> km
        8: 1.0e-6,       # mm -> km
    }
    f = conv.get(int(units_code), 1.0)  # 0/unknown -> leave unchanged
    return float(length_value) * float(f)


def _unit_code_to_km_factor(units_code: int) -> float:
    conv = {
        1: 1.609344,     # mile -> km
        2: 0.3048,       # kft -> km
        3: 1.0,          # km -> km
        4: 0.001,        # m -> km
        5: 0.0003048,    # ft -> km
        6: 2.54e-5,      # in -> km
        7: 1.0e-5,       # cm -> km
        8: 1.0e-6,       # mm -> km
    }
    return float(conv.get(int(units_code), 1.0))


def _line_units_name(units_code: int) -> str:
    names = {
        0: "none_or_default",
        1: "mi",
        2: "kft",
        3: "km",
        4: "m",
        5: "ft",
        6: "in",
        7: "cm",
        8: "mm",
    }
    return names.get(int(units_code), "unknown")


def _enrich_edges_with_basekv_and_length_km(edge_csv_path: Path, node_names_master: list[str]) -> None:
    df_edges = pd.read_csv(edge_csv_path)

    # Node endpoint base-kV metadata.
    node_to_base_kv = _node_base_kv_map(node_names_master)
    df_edges["from_base_kv"] = df_edges["from_node"].map(node_to_base_kv)
    df_edges["to_base_kv"] = df_edges["to_node"].map(node_to_base_kv)

    # Normalize line-edge length to km (transformer rows remain synthetic).
    line_names = sorted(
        {
            str(x)
            for x in df_edges["line_name"].dropna().astype(str).unique().tolist()
            if str(x).startswith("Line.")
        }
    )
    line_len_km: dict[str, float] = {}
    line_units_name: dict[str, str] = {}
    per_len_units_code: dict[str, int] = {}
    for elem in line_names:
        ln = elem.split(".", 1)[1]
        try:
            dss.Lines.Name(ln)
            length_raw = float(dss.Lines.Length())
            line_u = int(dss.Lines.Units())
            line_len_km[elem] = _line_length_to_km(length_raw, line_u)
            line_units_name[elem] = _line_units_name(line_u)

            # Per-length impedance units usually come from LineCode units when defined.
            per_u = int(line_u)
            lc = str(dss.Lines.LineCode()).strip()
            if lc:
                try:
                    dss.LineCodes.Name(lc)
                    lc_u = int(dss.LineCodes.Units())
                    if lc_u != 0:
                        per_u = lc_u
                except Exception:
                    pass
            per_len_units_code[elem] = int(per_u)
        except Exception:
            continue

    # Keep only one unit column in output, as requested.
    df_edges["length_unit"] = df_edges["line_name"].map(line_units_name).fillna("synthetic_or_unknown")

    length_km_series = df_edges["line_name"].map(line_len_km)

    # Keep backward-compatible column name but make values unit-consistent for line rows.
    is_line_row = df_edges["line_name"].astype(str).str.startswith("Line.")
    df_edges.loc[is_line_row & length_km_series.notna(), "length"] = length_km_series.loc[
        is_line_row & length_km_series.notna()
    ]

    # Recompute full R/X/C with consistent units for line rows.
    # R_per_len/X_per_len/C_per_len are in "per_len_units"; convert via km factors.
    line_rows = is_line_row & length_km_series.notna()
    if line_rows.any():
        per_units_series = df_edges.loc[line_rows, "line_name"].map(per_len_units_code).fillna(0).astype(int)
        km_per_perlen = per_units_series.map(_unit_code_to_km_factor).astype(float)
        # Convert per-unit-length quantities to per-km, then multiply by length_km.
        r_per_km = df_edges.loc[line_rows, "R_per_len"].astype(float) / km_per_perlen
        x_per_km = df_edges.loc[line_rows, "X_per_len"].astype(float) / km_per_perlen
        c_per_km = df_edges.loc[line_rows, "C_per_len"].astype(float) / km_per_perlen
        L_km = length_km_series.loc[line_rows].astype(float)
        df_edges.loc[line_rows, "R_full"] = r_per_km * L_km
        df_edges.loc[line_rows, "X_full"] = x_per_km * L_km
        df_edges.loc[line_rows, "C_full"] = c_per_km * L_km

    # Canonical tiny connector values requested for all none_or_default line-unit rows.
    none_mask = df_edges["length_unit"].astype(str) == "none_or_default"
    if none_mask.any():
        df_edges.loc[none_mask, "length"] = 0.001
        df_edges.loc[none_mask, "R_per_len"] = 0.001
        df_edges.loc[none_mask, "X_per_len"] = 0.01
        df_edges.loc[none_mask, "C_per_len"] = 0.0
        df_edges.loc[none_mask, "R_full"] = 1.0e-6
        df_edges.loc[none_mask, "X_full"] = 1.0e-5
        df_edges.loc[none_mask, "C_full"] = 0.0

    df_edges.to_csv(edge_csv_path, index=False)


def _compute_laplacian_pe_from_edges(
    *,
    node_names: list[str],
    edge_csv_path: Path,
    k: int,
    seed: int = 42,
    zero_eig_tol: float = 1e-8,
) -> np.ndarray:
    """
    Compute normalized Laplacian eigenvector PE with shape (N, k).
    Uses edge weight w = 1 / sqrt(R_full^2 + X_full^2), with floor at 1e-6.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(node_names)
    if n <= 1:
        raise ValueError("Need at least 2 nodes to compute PE.")
    if k >= n:
        raise ValueError(f"node_pe_k must be < N_nodes. Got k={k}, N={n}.")
    if zero_eig_tol <= 0:
        raise ValueError("zero_eig_tol must be > 0")

    node_to_local = {str(nm): i for i, nm in enumerate(node_names)}
    df = pd.read_csv(edge_csv_path)
    src_col = "from_node"
    dst_col = "to_node"
    r_col = "R_full"
    x_col = "X_full"
    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError("Edge CSV missing from_node/to_node columns for PE computation.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for _, row in df.iterrows():
        u_name = str(row[src_col]).strip()
        v_name = str(row[dst_col]).strip()
        if u_name not in node_to_local or v_name not in node_to_local:
            continue
        u = int(node_to_local[u_name])
        v = int(node_to_local[v_name])
        r = float(row[r_col]) if (r_col in df.columns and pd.notna(row[r_col])) else 0.0
        x = float(row[x_col]) if (x_col in df.columns and pd.notna(row[x_col])) else 0.0
        z = float(np.sqrt(r * r + x * x))
        w = 1.0 / max(z, 1e-6)
        rows.append(u)
        cols.append(v)
        data.append(w)

    if not data:
        raise RuntimeError("No usable edges for PE computation.")

    w_dir = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Make undirected/symmetric affinity.
    w = (w_dir + w_dir.T) * 0.5
    deg = np.asarray(w.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    d_inv_sqrt_mat = diags(d_inv_sqrt)
    l_unnorm = diags(deg) - w
    l_norm = d_inv_sqrt_mat @ l_unnorm @ d_inv_sqrt_mat

    # Solve a few extra low-end modes to robustly skip near-zero components.
    k_solve = min(n - 1, max(k + 1, k + 16))
    np.random.seed(int(seed))
    eigvals, eigvecs = eigsh(
        l_norm,
        k=int(k_solve),
        sigma=1e-8,
        which="LM",
        tol=1e-6,
        maxiter=50_000,
        ncv=min(n - 1, max(6 * int(k_solve) + 1, 60)),
    )
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    nontrivial_idx = np.where(np.abs(eigvals) > float(zero_eig_tol))[0]
    if nontrivial_idx.size < int(k):
        raise RuntimeError(
            f"Not enough nontrivial eigenpairs above tol={zero_eig_tol}. "
            f"Found {nontrivial_idx.size}, need {k}."
        )
    keep_idx = nontrivial_idx[: int(k)]
    pe = eigvecs[:, keep_idx]
    pe = (pe - pe.mean(axis=0, keepdims=True)) / (pe.std(axis=0, keepdims=True) + 1e-8)
    return pe.astype(np.float32)


def _apply_snapshot_with_pv(
    *,
    load_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    load_to_busph: dict[str, list[tuple[str, int, float]]],
    pv_names: list[str],
    base_pmpp: np.ndarray,
    pv_to_busph: dict[str, list[tuple[str, int, float]]],
    p_load_total_kw: float,
    q_load_total_kvar: float,
    p_pv_total_kw: float,
    m_load_t: float,
    m_pv_t: float,
    sigma_load: float,
    sigma_pv: float,
    rng: np.random.Generator,
) -> tuple[dict, dict, dict, dict, dict]:
    p_load_t = float(p_load_total_kw) * float(m_load_t)
    q_load_t = float(q_load_total_kvar) * float(m_load_t)

    base_p_sum = float(np.sum(base_kw))
    base_q_sum = float(np.sum(base_kvar))
    s_load_p = (p_load_t / base_p_sum) if base_p_sum > 0 else 1.0
    s_load_q = (q_load_t / base_q_sum) if base_q_sum > 0 else 1.0

    noise_p = np.maximum(0.0, 1.0 + rng.normal(0.0, float(sigma_load), size=len(load_names)))
    noise_q = np.maximum(0.0, 1.0 + rng.normal(0.0, float(sigma_load), size=len(load_names)))
    kw_set = base_kw * s_load_p * noise_p
    kvar_set = base_kvar * s_load_q * noise_q

    busphP_load: dict[tuple[str, int], float] = {}
    busphQ_load: dict[tuple[str, int], float] = {}
    for i, name in enumerate(load_names):
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw_set[i]))
        dss.Loads.kvar(float(kvar_set[i]))
        for (bus, ph, w) in load_to_busph.get(name, []):
            busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + float(kw_set[i]) * float(w)
            busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + float(kvar_set[i]) * float(w)

    # Set PV Pmpp scaling; OpenDSS daily irradiance shape sets time-varying output at solve.
    base_pmpp_sum = float(np.sum(base_pmpp))
    pmpp_scale = (float(p_pv_total_kw) / base_pmpp_sum) if base_pmpp_sum > 0 else 1.0
    noise_pv = np.maximum(0.0, 1.0 + rng.normal(0.0, float(sigma_pv), size=len(pv_names)))
    pmpp_set = base_pmpp * pmpp_scale * noise_pv
    for i, name in enumerate(pv_names):
        dss.PVsystems.Name(name)
        dss.PVsystems.Pmpp(float(pmpp_set[i]))

    # Nominal bus-phase PV P using profile multiplier; actual Q captured post-solve.
    busphP_pv_nominal: dict[tuple[str, int], float] = {}
    busphQ_pv_nominal: dict[tuple[str, int], float] = {}
    for i, name in enumerate(pv_names):
        p_nominal = float(pmpp_set[i]) * float(m_pv_t)
        for (bus, ph, w) in pv_to_busph.get(name, []):
            busphP_pv_nominal[(bus, ph)] = busphP_pv_nominal.get((bus, ph), 0.0) + p_nominal * float(w)
            busphQ_pv_nominal[(bus, ph)] = busphQ_pv_nominal.get((bus, ph), 0.0) + 0.0

    totals = {
        "P_load_time_kw": p_load_t,
        "Q_load_time_kvar": q_load_t,
        "P_pv_time_kw": float(p_pv_total_kw) * float(m_pv_t),
        "p_load_kw_set_total": float(np.sum(kw_set)),
        "q_load_kvar_set_total": float(np.sum(kvar_set)),
        "p_pv_pmpp_kw_set_total": float(np.sum(pmpp_set)),
    }
    return totals, busphP_load, busphQ_load, busphP_pv_nominal, busphQ_pv_nominal


def generate_original_style_dataset_8500_unbalanced(
    *,
    n_scenarios: int = 200,
    k_snapshots_per_scenario_total: int = 960,
    bins_by_profile: dict | None = None,
    include_anchors: bool = True,
    master_seed: int = 20260130,
    sigma_load: float = 0.03,
    sigma_pv: float = 0.03,
    bess_total_mva_mean: float = 4.0,
    bess_total_mva_sigma: float = 0.05,
    bess_num_nodes_min: int = 1,
    bess_num_nodes_max: int = 8,
    bess_q_frac_max: float = 0.44,
    bess_candidate_nodes_override: list[str] | None = None,
    node_pe_k: int = 0,
    node_pe_seed: int = 42,
    node_pe_zero_eig_tol: float = 1e-8,
    node_pe_from_csv: str | None = None,
    node_pe_save_csv: str | None = None,
    p_load_mean_kw: float = 13731.9,
    q_load_mean_kvar: float = 2610.15,
    p_load_scale_range: tuple[float, float] = (0.7, 1.3),
    q_load_scale_range: tuple[float, float] = (0.7, 1.3),
    p_pv_scale_range: tuple[float, float] = (0.7, 1.3),
    vmin_safe_pu: float = 0.85,
    vmax_safe_pu: float = 1.15,
    include_source_in_safe_band: bool = True,
    return_node_df: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

    if k_snapshots_per_scenario_total < 1:
        raise ValueError("k_snapshots_per_scenario_total must be >= 1")
    if not (0.0 < float(vmin_safe_pu) < float(vmax_safe_pu)):
        raise ValueError(f"Invalid safe voltage band: [{vmin_safe_pu}, {vmax_safe_pu}]")
    if float(sigma_load) < 0.0 or float(sigma_pv) < 0.0:
        raise ValueError("sigma_load and sigma_pv must be non-negative.")
    if float(bess_total_mva_mean) < 0.0 or float(bess_total_mva_sigma) < 0.0:
        raise ValueError("bess_total_mva_mean and bess_total_mva_sigma must be non-negative.")
    if int(bess_num_nodes_min) < 1 or int(bess_num_nodes_max) < int(bess_num_nodes_min):
        raise ValueError("Invalid BESS three-phase bus-count range.")
    if not (0.0 <= float(bess_q_frac_max) <= 1.0):
        raise ValueError("bess_q_frac_max must be within [0, 1].")
    if int(node_pe_k) < 0:
        raise ValueError("node_pe_k must be >= 0")
    if float(p_load_mean_kw) <= 0.0 or float(q_load_mean_kvar) <= 0.0:
        raise ValueError("p_load_mean_kw and q_load_mean_kvar must be positive.")

    _compile_8500_unbalanced_daily_setup()
    _detach_daily_loadshape_from_loads()

    node_names_all, _, _, _ = inj.get_all_bus_phase_nodes()
    node_names_graph = _filter_out_x_sx_nodes(node_names_all)
    if not node_names_graph:
        raise RuntimeError("No nodes left after filtering x*/sx* buses for graph artifacts.")
    node_to_idx_all = {n: i for i, n in enumerate(node_names_all)}
    node_set_all_lower = {str(n).lower() for n in node_names_all}
    if bess_candidate_nodes_override is None:
        raise ValueError("bess_candidate_nodes_override must be provided (manual BESS candidate list).")
    user_nodes_raw = [str(x).strip().lower() for x in bess_candidate_nodes_override if str(x).strip()]
    user_nodes_dedup = list(dict.fromkeys(user_nodes_raw))
    bess_candidate_nodes = [n for n in user_nodes_dedup if n in node_set_all_lower]
    n_bad = int(len(user_nodes_dedup) - len(bess_candidate_nodes))
    print(f"[diag] BESS candidate nodes (manual list): {len(bess_candidate_nodes)} valid, {n_bad} invalid")
    if not bess_candidate_nodes:
        raise ValueError("No valid BESS candidate nodes remain after validation against circuit nodes.")
    bess_candidate_bus_to_nodes = _candidate_three_phase_buses_from_nodes(bess_candidate_nodes)
    if not bess_candidate_bus_to_nodes:
        raise ValueError(
            "No full three-phase BESS candidate buses found. Provide candidates containing .1/.2/.3 for each bus."
        )
    dropped_non_3ph = len(set(n.split(".")[0] for n in bess_candidate_nodes)) - len(bess_candidate_bus_to_nodes)
    print(
        f"[diag] BESS candidate 3-phase buses: {len(bess_candidate_bus_to_nodes)} "
        f"(dropped_non_3ph_buses={max(0, int(dropped_non_3ph))})"
    )

    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_graph,
        edge_csv_path=str(EDGE_CSV),
        excluded_buses=(),
    )
    # Enrich static edges with endpoint base-kV and normalized length metadata.
    _enrich_edges_with_basekv_and_length_km(EDGE_CSV, node_names_graph)

    node_to_dist = lt_dist._compute_electrical_distance_from_source(node_names_graph, str(EDGE_CSV))
    node_to_base_kv = _node_base_kv_map(node_names_graph)
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
        pe = _compute_laplacian_pe_from_edges(
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
        if (not include_source_in_safe_band) and (b.startswith("sourcebus") or b.startswith("_hvmv_sub")):
            continue
        safe_band_eval_indices.append(i)
    if not safe_band_eval_indices:
        raise RuntimeError("No nodes available for safe-band evaluation.")

    # Use explicit profile files from the unbalanced model directory.
    mL = inj.read_profile_csv_two_col_noheader(str(MODEL_DIR / "5minDayShape.csv"), npts=NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(str(MODEL_DIR / "irr_day_001.csv"), npts=NPTS, debug=False)

    load_names, base_kw, base_kvar, load_to_busph = _collect_loads_and_maps()
    pv_names, base_pmpp, pv_to_busph = _collect_pv_maps()
    if len(load_names) == 0:
        raise RuntimeError("No loads found in unbalanced 8500 model.")
    if len(pv_names) == 0:
        raise RuntimeError("No PV systems found in unbalanced 8500 model.")

    reg_names = _discover_reg_controls()
    cap_names = _discover_capacitors()

    base_p_load = float(np.sum(base_kw))
    base_q_load = float(np.sum(base_kvar))
    base_p_pv = float(np.sum(base_pmpp))

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[dict] = []
    sample_id = 0
    skipped_nonconv = 0
    skipped_bad_v = 0
    total_v_outside_band = 0
    n_node_rows_written = 0

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
            _compile_8500_unbalanced_daily_setup()
            _detach_daily_loadshape_from_loads()

            load_names, base_kw, base_kvar, load_to_busph = _collect_loads_and_maps()
            pv_names, base_pmpp, pv_to_busph = _collect_pv_maps()

            # Use explicit user-controlled means as scenario baselines, then perturb by scale ranges.
            p_load = float(p_load_mean_kw) * float(rng_master.uniform(*p_load_scale_range))
            q_load = float(q_load_mean_kvar) * float(rng_master.uniform(*q_load_scale_range))
            p_pv = base_p_pv * float(rng_master.uniform(*p_pv_scale_range))
            if bess_candidate_bus_to_nodes:
                n_bess = int(
                    rng_master.integers(
                        int(bess_num_nodes_min),
                        int(min(int(bess_num_nodes_max), len(bess_candidate_bus_to_nodes))) + 1,
                    )
                )
                selected_bess_buses = [str(x) for x in rng_master.choice(sorted(bess_candidate_bus_to_nodes.keys()), size=n_bess, replace=False)]
                selected_bess_nodes = [
                    node for b in selected_bess_buses for node in bess_candidate_bus_to_nodes.get(b, [])
                ]
            else:
                n_bess = 0
                selected_bess_buses = []
                selected_bess_nodes = []
            bess_total_mva_s = float(
                max(0.0, float(bess_total_mva_mean) * (1.0 + float(rng_master.normal(0.0, float(bess_total_mva_sigma)))))
            )
            bess_total_kva_s = bess_total_mva_s * 1000.0
            bess_s_rated_kva_per_bus = (bess_total_kva_s / float(n_bess)) if n_bess > 0 else 0.0
            bess_s_rated_kva_per_node = (bess_s_rated_kva_per_bus / 3.0) if n_bess > 0 else 0.0
            bess_gen_by_node = _install_bess_generators(selected_bess_nodes) if n_bess > 0 else {}

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
                hr = int(t // 12)
                sec = int((t % 12) * 300)
                dss.Text.Command(f"set hour={hr} sec={sec}")

                # Sample BESS dispatch at this snapshot: random Q in [-qmax, qmax], P from capability.
                busphP_bess: dict[tuple[str, int], float] = {}
                busphQ_bess: dict[tuple[str, int], float] = {}
                busphS_bess: dict[tuple[str, int], float] = {}
                p_bess_total_set_kw = 0.0
                q_bess_total_set_kvar = 0.0
                for bus in selected_bess_buses:
                    phase_nodes = bess_candidate_bus_to_nodes.get(bus, [])
                    if len(phase_nodes) != 3:
                        continue
                    s_bus_kva = float(bess_s_rated_kva_per_bus)
                    q_bus_kvar = float(rng_solve.uniform(-float(bess_q_frac_max), float(bess_q_frac_max)) * s_bus_kva)
                    p_bus_mag_kw = float(np.sqrt(max(0.0, s_bus_kva * s_bus_kva - q_bus_kvar * q_bus_kvar)))
                    p_bus_sign = float(rng_solve.choice(np.array([-1.0, 1.0], dtype=float)))
                    p_bus_kw = float(p_bus_sign * p_bus_mag_kw)
                    p_phase_kw = p_bus_kw / 3.0
                    q_phase_kvar = q_bus_kvar / 3.0
                    s_phase_kva = s_bus_kva / 3.0

                    for node in phase_nodes:
                        bus_i, phs = str(node).split(".")
                        ph = int(phs)
                        gname = bess_gen_by_node.get(node, "")
                        if gname:
                            try:
                                dss.Generators.Name(gname)
                                dss.Generators.kW(float(p_phase_kw))
                                dss.Generators.kvar(float(q_phase_kvar))
                            except Exception:
                                pass
                        busphP_bess[(bus_i, ph)] = busphP_bess.get((bus_i, ph), 0.0) + p_phase_kw
                        busphQ_bess[(bus_i, ph)] = busphQ_bess.get((bus_i, ph), 0.0) + q_phase_kvar
                        busphS_bess[(bus_i, ph)] = busphS_bess.get((bus_i, ph), 0.0) + s_phase_kva
                        p_bess_total_set_kw += p_phase_kw
                        q_bess_total_set_kvar += q_phase_kvar

                totals, busphP_load, busphQ_load, busphP_pv_presolve, _ = _apply_snapshot_with_pv(
                    load_names=load_names,
                    base_kw=base_kw,
                    base_kvar=base_kvar,
                    load_to_busph=load_to_busph,
                    pv_names=pv_names,
                    base_pmpp=base_pmpp,
                    pv_to_busph=pv_to_busph,
                    p_load_total_kw=p_load,
                    q_load_total_kvar=q_load,
                    p_pv_total_kw=p_pv,
                    m_load_t=float(mL[t]),
                    m_pv_t=float(mPV[t]),
                    sigma_load=sigma_load,
                    sigma_pv=sigma_pv,
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

                # Actual per-PV P/Q after solve (includes VoltVar behavior where applicable).
                pv_totals_post = _read_pv_totals_post_solve_kw_kvar(pv_names)
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

                p_load_post_kw, q_load_post_kvar = _sum_loads_post_solve_kw_kvar()
                p_loss_post_kw, q_loss_post_kvar = _circuit_losses_kw_kvar()
                p_grid_post_kw, q_grid_post_kvar = _grid_upstream_post_kw_kvar()
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
                        "sigma_pv": float(sigma_pv),
                        "bess_total_mva_mean": float(bess_total_mva_mean),
                        "bess_total_mva_sigma": float(bess_total_mva_sigma),
                        "bess_total_mva_scenario": float(bess_total_mva_s),
                        "bess_num_nodes": int(n_bess),
                        "bess_num_3ph_buses": int(n_bess),
                        "bess_total_nodes_1ph_equiv": int(len(selected_bess_nodes)),
                        "bess_s_rated_kva_per_bus": float(bess_s_rated_kva_per_bus),
                        "bess_s_rated_kva_per_node": float(bess_s_rated_kva_per_node),
                        "bess_q_frac_max": float(bess_q_frac_max),
                        "bess_candidate_count": int(len(bess_candidate_bus_to_nodes)),
                        "bess_candidate_count_nodes": int(len(bess_candidate_nodes)),
                        "bess_buses_csv": ",".join(selected_bess_buses),
                        "bess_nodes_csv": ",".join(selected_bess_nodes),
                        "P_bess_set_total_kw": float(p_bess_total_set_kw),
                        "Q_bess_set_total_kvar": float(q_bess_total_set_kvar),
                        "m_loadshape": float(mL[t]),
                        "m_irradshape": float(mPV[t]),
                        "P_load_time_kw": float(totals["P_load_time_kw"]),
                        "Q_load_time_kvar": float(totals["Q_load_time_kvar"]),
                        "P_pv_time_kw": float(totals["P_pv_time_kw"]),
                        "p_load_kw_set_total": float(totals["p_load_kw_set_total"]),
                        "q_load_kvar_set_total": float(totals["q_load_kvar_set_total"]),
                        "p_pv_pmpp_kw_set_total": float(totals["p_pv_pmpp_kw_set_total"]),
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
                        **{
                            f"pv_{str(pv_name).lower()}_p_post_kw": float(pv_totals_post.get(pv_name, (0.0, 0.0))[0])
                            for pv_name in pv_names
                        },
                        **{
                            f"pv_{str(pv_name).lower()}_q_post_kvar": float(pv_totals_post.get(pv_name, (0.0, 0.0))[1])
                            for pv_name in pv_names
                        },
                        **_read_reg_control_state(reg_names),
                        **_read_capacitor_sample_fields(cap_names),
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
                            # Requested: PV node feature is pre-solve active power only.
                            "p_pv_kw": float(busphP_pv_presolve.get((bus, ph), 0.0)),
                            "p_bess_kw": float(busphP_bess.get((bus, ph), 0.0)),
                            "q_bess_kvar": float(busphQ_bess.get((bus, ph), 0.0)),
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
    df_node = pd.read_csv(NODE_CSV) if return_node_df else pd.DataFrame()

    print("\n[ORIGINAL-STYLE 8500 UNBALANCED DATASET] saved.")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  sample_meta: {SAMPLE_CSV}")
    print(f"  node_features_targets: {NODE_CSV}")
    print(f"  kept samples: {df_sample['sample_id'].nunique() if len(df_sample) else 0}")
    print(f"  skipped_nonconv={skipped_nonconv} skipped_badV={skipped_bad_v}")
    print(
        f"  safe_band=[{float(vmin_safe_pu):.3f}, {float(vmax_safe_pu):.3f}] "
        f"total_v_outside_safe_band={int(total_v_outside_band)}"
    )
    print(f"  node_rows_written={int(n_node_rows_written)}")
    print(f"  files: {NODE_INDEX_CSV.name}, {EDGE_CSV.name}, {SAMPLE_CSV.name}, {NODE_CSV.name}")
    return df_sample, df_node


if __name__ == "__main__":
    generate_original_style_dataset_8500_unbalanced(
        n_scenarios=200,
        k_snapshots_per_scenario_total=960,
        bins_by_profile={"load": 10, "pv": 10, "net": 10},
        include_anchors=True,
        master_seed=20260130,
        sigma_load=0.03,
        sigma_pv=0.03,
        bess_total_mva_mean=4.0,
        bess_total_mva_sigma=0.05,
        bess_num_nodes_min=1,
        bess_num_nodes_max=8,
        bess_q_frac_max=0.44,
        p_load_mean_kw=13731.9,
        q_load_mean_kvar=2610.15,
        p_load_scale_range=(0.7, 1.3),
        q_load_scale_range=(0.7, 1.3),
        p_pv_scale_range=(0.7, 1.3),
        vmin_safe_pu=0.85,
        vmax_safe_pu=1.15,
        include_source_in_safe_band=True,
        return_node_df=False,
    )

