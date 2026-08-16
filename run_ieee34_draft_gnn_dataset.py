"""
IEEE 34 Mirzaei dataset generation aligned with the MT-GPS paper draft.

Graph construction (paper §II):
  - Directed edges from network-only nodal Y (Line/Transformer/Capacitor/Reactor =
    lines, xfmrs, network shunts). Message j→i uses e_ji=[Re(Y_ij), Im(Y_ij)]
    stored in R_full/X_full. Loads/PV/Vsource are excluded from Y.
  - Node operating features: ZIP at V_ref (|V|=1, nominal angles 0/−120/+120°):
      p_P_kw, q_P_kvar, p_I_kw, q_I_kvar, p_Z_kw, q_Z_kvar, p_pv_kw
    Terminals from OpenDSS NodeRef; wye/LN vs true LL-delta via ground conductor;
    delta uses diag(V) H^T diag(H V)^{-1} with paper I/Z |HV| factors.
    Model 8 uses ZIPV P/I/Z mix; other models map to one ZIP bin (7 splits P/Q).
  - Laplacian PE: optional freeze via node_pe_from_csv (reuse Mirzaei/original pe_*);
    otherwise |Y|-weighted PE from the new edge graph.
  - Settled taps/caps/aux are meta targets; Y built at regulator tap=1.

Outputs the same chunk layout expected by train_da_gps_multitask_complex_voltage_gine.py.
"""
from __future__ import annotations

import csv
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import eigsh

import run_injection_dataset as inj
import run_loadtype_dataset as lt_dist
import run_original_style_dataset_8500_unbalanced as ds8500
import run_original_style_dataset_ieee34_mirzaei as ie34

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
ds8500 = importlib.reload(ds8500)
ie34 = importlib.reload(ie34)

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DSS_FILE = REPO_ROOT / "new dss from dr mirzaei" / "IEEE34_PV.dss"
NPTS = int(inj.NPTS)
STEP_MIN = float(inj.STEP_MIN)

_K_GNN2 = Path(r"K:\My Drive\datasets_gnn2")
try:
    import google.colab  # noqa: F401

    DEFAULT_OUT = Path("/content/drive/MyDrive/datasets_gnn2/original_ieee34_draft_yzip")
except ImportError:
    if _K_GNN2.exists():
        DEFAULT_OUT = _K_GNN2 / "original_ieee34_draft_yzip"
    else:
        DEFAULT_OUT = REPO_ROOT / "datasets_gnn2" / "original_ieee34_draft_yzip"

INCLUDED_Y_CLASSES = {"line", "transformer", "capacitor", "reactor"}

# Draft operating feature columns (7) + legacy zeros for schema compatibility.
ZIP_FEATURE_COLS = [
    "p_P_kw",
    "q_P_kvar",
    "p_I_kw",
    "q_I_kvar",
    "p_Z_kw",
    "q_Z_kvar",
    "p_pv_kw",
]


def _model_to_zip_channel(model_id: int) -> str:
    """Map a single OpenDSS load model → one ZIP bin (non-Model-8 loads)."""
    m = int(model_id)
    if m == 2:
        return "Z"
    if m == 5:
        return "I"
    if m == 7:
        # Const P, fixed-impedance Q: put Q in Z via ZIPV-style split below when possible;
        # fallback whole-load Z would be wrong for P — keep P-bin for undifferentiated use.
        return "P"
    # 1 = const-P, 3/4/6 ≈ P-dominated; Model 8 handled separately
    return "P"


def _zip_complex_components(
    *,
    p_set: float,
    q_set: float,
    model_id: int,
) -> dict[str, complex]:
    """
    Split a load setpoint into paper ZIP complex powers s_P, s_I, s_Z (kW + j kvar).

    Model 8 uses OpenDSS ZIPV = [Zp,Ip,Pp, Zq,Iq,Pq, Vcut].
    Other models place the full setpoint in one bin (1/3/4/6→P, 5→I, 2→Z).
    Model 7 (const P, fixed Z Q): P→P bin, Q→Z bin.
    """
    m = int(model_id)
    z = i = p = complex(0.0, 0.0)
    if m == 8:
        zipv = None
        try:
            zipv = list(dss.Loads.ZIPV)
        except Exception:
            zipv = None
        if zipv is not None and len(zipv) >= 6:
            zp, ip, pp = float(zipv[0]), float(zipv[1]), float(zipv[2])
            zq, iq, pq = float(zipv[3]), float(zipv[4]), float(zipv[5])
            z = complex(p_set * zp, q_set * zq)
            i = complex(p_set * ip, q_set * iq)
            p = complex(p_set * pp, q_set * pq)
            return {"P": p, "I": i, "Z": z}
        # Fall through if ZIPV missing
    if m == 7:
        return {"P": complex(p_set, 0.0), "I": 0j, "Z": complex(0.0, q_set)}
    ch = _model_to_zip_channel(m)
    s = complex(p_set, q_set)
    return {
        "P": s if ch == "P" else 0j,
        "I": s if ch == "I" else 0j,
        "Z": s if ch == "Z" else 0j,
    }


def _yorder_name_map() -> list[str]:
    return [str(x) for x in dss.Circuit.YNodeOrder()]


def _load_phase_terminals_from_noderef() -> list[tuple[str, int]]:
    """Active CktElement phase terminals via NodeRef → YNodeOrder (skips ground)."""
    yorder = _yorder_name_map()
    out: list[tuple[str, int]] = []
    for r in np.asarray(dss.CktElement.NodeRef(), dtype=np.int64).ravel():
        ri = int(r)
        if ri <= 0:
            continue
        idx = ri - 1
        if idx < 0 or idx >= len(yorder):
            continue
        nm = str(yorder[idx]).strip()
        if "." not in nm:
            continue
        bus, ph_s = nm.rsplit(".", 1)
        if not ph_s.isdigit():
            continue
        ph = int(ph_s)
        if ph not in (1, 2, 3):
            continue
        out.append((bus, ph))
    return out


def _load_has_ground_terminal() -> bool:
    return any(int(r) <= 0 for r in np.asarray(dss.CktElement.NodeRef(), dtype=np.int64).ravel())


def assemble_network_y_on_nodes(node_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Stamp network-only YPrim onto the given phase-node list (siemens).
    Excludes Load/Generator/PVSystem/Storage/Vsource.

    Some OpenDSS elements on large feeders (esp. IEEE 8500) raise
    ``NodeRef is not populated``; those elements are skipped with a count.
    """
    try:
        dss.Basic.AdvancedTypes(True)
    except Exception:
        pass

    y_order = [str(x) for x in dss.Circuit.YNodeOrder()]
    n = len(node_names)
    name_to_i = {str(n).strip().lower(): i for i, n in enumerate(node_names)}
    Y = lil_matrix((n, n), dtype=np.complex128)
    n_ok = 0
    n_skip = 0
    skip_examples: list[str] = []

    for element_name in dss.Circuit.AllElementNames():
        element_class = element_name.split(".", maxsplit=1)[0].lower()
        if element_class not in INCLUDED_Y_CLASSES:
            continue
        if dss.Circuit.SetActiveElement(element_name) < 0:
            continue
        try:
            if hasattr(dss.CktElement, "Enabled") and (not bool(dss.CktElement.Enabled())):
                n_skip += 1
                continue
        except Exception:
            pass

        try:
            node_ref = np.asarray(dss.CktElement.NodeRef(), dtype=np.int64).ravel()
            y_prim = np.asarray(dss.CktElement.YPrim(), dtype=np.complex128)
        except Exception as exc:
            n_skip += 1
            if len(skip_examples) < 8:
                skip_examples.append(f"{element_name} ({type(exc).__name__}: {exc})")
            continue

        local_order = int(node_ref.size)
        if local_order == 0 or y_prim.size == 0:
            n_skip += 1
            continue
        if y_prim.ndim == 1:
            # Classic interleaved Re/Im when AdvancedTypes is off / partial.
            if y_prim.size == 2 * local_order * local_order:
                raw = np.asarray(y_prim, dtype=float).ravel()
                y_prim = (raw[0::2] + 1j * raw[1::2]).reshape(
                    (local_order, local_order), order="F"
                )
            elif y_prim.size == local_order * local_order:
                y_prim = y_prim.reshape((local_order, local_order), order="F")
            else:
                n_skip += 1
                continue
        elif y_prim.shape != (local_order, local_order):
            if y_prim.size != local_order * local_order:
                n_skip += 1
                continue
            y_prim = y_prim.reshape((local_order, local_order), order="F")

        active = np.flatnonzero(node_ref > 0)
        gidx: list[int] = []
        loci: list[int] = []
        for loc in active:
            yi = int(node_ref[loc]) - 1
            if yi < 0 or yi >= len(y_order):
                continue
            nm = str(y_order[yi]).strip().lower()
            gi = name_to_i.get(nm)
            if gi is None:
                continue
            gidx.append(int(gi))
            loci.append(int(loc))

        if not gidx:
            n_skip += 1
            continue

        stamped = False
        for a, ia in zip(loci, gidx):
            for b, ib in zip(loci, gidx):
                val = y_prim[a, b]
                if val != 0:
                    Y[ia, ib] += val
                    stamped = True
        if stamped:
            n_ok += 1
        else:
            n_skip += 1

    try:
        dss.Basic.AdvancedTypes(False)
    except Exception:
        pass

    Y_csr = Y.tocsr()
    # lil/csr: use .nnz / dense diagonal — np.count_nonzero(sparse) is ambiguous.
    nnz_offdiag = int(Y_csr.nnz - np.count_nonzero(Y_csr.diagonal()))
    print(
        f"[assemble_network_y] elements_ok={n_ok} skipped={n_skip} "
        f"N={n} nnz_offdiag={nnz_offdiag}"
    )
    if skip_examples:
        print("[assemble_network_y] skip examples:")
        for s in skip_examples:
            print(f"  - {s}")
    if n_ok == 0:
        raise RuntimeError(
            "assemble_network_y_on_nodes: no network elements stamped into Y "
            f"(skipped={n_skip}). Check DSS compile / included classes."
        )

    return Y_csr.toarray(), node_names


def _edge_row_y(
    *,
    from_node: str,
    to_node: str,
    y: complex,
    u_idx: int,
    v_idx: int,
) -> dict:
    """One directed edge j→i with e=[Re(Y_ij), Im(Y_ij)] (paper draft)."""
    bi, pi = from_node.rsplit(".", 1) if "." in from_node else (from_node, "")
    bj, pj = to_node.rsplit(".", 1) if "." in to_node else (to_node, "")
    return {
        "from_node": from_node,
        "to_node": to_node,
        "from_bus": bi,
        "to_bus": bj,
        "phase": int(pi) if str(pi).isdigit() and str(pj).isdigit() and pi == pj else 0,
        "line_name": "Yprim.network",
        "linecode": "",
        "nph_line": 0,
        "length": 0.0,
        "R_per_len": 0.0,
        "X_per_len": 0.0,
        "C_per_len": 0.0,
        # Message j→i uses Y_ij (siemens); trainer reads R_full/X_full as edge_attr.
        "R_full": float(np.real(y)),
        "X_full": float(np.imag(y)),
        "C_full": 0.0,
        "y_re": float(np.real(y)),
        "y_im": float(np.imag(y)),
        "abs_y": float(abs(y)),
        "u_idx": int(u_idx),
        "v_idx": int(v_idx),
        "from_base_kv": np.nan,
        "to_base_kv": np.nan,
        "length_unit": "y_admittance_S",
        "edge_directed": 1,
    }


def export_y_edges_csv(
    Y: np.ndarray,
    node_names: list[str],
    edge_csv: Path,
    *,
    atol: float = 0.0,
) -> int:
    """
    Write directed coupled pairs: from_node=j, to_node=i, R_full/X_full = Re/Im(Y_ij).

    Matches the paper edge feature e_ji = [Re(Y_ij), Im(Y_ij)] for message j→i.
    """
    rows: list[dict] = []
    n = len(node_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yij = complex(Y[i, j])
            if abs(yij) <= atol:
                continue
            ni = str(node_names[i])
            nj = str(node_names[j])
            # CSV orientation: from=j (source), to=i (target) ↔ paper edge ji.
            rows.append(
                _edge_row_y(from_node=nj, to_node=ni, y=yij, u_idx=j, v_idx=i)
            )
    df = pd.DataFrame(rows)
    edge_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(edge_csv, index=False)
    return len(df)


def compute_laplacian_pe_from_y_edges(
    *,
    node_names: list[str],
    edge_csv_path: Path,
    k: int,
    seed: int = 42,
    zero_eig_tol: float = 1e-8,
) -> np.ndarray:
    """Normalized Laplacian PE with impedance-style affinity w = |Y| (=1/|Z| for series)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(node_names)
    if k >= n:
        raise ValueError(f"node_pe_k must be < N. Got k={k}, N={n}.")
    node_to_local = {str(nm): i for i, nm in enumerate(node_names)}
    df = pd.read_csv(edge_csv_path)
    # Directed catalogs list both (i,j) and (j,i); collapse to undirected affinity.
    pair_w: dict[tuple[int, int], float] = {}
    for _, row in df.iterrows():
        u = str(row["from_node"]).strip()
        v = str(row["to_node"]).strip()
        if u not in node_to_local or v not in node_to_local:
            continue
        r = float(row["R_full"]) if pd.notna(row.get("R_full")) else 0.0
        x = float(row["X_full"]) if pd.notna(row.get("X_full")) else 0.0
        w = float(np.hypot(r, x))
        if w <= 0.0:
            continue
        iu, iv = node_to_local[u], node_to_local[v]
        if iu == iv:
            continue
        a, b = (iu, iv) if iu < iv else (iv, iu)
        prev = pair_w.get((a, b))
        pair_w[(a, b)] = w if prev is None else 0.5 * (prev + w)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (iu, iv), w in pair_w.items():
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        data.extend([w, w])
    if not data:
        raise RuntimeError(f"No positive |Y| edges for PE in {edge_csv_path}")
    W = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    deg = np.asarray(W.sum(axis=1)).ravel()
    # Avoid div0 on isolated nodes
    d_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    Dmh = diags(d_inv_sqrt)
    L = diags(np.ones(n, dtype=np.float64)) - (Dmh @ W @ Dmh)
    # Smallest nontrivial eigenvectors of normalized Laplacian
    k_eigs = min(n - 2, max(k + 4, k + 1))
    try:
        eigvals, eigvecs = eigsh(L, k=k_eigs, which="SM", seed=int(seed))
    except TypeError:
        # older scipy: no seed=
        eigvals, eigvecs = eigsh(L, k=k_eigs, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    nontrivial = np.where(np.abs(eigvals) > float(zero_eig_tol))[0]
    if nontrivial.size < int(k):
        raise RuntimeError(
            f"Not enough nontrivial PE modes: found {nontrivial.size}, need {k}."
        )
    pe = eigvecs[:, nontrivial[: int(k)]]
    pe = (pe - pe.mean(axis=0, keepdims=True)) / (pe.std(axis=0, keepdims=True) + 1e-8)
    return pe.astype(np.float32)


# Nominal phase angles for V_ref (radians): a=0, b=-120°, c=+120°.
_PHASE_ANGLE_RAD = {1: 0.0, 2: -2.0 * np.pi / 3.0, 3: 2.0 * np.pi / 3.0}


def _vref_phasor(phase: int) -> complex:
    return complex(np.exp(1j * float(_PHASE_ANGLE_RAD[int(phase)])))


def _add_complex_to_busph(
    p_map: dict[tuple[str, int], float],
    q_map: dict[tuple[str, int], float],
    bus: str,
    ph: int,
    s: complex,
) -> None:
    key = (str(bus), int(ph))
    p_map[key] = p_map.get(key, 0.0) + float(np.real(s))
    q_map[key] = q_map.get(key, 0.0) + float(np.imag(s))


def _allocate_delta_branch(
    p_map: dict[tuple[str, int], float],
    q_map: dict[tuple[str, int], float],
    *,
    bus_i: str,
    ph_i: int,
    bus_j: str,
    ph_j: int,
    s_d: complex,
) -> None:
    """
    Paper delta→phase map for one oriented connection (i,j):
      S_node = diag(V) H^T diag(H V)^{-1} s_Δ
    with H row = [+1 at i, -1 at j], V = V_ref (|V|=1, nominal angles).
    """
    Vi = _vref_phasor(ph_i)
    Vj = _vref_phasor(ph_j)
    hv = Vi - Vj
    if abs(hv) < 1e-14:
        raise ZeroDivisionError(
            f"H V_ref ≈ 0 for delta branch {bus_i}.{ph_i}-{bus_j}.{ph_j}"
        )
    i_br = s_d / hv
    _add_complex_to_busph(p_map, q_map, bus_i, ph_i, Vi * i_br)
    _add_complex_to_busph(p_map, q_map, bus_j, ph_j, Vj * (-i_br))


def _scale_delta_zip_setpoint(s: complex, channel: str, ph_i: int, ph_j: int) -> complex:
    """
    Paper I/Z factors on delta nominal powers before the H map:
      I: s ⊙ |H V| / √3
      Z: s ⊙ |H V|^2 / 3
    At balanced |V|=1, |H V|=√3 ⇒ both factors are 1; kept explicit for fidelity.
    """
    if abs(s) == 0.0:
        return 0j
    hv = abs(_vref_phasor(ph_i) - _vref_phasor(ph_j))
    ch = str(channel).upper()
    if ch == "I":
        return s * (hv / np.sqrt(3.0))
    if ch == "Z":
        return s * ((hv * hv) / 3.0)
    return s


def _allocate_zip_on_terminals(
    out: dict[str, dict[tuple[str, int], float]],
    *,
    comps: dict[str, complex],
    terminals: list[tuple[str, int]],
    topology: str,
) -> None:
    """Place ZIP components onto phase nodes for wye or delta terminal sets."""
    if not terminals:
        return
    topo = str(topology).lower()
    if topo == "wye":
        n = float(len(terminals))
        for ch, s_tot in comps.items():
            if abs(s_tot) == 0.0:
                continue
            # At |V_ref|=1: I ⊙ |V| and Z ⊙ |V|^2 leave setpoints unchanged.
            s_each = s_tot / n
            pk, qk = f"p_{ch}_kw", f"q_{ch}_kvar"
            for bus, ph in terminals:
                _add_complex_to_busph(out[pk], out[qk], bus, ph, s_each)
        return

    if topo != "delta":
        raise ValueError(f"Unknown load topology {topology!r}")

    # Build LL legs: 1φ → one pair; 3φ → cyclic pairs on the three terminals.
    if len(terminals) == 2:
        legs = [(terminals[0], terminals[1])]
        n_leg = 1.0
    elif len(terminals) >= 3:
        t = terminals[:3]
        legs = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
        n_leg = 3.0
    else:
        raise ValueError(f"Delta load needs ≥2 phase terminals, got {terminals!r}")

    for ch, s_tot in comps.items():
        if abs(s_tot) == 0.0:
            continue
        s_leg = s_tot / n_leg
        pk, qk = f"p_{ch}_kw", f"q_{ch}_kvar"
        for (bi, phi), (bj, phj) in legs:
            s_eff = _scale_delta_zip_setpoint(s_leg, ch, phi, phj)
            _allocate_delta_branch(
                out[pk], out[qk], bus_i=bi, ph_i=phi, bus_j=bj, ph_j=phj, s_d=s_eff
            )


def zip_busph_features_at_vref(
    *,
    loads_dss: list[str] | None = None,
    model_by_device: dict[str, int] | None = None,
    # Legacy kwargs kept so older call sites keep working; ignored (NodeRef path).
    dev_to_dss_load: dict[str, str] | None = None,
    dev_to_busph_load: dict[str, list[tuple[str, int, float]]] | None = None,
) -> dict[str, dict[tuple[str, int], float]]:
    """
    Voltage-independent ZIP phase-node powers at paper V_ref.

    Terminals come from OpenDSS NodeRef→YNodeOrder (not heuristic DEVICE_TO_BUSPH):
      - phase nodes + ground ⇒ grounded-wye / LN map (equal split)
      - ≥2 phase nodes, no ground, Conn=Delta ⇒ H / V_ref delta map
      - Conn=Delta but only one phase + ground (IEEE34 1φ quirk) ⇒ LN on that phase
    ZIP bins follow OpenDSS Model / ZIPV (Model 8).
    """
    del dev_to_busph_load
    out = {
        "p_P_kw": {},
        "q_P_kvar": {},
        "p_I_kw": {},
        "q_I_kvar": {},
        "p_Z_kw": {},
        "q_Z_kvar": {},
    }
    model_by_device = model_by_device or {}

    if loads_dss is not None:
        load_names = list(loads_dss)
    elif dev_to_dss_load is not None:
        load_names = list(dev_to_dss_load.values())
    else:
        load_names = list(dss.Loads.AllNames())

    for ln in load_names:
        dss.Loads.Name(ln)
        if dss.Circuit.SetActiveElement(f"Load.{ln}") < 0:
            continue
        p_set = float(dss.Loads.kW())
        q_set = float(dss.Loads.kvar())
        if abs(complex(p_set, q_set)) == 0.0:
            continue

        key_l = str(ln).strip().lower()
        m = model_by_device.get(key_l, model_by_device.get(str(ln)))
        if m is None:
            m = int(dss.Loads.Model())
        comps = _zip_complex_components(p_set=p_set, q_set=q_set, model_id=int(m))

        terminals = _load_phase_terminals_from_noderef()
        if not terminals:
            continue
        has_gnd = _load_has_ground_terminal()
        is_delta = bool(dss.Loads.IsDelta())

        # True LL delta: no ground conductor and at least two phase terminals.
        if is_delta and (not has_gnd) and len(terminals) >= 2:
            topology = "delta"
        else:
            # Grounded wye, LN, or Conn=Delta with NodeRef[..., 0] (phase-to-ground).
            topology = "wye"

        _allocate_zip_on_terminals(out, comps=comps, terminals=terminals, topology=topology)

    return out


def _busph_get(d: dict, bus: str, ph: int, default: float = 0.0) -> float:
    bus_s = str(bus)
    for key in ((bus_s, int(ph)), (bus_s.lower(), int(ph)), (bus_s.upper(), int(ph))):
        if key in d:
            return float(d[key])
    return float(default)


def _force_nominal_taps_and_build_y(node_names_graph: list[str]) -> np.ndarray:
    """Freeze controls, set regulator taps to 1.0 when possible, solve, assemble Y."""
    dss.Text.Command("Set ControlMode=OFF")
    for xf in ie34.NATIVE_REG_XFMRS:
        try:
            dss.Text.Command(f"Transformer.{xf}.wdg=2 tap=1.0")
        except Exception:
            pass
    try:
        dss.Solution.Solve()
    except Exception:
        pass
    Y, _ = assemble_network_y_on_nodes(node_names_graph)
    return Y


def generate_ieee34_draft_dataset(
    *,
    out_dir: str | Path | None = None,
    n_scenarios: int = 50,
    k_snapshots_per_scenario_total: int = 40,
    bins_by_profile: dict | None = None,
    include_anchors: bool = True,
    master_seed: int = 3420230,
    sigma_load: float = 0.02,
    sigma_pv: float = 0.02,
    node_pe_k: int = 8,
    node_pe_seed: int = 42,
    node_pe_zero_eig_tol: float = 1e-8,
    node_pe_from_csv: str | Path | None = None,
    p_load_scale_range: tuple[float, float] = (0.95, 1.05),
    q_load_scale_range: tuple[float, float] = (0.95, 1.05),
    p_pv_scale_range: tuple[float, float] = (0.95, 1.05),
    vmin_safe_pu: float = 0.85,
    vmax_safe_pu: float = 1.10,
    include_source_in_safe_band: bool = False,
    write_mvagg_compat: bool = True,
    delete_raw_node_csv_after_mvagg: bool = False,
    control_mode: str = "static",
    randomize_zip_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bins_by_profile is None:
        bins_by_profile = {"load": 3, "pv": 3, "net": 3}
    mode = str(control_mode).strip().lower()
    if mode not in ("static", "off"):
        raise ValueError("control_mode must be 'static' or 'off'")
    if not DSS_FILE.is_file():
        raise FileNotFoundError(DSS_FILE)

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_csv = out_dir / "gnn_edges_phase_static.csv"
    node_csv = out_dir / "gnn_node_features_and_targets.csv"
    sample_csv = out_dir / "gnn_sample_meta.csv"
    node_index_csv = out_dir / "gnn_node_index_master.csv"
    mvagg_csv = out_dir / "gnn_node_features_and_targets_mvagg.csv"
    y_npz = out_dir / "Y_network_S.npy"
    y_order_csv = out_dir / "YNodeOrder_graph.csv"

    dss_path = inj.compile_once()
    inj.setup_daily()
    try:
        dss.Text.Command(f"Set ControlMode={'Static' if mode == 'static' else 'OFF'}")
        dss.Text.Command(f"Set MaxControlIter={int(inj.MAX_CONTROL_ITER)}")
    except Exception:
        pass

    node_names_all, _, _, _ = inj.get_all_bus_phase_nodes()
    node_names_graph = ie34._filter_graph_nodes(node_names_all)
    if not node_names_graph:
        raise RuntimeError("No graph nodes after filtering upstream buses.")
    node_to_idx_all = {n: i for i, n in enumerate(node_names_all)}
    print(
        f"[ieee34-draft] nodes_all={len(node_names_all)} nodes_graph={len(node_names_graph)} "
        f"control_mode={mode} out={out_dir}"
    )

    # Reference network-only Y at nominal taps (static edge catalog for the chunk).
    Y = _force_nominal_taps_and_build_y(node_names_graph)
    n_und = export_y_edges_csv(Y, node_names_graph, edge_csv)
    np.save(y_npz, Y)
    pd.DataFrame(
        {"matrix_index": np.arange(len(node_names_graph)), "opendss_node": node_names_graph}
    ).to_csv(y_order_csv, index=False)
    print(f"[ieee34-draft] Y shape={Y.shape} directed Y-edges={n_und} -> {edge_csv.name}")

    # AdvancedTypes(True) makes TotalPower/Losses return complex scalars; restore
    # classic list APIs for the rest of the snapshot pipeline.
    try:
        dss.Basic.AdvancedTypes(False)
    except Exception:
        pass

    # Do NOT run 8500 ohm/length enrich on Y-admittance edges: it overwrites
    # length_unit (breaks loader heuristics) and must never rewrite R_full/X_full.
    try:
        import pandas as _pd

        _ed = _pd.read_csv(edge_csv)
        _bmap = ds8500._node_base_kv_map(node_names_graph)
        _ed["from_base_kv"] = _ed["from_node"].map(_bmap)
        _ed["to_base_kv"] = _ed["to_node"].map(_bmap)
        _ed.to_csv(edge_csv, index=False)
    except Exception as exc:
        print(f"[ieee34-draft] basekv annotate skipped: {exc}")

    node_to_dist = lt_dist._compute_electrical_distance_from_source(
        node_names_graph, str(edge_csv)
    )
    node_to_base_kv = ds8500._node_base_kv_map(node_names_graph)
    node_index_df = pd.DataFrame(
        {
            "node": node_names_graph,
            "node_idx": np.arange(len(node_names_graph), dtype=int),
            "base_kv": [float(node_to_base_kv.get(n, np.nan)) for n in node_names_graph],
            "electrical_distance_ohm": [
                float(node_to_dist.get(n, np.nan)) for n in node_names_graph
            ],
        }
    )
    pe_src = str(node_pe_from_csv).strip() if node_pe_from_csv is not None else ""
    if pe_src:
        pe_path = Path(pe_src)
        if not pe_path.is_file():
            raise FileNotFoundError(f"node_pe_from_csv not found: {pe_path}")
        pe_df = pd.read_csv(pe_path)
        pe_cols = sorted([c for c in pe_df.columns if str(c).lower().startswith("pe_")])
        if not pe_cols:
            raise ValueError(f"No pe_* columns in {pe_path}")
        pe_df = pe_df.copy()
        pe_df["node"] = pe_df["node"].astype(str).str.strip().str.lower()
        node_index_df["node"] = node_index_df["node"].astype(str).str.strip().str.lower()
        pe_map = pe_df.set_index("node")[pe_cols]
        aligned = pe_map.reindex(node_index_df["node"].tolist())
        n_miss = int(aligned.isna().any(axis=1).sum())
        for c in pe_cols:
            node_index_df[c] = aligned[c].to_numpy(dtype=float)
        print(
            f"[ieee34-draft] PE frozen from {pe_path} "
            f"({len(pe_cols)} cols; missing_nodes={n_miss} filled NaN)"
        )
    elif int(node_pe_k) > 0:
        pe = compute_laplacian_pe_from_y_edges(
            node_names=node_names_graph,
            edge_csv_path=edge_csv,
            k=int(node_pe_k),
            seed=int(node_pe_seed),
            zero_eig_tol=float(node_pe_zero_eig_tol),
        )
        for j in range(int(node_pe_k)):
            node_index_df[f"pe_{j + 1}"] = pe[:, j]
        print(f"[ieee34-draft] PE k={int(node_pe_k)} from |Y|-weighted graph")
    node_index_df.to_csv(node_index_csv, index=False)

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = inj.read_profile_csv_two_col_noheader(
        inj.resolve_csv_path(csvL_token, dss_path), npts=NPTS, debug=False
    )
    mPV = inj.read_profile_csv_two_col_noheader(
        inj.resolve_csv_path(csvPV_token, dss_path), npts=NPTS, debug=False
    )

    safe_band_eval_indices = []
    for i, n in enumerate(node_names_all):
        b = n.split(".")[0].strip().lower()
        if (not include_source_in_safe_band) and b in {
            x.lower() for x in inj.EXCLUDED_UPSTREAM_BUSES
        }:
            continue
        safe_band_eval_indices.append(i)

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[dict] = []
    sample_id = 0
    skipped_nonconv = 0
    skipped_bad_v = 0

    p0 = float(inj.BASELINE["P_load_total_kw"])
    q0 = float(inj.BASELINE["Q_load_total_kvar"])
    pv0 = float(inj.BASELINE["P_pv_total_kw"])

    node_fieldnames = [
        "sample_id",
        "node",
        "node_idx",
        "bus",
        "phase",
        *ZIP_FEATURE_COLS,
        "p_bess_kw",
        "q_bess_kvar",
        # Keep legacy aggregate columns for debugging / ablations
        "p_load_kw",
        "q_load_kvar",
        "vmag_pu",
        "vang_deg",
    ]
    graph_node_to_idx = {n: i for i, n in enumerate(node_names_graph)}

    with open(node_csv, "w", newline="", encoding="utf-8") as f_node:
        node_writer = csv.DictWriter(f_node, fieldnames=node_fieldnames)
        node_writer.writeheader()

        for s in range(int(n_scenarios)):
            t0_s = time.time()
            dss.Basic.ClearAll()
            dss.Text.Command(f'compile "{dss_path}"')
            inj._apply_voltage_bases()
            inj.setup_daily()
            try:
                dss.Text.Command(
                    f"Set ControlMode={'Static' if mode == 'static' else 'OFF'}"
                )
                dss.Text.Command(f"Set MaxControlIter={int(inj.MAX_CONTROL_ITER)}")
            except Exception:
                pass

            _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
            loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(
                bus_to_phases
            )
            pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

            p_load = p0 * float(rng_master.uniform(*p_load_scale_range))
            q_load = q0 * float(rng_master.uniform(*q_load_scale_range))
            p_pv = pv0 * float(rng_master.uniform(*p_pv_scale_range))
            sigL = float(sigma_load)
            sigPV = float(sigma_pv)

            model_by_device = ie34._sample_and_apply_load_models(
                loads_dss,
                rng_master,
                randomize=bool(randomize_zip_models),
            )

            prof_net = (p_load * mL) - (p_pv * mPV)
            rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            times = inj.select_times_three_profiles(
                prof_load=mL,
                prof_pv=mPV,
                prof_net=prof_net,
                K_total=int(k_snapshots_per_scenario_total),
                bins_by_profile=bins_by_profile,
                include_anchors=include_anchors,
                rng=rng_times,
            )
            times_int = [int(x) for x in times]
            rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

            for t in times_int:
                inj.set_time_index(t)
                totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = (
                    inj.apply_snapshot_timeconditioned(
                        P_load_total_kw=p_load,
                        Q_load_total_kvar=q_load,
                        P_pv_total_kw=p_pv,
                        mL_t=float(mL[t]),
                        mPV_t=float(mPV[t]),
                        loads_dss=loads_dss,
                        dev_to_dss_load=dev_to_dss_load,
                        dev_to_busph_load=dev_to_busph_load,
                        pv_dss=pv_dss,
                        pv_to_dss=pv_to_dss,
                        pv_to_busph=pv_to_busph,
                        sigma_load=sigL,
                        sigma_pv=sigPV,
                        rng=rng_solve,
                    )
                )
                # Re-assert per-scenario models after setpoint write.
                ie34._reapply_load_models(model_by_device, loads_dss)
                zip_maps = zip_busph_features_at_vref(
                    loads_dss=loads_dss,
                    model_by_device=model_by_device,
                    # kept for call-site compatibility; H/V_ref path ignores weights
                    dev_to_dss_load=dev_to_dss_load,
                    dev_to_busph_load=dev_to_busph_load,
                )

                try:
                    dss.Solution.Solve()
                except Exception:
                    pass
                if not dss.Solution.Converged():
                    skipped_nonconv += 1
                    continue

                vm, va = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_all)
                vm_a = np.asarray(vm, float)
                band = vm_a[safe_band_eval_indices]
                band = band[np.isfinite(band)]
                if band.size == 0 or np.any(band < vmin_safe_pu) or np.any(
                    band > vmax_safe_pu
                ):
                    skipped_bad_v += 1
                    continue

                p_grid, q_grid = ds8500._grid_upstream_post_kw_kvar()
                p_loss, q_loss = ds8500._circuit_losses_kw_kvar()
                pv_post = ds8500._read_pv_totals_post_solve_kw_kvar(list(ie34.NATIVE_PVS))
                pv_post_norm = {str(k).lower(): v for k, v in pv_post.items()}
                for want in ie34.NATIVE_PVS:
                    pv_post_norm.setdefault(want, (0.0, 0.0))

                p_load_t = float(p_load) * float(mL[t])
                zip_shares = ie34._zip_p_shares_from_setpoints(
                    ie34._device_p_setpoints_from_totals(p_load_t),
                    model_by_device=model_by_device,
                )
                reg_taps = ie34._read_native_reg_taps()
                cap_fields = ie34._read_cap_kvar_post()

                sid = int(sample_id)
                sample_id += 1
                rows_sample.append(
                    {
                        "sample_id": sid,
                        "scenario_id": int(s),
                        "t_index": int(t),
                        "t_minutes": float(t * STEP_MIN),
                        "control_mode": mode,
                        "P_load_total_kw": float(p_load),
                        "Q_load_total_kvar": float(q_load),
                        "P_pv_total_kw": float(p_pv),
                        "sigma_load": float(sigL),
                        "sigma_pv": float(sigPV),
                        "m_loadshape": float(mL[t]),
                        "m_irradshape": float(mPV[t]),
                        "P_grid_upstream_post_kw": float(p_grid),
                        "Q_grid_upstream_post_kvar": float(q_grid),
                        "P_loss_total_post_kw": float(p_loss),
                        "Q_loss_total_post_kvar": float(q_loss),
                        "pv_pv850_p_post_kw": float(pv_post_norm.get("pv850", (0.0, 0.0))[0]),
                        "pv_pv850_q_post_kvar": float(pv_post_norm.get("pv850", (0.0, 0.0))[1]),
                        "pv_pv860_p_post_kw": float(pv_post_norm.get("pv860", (0.0, 0.0))[0]),
                        "pv_pv860_q_post_kvar": float(pv_post_norm.get("pv860", (0.0, 0.0))[1]),
                        **zip_shares,
                        **reg_taps,
                        **cap_fields,
                    }
                )

                for n in node_names_graph:
                    bus, ph_s = str(n).rsplit(".", 1)
                    ph = int(ph_s)
                    pl = _busph_get(busphP_load, bus, ph)
                    ql = _busph_get(busphQ_load, bus, ph)
                    ppv = _busph_get(busphP_pv, bus, ph)  # available DER at time t
                    i_all = node_to_idx_all.get(n)
                    if i_all is None:
                        i_all = next(
                            (
                                j
                                for j, nn in enumerate(node_names_all)
                                if str(nn).lower() == str(n).lower()
                            ),
                            None,
                        )
                    if i_all is None:
                        continue
                    node_writer.writerow(
                        {
                            "sample_id": sid,
                            "node": n,
                            "node_idx": int(graph_node_to_idx[n]),
                            "bus": bus,
                            "phase": ph,
                            "p_P_kw": _busph_get(zip_maps["p_P_kw"], bus, ph),
                            "q_P_kvar": _busph_get(zip_maps["q_P_kvar"], bus, ph),
                            "p_I_kw": _busph_get(zip_maps["p_I_kw"], bus, ph),
                            "q_I_kvar": _busph_get(zip_maps["q_I_kvar"], bus, ph),
                            "p_Z_kw": _busph_get(zip_maps["p_Z_kw"], bus, ph),
                            "q_Z_kvar": _busph_get(zip_maps["q_Z_kvar"], bus, ph),
                            "p_pv_kw": ppv,
                            "p_bess_kw": 0.0,
                            "q_bess_kvar": 0.0,
                            "p_load_kw": pl,
                            "q_load_kvar": ql,
                            "vmag_pu": float(vm_a[i_all]),
                            "vang_deg": float(va[i_all]),
                        }
                    )

            print(
                f"[ieee34-draft] scenario {s + 1}/{n_scenarios} done in {time.time() - t0_s:.1f}s "
                f"| samples so far={sample_id}",
                flush=True,
            )

    df_sample = pd.DataFrame(rows_sample)
    df_sample.to_csv(sample_csv, index=False)
    if write_mvagg_compat:
        # mvagg compat: copy draft node CSV (no split-phase agg on ieee34)
        df_node = pd.read_csv(node_csv)
        df_node.to_csv(mvagg_csv, index=False)
        if delete_raw_node_csv_after_mvagg and node_csv.is_file():
            try:
                node_csv.unlink()
            except Exception:
                pass
            df_node = pd.read_csv(mvagg_csv)
        else:
            df_node = pd.read_csv(node_csv)
    else:
        df_node = pd.read_csv(node_csv)

    print(
        f"[ieee34-draft] done samples={len(df_sample)} node_rows={len(df_node)} "
        f"skip_nonconv={skipped_nonconv} skip_bad_v={skipped_bad_v}"
    )
    print(f"[ieee34-draft] feature cols: {ZIP_FEATURE_COLS}")
    print(f"[ieee34-draft] edges use R_full/X_full = Re(Y)/Im(Y) [siemens]")
    return df_sample, df_node


if __name__ == "__main__":
    generate_ieee34_draft_dataset(
        out_dir=REPO_ROOT / "datasets_gnn2" / "_smoke_ieee34_draft_yzip",
        n_scenarios=1,
        k_snapshots_per_scenario_total=4,
        master_seed=42,
    )
