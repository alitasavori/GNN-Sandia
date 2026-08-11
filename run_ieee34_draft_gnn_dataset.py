"""
IEEE 34 Mirzaei dataset generation aligned with the MT-GPS paper draft.

Differences vs run_original_style_dataset_ieee34_mirzaei.py:
  - Edges from network-only nodal Y (Line/Transformer/Capacitor/Reactor YPrim),
    attributes [Re(Y_ij), Im(Y_ij)] stored in R_full/X_full for trainer compatibility.
  - Node operating features are voltage-independent ZIP channels at |V_ref|=1:
      p_P_kw, q_P_kvar, p_I_kw, q_I_kvar, p_Z_kw, q_Z_kvar, p_pv_kw
    (OpenDSS load models 1/4→P, 5→I, 2→Z; phase allocation via existing bus–phase weights).
  - Laplacian PE uses admittance weights w=|Y_ij|.

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
import run_original_style_dataset_906_lvtestcase as ds906
import run_original_style_dataset_ieee34_mirzaei as ie34

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
ds8500 = importlib.reload(ds8500)
ds906 = importlib.reload(ds906)
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
    """Map OpenDSS load model → ZIP bin used in the draft feature vector."""
    m = int(model_id)
    if m == 2:
        return "Z"
    if m == 5:
        return "I"
    # 1 = const-P, 4 = const-P / quad-Q (paper/Log(v) treat as P-bin), others → P
    return "P"


def assemble_network_y_on_nodes(node_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Stamp network-only YPrim onto the given phase-node list (siemens).
    Excludes Load/Generator/PVSystem/Storage/Vsource.
    """
    try:
        dss.Basic.AdvancedTypes(True)
    except Exception:
        pass

    y_order = [str(x) for x in dss.Circuit.YNodeOrder()]
    y_lower = {str(n).strip().lower(): i for i, n in enumerate(y_order)}
    n = len(node_names)
    name_to_i = {str(n).strip().lower(): i for i, n in enumerate(node_names)}
    Y = lil_matrix((n, n), dtype=np.complex128)

    for element_name in dss.Circuit.AllElementNames():
        element_class = element_name.split(".", maxsplit=1)[0].lower()
        if element_class not in INCLUDED_Y_CLASSES:
            continue
        if dss.Circuit.SetActiveElement(element_name) < 0:
            continue
        node_ref = np.asarray(dss.CktElement.NodeRef(), dtype=np.int64)
        y_prim = np.asarray(dss.CktElement.YPrim(), dtype=np.complex128)
        local_order = int(node_ref.size)
        if local_order == 0 or y_prim.size == 0:
            continue
        if y_prim.shape != (local_order, local_order):
            if y_prim.size != local_order * local_order:
                continue
            y_prim = y_prim.reshape((local_order, local_order), order="F")

        active = np.flatnonzero(node_ref > 0)
        # Map NodeRef → YNodeOrder name → graph index
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

        for a, ia in zip(loci, gidx):
            for b, ib in zip(loci, gidx):
                val = y_prim[a, b]
                if val != 0:
                    Y[ia, ib] += val

    return Y.tocsr().toarray(), node_names


def export_y_edges_csv(
    Y: np.ndarray,
    node_names: list[str],
    edge_csv: Path,
    *,
    atol: float = 0.0,
) -> int:
    """
    Write undirected coupled pairs with R_full=Re(Y_ij), X_full=Im(Y_ij).
    Trainer `_load_compacted_edges` reads R_full/X_full as edge_attr.
    """
    rows: list[dict] = []
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            yij = complex(Y[i, j])
            yji = complex(Y[j, i])
            if abs(yij) <= atol and abs(yji) <= atol:
                continue
            y = 0.5 * (yij + yji) if (abs(yij) > 0 and abs(yji) > 0) else (
                yij if abs(yij) >= abs(yji) else yji
            )
            ni = str(node_names[i])
            nj = str(node_names[j])
            bi, pi = ni.rsplit(".", 1) if "." in ni else (ni, "")
            bj, pj = nj.rsplit(".", 1) if "." in nj else (nj, "")
            rows.append(
                {
                    "from_node": ni,
                    "to_node": nj,
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
                    # Trainer edge_attr channels (draft: Re/Im Y)
                    "R_full": float(np.real(y)),
                    "X_full": float(np.imag(y)),
                    "C_full": 0.0,
                    "y_re": float(np.real(y)),
                    "y_im": float(np.imag(y)),
                    "abs_y": float(abs(y)),
                    "u_idx": i,
                    "v_idx": j,
                    "from_base_kv": np.nan,
                    "to_base_kv": np.nan,
                    "length_unit": "y_admittance_S",
                }
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
    """Normalized Laplacian PE with affinity w = |Y| = sqrt(R_full^2 + X_full^2)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(node_names)
    if k >= n:
        raise ValueError(f"node_pe_k must be < N. Got k={k}, N={n}.")
    node_to_local = {str(nm): i for i, nm in enumerate(node_names)}
    df = pd.read_csv(edge_csv_path)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
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
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        data.extend([w, w])
    if not data:
        raise RuntimeError("No usable Y-edges for PE.")
    w_mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    w_mat = 0.5 * (w_mat + w_mat.T)
    deg = np.asarray(w_mat.sum(axis=1)).ravel()
    d_inv = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    l_norm = diags(d_inv) @ (diags(deg) - w_mat) @ diags(d_inv)
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
    nontrivial = np.where(np.abs(eigvals) > float(zero_eig_tol))[0]
    if nontrivial.size < int(k):
        raise RuntimeError(
            f"Not enough nontrivial PE modes: found {nontrivial.size}, need {k}."
        )
    pe = eigvecs[:, nontrivial[: int(k)]]
    pe = (pe - pe.mean(axis=0, keepdims=True)) / (pe.std(axis=0, keepdims=True) + 1e-8)
    return pe.astype(np.float32)


def zip_busph_features_at_vref(
    *,
    dev_to_dss_load: dict[str, str],
    dev_to_busph_load: dict[str, list[tuple[str, int, float]]],
    model_by_device: dict[str, int],
) -> dict[str, dict[tuple[str, int], float]]:
    """
    Voltage-independent ZIP phase-node powers at |V_ref|=1.

    Each load's current kW/kvar setpoint is placed entirely in one ZIP bin
    according to its OpenDSS model, then spread with the existing bus–phase
    weights (wye / multi-phase allocation already encoded there).
    """
    out = {
        "p_P_kw": {},
        "q_P_kvar": {},
        "p_I_kw": {},
        "q_I_kvar": {},
        "p_Z_kw": {},
        "q_Z_kvar": {},
    }
    for dev_key, ln in dev_to_dss_load.items():
        dss.Loads.Name(ln)
        p_set = float(dss.Loads.kW())
        q_set = float(dss.Loads.kvar())
        m = model_by_device.get(str(dev_key), model_by_device.get(str(dev_key).lower()))
        if m is None:
            m = int(dss.Loads.Model())
        ch = _model_to_zip_channel(int(m))
        pk = f"p_{ch}_kw"
        qk = f"q_{ch}_kvar"
        for bus, ph, w in dev_to_busph_load.get(dev_key, []):
            key = (str(bus), int(ph))
            out[pk][key] = out[pk].get(key, 0.0) + p_set * float(w)
            out[qk][key] = out[qk].get(key, 0.0) + q_set * float(w)
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
    print(f"[ieee34-draft] Y shape={Y.shape} undirected Y-edges={n_und} -> {edge_csv.name}")

    # AdvancedTypes(True) makes TotalPower/Losses return complex scalars; restore
    # classic list APIs for the rest of the snapshot pipeline.
    try:
        dss.Basic.AdvancedTypes(False)
    except Exception:
        pass

    try:
        ds8500._enrich_edges_with_basekv_and_length_km(edge_csv, node_names_graph)
    except Exception as exc:
        print(f"[ieee34-draft] basekv enrich skipped: {exc}")

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
    if int(node_pe_k) > 0:
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
    dummy_meta = ds906._dummy_cap_reg_meta()

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
                    dev_to_dss_load=dev_to_dss_load,
                    dev_to_busph_load=dev_to_busph_load,
                    model_by_device=model_by_device,
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
                        **dummy_meta,
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
