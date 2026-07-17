"""
Original-style DA-GPS dataset on Mirzaei IEEE34_PV.dss.

Emits the same artifact layout as 906/8500 original-style generators so the
existing multitask trainer can load chunks without schema changes:

  gnn_node_index_master.csv
  gnn_edges_phase_static.csv
  gnn_sample_meta.csv
  gnn_node_features_and_targets.csv
  gnn_node_features_and_targets_mvagg.csv  (compat copy; no split-phase agg)

Node features: p_load_kw, q_load_kvar, p_pv_kw  (+ zero BESS cols)
Targets:       vmag_pu, vang_deg

Sample meta includes:
  - dummy 8500 TARGET_* cap/reg columns (trainer heads; keep lambda_cap/reg=0)
  - real system tokens: grid P/Q, PV850/860 post P/Q, ZIP/load-model P shares,
    fixed capacitor kvar, native regulator transformer taps (no RegControl)
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

import run_injection_dataset as inj
import run_loadtype_dataset as lt_dist
import run_original_style_dataset_8500_unbalanced as ds8500
import run_original_style_dataset_906_lvtestcase as ds906

inj = importlib.reload(inj)
lt_dist = importlib.reload(lt_dist)
ds8500 = importlib.reload(ds8500)
ds906 = importlib.reload(ds906)

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DSS_FILE = REPO_ROOT / "new dss from dr mirzaei" / "IEEE34_PV.dss"
NPTS = int(inj.NPTS)  # 288
STEP_MIN = float(inj.STEP_MIN)  # 5

_K_GNN2 = Path(r"K:\My Drive\datasets_gnn2")
_K_MYDRIVE = Path(r"K:\My Drive")
try:
    import google.colab  # noqa: F401

    OUT_DIR = Path("/content/drive/MyDrive/datasets_gnn2/original_ieee34_mirzaei")
except ImportError:
    if _K_GNN2.exists() or _K_MYDRIVE.exists():
        OUT_DIR = _K_GNN2 / "original_ieee34_mirzaei"
    elif Path(r"D:\datasets").exists():
        OUT_DIR = Path(r"D:\datasets\original_ieee34_mirzaei")
    else:
        OUT_DIR = REPO_ROOT / "datasets_gnn2" / "original_ieee34_mirzaei"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CSV = OUT_DIR / "gnn_edges_phase_static.csv"
NODE_CSV = OUT_DIR / "gnn_node_features_and_targets.csv"
SAMPLE_CSV = OUT_DIR / "gnn_sample_meta.csv"
NODE_INDEX_CSV = OUT_DIR / "gnn_node_index_master.csv"
MVAGG_CSV = OUT_DIR / "gnn_node_features_and_targets_mvagg.csv"

# Native Mirzaei regulator transformers (no RegControl objects).
NATIVE_REG_XFMRS = ("rega1", "rega2", "rega3", "regb1", "regb2", "regb3")
NATIVE_CAPS = ("c844", "c848")
NATIVE_PVS = ("pv850", "pv860")


def _filter_graph_nodes(node_names: list[str]) -> list[str]:
    excl = {b.lower() for b in inj.EXCLUDED_UPSTREAM_BUSES}
    out = []
    for n in node_names:
        bus = str(n).split(".")[0].strip().lower()
        if bus in excl:
            continue
        out.append(str(n))
    return out


def _zip_p_shares_from_setpoints(
    p_by_device: dict[str, float],
    model_by_device: dict[str, int] | None = None,
) -> dict[str, float]:
    """Fraction of set P in each DSS load model (1/2/4/5)."""
    totals = {1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}
    for name, p in p_by_device.items():
        if model_by_device is not None:
            m = model_by_device.get(str(name), model_by_device.get(str(name).lower()))
            if m is None:
                m = int(lt_dist.DEVICE_TO_MODEL.get(str(name), 1))
            else:
                m = int(m)
        else:
            m = int(lt_dist.DEVICE_TO_MODEL.get(str(name), 1))
        if m not in totals:
            m = 1
        totals[m] += max(float(p), 0.0)
    s = sum(totals.values())
    if s <= 1e-12:
        return {f"share_m{m}_p": 0.0 for m in (1, 2, 4, 5)}
    return {f"share_m{m}_p": float(totals[m] / s) for m in (1, 2, 4, 5)}


def _sample_and_apply_load_models(
    loads_dss: list[str],
    rng: np.random.Generator,
    *,
    randomize: bool,
) -> dict[str, int]:
    """Assign DSS Load.Model per device; optionally randomize away from Mirzaei defaults.

    When randomize=True, each load is drawn from {1,2,4,5} so ZIP-share system
    tokens vary across scenarios (default Mirzaei mix is otherwise fixed).
    Returns map keyed by DSS load name and lowercase alias.
    """
    allowed = (1, 2, 4, 5)
    model_by: dict[str, int] = {}
    for ln in loads_dss:
        key = str(ln)
        if randomize:
            m = int(rng.choice(allowed))
        else:
            m = int(
                lt_dist.DEVICE_TO_MODEL.get(
                    key, lt_dist.DEVICE_TO_MODEL.get(key.lower(), 1)
                )
            )
            if m not in allowed:
                m = 1
        try:
            dss.Loads.Name(ln)
            dss.Loads.Model(m)
        except Exception:
            pass
        model_by[key] = m
        model_by[key.lower()] = m
    return model_by


def _reapply_load_models(model_by_device: dict[str, int], loads_dss: list[str]) -> None:
    for ln in loads_dss:
        m = model_by_device.get(str(ln), model_by_device.get(str(ln).lower()))
        if m is None:
            continue
        try:
            dss.Loads.Name(ln)
            dss.Loads.Model(int(m))
        except Exception:
            pass


def _read_native_reg_taps() -> dict[str, float]:
    out = {}
    for nm in NATIVE_REG_XFMRS:
        tap = 1.0
        try:
            dss.Transformers.Name(nm)
            nwind = int(dss.Transformers.NumWindings())
            dss.Transformers.Wdg(min(2, nwind))
            tap = float(dss.Transformers.Tap())
        except Exception:
            tap = 1.0
        out[f"reg_{nm}_tap_pu"] = tap
    return out


def _read_cap_kvar_post() -> dict[str, float]:
    out = {}
    for nm in NATIVE_CAPS:
        q = 0.0
        try:
            dss.Capacitors.Name(nm)
            q = float(dss.Capacitors.kvar())
        except Exception:
            q = 0.0
        out[f"cap_{nm}_q_post_kvar"] = q
        # steps_on: fixed banks → 1 if kvar>0
        out[f"cap_{nm}_n_steps_on"] = float(1.0 if abs(q) > 1e-6 else 0.0)
    return out


def _device_p_setpoints_from_totals(
    p_load_t: float,
) -> dict[str, float]:
    """Approximate per-device set P using DEVICE_P_SHARE (same as injection)."""
    return {
        k: float(p_load_t) * float(v) for k, v in inj.DEVICE_P_SHARE.items()
    }


def _busph_get(d: dict, bus: str, ph: int, default: float = 0.0) -> float:
    bus_s = str(bus)
    for key in ((bus_s, int(ph)), (bus_s.lower(), int(ph)), (bus_s.upper(), int(ph))):
        if key in d:
            return float(d[key])
    return float(default)


def generate_original_style_dataset_ieee34_mirzaei(
    *,
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
    node_pe_from_csv: str | None = None,
    node_pe_save_csv: str | None = None,
    p_load_scale_range: tuple[float, float] = (0.95, 1.05),
    q_load_scale_range: tuple[float, float] = (0.95, 1.05),
    p_pv_scale_range: tuple[float, float] = (0.95, 1.05),
    vmin_safe_pu: float = 0.85,
    vmax_safe_pu: float = 1.10,
    include_source_in_safe_band: bool = False,
    return_node_df: bool = False,
    write_mvagg_compat: bool = True,
    delete_raw_node_csv_after_mvagg: bool = False,
    control_mode: str = "static",
    randomize_zip_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate original-style IEEE34 Mirzaei chunk for DA-GPS."""
    if bins_by_profile is None:
        bins_by_profile = {"load": 3, "pv": 3, "net": 3}
    if k_snapshots_per_scenario_total < 1:
        raise ValueError("k_snapshots_per_scenario_total must be >= 1")
    if not (0.0 < float(vmin_safe_pu) < float(vmax_safe_pu)):
        raise ValueError(f"Invalid safe voltage band: [{vmin_safe_pu}, {vmax_safe_pu}]")

    mode = str(control_mode).strip().lower()
    if mode not in ("static", "off"):
        raise ValueError("control_mode must be 'static' or 'off'")

    if not DSS_FILE.is_file():
        raise FileNotFoundError(DSS_FILE)

    dss_path = inj.compile_once()
    inj.setup_daily()
    try:
        dss.Text.Command(f"Set ControlMode={'Static' if mode == 'static' else 'OFF'}")
        dss.Text.Command(f"Set MaxControlIter={int(inj.MAX_CONTROL_ITER)}")
    except Exception:
        pass

    node_names_all, _, _, _ = inj.get_all_bus_phase_nodes()
    node_names_graph = _filter_graph_nodes(node_names_all)
    if not node_names_graph:
        raise RuntimeError("No graph nodes after filtering upstream buses.")
    node_to_idx_all = {n: i for i, n in enumerate(node_names_all)}
    print(
        f"[ieee34] nodes_all={len(node_names_all)} nodes_graph={len(node_names_graph)} "
        f"control_mode={mode}"
    )

    inj.extract_static_phase_edges_to_csv(
        node_names_master=node_names_graph,
        edge_csv_path=str(EDGE_CSV),
        excluded_buses=(),
    )
    ds8500._enrich_edges_with_basekv_and_length_km(EDGE_CSV, node_names_graph)

    node_to_dist = lt_dist._compute_electrical_distance_from_source(
        node_names_graph, str(EDGE_CSV)
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
        pe_df = pd.read_csv(pe_path)
        pe_cols = sorted([c for c in pe_df.columns if str(c).lower().startswith("pe_")])
        pe_df["node"] = pe_df["node"].astype(str).str.strip().str.lower()
        node_index_df["node"] = node_index_df["node"].astype(str).str.strip().str.lower()
        pe_map = pe_df.set_index("node")[pe_cols]
        aligned = pe_map.reindex(node_index_df["node"].tolist())
        for c in pe_cols:
            node_index_df[c] = aligned[c].to_numpy(dtype=float)
        print(f"[ieee34] loaded PE from {pe_path}")
    elif int(node_pe_k) > 0:
        pe = ds8500._compute_laplacian_pe_from_edges(
            node_names=node_names_graph,
            edge_csv_path=EDGE_CSV,
            k=int(node_pe_k),
            seed=int(node_pe_seed),
            zero_eig_tol=float(node_pe_zero_eig_tol),
        )
        for j in range(int(node_pe_k)):
            node_index_df[f"pe_{j + 1}"] = pe[:, j]
        print(f"[ieee34] computed PE k={int(node_pe_k)}")
        pe_save = str(node_pe_save_csv).strip() if node_pe_save_csv is not None else ""
        if pe_save:
            Path(pe_save).parent.mkdir(parents=True, exist_ok=True)
            node_index_df[
                ["node", *[f"pe_{j + 1}" for j in range(int(node_pe_k))]]
            ].to_csv(pe_save, index=False)

    node_index_df.to_csv(NODE_INDEX_CSV, index=False)

    # Profiles
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
    if not safe_band_eval_indices:
        raise RuntimeError("No nodes for safe-band evaluation.")

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
        "p_load_kw",
        "q_load_kvar",
        "p_pv_kw",
        "p_bess_kw",
        "q_bess_kvar",
        "vmag_pu",
        "vang_deg",
    ]
    graph_node_to_idx = {n: i for i, n in enumerate(node_names_graph)}

    with open(NODE_CSV, "w", newline="", encoding="utf-8") as f_node:
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
            # Optional extra noise bounds (defaults match injection RANGES)
            sigL = float(sigma_load) if sigma_load is not None else float(
                rng_master.uniform(*inj.RANGES["sigma_load"])
            )
            sigPV = float(sigma_pv) if sigma_pv is not None else float(
                rng_master.uniform(*inj.RANGES["sigma_pv"])
            )

            # Per-scenario ZIP / load-model assignment (varies share_m*_p tokens).
            model_by_device = _sample_and_apply_load_models(
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

                busphP_pv_act, busphQ_pv_act = inj.get_pv_actual_pq_by_busph(
                    pv_to_dss, pv_to_busph
                )
                p_grid, q_grid = ds8500._grid_upstream_post_kw_kvar()
                p_loss, q_loss = ds8500._circuit_losses_kw_kvar()
                pv_post = ds8500._read_pv_totals_post_solve_kw_kvar(list(NATIVE_PVS))
                # normalize keys
                pv_post_norm = {
                    str(k).lower(): v for k, v in pv_post.items()
                }
                for want in NATIVE_PVS:
                    pv_post_norm.setdefault(want, (0.0, 0.0))

                p_load_t = float(p_load) * float(mL[t])
                _reapply_load_models(model_by_device, loads_dss)
                zip_shares = _zip_p_shares_from_setpoints(
                    _device_p_setpoints_from_totals(p_load_t),
                    model_by_device=model_by_device,
                )
                reg_taps = _read_native_reg_taps()
                cap_fields = _read_cap_kvar_post()

                sid = int(sample_id)
                sample_id += 1
                row_meta = {
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
                    "pv_pv850_p_post_kw": float(pv_post_norm["pv850"][0]),
                    "pv_pv850_q_post_kvar": float(pv_post_norm["pv850"][1]),
                    "pv_pv860_p_post_kw": float(pv_post_norm["pv860"][0]),
                    "pv_pv860_q_post_kvar": float(pv_post_norm["pv860"][1]),
                    **zip_shares,
                    **reg_taps,
                    **cap_fields,
                    **dummy_meta,
                }
                rows_sample.append(row_meta)

                # node rows
                for n in node_names_graph:
                    bus, ph_s = str(n).rsplit(".", 1)
                    ph = int(ph_s)
                    pl = _busph_get(busphP_load, bus, ph)
                    ql = _busph_get(busphQ_load, bus, ph)
                    ppv = _busph_get(busphP_pv_act, bus, ph)
                    i_all = node_to_idx_all.get(n)
                    if i_all is None:
                        # case-insensitive node match
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
                            "p_load_kw": pl,
                            "q_load_kvar": ql,
                            "p_pv_kw": ppv,
                            "p_bess_kw": 0.0,
                            "q_bess_kvar": 0.0,
                            "vmag_pu": float(vm_a[i_all]),
                            "vang_deg": float(va[i_all]),
                        }
                    )

            print(
                f"[ieee34] scenario {s + 1}/{n_scenarios} done in {time.time() - t0_s:.1f}s "
                f"| samples so far={sample_id}",
                flush=True,
            )

    df_sample = pd.DataFrame(rows_sample)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    if write_mvagg_compat:
        ds906._write_mvagg_compat_from_raw(NODE_CSV, MVAGG_CSV)
        if delete_raw_node_csv_after_mvagg and NODE_CSV.is_file():
            try:
                NODE_CSV.unlink()
            except Exception:
                pass

    print(
        f"[ieee34] done samples={len(df_sample)} skipped_nonconv={skipped_nonconv} "
        f"skipped_bad_v={skipped_bad_v}"
    )
    print(f"  sample_meta: {SAMPLE_CSV}")
    print(f"  nodes: {NODE_CSV if NODE_CSV.is_file() else '(deleted after mvagg)'}")
    print(f"  mvagg: {MVAGG_CSV if write_mvagg_compat else '(skipped)'}")
    print(f"  edges: {EDGE_CSV}")
    print(f"  index: {NODE_INDEX_CSV}")

    df_node = pd.DataFrame()
    if return_node_df and NODE_CSV.is_file():
        df_node = pd.read_csv(NODE_CSV)
    return df_sample, df_node


if __name__ == "__main__":
    generate_original_style_dataset_ieee34_mirzaei(
        n_scenarios=2,
        k_snapshots_per_scenario_total=8,
        master_seed=3420230,
        write_mvagg_compat=True,
        delete_raw_node_csv_after_mvagg=False,
    )
