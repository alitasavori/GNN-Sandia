"""
One-stop experiment:

1) Generate a reduced-size "original + capacitor-Q" dataset:
   - Same as ORIGINAL (p_load, q_load, p_pv, q_pv) but adds:
       q_cap_kvar (pre-solve nominal capacitor kvar per node)
   - Uses 1/3 as many snapshots per scenario (default: 320 vs 960)
   - Writes to: datasets_gnn2/original_plus_cap/

2) Train a PF-identity GNN on this dataset and save the best checkpoint.

3) Run a 24h baseline day (sigma=0, inj.BASELINE) and compare:
   - OpenDSS voltage (reference)
   - A model trained on ORIGINAL dataset (user-provided checkpoint)
   - The newly trained ORIGINAL+CAP model

Outputs:
 - Dataset CSVs in datasets_gnn2/original_plus_cap/
 - Model checkpoint in models_gnn2/original_plus_cap/
 - Plots in output plots/original_plus_cap_compare_*.png

Usage (from repo root):
  python run_original_plus_cap_dataset_train_compare.py --original-ckpt models_gnn2/original/light_xwide_emb_depth4_best.pt
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import opendssdirect as dss

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_best7_train import PFIdentityGNN, train_one


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

OUT_DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "original_plus_cap")
OUT_MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "original_plus_cap")
OUT_PLOTS_DIR = os.path.join(BASE_DIR, "output plots")

EDGE_CSV = os.path.join(OUT_DATASET_DIR, "gnn_edges_phase_static.csv")
NODE_CSV = os.path.join(OUT_DATASET_DIR, "gnn_node_features_and_targets.csv")
SAMPLE_CSV = os.path.join(OUT_DATASET_DIR, "gnn_sample_meta.csv")
NODE_INDEX_CSV = os.path.join(OUT_DATASET_DIR, "gnn_node_index_master.csv")


FEAT_ORIGINAL_PLUS_CAP = ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar", "q_cap_kvar"]


def _ensure_dirs() -> None:
    os.makedirs(OUT_DATASET_DIR, exist_ok=True)
    os.makedirs(OUT_MODELS_DIR, exist_ok=True)
    os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
    # Ensure shared training output dir from run_gnn3_best7_train also exists,
    # so torch.save(...) in train_one does not fail on a fresh clone (e.g. Colab).
    os.makedirs("gnn3_best7_output", exist_ok=True)


def _get_master_95_nodes_and_index() -> tuple[list[str], dict[str, int]]:
    node_names_master, _, _, _ = inj.get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}
    return node_names_master, node_to_idx_master


def _get_89_nodes_from_master(node_names_master: list[str]) -> list[str]:
    return [n for n in node_names_master if n.split(".")[0] not in inj.EXCLUDED_UPSTREAM_BUSES]


def generate_dataset_original_plus_cap(
    *,
    n_scenarios: int = 200,
    k_snapshots_per_scenario_total: int = 320,  # 1/3 of 960
    bins_by_profile: dict | None = None,
    include_anchors: bool = True,
    master_seed: int = 20260130,
    loadshape_name: str = "5minDayShape",
    irradshape_name: str = "IrradShape",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset generator: original features + q_cap_kvar (pre-solve nominal per-node)."""
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

    _ensure_dirs()
    dss_path = inj.compile_once()
    inj.setup_daily()

    node_names_master, node_to_idx_master = _get_master_95_nodes_and_index()
    pd.DataFrame(
        {"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}
    ).to_csv(NODE_INDEX_CSV, index=False)

    inj.extract_static_phase_edges_to_csv(node_names_master=node_names_master, edge_csv_path=EDGE_CSV)

    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, irradshape_name)
    csvL = inj.resolve_csv_path(csvL_token, dss_path)
    csvPV = inj.resolve_csv_path(csvPV_token, dss_path)
    mL = inj.read_profile_csv_two_col_noheader(csvL, npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(csvPV, npts=inj.NPTS, debug=False)

    rng_master = np.random.default_rng(master_seed)
    rows_sample: list[dict] = []
    rows_node: list[dict] = []
    sample_id = 0
    kept = 0
    skipped_nonconv = 0
    skipped_badV = 0

    for s in range(n_scenarios):
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        inj._apply_voltage_bases()
        inj.setup_daily()

        node_names_s, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
        if len(node_names_s) != len(node_names_master):
            raise RuntimeError(f"Scenario {s}: node count mismatch")

        loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
        pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()

        sc = inj.sample_scenario_from_baseline(inj.BASELINE, inj.RANGES, rng_master)
        P_load = sc["P_load_total_kw"]
        Q_load = sc["Q_load_total_kvar"]
        P_pv = sc["P_pv_total_kw"]
        sigL = sc["sigma_load"]
        sigPV = sc["sigma_pv"]
        prof_load, prof_pv = mL, mPV
        prof_net = (P_load * mL) - (P_pv * mPV)

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
        times = [int(t) for t in times]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        for t in times:
            inj.set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = inj.apply_snapshot_timeconditioned(
                P_load_total_kw=P_load,
                Q_load_total_kvar=Q_load,
                P_pv_total_kw=P_pv,
                mL_t=float(mL[t]),
                mPV_t=float(mPV[t]),
                loads_dss=loads_dss,
                dev_to_dss_load=dev_to_dss_load,
                dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss,
                pv_to_dss=pv_to_dss,
                pv_to_busph=pv_to_busph,
                sigma_load=float(sigL),
                sigma_pv=float(sigPV),
                rng=rng_solve,
            )

            try:
                dss.Solution.Solve()
            except Exception:
                pass
            if not dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
            vmag_m, vang_m = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            vmag_arr = np.asarray(vmag_m, dtype=float)
            if (
                (not np.isfinite(vmag_arr).all())
                or (vmag_arr.min() < inj.VMAG_PU_MIN)
                or (vmag_arr.max() > inj.VMAG_PU_MAX)
            ):
                skipped_badV += 1
                continue

            vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

            rows_sample.append(
                {
                    "sample_id": sample_id,
                    "scenario_id": s,
                    "t_index": t,
                    "t_minutes": t * inj.STEP_MIN,
                    "P_load_total_kw": float(P_load),
                    "Q_load_total_kvar": float(Q_load),
                    "P_pv_total_kw": float(P_pv),
                    "sigma_load": float(sigL),
                    "sigma_pv": float(sigPV),
                    "m_loadshape": float(mL[t]),
                    "m_irradshape": float(mPV[t]),
                    "P_load_time_kw": float(totals["P_load_time_kw"]),
                    "Q_load_time_kvar": float(totals["Q_load_time_kvar"]),
                    "P_pv_time_kw": float(totals["P_pv_time_kw"]),
                    "p_load_kw_set_total": float(totals["p_load_kw_set_total"]),
                    "q_load_kvar_set_total": float(totals["q_load_kvar_set_total"]),
                    "p_pv_pmpp_kw_set_total": float(totals["p_pv_pmpp_kw_set_total"]),
                    "prof_load": float(prof_load[t]),
                    "prof_pv": float(prof_pv[t]),
                    "prof_net": float(prof_net[t]),
                }
            )

            for n in node_names_master:
                bus, phs = n.split(".")
                ph = int(phs)
                if bus in inj.EXCLUDED_UPSTREAM_BUSES:
                    continue

                p_load_node = float(busphP_load.get((bus, ph), 0.0))
                q_load_node = float(busphQ_load.get((bus, ph), 0.0))
                p_pv_node = float(busphP_pv_actual.get((bus, ph), 0.0))
                q_pv_node = float(busphQ_pv_actual.get((bus, ph), 0.0))
                q_cap_node = float(inj.cap_q_kvar_per_node(bus, ph))  # pre-solve nominal per-node
                vm, va = vdict_m.get(n, (np.nan, np.nan))

                rows_node.append(
                    {
                        "sample_id": sample_id,
                        "node": n,
                        "node_idx": int(node_to_idx_master[n]),
                        "bus": bus,
                        "phase": int(ph),
                        "p_load_kw": p_load_node,
                        "q_load_kvar": q_load_node,
                        "p_pv_kw": p_pv_node,
                        "q_pv_kvar": q_pv_node,
                        "q_cap_kvar": q_cap_node,
                        "vmag_pu": float(vm),
                        "vang_deg": float(va),
                    }
                )

            sample_id += 1
            kept += 1

        print(
            f"[scenario {s+1}/{n_scenarios}] kept={kept} "
            f"skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} "
            f"Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}"
        )

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV, index=False)
    df_node.to_csv(NODE_CSV, index=False)
    return df_sample, df_node


@dataclass
class LoadedModel:
    model: PFIdentityGNN
    node_in_dim: int
    use_phase_onehot: bool


def _load_model(ckpt_path: str, device: str) -> LoadedModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mdl = PFIdentityGNN(
        num_nodes=int(cfg["N"]),
        num_edges=int(cfg["E"]),
        node_in_dim=int(cfg["node_in_dim"]),
        edge_in_dim=int(cfg["edge_in_dim"]),
        out_dim=int(cfg["out_dim"]),
        node_emb_dim=int(cfg["node_emb_dim"]),
        edge_emb_dim=int(cfg["edge_emb_dim"]),
        h_dim=int(cfg["h_dim"]),
        num_layers=int(cfg["num_layers"]),
        use_norm=bool(cfg.get("use_norm", False)),
    ).to(device)
    mdl.load_state_dict(ckpt["state_dict"])
    mdl.eval()
    return LoadedModel(model=mdl, node_in_dim=int(cfg["node_in_dim"]), use_phase_onehot=bool(cfg.get("use_phase_onehot", False)))


def _build_x_original(node_names_89: list[str], busphP_load, busphQ_load, busphP_pv_actual, busphQ_pv_actual) -> np.ndarray:
    X = np.zeros((len(node_names_89), 4), dtype=np.float32)
    for i, n in enumerate(node_names_89):
        bus, phs = n.split(".")
        ph = int(phs)
        X[i, 0] = float(busphP_load.get((bus, ph), 0.0))
        X[i, 1] = float(busphQ_load.get((bus, ph), 0.0))
        X[i, 2] = float(busphP_pv_actual.get((bus, ph), 0.0))
        X[i, 3] = float(busphQ_pv_actual.get((bus, ph), 0.0))
    return X


def _build_x_original_plus_cap(node_names_89: list[str], busphP_load, busphQ_load, busphP_pv_actual, busphQ_pv_actual) -> np.ndarray:
    X = np.zeros((len(node_names_89), 5), dtype=np.float32)
    for i, n in enumerate(node_names_89):
        bus, phs = n.split(".")
        ph = int(phs)
        X[i, 0] = float(busphP_load.get((bus, ph), 0.0))
        X[i, 1] = float(busphQ_load.get((bus, ph), 0.0))
        X[i, 2] = float(busphP_pv_actual.get((bus, ph), 0.0))
        X[i, 3] = float(busphQ_pv_actual.get((bus, ph), 0.0))
        X[i, 4] = float(inj.cap_q_kvar_per_node(bus, ph))
    return X


@torch.no_grad()
def compare_daily_profiles(
    *,
    original_ckpt: str,
    plus_cap_ckpt: str,
    nodes_to_plot: list[str] | None,
    device: str,
) -> None:
    """Run a 24h baseline day and save plots + print MAE/RMSE."""
    _ensure_dirs()

    # Common 89-node list (deterministic, matches train_one mapping)
    node_names_master, _ = _get_master_95_nodes_and_index()
    node_names_89 = _get_89_nodes_from_master(node_names_master)
    node_set_89 = set(node_names_89)
    if nodes_to_plot:
        nodes_to_plot = [n for n in nodes_to_plot if n in node_set_89]
    # If no explicit list is provided (or all were invalid), default to all 89 nodes
    if not nodes_to_plot:
        nodes_to_plot = list(node_names_89)

    # Load models
    mdl_orig = _load_model(original_ckpt, device=device)
    mdl_cap = _load_model(plus_cap_ckpt, device=device)
    if mdl_orig.node_in_dim != 4:
        raise ValueError(f"Expected ORIGINAL ckpt node_in_dim=4, got {mdl_orig.node_in_dim} ({original_ckpt})")
    if mdl_cap.node_in_dim != 5:
        raise ValueError(f"Expected PLUS-CAP ckpt node_in_dim=5, got {mdl_cap.node_in_dim} ({plus_cap_ckpt})")

    # Need the static graph tensors from the PLUS-CAP checkpoint (same topology)
    ckpt_static = torch.load(plus_cap_ckpt, map_location="cpu", weights_only=False)
    edge_index = ckpt_static["edge_index"].to(device)
    edge_attr = ckpt_static["edge_attr"].to(device)
    edge_id = ckpt_static["edge_id"].to(device)

    # DSS baseline day setup (sigma=0)
    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvL_token, dss_path), npts=inj.NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvPV_token, dss_path), npts=inj.NPTS, debug=False)

    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng_det = np.random.default_rng(0)

    # Time series containers
    T = inj.NPTS
    t_hours = np.arange(T) * inj.STEP_MIN / 60.0
    V_dss = np.full((T, len(node_names_89)), np.nan, dtype=np.float32)
    V_orig = np.full((T, len(node_names_89)), np.nan, dtype=np.float32)
    V_cap = np.full((T, len(node_names_89)), np.nan, dtype=np.float32)

    node_to_idx = {n: i for i, n in enumerate(node_names_89)}

    for t in range(T):
        inj.set_time_index(t)
        inj.apply_snapshot_timeconditioned(
            P_load_total_kw=float(inj.BASELINE["P_load_total_kw"]),
            Q_load_total_kvar=float(inj.BASELINE["Q_load_total_kvar"]),
            P_pv_total_kw=float(inj.BASELINE["P_pv_total_kw"]),
            mL_t=float(mL[t]),
            mPV_t=float(mPV[t]),
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=0.0,
            sigma_pv=0.0,
            rng=rng_det,
        )
        try:
            dss.Solution.Solve()
        except Exception:
            pass
        if not dss.Solution.Converged():
            continue

        # Build baseline features (post-solve PV P/Q)
        busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)

        # Reconstruct busphP_load/Q_load from the applied snapshot (deterministic, sigma=0) by re-applying helper:
        # We already set the DSS loads via apply_snapshot_timeconditioned; to keep feature alignment with training,
        # we reuse the same aggregation dictionaries returned by lt._apply_snapshot_with_per_type (no extra noise).
        _, busphP_load, busphQ_load, _, _, _ = lt._apply_snapshot_with_per_type(
            P_load_total_kw=float(inj.BASELINE["P_load_total_kw"]),
            Q_load_total_kvar=float(inj.BASELINE["Q_load_total_kvar"]),
            P_pv_total_kw=float(inj.BASELINE["P_pv_total_kw"]),
            mL_t=float(mL[t]),
            mPV_t=float(mPV[t]),
            loads_dss=loads_dss,
            dev_to_dss_load=dev_to_dss_load,
            dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss,
            pv_to_dss=pv_to_dss,
            pv_to_busph=pv_to_busph,
            sigma_load=0.0,
            sigma_pv=0.0,
            rng=rng_det,
        )

        # DSS voltages (targets)
        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names_89)
        V_dss[t, :] = np.asarray(vmag_m, dtype=np.float32)

        X_orig = _build_x_original(node_names_89, busphP_load, busphQ_load, busphP_pv_actual, busphQ_pv_actual)
        X_cap = _build_x_original_plus_cap(node_names_89, busphP_load, busphQ_load, busphP_pv_actual, busphQ_pv_actual)

        x1 = torch.tensor(X_orig, dtype=torch.float32, device=device)
        x2 = torch.tensor(X_cap, dtype=torch.float32, device=device)

        from torch_geometric.data import Data

        g1 = Data(x=x1, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=len(node_names_89))
        g2 = Data(x=x2, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=len(node_names_89))

        y1 = mdl_orig.model(g1).detach().cpu().numpy().astype(np.float32).reshape(-1)
        y2 = mdl_cap.model(g2).detach().cpu().numpy().astype(np.float32).reshape(-1)
        V_orig[t, :] = y1
        V_cap[t, :] = y2

    # Report MAE/RMSE over all nodes/time (finite)
    def _metrics(vhat: np.ndarray) -> tuple[float, float]:
        ok = np.isfinite(V_dss) & np.isfinite(vhat)
        if ok.sum() == 0:
            return float("nan"), float("nan")
        err = (vhat[ok] - V_dss[ok]).astype(np.float64)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        return mae, rmse

    mae_o, rmse_o = _metrics(V_orig)
    mae_c, rmse_c = _metrics(V_cap)
    print("\n=== 24h Baseline (sigma=0) metrics over all nodes/time ===")
    print(f"  ORIGINAL ckpt:       MAE={mae_o:.6f}  RMSE={rmse_o:.6f}  ({original_ckpt})")
    print(f"  ORIGINAL+CAP ckpt:   MAE={mae_c:.6f}  RMSE={rmse_c:.6f}  ({plus_cap_ckpt})")

    # Plot selected nodes
    for n in nodes_to_plot:
        idx = node_to_idx[n]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_hours, V_dss[:, idx], "b-", linewidth=2, label="OpenDSS |V| (pu)")
        ax.plot(t_hours, V_orig[:, idx], color="orange", linestyle="--", linewidth=1.5, label=f"Original GNN (MAE={np.nanmean(np.abs(V_orig[:, idx]-V_dss[:, idx])):.4f})")
        ax.plot(t_hours, V_cap[:, idx], "g:", linewidth=1.5, label=f"Orig+Cap GNN (MAE={np.nanmean(np.abs(V_cap[:, idx]-V_dss[:, idx])):.4f})")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Voltage magnitude (pu)")
        ax.set_title(f"24h voltage @ {n}")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUT_PLOTS_DIR, f"original_plus_cap_compare_{n.replace('.', '_')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-ckpt", required=True, help="Checkpoint trained on datasets_gnn2/original (node_in_dim=4).")
    p.add_argument("--n-scenarios", type=int, default=200)
    p.add_argument("--k-per-scenario", type=int, default=320, help="Snapshots per scenario (default 320 = 1/3 of 960).")
    p.add_argument("--seed", type=int, default=20260130)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument(
        "--plot-nodes",
        nargs="*",
        default=None,
        help="Optional list of nodes to plot; if omitted, plots are saved for all 89 nodes.",
    )
    # Training hyperparams (single fixed candidate; you can edit/extend later)
    p.add_argument("--cfg-name", default="original_plus_cap_h128_depth4")
    p.add_argument("--h-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-emb", type=int, default=16)
    p.add_argument("--e-emb", type=int, default=8)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("ORIGINAL + CAP: generate 1/3 dataset, train, and compare 24h profile")
    print("=" * 80)
    print(f"Dataset out: {OUT_DATASET_DIR}")
    print(f"Models out : {OUT_MODELS_DIR}")
    print(f"Plots out  : {OUT_PLOTS_DIR}")
    print(f"Original ckpt: {args.original_ckpt}")

    # 1) Generate dataset
    print("\n[1/3] Generating dataset...")
    generate_dataset_original_plus_cap(
        n_scenarios=int(args.n_scenarios),
        k_snapshots_per_scenario_total=int(args.k_per_scenario),
        bins_by_profile={"load": 10, "pv": 10, "net": 10},
        include_anchors=True,
        master_seed=int(args.seed),
    )
    print(f"[done] Dataset saved under {OUT_DATASET_DIR}")

    # 2) Train model
    print("\n[2/3] Training model on ORIGINAL+CAP dataset...")
    ckpt_path = train_one(
        block_id=1205,
        cfg_name=str(args.cfg_name),
        out_dir=OUT_DATASET_DIR,
        feature_cols=FEAT_ORIGINAL_PLUS_CAP,
        target_col="vmag_pu",
        n_emb=int(args.n_emb),
        e_emb=int(args.e_emb),
        h_dim=int(args.h_dim),
        n_layers=int(args.n_layers),
        use_norm=False,
        use_phase_onehot=False,
        early_stop=True,
    )
    if ckpt_path is None:
        raise RuntimeError("Training failed or was skipped; no checkpoint produced.")

    out_ckpt = os.path.join(OUT_MODELS_DIR, f"{args.cfg_name}_best.pt")
    torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), out_ckpt)
    print(f"[saved] Trained ORIGINAL+CAP checkpoint -> {out_ckpt}")

    # 3) Compare baseline 24h overlays
    print("\n[3/3] Comparing daily profiles (OpenDSS vs Original vs Original+Cap)...")
    compare_daily_profiles(
        original_ckpt=str(args.original_ckpt),
        plus_cap_ckpt=str(out_ckpt),
        nodes_to_plot=list(args.plot_nodes),
        device=str(args.device),
    )

    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

