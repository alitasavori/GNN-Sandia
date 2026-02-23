"""
24h voltage profile overlay for the 7 best GNN3 models.
Compares OpenDSS solve time vs GNN inference time for each model.
Run from repo root. Requires: trained checkpoints in gnn3_best7_output/,
opendssdirect, run_injection_dataset, run_loadtype_dataset, run_deltav_dataset.
Saves overlay plots to gnn3_best7_output/ (same folder as checkpoints).
"""
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opendssdirect as dss
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_deltav_dataset import _apply_snapshot_zero_pv
from run_gnn3_best7_train import PFIdentityGNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

CAP_Q_KVAR = {"844": 100.0, "848": 150.0}
SOURCE_BUSES = ("sourcebus", "800")
DIR_LOADTYPE = "gnn_samples_loadtype_full"
OUTPUT_DIR = "gnn3_best7_output"
NPTS = 288
STEP_MIN = 5
P_BASE, Q_BASE, PV_BASE = 1415.2, 835.2, 1000.0
OBSERVED_NODE = "840.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_bus_to_phases_from_master_nodes(node_names_master):
    bus_to_phases = {}
    for n in node_names_master:
        bus, phs = n.split(".")
        ph = int(phs)
        bus_to_phases.setdefault(bus, set()).add(ph)
    return {b: sorted(list(s)) for b, s in bus_to_phases.items()}


def build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv):
    """Original (3 feat): p_load_kw, q_load_kvar, p_pv_kw per node."""
    X = np.zeros((len(node_names_master), 3), dtype=np.float32)
    for i, n in enumerate(node_names_master):
        bus, phs = n.split(".")
        ph = int(phs)
        X[i, 0] = float(busphP_load.get((bus, ph), 0.0))
        X[i, 1] = float(busphQ_load.get((bus, ph), 0.0))
        X[i, 2] = float(busphP_pv.get((bus, ph), 0.0))
    return X


def build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid):
    """Injection (2 feat): p_inj_kw, q_inj_kvar per node. Source bus uses grid P/Q; others: p_inj=p_pv-p_load, q_inj=-q_load+cap_Q."""
    P_grid_per_ph = P_grid / 3.0
    Q_grid_per_ph = Q_grid / 3.0
    X = np.zeros((len(node_names_master), 2), dtype=np.float32)
    for i, n in enumerate(node_names_master):
        bus, phs = n.split(".")
        ph = int(phs)
        p_load = float(busphP_load.get((bus, ph), 0.0))
        q_load = float(busphQ_load.get((bus, ph), 0.0))
        p_pv = float(busphP_pv.get((bus, ph), 0.0))
        if bus == "sourcebus":
            p_inj = P_grid_per_ph
            q_inj = Q_grid_per_ph
        else:
            p_inj = p_pv - p_load
            q_inj = -q_load + float(CAP_Q_KVAR.get(bus, 0.0))
        X[i, 0] = p_inj
        X[i, 1] = q_inj
    return X


def build_gnn_x_loadtype(node_names_master, busph_per_type, busphP_pv, node_to_electrical_dist,
                         p_sys_balance, q_sys_balance):
    """Load-type (13 feat): electrical_distance, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv, p_sys, q_sys."""
    X = np.zeros((len(node_names_master), 13), dtype=np.float32)
    for i, n in enumerate(node_names_master):
        bus, phs = n.split(".")
        ph = int(phs)
        m1_p = float(busph_per_type[1][0].get((bus, ph), 0.0))
        m1_q = float(busph_per_type[1][1].get((bus, ph), 0.0))
        m2_p = float(busph_per_type[2][0].get((bus, ph), 0.0))
        m2_q = float(busph_per_type[2][1].get((bus, ph), 0.0))
        m4_p = float(busph_per_type[4][0].get((bus, ph), 0.0))
        m4_q = float(busph_per_type[4][1].get((bus, ph), 0.0))
        m5_p = float(busph_per_type[5][0].get((bus, ph), 0.0))
        m5_q = float(busph_per_type[5][1].get((bus, ph), 0.0))
        q_cap = float(CAP_Q_KVAR.get(bus, 0.0))
        p_pv = float(busphP_pv.get((bus, ph), 0.0))
        X[i, 0] = float(node_to_electrical_dist.get(n, 0.0))
        X[i, 1], X[i, 2] = m1_p, m1_q
        X[i, 3], X[i, 4] = m2_p, m2_q
        X[i, 5], X[i, 6] = m4_p, m4_q
        X[i, 7], X[i, 8] = m5_p, m5_q
        X[i, 9] = q_cap
        X[i, 10] = p_pv
        X[i, 11] = p_sys_balance
        X[i, 12] = q_sys_balance
    return X


def get_all_node_voltage_pu_and_angle_dict():
    node_names = list(dss.Circuit.AllNodeNames())
    nodes_by_bus = {}
    for n in node_names:
        if "." not in n:
            continue
        bus, ph = n.split(".")
        try:
            iph = int(ph)
        except Exception:
            continue
        if iph not in (1, 2, 3):
            continue
        nodes_by_bus.setdefault(bus, set()).add(iph)

    bus_cache = {}
    for bus in nodes_by_bus.keys():
        dss.Circuit.SetActiveBus(bus)
        bus_nodes = list(dss.Bus.Nodes())
        if hasattr(dss.Bus, "puVmagAngle"):
            arr = list(dss.Bus.puVmagAngle())
            mags = arr[0::2]
            angs = arr[1::2]
            tmp = {int(nn): (float(m), float(a)) for nn, m, a in zip(bus_nodes, mags, angs)}
        else:
            arr = list(dss.Bus.VMagAngle())
            mags_v = arr[0::2]
            angs = arr[1::2]
            kvbase = float(dss.Bus.kVBase())
            vbase = kvbase * 1000.0 if kvbase > 0 else np.nan
            tmp = {int(nn): (float(mv) / vbase, float(a)) for nn, mv, a in zip(bus_nodes, mags_v, angs)}
        bus_cache[bus] = tmp

    out = {}
    for bus, phs in nodes_by_bus.items():
        for ph in phs:
            nm = f"{bus}.{ph}"
            if ph in bus_cache.get(bus, {}):
                out[nm] = bus_cache[bus][ph]
    return out


def _strip_quotes(s):
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def find_loadshape_csv_in_dss(dss_path, loadshape_name):
    txt = open(dss_path, "r", encoding="utf-8", errors="ignore").read()
    pat = re.compile(rf"(?im)^\s*new\s+loadshape\.{re.escape(loadshape_name)}\b.*?$", re.MULTILINE)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Could not find 'New Loadshape.{loadshape_name}' in DSS file.")
    line = m.group(0)
    m2 = re.search(r"(?i)\bcsvfile\s*=\s*([^\s]+)", line)
    if not m2:
        raise RuntimeError(f"Loadshape.{loadshape_name} found, but no csvfile=... on that line:\n{line}")
    csv_token = _strip_quotes(m2.group(1))
    return csv_token, line


def resolve_csv_path(csv_token, dss_path):
    csv_token = csv_token.replace("\\", "/")
    if os.path.isabs(csv_token):
        return csv_token
    base = os.path.dirname(dss_path)
    return os.path.abspath(os.path.join(base, csv_token))


def read_profile_csv_two_col_noheader(csv_path, npts=NPTS):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise RuntimeError(f"{csv_path} must have at least 2 columns (time,value).")
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(y)
    y = y[ok]
    if len(y) < npts:
        raise RuntimeError(f"{csv_path} has {len(y)} valid points < required {npts}")
    return y[:npts]


def load_model_for_inference(path, device=None):
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location="cpu")
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
    static = {
        "config": cfg,
        "edge_index": ckpt["edge_index"],
        "edge_attr": ckpt["edge_attr"],
        "edge_id": ckpt["edge_id"],
        "N": int(cfg["N"]),
        "best_rmse": ckpt.get("best_rmse"),
    }
    return mdl, static


def _parse_phase_from_node_name(name):
    m = re.search(r"\.(\d+)$", str(name))
    return int(m.group(1)) - 1 if m else 0


@torch.no_grad()
def voltage_profile_overlay_24h(ckpt_path, scenario_name, device=None, verbose=True):
    """Run 24h overlay for one model. Returns (df, dss_total_s, gnn_total_s)."""
    if device is None:
        device = DEVICE

    model, static = load_model_for_inference(ckpt_path, device=device)
    cfg = static["config"]
    node_in_dim = int(cfg["node_in_dim"])
    target_col = cfg.get("target_col", "vmag_pu")
    use_phase_onehot = bool(cfg.get("use_phase_onehot", False))
    dataset_dir = cfg.get("dataset", DIR_LOADTYPE)
    is_deltav = target_col == "vmag_delta_pu"

    node_index_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(node_index_csv):
        raise FileNotFoundError(f"Missing {node_index_csv}")
    master_df = pd.read_csv(node_index_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx").reset_index(drop=True)
    node_names_master = master_df["node"].astype(str).tolist()
    N_expected = static["N"]
    if len(node_names_master) != N_expected:
        raise RuntimeError(f"MASTER node count {len(node_names_master)} != model expects {N_expected}.")

    if OBSERVED_NODE not in set(node_names_master):
        raise RuntimeError(f"observed_node='{OBSERVED_NODE}' not in MASTER.")
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    obs_idx = node_to_idx[OBSERVED_NODE]

    dss_path = inj.compile_once()
    inj.setup_daily()

    csvL_token, _ = find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = read_profile_csv_two_col_noheader(resolve_csv_path(csvL_token, dss_path), npts=NPTS)
    mPV = read_profile_csv_two_col_noheader(resolve_csv_path(csvPV_token, dss_path), npts=NPTS)

    bus_to_phases = build_bus_to_phases_from_master_nodes(node_names_master)
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng_det = np.random.default_rng(0)
    edge_csv = os.path.join(dataset_dir, "gnn_edges_phase_static.csv")
    if not os.path.exists(edge_csv):
        edge_csv = os.path.join(DIR_LOADTYPE, "gnn_edges_phase_static.csv")
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(node_names_master, edge_csv)

    phase_map = None
    if use_phase_onehot:
        phase_map = np.array([_parse_phase_from_node_name(n) for n in node_names_master], dtype=np.int64)
        ph_oh = np.eye(3, dtype=np.float32)[phase_map]

    edge_index = static["edge_index"].to(device)
    edge_attr = static["edge_attr"].to(device)
    edge_id = static["edge_id"].to(device)

    dss_solve_times = []
    gnn_times = []
    t_hours, vmag_dss, vmag_gnn = [], [], []
    nonconv = 0
    use_cuda_timer = device.startswith("cuda") and torch.cuda.is_available()

    for t in range(NPTS):
        inj.set_time_index(t)
        dss_time_step = 0.0

        if is_deltav:
            totals_z, busphP_load, busphQ_load, busphP_pv_z, busphQ_pv_z, busph_per_type = _apply_snapshot_zero_pv(
                P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, mL_t=float(mL[t]),
                loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
                sigma_load=0.0, rng=rng_det,
            )
            t0_dss = time.perf_counter()
            dss.Solution.Solve()
            dss_time_step += time.perf_counter() - t0_dss
            if not dss.Solution.Converged():
                nonconv += 1
                t_hours.append(t * STEP_MIN / 60.0)
                vmag_dss.append(np.nan)
                vmag_gnn.append(np.nan)
                gnn_times.append(np.nan)
                dss_solve_times.append(dss_time_step)
                continue
            vdict_z = get_all_node_voltage_pu_and_angle_dict()
            vmag_zero = np.array([float(vdict_z.get(n, (np.nan, 0))[0]) for n in node_names_master], dtype=np.float32)

        _, busphP_load, busphQ_load, busphP_pv, busphQ_pv, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE, Q_load_total_kvar=Q_BASE, P_pv_total_kw=PV_BASE,
            mL_t=float(mL[t]), mPV_t=float(mPV[t]),
            loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
            pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
            sigma_load=0.0, sigma_pv=0.0, rng=rng_det,
        )

        t0_dss = time.perf_counter()
        dss.Solution.Solve()
        dss_time_step += time.perf_counter() - t0_dss
        dss_solve_times.append(dss_time_step)

        if not dss.Solution.Converged():
            nonconv += 1
            t_hours.append(t * STEP_MIN / 60.0)
            vmag_dss.append(np.nan)
            vmag_gnn.append(np.nan)
            gnn_times.append(np.nan)
            continue

        vdict = get_all_node_voltage_pu_and_angle_dict()
        vm_dss, _ = vdict[OBSERVED_NODE]

        if dataset_dir == "gnn_samples_out":
            X = build_gnn_x_original(node_names_master, busphP_load, busphQ_load, busphP_pv)
        elif dataset_dir == "gnn_samples_inj_full":
            pwr = dss.Circuit.TotalPower()
            P_grid = -float(pwr[0])
            Q_grid = -float(pwr[1])
            X = build_gnn_x_injection(node_names_master, busphP_load, busphQ_load, busphP_pv, P_grid, Q_grid)
        else:
            sum_p_load = float(sum(busphP_load.values()))
            sum_q_load = float(sum(busphQ_load.values()))
            sum_p_pv = float(sum(busphP_pv.values()))
            sum_q_cap = float(sum(CAP_Q_KVAR.values()))
            p_sys_balance = sum_p_load - sum_p_pv
            q_sys_balance = sum_q_load - sum_q_cap
            X = build_gnn_x_loadtype(
                node_names_master, busph_per_type, busphP_pv,
                node_to_electrical_dist, p_sys_balance, q_sys_balance,
            )
        if is_deltav:
            X = np.concatenate([X, vmag_zero[:, None]], axis=-1)
        if use_phase_onehot:
            X = np.concatenate([X, ph_oh], axis=-1)

        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        g = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, edge_id=edge_id, num_nodes=N_expected)

        if use_cuda_timer:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yhat = model(g)
            end.record()
            torch.cuda.synchronize()
            gnn_ms = float(start.elapsed_time(end))
            gnn_times.append(gnn_ms / 1000.0)
        else:
            t0_gnn = time.perf_counter()
            yhat = model(g)
            gnn_times.append(time.perf_counter() - t0_gnn)

        if is_deltav:
            delta_pred = float(yhat[obs_idx, 0].item())
            vm_gnn = vmag_zero[obs_idx] + delta_pred
        else:
            vm_gnn = float(yhat[obs_idx, 0].item())

        t_hours.append(t * STEP_MIN / 60.0)
        vmag_dss.append(float(vm_dss))
        vmag_gnn.append(vm_gnn)

    dss_total = float(np.nansum(dss_solve_times))
    gnn_total = float(np.nansum(gnn_times))

    if verbose:
        print(f"  {scenario_name}: OpenDSS total={dss_total:.3f}s | GNN total={gnn_total:.3f}s | nonconv={nonconv}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, vmag_dss, label="OpenDSS |V| (pu)")
    ax.plot(t_hours, vmag_gnn, label="GNN |V| (pu)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"Block {scenario_name}: Voltage Profile @ {OBSERVED_NODE} (24h)")
    ax.grid(True)
    ax.legend()
    best_rmse = static.get("best_rmse")
    gnn_desc = f"GNN: h={cfg['h_dim']} layers={cfg['num_layers']} target={target_col}"
    if best_rmse is not None:
        gnn_desc += f" | RMSE={best_rmse:.6f}"
    fig.text(0.02, 0.02, gnn_desc, fontsize=8, family="monospace",
             verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    img_path = os.path.join(OUTPUT_DIR, f"overlay_24h_block{scenario_name}.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {img_path}")

    out = pd.DataFrame({
        "hour": t_hours, "vmag_dss_pu": vmag_dss, "vmag_gnn_pu": vmag_gnn,
        "t_dss_solve_s": dss_solve_times, "t_gnn_forward_s": gnn_times,
    })
    return out, dss_total, gnn_total


def main():
    print("=" * 70)
    print("GNN3 BEST 7: 24h voltage profile overlay (OpenDSS vs GNN)")
    print("=" * 70)
    results = {}
    for block_id in range(1, 8):
        ckpt_path = os.path.join(OUTPUT_DIR, f"block{block_id}.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP Block {block_id}: checkpoint not found")
            continue
        print(f"\n>>> Block {block_id}")
        try:
            df, dss_s, gnn_s = voltage_profile_overlay_24h(
                ckpt_path, scenario_name=str(block_id), device=DEVICE, verbose=True
            )
            results[f"block{block_id}"] = (df, dss_s, gnn_s)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("INFERENCE SPEEDS (24h = 288 steps per model)")
    print("=" * 70)
    for name, (df, dss_s, gnn_s) in results.items():
        gnn_mean_ms = df["t_gnn_forward_s"].mean() * 1000
        print(f"  {name}: OpenDSS total={dss_s:.3f}s | GNN total={gnn_s:.3f}s | GNN mean/step={gnn_mean_ms:.2f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
