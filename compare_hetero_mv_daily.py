"""
Daily OpenDSS vs hetero-MV checkpoint comparison (8500 feeder).

Outputs:
  - Per-node 24h plots for requested nodes.
  - Per-node MAE CSV.
  - All-node error histogram.
  - Printed global MAE/RMSE vs OpenDSS.

Interpretation:
  - Each PyG node type (upstream, downstream, capacitor, load) has its own readout head.
    If training used --target-node-types load, only the load head is trained; other heads
    receive no gradient and stay near ~0, so predictions at upstream/capacitor/downstream
    buses look "flat" near the bottom of the plot.
  - Use --nodes with bus names that are in gnn_node_index_master and appear in the hetero
    node CSVs; the script maps each name to its storage type for the GNN.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import opendssdirect as dss

import run_injection_dataset as inj
import search_hetero_mv_gnn_architectures as hm
import run_daily_aggregate_dataset_8500 as rd8500


def _load_model(cfg_name: str, state_dict: dict, edge_index_dict: dict, device: torch.device) -> tuple[torch.nn.Module, bool]:
    use_gine = "gine" in cfg_name
    if use_gine:
        core = hm.HeteroTypedGINE(hm.NODE_TYPES, hm.IN_DIMS, 80, 3, 0.1, edge_index_dict).to(device)
        clean_sd = {k.replace("core.", "", 1): v for k, v in state_dict.items()}
        core.load_state_dict(clean_sd, strict=False)
        model = core
    elif "gat" in cfg_name:
        model = hm.HeteroTypedGAT(hm.NODE_TYPES, hm.IN_DIMS, 128, 4, 2, 0.1, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    elif "4x64" in cfg_name:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 64, 4, 0.15, True, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    elif "3x112" in cfg_name:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 112, 3, 0.05, False, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = hm.HeteroTypedSAGE(hm.NODE_TYPES, hm.IN_DIMS, 96, 2, 0.0, False, edge_index_dict).to(device)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, use_gine


def _build_feature_meta(nodes_dir: Path, name_to_gidx: dict[str, int], global_type: dict[int, str], g2l: dict[str, dict[int, int]], counts: dict[str, int]) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, tuple[str, int]], dict[str, dict[str, int]]]:
    typed_name = {t: [""] * counts[t] for t in hm.NODE_TYPES}
    typed_dist = {t: np.zeros(counts[t], dtype=np.float32) for t in hm.NODE_TYPES}
    rep: dict[int, dict[str, float | str]] = {}

    use_cols = ["node", "node_idx", "electrical_distance_ohm", "p_load_kw", "q_load_kvar", "q_capacitor_bank"]
    for kind, rel in hm.NODE_FILES.items():
        path = nodes_dir / rel
        try:
            reader = pd.read_csv(path, usecols=lambda c: c in use_cols, chunksize=400_000) if kind == "load" else [pd.read_csv(path, usecols=lambda c: c in use_cols)]
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied opening CSV: {path}\n"
                "Close this file in Excel (and any preview panes), then rerun."
            ) from e
        try:
            iterator = reader
            for chunk in iterator:
                for r in chunk.itertuples(index=False):
                    g = int(float(r.node_idx)) if pd.notna(r.node_idx) else name_to_gidx.get(str(r.node).strip().lower())
                    if g is None:
                        continue
                    g = int(g)
                    if g not in rep:
                        rep[g] = {
                            "node": str(r.node).strip().lower(),
                            "electrical_distance_ohm": float(getattr(r, "electrical_distance_ohm")) if pd.notna(getattr(r, "electrical_distance_ohm")) else 0.0,
                        }
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while reading CSV: {path}\n"
                "Close this file in Excel (and any preview panes), then rerun."
            ) from e

    for g, t in global_type.items():
        li = g2l[t][g]
        typed_name[t][li] = str(rep.get(g, {}).get("node", ""))
        typed_dist[t][li] = float(rep.get(g, {}).get("electrical_distance_ohm", 0.0))

    dss_to_typed: dict[str, tuple[str, int]] = {}
    for t in hm.NODE_TYPES:
        for i, n in enumerate(typed_name[t]):
            if n:
                dss_to_typed[n] = (t, i)

    typed_to_dss_idx: dict[str, dict[str, int]] = {t: {} for t in hm.NODE_TYPES}
    return typed_name, typed_dist, dss_to_typed, typed_to_dss_idx


def run_compare(
    checkpoint: Path,
    dataset_dir: Path,
    node_index: Path,
    out_dir: Path,
    plot_nodes: list[str],
    npts: int,
    step_min: int,
    ymin: float,
    ymax: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_dir = dataset_dir / "edges"
    nodes_dir = dataset_dir / "nodes"
    catalog = pd.read_csv(edges_dir / "hetero_mv_edge_catalog.csv")
    line_attr = pd.read_csv(edges_dir / "hetero_mv_line_edge_attr.csv")

    name_to_gidx = hm._read_node_idx_master(node_index)
    extra_names: set[str] = set()
    for fn in hm.NODE_FILES.values():
        df = pd.read_csv(nodes_dir / fn, usecols=["node"])
        extra_names.update(df["node"].astype(str).str.strip().tolist())

    g_list = hm._collect_global_node_indices(catalog, name_to_gidx, extra_names)
    membership = hm._membership_by_csv(nodes_dir)
    g2l, global_type, counts, edge_index_dict_cpu, line_ea_cpu = hm._build_typed_topology(catalog, line_attr, g_list, membership)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict_cpu.items()}
    line_ea = {k: v.to(device) for k, v in line_ea_cpu.items()}

    typed_name, typed_dist, dss_to_typed, _ = _build_feature_meta(nodes_dir, name_to_gidx, global_type, g2l, counts)

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_name = str(ck.get("cfg_name", checkpoint.stem))
    model, use_gine = _load_model(cfg_name, ck["state_dict"], edge_index_dict, device)
    print(f"[compare_hetero_mv_daily] model={cfg_name} device={device}")

    # Use the same baseline setup as run_daily_aggregate_dataset_8500.py:
    # compile daily entrypoint, detach Daily from loads, apply explicit mL[t] scaling.
    rd8500._compile_8500_daily_setup()
    rd8500._detach_daily_loadshape_from_loads()
    loads, load_to_busph = rd8500._collect_loads_and_maps()
    base_kw = np.array([float(d["kw"]) for d in loads], dtype=np.float64)
    base_kvar = np.array([float(d["kvar"]) for d in loads], dtype=np.float64)
    base_names = [str(d["name"]) for d in loads]
    mL = rd8500._daily_profile_5min(npts=npts)

    all_nodes = []
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

    typed_to_dss_idx: dict[str, np.ndarray] = {t: np.full(counts[t], -1, dtype=np.int64) for t in hm.NODE_TYPES}
    for n, (t, li) in dss_to_typed.items():
        if n in node_to_idx:
            typed_to_dss_idx[t][li] = node_to_idx[n]

    t_hours = np.arange(npts, dtype=np.float32) * (step_min / 60.0)
    v_dss = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)
    v_gnn = np.full((npts, len(all_nodes)), np.nan, dtype=np.float32)

    def _make_x_dict() -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for t in hm.NODE_TYPES:
            out[t] = np.zeros((counts[t], hm.IN_DIMS[t]), dtype=np.float32)
            if "electrical_distance_ohm" in hm.TYPE_FEAT_COLS[t]:
                ci = hm.TYPE_FEAT_COLS[t].index("electrical_distance_ohm")
                out[t][:, ci] = typed_dist[t]
        return out

    def _make_edge_attr_dict() -> dict[tuple[str, str, str], torch.Tensor]:
        out = dict(line_ea)
        for k, ei in edge_index_dict.items():
            if k[1] == "reg":
                out[k] = torch.zeros((ei.shape[1], 1), dtype=torch.float32, device=device)
        return out

    n_nonconv = 0
    scenario_scale = 1.0
    for i in range(npts):
        hr = int(i // 12)
        sec = int((i % 12) * (step_min * 60))
        dss.Text.Command(f"set hour={hr} sec={sec}")
        m_t = float(mL[i])
        total_scale_t = scenario_scale * m_t
        kw_set = base_kw * total_scale_t
        kvar_set = base_kvar * total_scale_t
        busphP_load: dict[tuple[str, int], float] = {}
        busphQ_load: dict[tuple[str, int], float] = {}
        for j, name in enumerate(base_names):
            dss.Loads.Name(name)
            dss.Loads.kW(float(kw_set[j]))
            dss.Loads.kvar(float(kvar_set[j]))
            for (bus, ph, w) in load_to_busph[name]:
                busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + float(kw_set[j]) * float(w)
                busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + float(kvar_set[j]) * float(w)

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            n_nonconv += 1
            continue

        vmag, _ = inj.get_all_node_voltage_pu_and_angle_filtered(all_nodes)
        v_dss[i, :] = np.asarray(vmag, dtype=np.float32)

        x_np = _make_x_dict()

        for (bus, ph), pval in busphP_load.items():
            node = f"{str(bus).strip().lower()}.{int(ph)}"
            tp = dss_to_typed.get(node)
            if tp is not None and tp[0] == "load":
                x_np["load"][tp[1], 0] = float(pval)
        for (bus, ph), qval in busphQ_load.items():
            node = f"{str(bus).strip().lower()}.{int(ph)}"
            tp = dss_to_typed.get(node)
            if tp is not None and tp[0] == "load":
                x_np["load"][tp[1], 1] = float(qval)

        dss.Capacitors.First()
        while True:
            cn = dss.Capacitors.Name()
            dss.Circuit.SetActiveElement(f"Capacitor.{cn}")
            buses = dss.CktElement.BusNames()
            if buses and len(buses) > 0:
                b = str(buses[0]).split(".")[0].strip().lower()
                try:
                    qnom = float(dss.Capacitors.kvar())
                    st = dss.Capacitors.States()
                    if isinstance(st, (list, tuple, np.ndarray)):
                        on = bool(np.any(np.asarray(st, dtype=float) > 0.5))
                    else:
                        on = float(st) > 0.5
                    q_now = qnom if on else 0.0
                except Exception:
                    q_now = 0.0
                for ph in (1, 2, 3):
                    node = f"{b}.{ph}"
                    tp = dss_to_typed.get(node)
                    if tp is not None and tp[0] == "capacitor":
                        li = tp[1]
                        x_np["capacitor"][li, 0] += q_now / 3.0
            if not dss.Capacitors.Next():
                break

        x_dict = {t: torch.from_numpy(x_np[t]).to(device) for t in hm.NODE_TYPES}
        with torch.no_grad():
            if use_gine:
                pred = model(x_dict, edge_index_dict, _make_edge_attr_dict())
            else:
                pred = model(x_dict, edge_index_dict)

        for t in hm.NODE_TYPES:
            arr = pred[t].detach().cpu().numpy()
            idxs = typed_to_dss_idx[t]
            good = idxs >= 0
            v_gnn[i, idxs[good]] = arr[good]

        if (i + 1) % 24 == 0:
            print(f"[{i + 1}/{npts}] collected", flush=True)

    mask = np.isfinite(v_dss) & np.isfinite(v_gnn)
    mae = float(np.mean(np.abs(v_dss[mask] - v_gnn[mask])))
    rmse = float(np.sqrt(np.mean((v_dss[mask] - v_gnn[mask]) ** 2)))
    print(f"\nOverall: MAE={mae:.6f} pu  RMSE={rmse:.6f} pu  n_points={int(mask.sum())} nonconv={n_nonconv}")

    node_rows = []
    for i, n in enumerate(all_nodes):
        m = np.isfinite(v_dss[:, i]) & np.isfinite(v_gnn[:, i])
        if m.any():
            node_rows.append((n, float(np.mean(np.abs(v_dss[m, i] - v_gnn[m, i])))))
    df_mae = pd.DataFrame(node_rows, columns=["node", "mae"]).sort_values("mae", ascending=False)
    df_mae.to_csv(out_dir / f"daily_mae_per_node_{cfg_name}.csv", index=False)

    for n in [str(x).strip().lower() for x in plot_nodes if str(x).strip().lower() in node_to_idx]:
        i = node_to_idx[n]
        tp = dss_to_typed.get(n)
        if tp is not None and ck_target_types is not None and tp[0] not in ck_target_types:
            print(
                f"[compare_hetero_mv_daily] warning: node {n!r} is hetero type {tp[0]!r} but checkpoint "
                f"only supervised {sorted(ck_target_types)} — GNN curve is not meaningful."
            )
        m = np.isfinite(v_dss[:, i]) & np.isfinite(v_gnn[:, i])
        n_mae = float(np.mean(np.abs(v_dss[m, i] - v_gnn[m, i]))) if m.any() else np.nan
        plt.figure(figsize=(10, 4.2))
        plt.plot(t_hours, v_dss[:, i], linewidth=2.0, label="OpenDSS baseline")
        plt.plot(t_hours, v_gnn[:, i], "--", linewidth=1.6, label=f"{cfg_name} (MAE={n_mae:.4f})")
        plt.xlabel("Hour of day")
        plt.ylabel("Voltage magnitude (pu)")
        plt.title(f"24h voltage @ {n}")
        plt.ylim(ymin, ymax)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"daily_compare_{cfg_name}_{n.replace('.', '_')}.png", dpi=160)
        plt.show()

    err = np.abs(v_dss[mask] - v_gnn[mask])
    plt.figure(figsize=(8.2, 4.2))
    plt.hist(err, bins=120, alpha=0.9)
    plt.xlabel("|V_gnn - V_dss| (pu)")
    plt.ylabel("Count")
    plt.title(f"Error distribution: {cfg_name} vs OpenDSS")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"daily_error_hist_{cfg_name}.png", dpi=170)
    plt.show()

    print("\nSaved:", out_dir.resolve())
    print(df_mae.head(10).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="8500 daily OpenDSS vs hetero checkpoint comparison")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to hetero *_best.pt checkpoint")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets_gnn2/loadtype_8500_dailyagg/Heterogenous GNN dataset"),
    )
    p.add_argument(
        "--node-index",
        type=Path,
        default=Path("datasets_gnn2/loadtype_8500_dailyagg/gnn_node_index_master.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("gnn2_daily_compare_8500_output"))
    p.add_argument("--nodes", type=str, default="m1026891.1,m1026891.2,m1026891.3", help="Comma-separated nodes to plot")
    p.add_argument("--npts", type=int, default=288)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--ymin", type=float, default=0.85)
    p.add_argument("--ymax", type=float, default=1.10)
    args = p.parse_args()

    run_compare(
        checkpoint=args.checkpoint.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        node_index=args.node_index.resolve(),
        out_dir=args.out_dir.resolve(),
        plot_nodes=[x.strip() for x in args.nodes.split(",") if x.strip()],
        npts=int(args.npts),
        step_min=int(args.step_min),
        ymin=float(args.ymin),
        ymax=float(args.ymax),
    )


if __name__ == "__main__":
    main()

