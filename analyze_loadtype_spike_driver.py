from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import run_injection_dataset as inj
import run_loadtype_dataset as lt
from run_gnn3_best7_train import LOADTYPE_FEAT
from run_gnn3_overlay_7 import (
    BASE_DIR,
    DIR_LOADTYPE,
    NPTS,
    P_BASE,
    PV_BASE,
    Q_BASE,
    STEP_MIN,
    build_gnn_x_loadtype,
    load_model_for_inference,
)

inj = importlib.reload(inj)
lt = importlib.reload(lt)

os.chdir(BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "gnn3_best7_output", "spike_driver")
DEFAULT_CKPT = os.path.join("gnn2_architecture_search", "loadtype", "best.pt")
VOLT_VAR_RELATED = {"p_pv_kw", "q_pv_kvar", "p_sys_balance_kw", "q_sys_balance_kvar"}


@dataclass
class DailyRun:
    node_names: list[str]
    hours: np.ndarray
    x14: np.ndarray
    v_dss: np.ndarray
    v_pred: np.ndarray
    valid_steps: np.ndarray
    node_in_dim: int
    use_phase_onehot: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose which load-type model input most drives a daily voltage spike."
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Load-type checkpoint to analyze.")
    p.add_argument("--node", required=True, help="Target node name, e.g. 858.2")
    p.add_argument("--focus-start", type=float, default=15.0, help="Window start hour.")
    p.add_argument("--focus-end", type=float, default=17.0, help="Window end hour.")
    p.add_argument("--top-k", type=int, default=3, help="Analyze this many spike timesteps in the window.")
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Defaults to gnn3_best7_output/spike_driver/...",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Defaults to gnn3_best7_output/spike_driver/...",
    )
    return p.parse_args()


def _sanitize_node(node: str) -> str:
    return node.replace(".", "_")


def _resolve_local_dataset_dir(dataset_dir: str | None) -> str:
    candidates: list[str] = []
    if dataset_dir:
        candidates.append(str(dataset_dir))
        norm = str(dataset_dir).replace("\\", os.sep).replace("/", os.sep)
        candidates.append(norm)
        tail = os.path.basename(norm.rstrip(os.sep))
        if tail:
            candidates.append(os.path.join(BASE_DIR, "datasets_gnn2", tail))
    candidates.append(DIR_LOADTYPE)

    seen = set()
    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if cand_abs in seen:
            continue
        seen.add(cand_abs)
        if os.path.exists(os.path.join(cand_abs, "gnn_node_index_master.csv")):
            return cand_abs

    raise FileNotFoundError(
        "Could not resolve a local dataset directory for this checkpoint. "
        f"Tried: {candidates}"
    )


def _resolve_reduced_node_list(dataset_dir: str, expected_n: int = 89) -> list[str]:
    node_csv = os.path.join(dataset_dir, "gnn_node_features_and_targets.csv")
    master_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(node_csv) or not os.path.exists(master_csv):
        raise FileNotFoundError(
            f"Need {node_csv} and {master_csv} to resolve the reduced node list."
        )
    df_n = pd.read_csv(node_csv)
    df_n["node_idx"] = pd.to_numeric(df_n["node_idx"], errors="raise").astype(int)
    kept_node_ids = sorted(df_n["node_idx"].unique())
    if len(kept_node_ids) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} reduced nodes per sample, got {len(kept_node_ids)}."
        )
    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    idx_to_name = master_df.set_index("node_idx")["node"].astype(str).to_dict()
    return [idx_to_name[idx] for idx in kept_node_ids]


def _get_full_master_node_list(dataset_dir: str) -> list[str]:
    master_csv = os.path.join(dataset_dir, "gnn_node_index_master.csv")
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Need {master_csv} for the full master node list.")
    master_df = pd.read_csv(master_csv)
    master_df["node_idx"] = pd.to_numeric(master_df["node_idx"], errors="raise").astype(int)
    master_df = master_df.sort_values("node_idx")
    return master_df["node"].astype(str).tolist()


def _nearest_valid_neighbors(valid_steps: np.ndarray, t: int) -> tuple[int | None, int | None]:
    prev_t = next((k for k in range(t - 1, -1, -1) if valid_steps[k]), None)
    next_t = next((k for k in range(t + 1, len(valid_steps)) if valid_steps[k]), None)
    return prev_t, next_t


def _neighbor_mean(series: np.ndarray, valid_steps: np.ndarray, t: int) -> float:
    prev_t, next_t = _nearest_valid_neighbors(valid_steps, t)
    vals = []
    if prev_t is not None:
        vals.append(float(series[prev_t]))
    if next_t is not None:
        vals.append(float(series[next_t]))
    if not vals:
        return float(series[t])
    return float(np.mean(vals))


def _reference_x14(x14: np.ndarray, valid_steps: np.ndarray, t: int) -> np.ndarray:
    prev_t, next_t = _nearest_valid_neighbors(valid_steps, t)
    if prev_t is None and next_t is None:
        raise ValueError(f"No valid neighboring timestep found for t={t}.")
    if prev_t is None:
        return x14[next_t].copy()
    if next_t is None:
        return x14[prev_t].copy()
    return 0.5 * (x14[prev_t] + x14[next_t])


def _model_input_from_x14(x14: np.ndarray, node_names: list[str], use_phase_onehot: bool) -> np.ndarray:
    if not use_phase_onehot:
        return x14.astype(np.float32, copy=False)
    phase_map = np.array([int(n.split(".")[-1]) - 1 for n in node_names], dtype=np.int64)
    ph_oh = np.eye(3, dtype=np.float32)[phase_map]
    return np.concatenate([x14, ph_oh], axis=-1).astype(np.float32, copy=False)


def _predict(model, static: dict, x14: np.ndarray, node_names: list[str], device: str) -> np.ndarray:
    x_model = _model_input_from_x14(x14, node_names, use_phase_onehot=bool(static["config"].get("use_phase_onehot", False)))
    x_t = torch.tensor(x_model, dtype=torch.float32, device=device)
    g = Data(
        x=x_t,
        edge_index=static["edge_index"].to(device),
        edge_attr=static["edge_attr"].to(device),
        edge_id=static["edge_id"].to(device),
        num_nodes=int(static["N"]),
    )
    with torch.no_grad():
        return model(g)[:, 0].detach().cpu().numpy()


def _run_daily_profile(ckpt_path: str, device: str) -> DailyRun:
    model, static = load_model_for_inference(ckpt_path, device=device)
    cfg = static["config"]
    node_in_dim = int(cfg["node_in_dim"])
    if node_in_dim not in (14, 17):
        raise ValueError(
            f"Strict spike analysis only supports load-type checkpoints with node_in_dim 14 or 17, got {node_in_dim}."
        )

    dataset_dir = _resolve_local_dataset_dir(cfg.get("dataset", DIR_LOADTYPE))
    edge_csv = os.path.join(dataset_dir, "gnn_edges_phase_static.csv")
    node_names = _resolve_reduced_node_list(dataset_dir)
    full_node_list = _get_full_master_node_list(dataset_dir)
    node_to_electrical_dist = lt._compute_electrical_distance_from_source(full_node_list, edge_csv)

    dss_path = inj.compile_once()
    inj.setup_daily()
    csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
    csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
    mL = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvL_token, dss_path), npts=NPTS, debug=False)
    mPV = inj.read_profile_csv_two_col_noheader(inj.resolve_csv_path(csvPV_token, dss_path), npts=NPTS, debug=False)

    _, _, _, bus_to_phases = inj.get_all_bus_phase_nodes()
    loads_dss, dev_to_dss_load, dev_to_busph_load = inj.build_load_device_maps(bus_to_phases)
    pv_dss, pv_to_dss, pv_to_busph = inj.build_pv_device_maps()
    rng = np.random.default_rng(0)

    x14_all = np.full((NPTS, len(node_names), len(LOADTYPE_FEAT)), np.nan, dtype=np.float32)
    v_dss = np.full((NPTS, len(node_names)), np.nan, dtype=np.float32)
    v_pred = np.full((NPTS, len(node_names)), np.nan, dtype=np.float32)
    valid_steps = np.zeros(NPTS, dtype=bool)

    for t in range(NPTS):
        inj.set_time_index(t)
        _, busphP_load, busphQ_load, _busphP_pv_nom, _busphQ_pv_nom, busph_per_type = lt._apply_snapshot_with_per_type(
            P_load_total_kw=P_BASE,
            Q_load_total_kvar=Q_BASE,
            P_pv_total_kw=PV_BASE,
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
            rng=rng,
        )
        inj.dss.Solution.Solve()
        if not inj.dss.Solution.Converged():
            continue

        busphP_pv_actual, busphQ_pv_actual = inj.get_pv_actual_pq_by_busph(pv_to_dss, pv_to_busph)
        vmag_m, _ = inj.get_all_node_voltage_pu_and_angle_filtered(node_names)
        x14 = build_gnn_x_loadtype(
            node_names,
            busph_per_type,
            busphP_pv_actual,
            node_to_electrical_dist,
            p_sys_balance=float(sum(busphP_load.values()) - sum(busphP_pv_actual.values())),
            q_sys_balance=float(
                sum(busphQ_load.values())
                - sum(busphQ_pv_actual.values())
                - inj.total_cap_q_kvar(node_names)
            ),
            busphQ_pv=busphQ_pv_actual,
        )
        pred = _predict(model, static, x14, node_names, device)

        x14_all[t] = x14
        v_dss[t] = np.asarray(vmag_m, dtype=np.float32)
        v_pred[t] = pred.astype(np.float32)
        valid_steps[t] = True

    return DailyRun(
        node_names=node_names,
        hours=np.arange(NPTS, dtype=np.float32) * (STEP_MIN / 60.0),
        x14=x14_all,
        v_dss=v_dss,
        v_pred=v_pred,
        valid_steps=valid_steps,
        node_in_dim=node_in_dim,
        use_phase_onehot=(node_in_dim == 17),
    )


def _pick_spike_steps(run: DailyRun, node_idx: int, focus_start: float, focus_end: float, top_k: int) -> list[int]:
    mask = (
        run.valid_steps
        & np.isfinite(run.v_pred[:, node_idx])
        & np.isfinite(run.v_dss[:, node_idx])
        & (run.hours >= focus_start)
        & (run.hours <= focus_end)
    )
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        raise ValueError("No valid timesteps found in the requested focus window.")

    scored: list[tuple[float, int]] = []
    series = run.v_pred[:, node_idx]
    for t in candidates:
        spike_mag = abs(float(series[t]) - _neighbor_mean(series, run.valid_steps, t))
        error_mag = abs(float(run.v_pred[t, node_idx]) - float(run.v_dss[t, node_idx]))
        scored.append((spike_mag + 0.5 * error_mag, int(t)))
    scored.sort(reverse=True)
    return [t for _, t in scored[: max(1, top_k)]]


def _counterfactual_for_feature(
    run: DailyRun,
    model,
    static: dict,
    device: str,
    node_idx: int,
    t: int,
    feature_idxs: tuple[int, ...],
) -> dict:
    base_x14 = run.x14[t]
    ref_x14 = _reference_x14(run.x14, run.valid_steps, t)
    cf_x14 = base_x14.copy()
    changed = {}
    for idx in feature_idxs:
        cf_x14[:, idx] = ref_x14[:, idx]
        changed[LOADTYPE_FEAT[idx]] = {
            "target_node_base": float(base_x14[node_idx, idx]),
            "target_node_ref": float(ref_x14[node_idx, idx]),
            "mean_abs_graph_delta": float(np.mean(np.abs(base_x14[:, idx] - ref_x14[:, idx]))),
        }

    cf_pred = _predict(model, static, cf_x14, run.node_names, device)
    base_pred = float(run.v_pred[t, node_idx])
    dss_val = float(run.v_dss[t, node_idx])
    base_spike = abs(base_pred - _neighbor_mean(run.v_pred[:, node_idx], run.valid_steps, t))
    cf_spike = abs(float(cf_pred[node_idx]) - _neighbor_mean(run.v_pred[:, node_idx], run.valid_steps, t))
    base_error = abs(base_pred - dss_val)
    cf_error = abs(float(cf_pred[node_idx]) - dss_val)
    return {
        "features": [LOADTYPE_FEAT[idx] for idx in feature_idxs],
        "base_pred": base_pred,
        "cf_pred": float(cf_pred[node_idx]),
        "dss": dss_val,
        "base_error": base_error,
        "cf_error": cf_error,
        "error_reduction": base_error - cf_error,
        "base_spike": base_spike,
        "cf_spike": cf_spike,
        "spike_reduction": base_spike - cf_spike,
        "changed": changed,
    }


def _default_output_paths(node: str) -> tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"loadtype_spike_driver_{_sanitize_node(node)}"
    return (
        os.path.join(OUTPUT_DIR, f"{stem}.json"),
        os.path.join(OUTPUT_DIR, f"{stem}.csv"),
    )


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.abspath(args.ckpt)
    json_path, csv_path = _default_output_paths(args.node)
    if args.output_json:
        json_path = os.path.abspath(args.output_json)
    if args.output_csv:
        csv_path = os.path.abspath(args.output_csv)

    print("=" * 72)
    print("Load-type spike driver analysis")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Target node: {args.node}")
    print(f"Focus window: {args.focus_start:.2f}h to {args.focus_end:.2f}h")
    print("=" * 72)

    run = _run_daily_profile(ckpt_path, device=device)
    if args.node not in run.node_names:
        raise ValueError(f"Node '{args.node}' not found in model node list.")
    node_idx = run.node_names.index(args.node)

    model, static = load_model_for_inference(ckpt_path, device=device)
    spike_steps = _pick_spike_steps(run, node_idx, args.focus_start, args.focus_end, args.top_k)

    print("\nSelected spike timesteps:")
    for t in spike_steps:
        pred = float(run.v_pred[t, node_idx])
        dss = float(run.v_dss[t, node_idx])
        spike_mag = abs(pred - _neighbor_mean(run.v_pred[:, node_idx], run.valid_steps, t))
        print(
            f"  t={t:03d} | hour={run.hours[t]:5.2f} | pred={pred:.5f} | dss={dss:.5f} | "
            f"abs_err={abs(pred - dss):.5f} | local_spike={spike_mag:.5f}"
        )

    rows: list[dict] = []
    aggregate_single: dict[str, float] = {name: 0.0 for name in LOADTYPE_FEAT}
    aggregate_pair: dict[tuple[str, str], float] = {}

    for t in spike_steps:
        single_results = []
        for idx, feat in enumerate(LOADTYPE_FEAT):
            res = _counterfactual_for_feature(run, model, static, device, node_idx, t, (idx,))
            rows.append(
                {
                    "hour": float(run.hours[t]),
                    "t_index": int(t),
                    "kind": "single",
                    "feature_1": feat,
                    "feature_2": "",
                    "base_pred": res["base_pred"],
                    "cf_pred": res["cf_pred"],
                    "dss": res["dss"],
                    "base_error": res["base_error"],
                    "cf_error": res["cf_error"],
                    "error_reduction": res["error_reduction"],
                    "base_spike": res["base_spike"],
                    "cf_spike": res["cf_spike"],
                    "spike_reduction": res["spike_reduction"],
                    "target_node_base": res["changed"][feat]["target_node_base"],
                    "target_node_ref": res["changed"][feat]["target_node_ref"],
                    "mean_abs_graph_delta": res["changed"][feat]["mean_abs_graph_delta"],
                }
            )
            aggregate_single[feat] += res["error_reduction"]
            single_results.append((res["error_reduction"], res["spike_reduction"], idx))

        single_results.sort(reverse=True)
        top_single_idxs = [idx for _, _, idx in single_results[:3]]
        for idx_a, idx_b in itertools.combinations(top_single_idxs, 2):
            res = _counterfactual_for_feature(run, model, static, device, node_idx, t, (idx_a, idx_b))
            f1, f2 = LOADTYPE_FEAT[idx_a], LOADTYPE_FEAT[idx_b]
            pair_key = tuple(sorted((f1, f2)))
            rows.append(
                {
                    "hour": float(run.hours[t]),
                    "t_index": int(t),
                    "kind": "pair",
                    "feature_1": f1,
                    "feature_2": f2,
                    "base_pred": res["base_pred"],
                    "cf_pred": res["cf_pred"],
                    "dss": res["dss"],
                    "base_error": res["base_error"],
                    "cf_error": res["cf_error"],
                    "error_reduction": res["error_reduction"],
                    "base_spike": res["base_spike"],
                    "cf_spike": res["cf_spike"],
                    "spike_reduction": res["spike_reduction"],
                    "target_node_base": np.nan,
                    "target_node_ref": np.nan,
                    "mean_abs_graph_delta": np.nan,
                }
            )
            aggregate_pair[pair_key] = aggregate_pair.get(pair_key, 0.0) + res["error_reduction"]

    df = pd.DataFrame(rows).sort_values(
        ["t_index", "kind", "error_reduction", "spike_reduction"],
        ascending=[True, True, False, False],
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    single_rank = sorted(aggregate_single.items(), key=lambda kv: kv[1], reverse=True)
    pair_rank = sorted(aggregate_pair.items(), key=lambda kv: kv[1], reverse=True)
    best_single_name, best_single_gain = single_rank[0]
    best_pair_name, best_pair_gain = (pair_rank[0] if pair_rank else ((), float("-inf")))

    if best_pair_name and best_pair_gain > best_single_gain * 1.15:
        likely_driver = {
            "kind": "pair",
            "features": list(best_pair_name),
            "reason": "combined feature replacement reduces error more than any single input",
        }
    else:
        likely_driver = {
            "kind": "single",
            "features": [best_single_name],
            "reason": "single-feature replacement gives the largest aggregate error reduction",
        }

    volt_var_overlap = [f for f in likely_driver["features"] if f in VOLT_VAR_RELATED]
    if volt_var_overlap:
        likely_driver["volt_var_related"] = True
        likely_driver["interpretation"] = (
            "Top driver overlaps the Volt-Var-related feature set "
            f"({', '.join(sorted(VOLT_VAR_RELATED))})."
        )
    else:
        likely_driver["volt_var_related"] = False
        likely_driver["interpretation"] = "Top driver is not purely Volt-Var-related; another modeled signal may dominate."

    summary = {
        "checkpoint": ckpt_path,
        "node": args.node,
        "focus_window_hours": [args.focus_start, args.focus_end],
        "node_in_dim": run.node_in_dim,
        "selected_t_indices": spike_steps,
        "selected_hours": [float(run.hours[t]) for t in spike_steps],
        "likely_driver": likely_driver,
        "single_feature_ranking": [
            {"feature": feat, "aggregate_error_reduction": float(gain)}
            for feat, gain in single_rank
        ],
        "pair_feature_ranking": [
            {"features": list(pair), "aggregate_error_reduction": float(gain)}
            for pair, gain in pair_rank
        ],
        "csv_path": csv_path,
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop single features by aggregate error reduction:")
    for feat, gain in single_rank[:5]:
        print(f"  {feat:>20s} : {gain:+.6f}")
    if pair_rank:
        print("\nTop feature pairs by aggregate error reduction:")
        for pair, gain in pair_rank[:3]:
            print(f"  {pair[0]} + {pair[1]} : {gain:+.6f}")

    print("\nLikely spike driver:")
    print(f"  kind        : {likely_driver['kind']}")
    print(f"  features    : {', '.join(likely_driver['features'])}")
    print(f"  reason      : {likely_driver['reason']}")
    print(f"  interpretation: {likely_driver['interpretation']}")
    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved JSON -> {json_path}")


if __name__ == "__main__":
    main()
