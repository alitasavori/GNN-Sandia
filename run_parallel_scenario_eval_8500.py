"""Parallel scenario evaluation on IEEE 8500: OpenDSS Solve() vs DA-GPS batched GPU forward.

Times independent operating points (Monte Carlo / hosting-capacity style):
  - OpenDSS CPU: sequential independent ``Solve()`` only (apply excluded from timed window
    is optional; by default we time Solve-only after apply, matching Method A paper line).
  - DA-GPS GPU: batched ``model(Batch)`` forward; micro-batches if full batch OOMs.
    Per-case time = total forward wall / N.

Example:
  python -u run_parallel_scenario_eval_8500.py \\
    --run-dir .../da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE \\
    --cache-pt .../run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt \\
    --batch-sizes 1000,2000,5000 --repeats 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import opendssdirect as dss
import torch
from torch_geometric.data import Batch, Data

from compare_gnn_inference_utils import (
    configure_cuda_inference,
    maybe_torch_compile,
)
from compare_mv_daily_timing import sync_inference_device
from compare_opendss_snapshot_helpers import (
    apply_explicit_loads_and_pv_pmpp,
    collect_unscaled_load_bases,
    discover_pv_system_names,
    read_pv_base_pmpp_kw,
    reassert_snapshot_before_each_solve,
    setup_da_gps_snapshot_opendss,
)
from nonunique_notebook_bootstrap import (
    resolve_cache_pt,
    resolve_feeder_checkpoint,
    resolve_feeder_run_dir,
    resolve_inference_device,
    resolve_notebook_repo,
)
from nonunique_opendss_daily import resolve_da_gps_device
from run_da_gps_daily_opendss_compare import (
    _resolve_default_edge_csv,
    _state_dict_is_legacy_edgeattn,
)
from train_da_gps_multitask_complex_voltage import DAGPSModel as DAGPSModelEdgeAttn
from train_da_gps_multitask_complex_voltage_gine import DAGPSModel as DAGPSModelGine
from train_da_gps_multitask_complex_voltage_gine import PlainNodeMLP
from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges

REPO_ROOT = Path(__file__).resolve().parent


def _find_master_dss(repo: Path) -> Path:
    cands = [
        repo / "8500 nodes with solar unbalanced" / "Master-PV2MW-inv.dss",
        repo / "8500-node" / "Master.dss",
    ]
    for p in cands:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("IEEE 8500 OpenDSS master not found")


def _compile_opendss(master: Path) -> None:
    model_dir = master.parent
    dss.Basic.ClearAll()
    dss.Text.Command(f"set datapath={model_dir}")
    dss.Text.Command(f"compile [{master.name}]")
    setup_da_gps_snapshot_opendss(npts=288, step_min=5.0)
    try:
        dss.Text.Command("set maxcontroliter=20000")
    except Exception:
        pass


def _capture_pv_base() -> tuple[list[str], dict[str, float]]:
    names = discover_pv_system_names()
    base = read_pv_base_pmpp_kw(names)
    return names, base


def _load_scales(n: int, profile: Path | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if profile is not None and profile.is_file():
        arr = np.loadtxt(profile, delimiter=",")
        if arr.ndim == 1:
            m = arr.astype(np.float64)
        else:
            m = arr[:, 1].astype(np.float64) if arr.shape[1] >= 2 else arr[:, 0].astype(np.float64)
        m = m[np.isfinite(m)]
        if m.size == 0:
            raise RuntimeError(f"empty load profile: {profile}")
        idx = rng.integers(0, m.size, size=n)
        return m[idx]
    # Fallback: training-like load multipliers
    return rng.uniform(0.56, 0.98, size=n).astype(np.float64)


def _time_opendss_solves(
    *,
    n: int,
    scales: np.ndarray,
    irr: np.ndarray,
    base_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    pv_names: list[str],
    pv_base: dict[str, float],
    repeats: int,
    warmup: int,
) -> dict:
    # Warmup
    for i in range(min(warmup, n)):
        apply_explicit_loads_and_pv_pmpp(
            base_names=base_names,
            base_kw=base_kw,
            base_kvar=base_kvar,
            m_t=float(scales[i]),
            pv_names=pv_names,
            pv_base_pmpp_kw=pv_base,
            ir_t=float(irr[i]),
        )
        reassert_snapshot_before_each_solve()
        dss.Solution.Solve()

    totals: list[float] = []
    for _rep in range(repeats):
        t_sum = 0.0
        for i in range(n):
            apply_explicit_loads_and_pv_pmpp(
                base_names=base_names,
                base_kw=base_kw,
                base_kvar=base_kvar,
                m_t=float(scales[i]),
                pv_names=pv_names,
                pv_base_pmpp_kw=pv_base,
                ir_t=float(irr[i]),
            )
            reassert_snapshot_before_each_solve()
            t0 = time.perf_counter()
            dss.Solution.Solve()
            t_sum += time.perf_counter() - t0
        totals.append(t_sum)
    batch_s = float(np.mean(totals))
    return {
        "batch_time_s": batch_s,
        "time_per_case_ms": 1000.0 * batch_s / max(n, 1),
        "repeats": repeats,
        "rep_batch_times_s": totals,
    }


def _pick_state_dict(bundle: dict) -> dict:
    for k in ("best_model_state_dict", "model_state_dict", "state_dict"):
        sd = bundle.get(k)
        if isinstance(sd, dict) and sd:
            return sd
    raise RuntimeError("checkpoint missing model state dict")


def _build_model_from_checkpoint(
    *,
    ckpt_path: Path,
    cache: dict,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
):
    bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(bundle)}")
    state = _pick_state_dict(bundle)
    hp = bundle.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        hp = {}

    x0 = cache["x"]
    N = int(x0.shape[1])
    n_feat = int(x0.shape[2])
    n_layers = int(hp.get("layers", hp.get("n_layers", 4)))
    hidden = int(hp.get("hidden", 128))
    heads = int(hp.get("heads", 4))
    node_emb_dim = int(hp.get("node_emb_dim", 2))
    edge_emb_dim = int(hp.get("edge_emb_dim", 4))
    n_cap = int(hp.get("n_cap", bundle.get("n_cap", 10)) or 10)
    n_reg = int(hp.get("n_reg", bundle.get("n_reg", 12)) or 12)
    n_sys = int(hp.get("n_system", bundle.get("n_system", 4)) or 4)
    n_pv_aux = int(hp.get("n_pv_aux", bundle.get("n_pv_aux", 0)) or 0)
    per_node_heads = bool(hp.get("per_node_heads", True))
    per_device_cap_head = bool(hp.get("per_device_cap_head", True))
    per_device_reg_head = bool(hp.get("per_device_reg_head", True))
    dropout = float(hp.get("dropout", 0.1))
    if bool(hp.get("disable_dropout", False)):
        dropout = 0.0
    model_type = str(bundle.get("model_type", "gine") or "gine").strip().lower()
    global_attn_mode = str(bundle.get("global_attn_mode", "tokens") or "tokens").strip().lower()
    if global_attn_mode not in ("tokens", "full_node"):
        global_attn_mode = "tokens"
    reg_loss_mode = str(bundle.get("reg_loss_mode", hp.get("reg_loss_mode", "ce")) or "ce").lower()
    reg_nclasses = bundle.get("reg_nclasses")
    use_legacy = _state_dict_is_legacy_edgeattn(state) and model_type != "mlp"

    if model_type == "mlp":
        model = PlainNodeMLP(
            n_nodes=N,
            node_in_dim=n_feat,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            node_emb_dim=node_emb_dim,
            per_node_heads=per_node_heads,
            n_cap=n_cap,
            n_reg=n_reg,
            n_pv_aux=n_pv_aux,
        )
    elif use_legacy:
        model = DAGPSModelEdgeAttn(
            n_nodes=N,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            heads=heads,
            n_layers=n_layers,
            n_cap=n_cap,
            n_reg=n_reg,
            n_system=n_sys,
            node_in_dim=n_feat,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            edge_dim=int(edge_attr.shape[1]),
            dropout=dropout,
            gradient_checkpointing=False,
            per_node_heads=per_node_heads,
            per_device_cap_head=per_device_cap_head,
            per_device_reg_head=per_device_reg_head,
        )
    else:
        model = DAGPSModelGine(
            n_nodes=N,
            num_edges=int(edge_index.shape[1]),
            hidden=hidden,
            heads=heads,
            n_layers=n_layers,
            n_cap=n_cap,
            n_reg=n_reg,
            n_system=n_sys,
            node_in_dim=n_feat,
            node_emb_dim=node_emb_dim,
            edge_emb_dim=edge_emb_dim,
            edge_dim=int(edge_attr.shape[1]),
            dropout=dropout,
            gradient_checkpointing=False,
            per_node_heads=per_node_heads,
            per_device_cap_head=per_device_cap_head,
            per_device_reg_head=per_device_reg_head,
            n_pv_aux=n_pv_aux,
            reg_nclasses=reg_nclasses if reg_loss_mode == "ce" else None,
            global_attn_mode=global_attn_mode,
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    model = maybe_torch_compile(model, label="parallel_eval", device=device)
    model.to(device)
    return model, bundle, N, n_feat


def _normalize_x(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    # Match daily / training: z-score continuous leading dims present in both.
    out = x.clone()
    d = min(int(out.shape[-1]), int(x_mean.numel()), int(x_std.numel()))
    mean = x_mean.reshape(-1)[:d]
    std = torch.clamp(x_std.reshape(-1)[:d], min=1e-6)
    out[..., :d] = (out[..., :d] - mean) / std
    return out


def _make_batch(
    xs: list[torch.Tensor],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> Batch:
    data_list = [
        Data(
            x=x.to(device),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=int(x.shape[0]),
        )
        for x in xs
    ]
    return Batch.from_data_list(data_list)


@torch.no_grad()
def _time_gnn_batched(
    *,
    model,
    x_norm: torch.Tensor,
    sample_idx: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    microbatch: int,
    repeats: int,
    warmup: int,
) -> dict:
    n = int(sample_idx.shape[0])
    mb = max(1, int(microbatch))

    def _one_pass() -> float:
        t_sum = 0.0
        for s in range(0, n, mb):
            e = min(n, s + mb)
            xs = [x_norm[int(i)] for i in sample_idx[s:e]]
            batch = _make_batch(xs, edge_index, edge_attr, device)
            sync_inference_device(device)
            t0 = time.perf_counter()
            _ = model(batch)
            sync_inference_device(device)
            t_sum += time.perf_counter() - t0
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return t_sum

    # Warmup
    for _ in range(max(1, warmup)):
        _ = _one_pass()

    totals: list[float] = []
    for _rep in range(repeats):
        totals.append(_one_pass())
    batch_s = float(np.mean(totals))
    return {
        "batch_time_s": batch_s,
        "time_per_case_ms": 1000.0 * batch_s / max(n, 1),
        "microbatch": mb,
        "n_microbatches": int(math.ceil(n / mb)),
        "repeats": repeats,
        "rep_batch_times_s": totals,
    }


def _auto_microbatch(
    *,
    model,
    x_norm: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    want: int,
    candidates: list[int],
) -> int:
    if device.type != "cuda":
        return min(want, 8)
    probe = x_norm[:1].detach()
    for mb in sorted(set(candidates), reverse=True):
        mb = int(mb)
        if mb < 1:
            continue
        try:
            xs = [probe[0] for _ in range(mb)]
            batch = _make_batch(xs, edge_index, edge_attr, device)
            sync_inference_device(device)
            _ = model(batch)
            sync_inference_device(device)
            del batch
            torch.cuda.empty_cache()
            return mb
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    return 1


def run(
    *,
    repo: Path,
    run_dir: Path,
    cache_pt: Path,
    batch_sizes: list[int],
    repeats: int,
    device: str,
    microbatch: int,
    seed: int,
    load_profile: Path | None,
    irr_profile: Path | None,
    out_dir: Path,
    skip_opendss: bool,
    skip_gnn: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_s = resolve_da_gps_device(device)
    device_t = torch.device(dev_s)
    configure_cuda_inference(device_t)

    ckpt = resolve_feeder_checkpoint(run_dir)
    print(f"[parallel_eval] device={device_t} run_dir={run_dir}", flush=True)
    print(f"[parallel_eval] checkpoint={ckpt}", flush=True)
    print(f"[parallel_eval] cache_pt={cache_pt}", flush=True)

    cache = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if "x" not in cache:
        raise RuntimeError(f"cache missing x: {cache_pt}")
    x_all = cache["x"].float()
    n_avail = int(x_all.shape[0])
    node_to_local = cache.get("node_to_local") or {}

    # Edges / model
    report = {}
    for name in ("da_gps_report.json", "da_gps_run_manifest.json"):
        p = run_dir / name
        if p.is_file():
            report = json.loads(p.read_text(encoding="utf-8"))
            break
    hp = report.get("hyperparameters") or {}
    edge_csv = _resolve_default_edge_csv(report, hp, cache_pt)
    edge_index, edge_attr = _load_compacted_edges(edge_csv, node_to_local)
    edge_index = edge_index.to(device_t)
    edge_attr = edge_attr.float().to(device_t)

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=False).float()
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=False).float()

    model = None
    if not skip_gnn:
        model, bundle, N, n_feat = _build_model_from_checkpoint(
            ckpt_path=ckpt,
            cache=cache,
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(),
            device=device_t,
        )
        print(
            f"[parallel_eval] GNN ready: N={N} n_feat={n_feat} "
            f"model_type={bundle.get('model_type', 'gine')}",
            flush=True,
        )
        x_norm = _normalize_x(x_all, x_mean, x_std)
    else:
        x_norm = x_all

    # OpenDSS setup
    base_names = base_kw = base_kvar = None
    pv_names: list[str] = []
    pv_base: dict[str, float] = {}
    if not skip_opendss:
        master = _find_master_dss(repo)
        print(f"[parallel_eval] OpenDSS compile: {master}", flush=True)
        _compile_opendss(master)
        base_names, base_kw, base_kvar = collect_unscaled_load_bases()
        pv_names, pv_base = _capture_pv_base()
        # one dummy solve to settle
        dss.Solution.Solve()

    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for n in batch_sizes:
        n = int(n)
        if n > n_avail and not skip_gnn:
            print(
                f"[parallel_eval] WARNING: requested N={n} > cache samples={n_avail}; "
                "sampling with replacement for GNN indices.",
                flush=True,
            )
        sample_idx = rng.integers(0, max(n_avail, 1), size=n)
        scales = _load_scales(n, load_profile, seed=seed + n)
        if irr_profile is not None and irr_profile.is_file():
            irr = _load_scales(n, irr_profile, seed=seed + 17 + n)
        else:
            irr = rng.uniform(0.0, 1.0, size=n).astype(np.float64)

        print("=" * 72, flush=True)
        print(f"[parallel_eval] N={n} repeats={repeats}", flush=True)

        od = None
        if not skip_opendss:
            assert base_names is not None and base_kw is not None and base_kvar is not None
            od = _time_opendss_solves(
                n=n,
                scales=scales,
                irr=irr,
                base_names=base_names,
                base_kw=base_kw,
                base_kvar=base_kvar,
                pv_names=pv_names,
                pv_base=pv_base,
                repeats=repeats,
                warmup=3,
            )
            print(
                f"  OpenDSS Solve: batch={od['batch_time_s']:.3f}s  "
                f"per_case={od['time_per_case_ms']:.3f} ms",
                flush=True,
            )

        gn = None
        if not skip_gnn:
            assert model is not None
            mb = int(microbatch)
            if mb <= 0:
                mb = _auto_microbatch(
                    model=model,
                    x_norm=x_norm,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    device=device_t,
                    want=min(n, 64),
                    candidates=[64, 32, 16, 8, 4, 2, 1],
                )
                print(f"  GNN auto microbatch={mb}", flush=True)
            gn = _time_gnn_batched(
                model=model,
                x_norm=x_norm,
                sample_idx=sample_idx,
                edge_index=edge_index,
                edge_attr=edge_attr,
                device=device_t,
                microbatch=mb,
                repeats=repeats,
                warmup=2,
            )
            print(
                f"  DA-GPS forward: batch={gn['batch_time_s']:.3f}s  "
                f"per_case={gn['time_per_case_ms']:.3f} ms  "
                f"(microbatch={gn['microbatch']}, n_mb={gn['n_microbatches']})",
                flush=True,
            )

        row = {
            "n_cases": n,
            "opendss": od,
            "da_gps": gn,
        }
        rows.append(row)

    summary = {
        "feeder": "8500",
        "device": str(device_t),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "cache_pt": str(cache_pt),
        "batch_sizes": batch_sizes,
        "repeats": repeats,
        "seed": seed,
        "rows": rows,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    out_json = out_dir / "parallel_scenario_eval_8500.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Parallel evaluation table (fill-in) ===", flush=True)
    print(f"{'Cases':>8} | {'OD batch s':>10} | {'OD ms/case':>10} | {'GNN batch s':>11} | {'GNN ms/case':>11}", flush=True)
    for r in rows:
        od = r.get("opendss") or {}
        gn = r.get("da_gps") or {}
        print(
            f"{r['n_cases']:>8} | "
            f"{od.get('batch_time_s', float('nan')):10.3f} | "
            f"{od.get('time_per_case_ms', float('nan')):10.3f} | "
            f"{gn.get('batch_time_s', float('nan')):11.3f} | "
            f"{gn.get('time_per_case_ms', float('nan')):11.3f}",
            flush=True,
        )
    print(f"\n[parallel_eval] wrote {out_json}", flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel scenario eval: OpenDSS Solve vs DA-GPS batched forward")
    p.add_argument("--repo", type=str, default="")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--cache-pt", type=str, default="")
    p.add_argument("--batch-sizes", type=str, default="1000,2000,5000")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--microbatch", type=int, default=0, help="0 = auto (probe GPU memory)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-profile", type=str, default="")
    p.add_argument("--irr-profile", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--skip-opendss", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Use batch sizes 50,100,200 and repeats=1")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo = resolve_notebook_repo(Path(args.repo) if str(args.repo).strip() else None)
    if str(args.run_dir).strip():
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = resolve_feeder_run_dir(repo, "8500")
    if str(args.cache_pt).strip():
        cache_pt = Path(args.cache_pt).expanduser().resolve()
    else:
        cache_pt = resolve_cache_pt(repo, "8500")

    if args.smoke:
        batch_sizes = [50, 100, 200]
        repeats = 1
    else:
        batch_sizes = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]
        repeats = int(args.repeats)

    day1 = repo / "a representativ days"
    load_p = Path(args.load_profile).expanduser() if str(args.load_profile).strip() else day1 / "load_day_004.csv"
    irr_p = Path(args.irr_profile).expanduser() if str(args.irr_profile).strip() else day1 / "irr_day_004.csv"
    if not load_p.is_file():
        load_p = None
    if not irr_p.is_file():
        irr_p = None

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else repo / "parallel_scenario_eval_8500_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    device = resolve_inference_device(args.device)

    run(
        repo=repo,
        run_dir=run_dir,
        cache_pt=cache_pt,
        batch_sizes=batch_sizes,
        repeats=repeats,
        device=device,
        microbatch=int(args.microbatch),
        seed=int(args.seed),
        load_profile=load_p,
        irr_profile=irr_p,
        out_dir=out_dir,
        skip_opendss=bool(args.skip_opendss),
        skip_gnn=bool(args.skip_gnn),
    )


if __name__ == "__main__":
    main()
