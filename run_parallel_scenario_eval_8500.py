"""Parallel scenario evaluation on IEEE 8500: OpenDSS Solve() vs DA-GPS batched forward.

Times independent operating points (Monte Carlo / hosting-capacity style):
  - OpenDSS: sequential independent ``Solve()`` only — always CPU.
  - DA-GPS: one forward per batch size B in ``--batch-sizes`` (CUDA or CPU).
    On CUDA/host OOM, that B is marked OOM and larger B's are skipped for GNN
    (OpenDSS continues). Per-case time = batch wall / B.

Example:
  python -u run_parallel_scenario_eval_8500.py \
    --run-dir .../da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE \
    --batch-sizes 8,16,32,64,128,256,512 --repeats 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
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
    _infer_reg_nclasses_from_state_dict,
    _resolve_da_gps_checkpoint,
    _resolve_default_edge_csv,
    _resolve_reg_loss_mode,
    _state_dict_is_legacy_edgeattn,
    _state_dict_per_device_cap,
    _state_dict_per_device_reg,
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
    """Compile 8500 PV master the same way Method A / daily compare does.

    Newer OpenDSSDirect builds reject ``set datapath=...`` before a circuit exists
    (error #301). Use ``cd`` + absolute ``redirect`` instead.
    """
    master = master.resolve()
    model_dir = master.parent
    dss.Basic.ClearAll()
    dss.Text.Command(f'cd "{model_dir}"')
    dss.Text.Command(f'redirect "{master}"')
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


def _infer_n_layers_from_state(state: dict) -> int | None:
    idxs: set[int] = set()
    for k in state:
        if k.startswith("blocks."):
            try:
                idxs.add(int(k.split(".")[1]))
            except Exception:
                continue
    return (max(idxs) + 1) if idxs else None


def _align_x_to_n_feat(x: torch.Tensor, n_feat: int) -> torch.Tensor:
    """Pad/truncate last dim so cache ``x`` matches checkpoint ``node_in`` width."""
    cur = int(x.shape[-1])
    if cur == n_feat:
        return x
    if cur > n_feat:
        print(f"[parallel_eval] truncating cache x dim {cur} → {n_feat}", flush=True)
        return x[..., :n_feat].contiguous()
    print(
        f"[parallel_eval] padding cache x dim {cur} → {n_feat} with zeros "
        "(missing PE/static cols vs training)",
        flush=True,
    )
    pad = torch.zeros(*x.shape[:-1], n_feat - cur, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _align_edge_attr(edge_attr: torch.Tensor, edge_dim: int) -> torch.Tensor:
    cur = int(edge_attr.shape[-1]) if edge_attr.numel() else 0
    if cur == edge_dim:
        return edge_attr
    if cur > edge_dim:
        print(f"[parallel_eval] truncating edge_attr dim {cur} → {edge_dim}", flush=True)
        return edge_attr[..., :edge_dim].contiguous()
    if edge_attr.numel() == 0:
        return torch.zeros((0, edge_dim), dtype=torch.float32)
    print(f"[parallel_eval] padding edge_attr dim {cur} → {edge_dim}", flush=True)
    pad = torch.zeros(edge_attr.shape[0], edge_dim - cur, dtype=edge_attr.dtype)
    return torch.cat([edge_attr, pad], dim=-1)


def _prefer_multisample_cache(repo: Path, preferred: Path, *, want_n_feat: int | None) -> Path:
    """Prefer a multi-sample pack matching training feature width over slim ref0 packs."""
    cands: list[Path] = [preferred]
    from nonunique_notebook_bootstrap import FEEDER_CACHE_DIRS, FEEDER_CACHE_PT_NAMES, _datasets_gnn2_roots

    names = FEEDER_CACHE_PT_NAMES.get("8500", ())
    dirs = FEEDER_CACHE_DIRS.get("8500", ())
    for root in _datasets_gnn2_roots(repo):
        for drel in dirs:
            folder = root / drel
            try:
                folder_ok = folder.is_dir()
            except OSError as e:
                print(f"[parallel_eval] skip unreadable cache dir {folder}: {e}", flush=True)
                continue
            if not folder_ok:
                continue
            for name in names:
                cands.append(folder / name)
            try:
                cands.extend(sorted(folder.glob("run_001*__full__*.pt")))
            except OSError as e:
                print(f"[parallel_eval] skip glob in {folder}: {e}", flush=True)
        for rel in (
            "datasets_gnn2_from pc/run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt",
            "datasets_gnn2_from pc/run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt",
        ):
            cands.append(repo / rel)

    best: Path | None = None
    best_score = (-1, -1)  # (n_samples, feat_match)
    seen: set[Path] = set()
    for raw in cands:
        try:
            p = raw.expanduser().resolve()
        except OSError as e:
            print(f"[parallel_eval] skip unresolvable cache cand {raw}: {e}", flush=True)
            continue
        if p in seen:
            continue
        seen.add(p)
        try:
            if not p.is_file():
                continue
        except OSError as e:
            # Common on Colab Shared-Drive FUSE: Errno 107 Transport endpoint is not connected
            print(f"[parallel_eval] skip unreadable cache cand {p}: {e}", flush=True)
            continue
        try:
            z = torch.load(p, map_location="cpu", weights_only=False)
            x = z.get("x")
            if not torch.is_tensor(x) or x.ndim != 3:
                continue
            n_s, _n, f = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
            feat_ok = 1 if (want_n_feat is None or f == int(want_n_feat)) else 0
            # Feature-width match must dominate sample count (wrong-width packs are truncated).
            score = (feat_ok, n_s)
            if score > best_score:
                best_score = score
                best = p
                if n_s >= 1000 and feat_ok == 1:
                    break
        except OSError as e:
            print(f"[parallel_eval] skip unreadable cache load {p}: {e}", flush=True)
            continue
        except Exception:
            continue
    if best is None:
        # Fall back to preferred if still readable; else raise a clear error.
        try:
            pref = preferred.expanduser().resolve()
            if pref.is_file():
                return pref
        except OSError as e:
            raise FileNotFoundError(
                f"No usable DA-GPS cache pack found (Drive may be disconnected: {e}). "
                "Remount Drive or set CACHE_PT_OVERRIDE to a local .pt under /content."
            ) from e
        raise FileNotFoundError(
            f"No usable DA-GPS cache pack found near preferred={preferred}. "
            "Remount Drive or set CACHE_PT_OVERRIDE."
        )
    if best.resolve() != preferred.resolve():
        print(f"[parallel_eval] using cache pack {best} (score samples/feat={best_score})", flush=True)
    return best


def _build_model_from_checkpoint(
    *,
    ckpt_path: Path,
    run_dir: Path,
    cache: dict,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
    x_mean: torch.Tensor,
):
    """Rebuild DAGPSModel using the same metadata path as Method A daily compare."""
    bundle, state = _resolve_da_gps_checkpoint(ckpt_path, run_dir, run_dir)
    hp = bundle.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        hp = {}

    x0 = cache["x"]
    N = int(x0.shape[1])

    # Prefer authoritative fields from checkpoint bundle / state_dict (not silent defaults).
    n_cap = int(bundle["n_cap"])
    n_reg = int(bundle["n_reg"])
    n_sys = int(bundle.get("n_system_tokens", bundle.get("n_system", hp.get("n_system", 4))))
    hidden = int(bundle.get("hidden", hp.get("hidden", 128)))
    n_layers = int(bundle.get("layers", hp.get("layers", 4)))
    inferred_L = _infer_n_layers_from_state(state)
    if inferred_L is not None and inferred_L != n_layers:
        print(f"[parallel_eval] layers from state_dict blocks={inferred_L} (meta had {n_layers})", flush=True)
        n_layers = inferred_L
    heads = int(bundle.get("heads", hp.get("heads", 4)))
    node_emb_dim = int(bundle.get("node_emb_dim", hp.get("node_emb_dim", 0)) or 0)
    edge_emb_dim = int(bundle.get("edge_emb_dim", hp.get("edge_emb_dim", 0)) or 0)
    n_pv_aux = int(bundle.get("n_pv_aux", 0) or 0)
    per_node_heads = bool(bundle.get("per_node_heads", False)) or (
        "volt_W" in state and state["volt_W"] is not None
    )
    per_device_cap_head = bool(bundle.get("per_device_cap_head", False)) or _state_dict_per_device_cap(state)
    per_device_reg_head = bool(bundle.get("per_device_reg_head", False)) or _state_dict_per_device_reg(state)
    dropout = float(hp.get("dropout", 0.1))
    if bool(hp.get("disable_dropout", False)):
        dropout = 0.0
    model_type = str(bundle.get("model_type", "gine") or "gine").strip().lower()
    global_attn_mode = str(bundle.get("global_attn_mode", "tokens") or "tokens").strip().lower()
    if global_attn_mode not in ("tokens", "full_node"):
        global_attn_mode = "tokens"
    reg_loss_mode = _resolve_reg_loss_mode(bundle, state)
    reg_nclasses = bundle.get("reg_nclasses")
    if not (isinstance(reg_nclasses, (list, tuple)) and len(reg_nclasses) == n_reg):
        reg_nclasses = _infer_reg_nclasses_from_state_dict(state, n_reg)
    use_legacy = _state_dict_is_legacy_edgeattn(state) and model_type != "mlp"

    # Feature / edge widths from weights (authoritative).
    # node_in.0 is Linear(node_in_dim + node_emb_dim → hidden); do NOT treat shape[1] as n_feat alone.
    ne = state.get("node_emb.weight")
    if torch.is_tensor(ne) and ne.ndim == 2:
        node_emb_dim = int(ne.shape[1])
    elif "node_emb.weight" not in state:
        node_emb_dim = 0

    w_in = state.get("node_in.0.weight")
    if torch.is_tensor(w_in):
        hidden = int(w_in.shape[0])
        in_combined = int(w_in.shape[1])
        n_feat = in_combined - int(node_emb_dim)
        if n_feat <= 0:
            raise RuntimeError(
                f"Invalid node_in width={in_combined} with node_emb_dim={node_emb_dim}"
            )
    else:
        n_feat = int(x_mean.reshape(-1).numel())

    # Edge CSV width is the physical edge_dim; edge_emb is concatenated inside the model.
    ee = state.get("edge_emb.weight")
    if torch.is_tensor(ee) and ee.ndim == 2:
        edge_emb_dim = int(ee.shape[1])
    elif "edge_emb.weight" not in state:
        edge_emb_dim = 0
    edge_dim = int(edge_attr.shape[1]) if edge_attr.numel() else 2
    edge_attr = _align_edge_attr(edge_attr.float(), edge_dim)

    # token_latent rows = n_cap + n_reg + n_system (not n_system alone).
    tl = state.get("token_latent")
    if torch.is_tensor(tl) and tl.ndim == 2:
        hidden = int(tl.shape[1])
        g_tok = int(tl.shape[0])
        inferred_sys = g_tok - int(n_cap) - int(n_reg)
        if inferred_sys < 0:
            raise RuntimeError(
                f"token_latent rows={g_tok} < n_cap+n_reg={n_cap + n_reg}"
            )
        if inferred_sys != n_sys:
            print(
                f"[parallel_eval] n_system from token_latent={inferred_sys} "
                f"(meta had {n_sys}; g_tokens={g_tok}=cap{n_cap}+reg{n_reg}+sys)",
                flush=True,
            )
        n_sys = inferred_sys

    print(
        f"[parallel_eval] arch L={n_layers} h={hidden} n_feat={n_feat} edge_dim={edge_dim} "
        f"node_emb={node_emb_dim} n_sys={n_sys} g_tokens={n_cap + n_reg + n_sys} "
        f"per_node_heads={per_node_heads} global_attn={global_attn_mode} reg_loss={reg_loss_mode}",
        flush=True,
    )

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
            edge_dim=edge_dim,
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
            edge_dim=edge_dim,
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
    return model, bundle, N, n_feat, edge_attr


def _normalize_x(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    """Z-score using training norms; lengths already aligned to ``n_feat``."""
    out = x.clone()
    mean = x_mean.reshape(-1)
    std = torch.clamp(x_std.reshape(-1), min=1e-6)
    d = min(int(out.shape[-1]), int(mean.numel()), int(std.numel()))
    out[..., :d] = (out[..., :d] - mean[:d]) / std[:d]
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


@torch.no_grad()
def _mae_gnn_vs_cache(
    *,
    model,
    x_norm: torch.Tensor,
    y_ri: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_idx: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """|V| / angle MAE of DA-GPS vs cache ``y_ri`` labels (same samples as the timed batch).

    Timing OpenDSS cases use independent load/irr draws and are not paired with these
    GNN inputs — accuracy here is model-vs-cache-GT on the batched scenarios.
    """
    from train_da_gps_multitask_complex_voltage_gine import _metrics_voltage

    xs = [x_norm[int(i)] for i in sample_idx]
    batch = _make_batch(xs, edge_index, edge_attr, device)
    sync_inference_device(device)
    v_n, *_rest = model(batch)
    sync_inference_device(device)
    B = int(sample_idx.shape[0])
    N = int(x_norm.shape[1])
    v_n = v_n.view(B, N, 2).float()
    ym = y_mean.reshape(N, 2).to(device=device, dtype=torch.float32)
    ys = y_std.reshape(N, 2).to(device=device, dtype=torch.float32).clamp_min(1e-12)
    pred_ri = v_n * ys.unsqueeze(0) + ym.unsqueeze(0)
    true_ri = y_ri[sample_idx.astype(np.int64)].to(device=device, dtype=torch.float32)
    if true_ri.ndim == 2:
        true_ri = true_ri.view(B, N, 2)
    met = _metrics_voltage(pred_ri.cpu(), true_ri.cpu())
    del batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return met


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or isinstance(exc, MemoryError)
    )


def run(
    *,
    repo: Path,
    run_dir: Path,
    cache_pt: Path,
    batch_sizes: list[int],
    repeats: int,
    device: str,
    seed: int,
    load_profile: Path | None,
    irr_profile: Path | None,
    out_dir: Path,
    skip_opendss: bool,
    skip_gnn: bool,
    stop_on_oom: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_s = resolve_da_gps_device(device)
    device_t = torch.device(dev_s)
    configure_cuda_inference(device_t)

    ckpt = resolve_feeder_checkpoint(run_dir)
    print(f"[parallel_eval] device={device_t} run_dir={run_dir}", flush=True)
    print(f"[parallel_eval] checkpoint={ckpt}", flush=True)

    # Peek raw feature width from x_mean.pt (matches training cache); fall back to weight math.
    want_n_feat: int | None = None
    try:
        xm = run_dir / "x_mean.pt"
        if xm.is_file():
            want_n_feat = int(
                torch.load(xm, map_location="cpu", weights_only=False).reshape(-1).numel()
            )
        else:
            peek = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = peek.get("best_model_state_dict") or peek.get("model_state_dict") or {}
            if isinstance(sd, dict):
                emb_d = 0
                ne = sd.get("node_emb.weight")
                if torch.is_tensor(ne) and ne.ndim == 2:
                    emb_d = int(ne.shape[1])
                w_in = sd.get("node_in.0.weight")
                if torch.is_tensor(w_in):
                    want_n_feat = int(w_in.shape[1]) - emb_d
                    if want_n_feat <= 0:
                        want_n_feat = None
    except Exception:
        want_n_feat = None

    cache_pt = _prefer_multisample_cache(repo, cache_pt, want_n_feat=want_n_feat)
    print(f"[parallel_eval] cache_pt={cache_pt}", flush=True)

    cache = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if "x" not in cache:
        raise RuntimeError(f"cache missing x: {cache_pt}")
    x_all = cache["x"].float()
    n_avail = int(x_all.shape[0])
    node_to_local = cache.get("node_to_local") or {}
    y_ri_all = cache.get("y_ri")
    if torch.is_tensor(y_ri_all):
        y_ri_all = y_ri_all.float()
    else:
        y_ri_all = None
        print("[parallel_eval] WARNING: cache missing y_ri — MAE disabled", flush=True)
    if n_avail < 2:
        print(
            f"[parallel_eval] WARNING: cache has only {n_avail} sample(s); "
            "GNN batching will resample with replacement. Prefer a full run_001* pack.",
            flush=True,
        )

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
    edge_attr = edge_attr.float()

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=False).float().reshape(-1)
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=False).float().reshape(-1)
    y_mean = torch.load(run_dir / "y_mean.pt", map_location="cpu", weights_only=False).float().reshape(-1)
    y_std = torch.load(run_dir / "y_std.pt", map_location="cpu", weights_only=False).float().reshape(-1)

    model = None
    n_feat = int(x_all.shape[-1])
    if not skip_gnn:
        model, bundle, N, n_feat, edge_attr = _build_model_from_checkpoint(
            ckpt_path=ckpt,
            run_dir=run_dir,
            cache=cache,
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(),
            device=device_t,
            x_mean=x_mean,
        )
        edge_attr = edge_attr.to(device_t)
        x_all = _align_x_to_n_feat(x_all, n_feat)
        if int(x_mean.numel()) != n_feat:
            print(
                f"[parallel_eval] aligning x_mean/std length {int(x_mean.numel())} → {n_feat}",
                flush=True,
            )
            if int(x_mean.numel()) < n_feat:
                x_mean = torch.cat([x_mean, torch.zeros(n_feat - int(x_mean.numel()))])
                x_std = torch.cat([x_std, torch.ones(n_feat - int(x_std.numel()))])
            else:
                x_mean = x_mean[:n_feat]
                x_std = x_std[:n_feat]
        print(
            f"[parallel_eval] GNN ready: N={N} n_feat={n_feat} n_cache_samples={n_avail} "
            f"model_type={bundle.get('model_type', 'gine')}",
            flush=True,
        )
        x_norm = _normalize_x(x_all, x_mean, x_std)
    else:
        edge_attr = edge_attr.to(device_t)
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
    gnn_oom_stop = False
    max_gnn_batch: int | None = None

    for n in batch_sizes:
        n = int(n)
        if n > n_avail and not skip_gnn:
            print(
                f"[parallel_eval] WARNING: requested batch={n} > cache samples={n_avail}; "
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
        print(
            f"[parallel_eval] batch_size={n}  (OpenDSS: {n} sequential Solve; "
            f"DA-GPS: one forward of {n})  repeats={repeats}  device={device_t}",
            flush=True,
        )

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
                f"  OpenDSS Solve (CPU): batch={od['batch_time_s']:.3f}s  "
                f"per_case={od['time_per_case_ms']:.3f} ms",
                flush=True,
            )

        gn = None
        if not skip_gnn:
            assert model is not None
            if gnn_oom_stop:
                gn = {
                    "status": "skipped_after_oom",
                    "batch_size": n,
                    "device": str(device_t),
                }
                print("  DA-GPS: skipped (prior OOM)", flush=True)
            else:
                # Full batch: one forward of size n (no micro-splitting).
                try:
                    gn = _time_gnn_batched(
                        model=model,
                        x_norm=x_norm,
                        sample_idx=sample_idx,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        device=device_t,
                        microbatch=n,
                        repeats=repeats,
                        warmup=2,
                    )
                    gn["status"] = "ok"
                    gn["batch_size"] = n
                    gn["device"] = str(device_t)
                    max_gnn_batch = n
                    if y_ri_all is not None:
                        try:
                            mae = _mae_gnn_vs_cache(
                                model=model,
                                x_norm=x_norm,
                                y_ri=y_ri_all,
                                y_mean=y_mean,
                                y_std=y_std,
                                sample_idx=sample_idx,
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                device=device_t,
                            )
                            gn["mae"] = mae
                            print(
                                f"  DA-GPS MAE vs cache y_ri: |V|={mae['mae_vmag_pu']:.6f} pu  "
                                f"angle={mae['mae_angle_deg']:.4f} deg  "
                                f"RMSE|V|={mae['rmse_vmag_pu']:.6f}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"  DA-GPS MAE skipped: {e}", flush=True)
                    speedup = (
                        float(od["batch_time_s"]) / float(gn["batch_time_s"])
                        if od is not None and float(gn["batch_time_s"]) > 0
                        else float("nan")
                    )
                    print(
                        f"  DA-GPS forward ({device_t}): batch={gn['batch_time_s']:.3f}s  "
                        f"per_case={gn['time_per_case_ms']:.3f} ms  "
                        f"speedup_vs_OD={speedup:.2f}x",
                        flush=True,
                    )
                except (RuntimeError, MemoryError) as e:
                    if not _is_oom_error(e):
                        raise
                    if device_t.type == "cuda":
                        torch.cuda.empty_cache()
                    gn = {
                        "status": "oom",
                        "batch_size": n,
                        "device": str(device_t),
                        "error": str(e)[:500],
                    }
                    print(f"  DA-GPS: OOM at batch_size={n} ({device_t})", flush=True)
                    if stop_on_oom:
                        gnn_oom_stop = True
                        print(
                            "  [parallel_eval] stopping further GNN batch sizes after OOM "
                            "(OpenDSS continues).",
                            flush=True,
                        )

        row = {
            "n_cases": n,
            "batch_size": n,
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
        "stop_on_oom": bool(stop_on_oom),
        "max_gnn_batch_ok": max_gnn_batch,
        "rows": rows,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_json = out_dir / "parallel_scenario_eval_8500.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Parallel evaluation table (batch-size sweep) ===", flush=True)
    print(
        f"{'B':>6} | {'OD batch s':>10} | {'OD ms/case':>10} | "
        f"{'GNN batch s':>11} | {'GNN ms/case':>11} | {'speedup':>8} | "
        f"{'|V| MAE':>10} | {'ang MAE':>8} | status",
        flush=True,
    )
    for r in rows:
        od = r.get("opendss") or {}
        gn = r.get("da_gps") or {}
        status = str(gn.get("status", "n/a"))
        od_s = od.get("batch_time_s", float("nan"))
        gn_s = gn.get("batch_time_s", float("nan"))
        mae = gn.get("mae") or {}
        try:
            sp = float(od_s) / float(gn_s) if float(gn_s) > 0 else float("nan")
        except Exception:
            sp = float("nan")
        print(
            f"{r['batch_size']:>6} | "
            f"{float(od_s) if od else float('nan'):10.3f} | "
            f"{float(od.get('time_per_case_ms', float('nan'))):10.3f} | "
            f"{float(gn_s) if 'batch_time_s' in gn else float('nan'):11.3f} | "
            f"{float(gn.get('time_per_case_ms', float('nan'))):11.3f} | "
            f"{sp:8.2f} | "
            f"{float(mae.get('mae_vmag_pu', float('nan'))):10.6f} | "
            f"{float(mae.get('mae_angle_deg', float('nan'))):8.4f} | {status}",
            flush=True,
        )
    if max_gnn_batch is not None:
        print(f"\n[parallel_eval] max DA-GPS batch that fit: {max_gnn_batch}", flush=True)
    print(f"[parallel_eval] wrote {out_json}", flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallel scenario eval: OpenDSS Solve (CPU) vs DA-GPS full-batch forward"
    )
    p.add_argument("--repo", type=str, default="")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--cache-pt", type=str, default="")
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16,32,64,128,256,512",
        help="Comma list of full GPU/CPU batch sizes to sweep (one forward each)",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="DA-GPS device (cuda or cpu). OpenDSS Solve is always CPU.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-profile", type=str, default="")
    p.add_argument("--irr-profile", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--skip-opendss", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick check: batch sizes 8,16 and repeats=1",
    )
    p.add_argument(
        "--stop-on-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After first GNN OOM, skip larger GNN batches (OpenDSS still runs)",
    )
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
        batch_sizes = [8, 16]
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
        seed=int(args.seed),
        load_profile=load_p,
        irr_profile=irr_p,
        out_dir=out_dir,
        skip_opendss=bool(args.skip_opendss),
        skip_gnn=bool(args.skip_gnn),
        stop_on_oom=bool(args.stop_on_oom),
    )


if __name__ == "__main__":
    main()
