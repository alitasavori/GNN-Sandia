"""Train PowerFlowMultiNet — oracle device states (GENConv baseline).

Faithful to arXiv:2403.00892v3 framing where possible:
  - Physical-bus multigraph; taps/caps are OpenDSS-settled *inputs* (not predicted).
  - Joint MSE on bus V/φ and substation P/Q (default ``--lambda_sub 1``).
  - Adam + MultiStepLR at 50%% / 80%% of max epochs (gamma=0.1; schedule is an
    implementation assumption — paper says Adam lr=0.001).
  - Default epochs=50 (launcher), MultiStepLR milestones at 50%/80% of max epochs
    (→ 25, 40 when epochs=50); effective batch ≈128 (batch_size × grad_accum).

Artifacts (per OUT_DIR):
  - pfmn_oracle_best.pt
  - training_last.pt
  - pfmn_report.json
  - run_manifest.json (+ pfmn_run_manifest.json alias)

Logging mirrors DA-GPS chunk_parent style (``[pfmn chunk_parent]`` + train_pool_eval).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import gc
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from build_powerflowmultinet_graph import (
    EDGE_CONT_IDX,
    EDGE_FEAT_DIM,
    NODE_CONT_IDX,
    NODE_FEAT_DIM,
    load_pfmn_chunk_tensors,
    materialize_edge_attr,
)
from powerflowmultinet_model import PowerFlowMultiNet
from train_interactive_pause import (
    add_interactive_pause_args,
    ask_continue_or_stop,
    should_interactive_pause,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Bump when tensor / feature schema changes so Colab Drive caches rebuild.
_CACHE_SUFFIX = "__pfmn_oracle_v2.pt"


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(spec: str) -> torch.device:
    s = str(spec or "auto").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sorted_run_chunks(chunk_parent: Path, glob_pat: str) -> list[Path]:
    glob_pat = str(glob_pat).strip()
    if "," in glob_pat:
        allowed = {s.strip() for s in glob_pat.split(",") if s.strip()}
        chunks = sorted(
            (p for p in chunk_parent.iterdir() if p.is_dir() and p.name in allowed),
            key=lambda p: p.name,
        )
        missing = allowed - {p.name for p in chunks}
        if missing:
            raise FileNotFoundError(f"Missing chunk folders: {sorted(missing)[:5]}")
        return chunks
    return sorted(
        (p for p in chunk_parent.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, glob_pat)),
        key=lambda p: p.name,
    )


def _cache_path(cache_dir: Path, chunk_dir: Path) -> Path:
    return cache_dir / f"{chunk_dir.name}{_CACHE_SUFFIX}"


def _load_or_build_chunk(
    chunk_dir: Path,
    cache_dir: Path,
    *,
    nodes_csv: str,
    edges_csv: str,
    meta_csv: str,
    rebuild: bool,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, chunk_dir)
    if cp.is_file() and not rebuild:
        print(f"[pfmn cache] load {cp.name}", flush=True)
        return torch.load(cp, map_location="cpu", weights_only=False)
    nodes = chunk_dir / nodes_csv
    edges = chunk_dir / edges_csv
    meta = chunk_dir / meta_csv
    for p in (nodes, edges, meta):
        if not p.is_file():
            raise FileNotFoundError(p)
    print(f"[pfmn cache] build {chunk_dir.name}", flush=True)
    pack = load_pfmn_chunk_tensors(nodes, edges, meta)
    pack["chunk_name"] = chunk_dir.name
    pack["cache_schema"] = "pfmn_oracle_v2"
    torch.save(pack, cp)
    print(f"[pfmn cache] wrote {cp}", flush=True)
    return pack


class PackLRU:
    """Load chunk caches on demand so 8500 does not keep all packs in RAM."""

    def __init__(self, paths: list[Path], *, max_open: int = 2, gc_every: int = 8):
        self.paths = [Path(p) for p in paths]
        self.max_open = max(1, int(max_open))
        self.gc_every = max(1, int(gc_every))
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.n_hits = 0
        self.n_loads = 0
        self.n_evictions = 0

    def __len__(self) -> int:
        return len(self.paths)

    def reset_stats(self) -> None:
        self.n_hits = 0
        self.n_loads = 0
        self.n_evictions = 0

    def get(self, i: int) -> dict:
        i = int(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            self.n_hits += 1
            return self._cache[i]
        while len(self._cache) >= self.max_open:
            self._cache.popitem(last=False)
            self.n_evictions += 1
            # Full GC on every miss was catastrophic for shuffle+lru=2; amortize.
            if self.n_evictions % self.gc_every == 0:
                gc.collect()
        pack = torch.load(self.paths[i], map_location="cpu", weights_only=False)
        self._cache[i] = pack
        self._cache.move_to_end(i)
        self.n_loads += 1
        return pack

    def peek_meta(self, i: int) -> dict:
        """Load pack once to read shapes, then drop if not caching permanently."""
        p = self.get(i)
        return {
            "n_samples": int(p["x"].shape[0]),
            "n_bus": int(p["x"].shape[1]),
            "n_edge": int(p["edge_index"].shape[1]),
            "node_dim": int(p.get("node_dim", p["x"].shape[-1])),
            "edge_dim": int(p.get("edge_dim", EDGE_FEAT_DIM)),
            "state_dim": int(p.get("state_dim", p["device_state"].shape[-1])),
        }


class PackAwareBatchSampler(Sampler[list[int]]):
    """Form batches within one chunk pack so PackLRU stays hot.

    Shuffle (optional) happens *within* each pack and across pack order, not
    by mixing arbitrary (chunk, sample) pairs — that caused silent torch.load
    thrashing with pack_lru=2 on 8500.
    """

    def __init__(
        self,
        sample_index: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        by_pack: dict[int, list[int]] = {}
        for di, (pi, _si) in enumerate(sample_index):
            by_pack.setdefault(int(pi), []).append(int(di))
        self._by_pack = by_pack
        self._len = sum(math.ceil(len(v) / self.batch_size) for v in by_pack.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._len)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 10007)
        packs = list(self._by_pack.keys())
        if self.shuffle:
            rng.shuffle(packs)
        for pi in packs:
            idxs = list(self._by_pack[pi])
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                yield idxs[i : i + self.batch_size]


def _as_int_list(v) -> list[int] | None:
    if v is None:
        return None
    if torch.is_tensor(v):
        return [int(x) for x in v.detach().cpu().reshape(-1).tolist()]
    if isinstance(v, (list, tuple, np.ndarray)):
        return [int(x) for x in list(v)]
    return None


def _build_sample_pairs(
    packs_meta: list[dict],
    *,
    dedupe_sample_ids: bool,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Build (pack_i, local_i) pairs; optionally keep first occurrence of each sample_id."""
    all_pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    raw = 0
    dropped = 0
    missing_sid = 0
    for ci, meta in enumerate(packs_meta):
        n_s = int(meta["n_samples"])
        raw += n_s
        sids = meta.get("sample_ids")
        if dedupe_sample_ids and sids is not None and len(sids) == n_s:
            for si, sid in enumerate(sids):
                if sid in seen:
                    dropped += 1
                    continue
                seen.add(sid)
                all_pairs.append((ci, si))
        else:
            if dedupe_sample_ids:
                missing_sid += 1
            all_pairs.extend((ci, s) for s in range(n_s))
    stats = {
        "raw": int(raw),
        "kept": int(len(all_pairs)),
        "dropped_dup_sid": int(dropped),
        "packs_missing_sample_ids": int(missing_sid),
        "unique_sample_ids": int(len(seen)),
    }
    return all_pairs, stats


class PfmnData(Data):
    """Stack graph-level tensors on a new batch dim (device_state, y_sub)."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("device_state", "y_sub"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class PfmnOracleDataset(Dataset):
    def __init__(
        self,
        packs: PackLRU,
        sample_index: list[tuple[int, int]],
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        e_mean: torch.Tensor,
        e_std: torch.Tensor,
        sub_mean: torch.Tensor,
        sub_std: torch.Tensor,
    ):
        self.packs = packs
        self.sample_index = sample_index
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
        self.e_mean, self.e_std = e_mean, e_std
        self.sub_mean, self.sub_std = sub_mean, sub_std

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, i: int) -> PfmnData:
        pi, si = self.sample_index[i]
        p = self.packs.get(pi)
        x = p["x"][si].clone()
        for j in NODE_CONT_IDX:
            x[:, j] = (x[:, j] - self.x_mean[j]) / self.x_std[j]
        y = (p["y_voltage"][si] - self.y_mean) / self.y_std
        mask = p["y_voltage_mask"][si]
        y_sub = (p["y_substation"][si] - self.sub_mean) / self.sub_std
        ea = materialize_edge_attr(p["edge_attr_static"], p["edge_tap_reg_idx"], p["reg_taps"][si])
        for j in EDGE_CONT_IDX:
            ea[:, j] = (ea[:, j] - self.e_mean[j]) / self.e_std[j]
        ds = p["device_state"][si]
        return PfmnData(
            x=x,
            edge_index=p["edge_index"],
            edge_attr=ea,
            y=y,
            y_mask=mask,
            y_sub=y_sub,
            device_state=ds,
        )


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n - 2, 1)
    n_val = min(max(n_val, 1), n - n_train - 1) if n >= 3 else 1
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_val - 1)
        n_train = n - n_val - n_test
    tr = perm[:n_train]
    va = perm[n_train : n_train + n_val]
    te = perm[n_train + n_val :]
    return tr, va, te


def _fit_norm_stats(
    packs: PackLRU,
    train_pairs: list[tuple[int, int]],
    *,
    max_samples: int = 4096,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Stream means/stds without concatenating all graphs (avoids Colab OOM)."""
    pairs = train_pairs
    if len(pairs) > int(max_samples) > 0:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(len(pairs), size=int(max_samples), replace=False)
        pairs = [pairs[int(i)] for i in pick]
        print(
            f"[pfmn] norm stats on {len(pairs)}/{len(train_pairs)} train samples "
            f"(cap={max_samples})",
            flush=True,
        )

    x_sum = torch.zeros(NODE_FEAT_DIM, dtype=torch.float64)
    x_sumsq = torch.zeros(NODE_FEAT_DIM, dtype=torch.float64)
    x_count = 0
    y_sum = torch.zeros(6, dtype=torch.float64)
    y_sumsq = torch.zeros(6, dtype=torch.float64)
    y_count = 0
    e_sum = torch.zeros(EDGE_FEAT_DIM, dtype=torch.float64)
    e_sumsq = torch.zeros(EDGE_FEAT_DIM, dtype=torch.float64)
    e_count = 0
    sub_sum = torch.zeros(6, dtype=torch.float64)
    sub_sumsq = torch.zeros(6, dtype=torch.float64)
    sub_count = 0

    for pi, si in pairs:
        p = packs.get(pi)
        x = p["x"][si].double()
        y = p["y_voltage"][si].double()
        sub = p["y_substation"][si].double()
        ea = materialize_edge_attr(p["edge_attr_static"], p["edge_tap_reg_idx"], p["reg_taps"][si]).double()
        x_sum += x.sum(dim=0)
        x_sumsq += x.pow(2).sum(dim=0)
        x_count += int(x.shape[0])
        y_sum += y.sum(dim=0)
        y_sumsq += y.pow(2).sum(dim=0)
        y_count += int(y.shape[0])
        e_sum += ea.sum(dim=0)
        e_sumsq += ea.pow(2).sum(dim=0)
        e_count += int(ea.shape[0])
        sub_sum += sub
        sub_sumsq += sub.pow(2)
        sub_count += 1

    def _mean_std(sum_, sumsq, n: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = max(n, 1)
        mean = (sum_ / n).float()
        var = (sumsq / n - (sum_ / n).pow(2)).clamp_min(0.0)
        std = var.sqrt().float().clamp_min(1e-6)
        if mean.numel() != dim:
            mean = mean.view(dim)
            std = std.view(dim)
        return mean, std

    x_mean = torch.zeros(NODE_FEAT_DIM)
    x_std = torch.ones(NODE_FEAT_DIM)
    xm, xs = _mean_std(x_sum, x_sumsq, x_count, NODE_FEAT_DIM)
    for j in NODE_CONT_IDX:
        x_mean[j] = xm[j]
        x_std[j] = xs[j]

    y_mean, y_std = _mean_std(y_sum, y_sumsq, y_count, 6)

    e_mean = torch.zeros(EDGE_FEAT_DIM)
    e_std = torch.ones(EDGE_FEAT_DIM)
    em, es = _mean_std(e_sum, e_sumsq, e_count, EDGE_FEAT_DIM)
    for j in EDGE_CONT_IDX:
        e_mean[j] = em[j]
        e_std[j] = es[j]

    sub_mean, sub_std = _mean_std(sub_sum, sub_sumsq, sub_count, 6)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "e_mean": e_mean,
        "e_std": e_std,
        "sub_mean": sub_mean,
        "sub_std": sub_std,
    }


def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(pred.dtype)
    denom = w.sum().clamp_min(1.0)
    return ((pred - tgt).pow(2) * w).sum() / denom


def _denorm(y_n: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y_n * std + mean


def _batch_y_sub(batch) -> torch.Tensor:
    ys = batch.y_sub
    if ys.dim() == 1:
        ys = ys.view(-1, 6)
    return ys


def _joint_loss(
    pred_v: torch.Tensor,
    pred_sub: torch.Tensor | None,
    batch,
    *,
    lambda_sub: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, volt, sub) losses. Sub term is 0 when λ_sub=0 or no head."""
    loss_v = _masked_mse(pred_v, batch.y, batch.y_mask)
    loss_s = pred_v.new_zeros(())
    if pred_sub is not None:
        loss_s = F.mse_loss(pred_sub, _batch_y_sub(batch))
    loss = loss_v + float(lambda_sub) * loss_s if float(lambda_sub) > 0 else loss_v
    return loss, loss_v, loss_s


@torch.no_grad()
def evaluate(
    model: PowerFlowMultiNet,
    loader: DataLoader,
    device: torch.device,
    norms: dict[str, torch.Tensor],
    *,
    lambda_sub: float,
) -> dict[str, float]:
    model.eval()
    tot_loss = 0.0
    volt_loss = 0.0
    sub_loss = 0.0
    n_batches = 0
    abs_vmag: list[torch.Tensor] = []
    abs_ang: list[torch.Tensor] = []
    mse_re: list[torch.Tensor] = []
    mse_im: list[torch.Tensor] = []
    mse_re_n: list[torch.Tensor] = []
    mse_im_n: list[torch.Tensor] = []
    abs_sub_p: list[torch.Tensor] = []
    abs_sub_q: list[torch.Tensor] = []
    y_true_mag: list[torch.Tensor] = []
    y_pred_mag: list[torch.Tensor] = []

    y_mean = norms["y_mean"].to(device)
    y_std = norms["y_std"].to(device)
    sub_mean = norms["sub_mean"].to(device)
    sub_std = norms["sub_std"].to(device)

    for batch in loader:
        batch = batch.to(device)
        pred_v, pred_sub = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.device_state, batch.batch
        )
        loss, loss_v, loss_s = _joint_loss(pred_v, pred_sub, batch, lambda_sub=lambda_sub)
        tot_loss += float(loss.item())
        volt_loss += float(loss_v.item())
        sub_loss += float(loss_s.item())
        n_batches += 1

        pred_phys = _denorm(pred_v, y_mean, y_std)
        true_phys = _denorm(batch.y, y_mean, y_std)
        m = batch.y_mask > 0.5
        for ph in range(3):
            mm = m[:, 2 * ph]
            ma = m[:, 2 * ph + 1]
            if mm.any():
                pv = pred_phys[mm, 2 * ph]
                tv = true_phys[mm, 2 * ph]
                abs_vmag.append((pv - tv).abs())
                y_pred_mag.append(pv)
                y_true_mag.append(tv)
                if ma.any() and int(ma.sum()) == int(mm.sum()):
                    pa = pred_phys[ma, 2 * ph + 1]
                    ta = true_phys[ma, 2 * ph + 1]
                    pr = pv * torch.cos(pa)
                    pi_ = pv * torch.sin(pa)
                    tr = tv * torch.cos(ta)
                    ti = tv * torch.sin(ta)
                    mse_re.append((pr - tr).pow(2))
                    mse_im.append((pi_ - ti).pow(2))
                    pvn = pred_v[mm, 2 * ph]
                    tvn = batch.y[mm, 2 * ph]
                    pan = pred_v[ma, 2 * ph + 1]
                    tan = batch.y[ma, 2 * ph + 1]
                    mse_re_n.append((pvn * torch.cos(pan) - tvn * torch.cos(tan)).pow(2))
                    mse_im_n.append((pvn * torch.sin(pan) - tvn * torch.sin(tan)).pow(2))
            if ma.any():
                d = pred_phys[ma, 2 * ph + 1] - true_phys[ma, 2 * ph + 1]
                d = (d + math.pi) % (2 * math.pi) - math.pi
                abs_ang.append(d.abs() * (180.0 / math.pi))

        if pred_sub is not None:
            ys = _batch_y_sub(batch)
            pred_s = _denorm(pred_sub, sub_mean, sub_std)
            true_s = _denorm(ys, sub_mean, sub_std)
            # channels: P_a,Q_a,P_b,Q_b,P_c,Q_c
            abs_sub_p.append((pred_s[:, 0::2] - true_s[:, 0::2]).abs().reshape(-1))
            abs_sub_q.append((pred_s[:, 1::2] - true_s[:, 1::2]).abs().reshape(-1))

    def _cat_mean(xs: list[torch.Tensor]) -> float:
        if not xs:
            return float("nan")
        return float(torch.cat(xs).mean().item())

    mae_vmag = _cat_mean(abs_vmag)
    mae_ang = _cat_mean(abs_ang)
    mse_ri = float("nan")
    if mse_re and mse_im:
        mse_ri = float(0.5 * (torch.cat(mse_re).mean() + torch.cat(mse_im).mean()).item())
    mse_ri_n = float("nan")
    if mse_re_n and mse_im_n:
        mse_ri_n = float(0.5 * (torch.cat(mse_re_n).mean() + torch.cat(mse_im_n).mean()).item())

    r2 = float("nan")
    if y_true_mag:
        yt = torch.cat(y_true_mag)
        yp = torch.cat(y_pred_mag)
        ss_res = (yt - yp).pow(2).sum()
        ss_tot = (yt - yt.mean()).pow(2).sum().clamp_min(1e-12)
        r2 = float((1.0 - ss_res / ss_tot).item())

    nb = max(n_batches, 1)
    return {
        "loss_total": tot_loss / nb,
        "loss_volt": volt_loss / nb,
        "loss_sub": sub_loss / nb,
        "loss_tot": tot_loss / nb,
        "mae_vmag_pu": mae_vmag,
        "mae_angle_deg": mae_ang,
        "mae_sub_p": _cat_mean(abs_sub_p),
        "mae_sub_q": _cat_mean(abs_sub_q),
        "mse_ri": mse_ri,
        "mse_ri_normalized": mse_ri_n if mse_ri_n == mse_ri_n else mse_ri,
        "r2_vmag_mean": r2,
    }


def _print_metrics_line(tag: str, met: dict[str, float]) -> None:
    mse_ri = float(met.get("mse_ri_normalized", met.get("mse_ri", float("nan"))))
    sub_p = float(met.get("mae_sub_p", float("nan")))
    sub_q = float(met.get("mae_sub_q", float("nan")))
    print(
        f"{tag} |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"sub P MAE={sub_p:.6f}  sub Q MAE={sub_q:.6f}  "
        f"Re/Im MSE(nrm)={mse_ri:.6f}  r2_mean={met['r2_vmag_mean']:.6f}  "
        f"tot={met['loss_total']:.6f}  volt={met['loss_volt']:.6f}  sub={met.get('loss_sub', float('nan')):.6f}",
        flush=True,
    )


def _print_eval_section(label: str, phase: str, val_met: dict, test_met: dict) -> None:
    print(f"\n=== {label} ({phase}) ===", flush=True)
    _print_metrics_line("Val ", val_met)
    _print_metrics_line("Test", test_met)


def _format_epoch_log(
    *,
    epoch: int,
    epochs: int,
    train_tot: float,
    train_volt: float,
    train_sub: float,
    val_met: dict[str, float] | None,
    best_val: float,
    best_epoch: int,
    bad: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> str:
    line = (
        f"[pfmn chunk_parent] epoch {epoch:4d}/{epochs} "
        f"| train_tot={train_tot:.6f} train_volt={train_volt:.6f} train_sub={train_sub:.6f}"
    )
    if val_met is not None:
        line += (
            f" | val_tot={val_met['loss_total']:.6f} val_volt={val_met['loss_volt']:.6f} "
            f"val_sub={val_met.get('loss_sub', float('nan')):.6f} "
            f"| val_r2_mean={val_met['r2_vmag_mean']:.6f} "
            f"val_|V|_MAE={val_met['mae_vmag_pu']:.6f} "
            f"val_ang_MAE={val_met['mae_angle_deg']:.6f}"
        )
    best_s = f"{best_val:.6f}" if best_val < float("inf") else "inf"
    line += (
        f" | best={best_s} @ epoch {best_epoch} "
        f"| epochs_since_best={bad} patience={patience} min_delta={min_delta:g} "
        f"| lr={lr:.2e}"
    )
    return line


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _multistep_milestones(epochs: int) -> list[int]:
    """Paper-faithful assumption: drop LR at 50% and 80% of max epochs."""
    e = max(1, int(epochs))
    m1 = max(1, int(round(0.5 * e)))
    m2 = max(m1 + 1, int(round(0.8 * e)))
    return [m1, m2]


def train(args: argparse.Namespace) -> Path:
    _configure_stdout()
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    chunk_parent = Path(args.chunk_parent).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = _sorted_run_chunks(chunk_parent, args.chunk_subdir_glob)
    if not chunks:
        raise RuntimeError(f"No chunks under {chunk_parent} glob={args.chunk_subdir_glob!r}")
    print(
        f"[pfmn chunk_parent] feeder_chunks={len(chunks)} parent={chunk_parent} cache={cache_dir}",
        flush=True,
    )

    packs_paths: list[Path] = []
    packs_meta: list[dict] = []
    for ci, ch in enumerate(chunks):
        pack = _load_or_build_chunk(
            ch,
            cache_dir,
            nodes_csv=args.nodes_csv,
            edges_csv=args.edge_catalog_csv,
            meta_csv=args.meta_csv,
            rebuild=bool(args.rebuild_cache),
        )
        cp = _cache_path(cache_dir, ch)
        packs_paths.append(cp)
        packs_meta.append(
            {
                "n_samples": int(pack["x"].shape[0]),
                "sample_ids": _as_int_list(pack.get("sample_ids")),
                "chunk_name": str(pack.get("chunk_name", ch.name)),
            }
        )
        # Drop in-memory pack immediately — PackLRU reloads from disk as needed.
        del pack
        gc.collect()

    pack_lru = max(1, int(getattr(args, "pack_lru", 2)))
    packs = PackLRU(packs_paths, max_open=pack_lru)
    dedupe = bool(getattr(args, "dedupe_sample_ids", True))
    all_pairs, dedupe_stats = _build_sample_pairs(packs_meta, dedupe_sample_ids=dedupe)
    if dedupe:
        print(
            f"[pfmn] sample_id dedupe: raw={dedupe_stats['raw']} kept={dedupe_stats['kept']} "
            f"dropped_dup={dedupe_stats['dropped_dup_sid']} "
            f"unique_ids={dedupe_stats['unique_sample_ids']} "
            f"packs_missing_ids={dedupe_stats['packs_missing_sample_ids']}",
            flush=True,
        )

    if 0 < float(args.sample_frac) < 1.0:
        rng = np.random.default_rng(int(args.seed))
        k = max(3, int(round(len(all_pairs) * float(args.sample_frac))))
        pick = rng.choice(len(all_pairs), size=min(k, len(all_pairs)), replace=False)
        all_pairs = [all_pairs[int(i)] for i in pick]

    tr_i, va_i, te_i = _split_indices(len(all_pairs), args.train_frac, args.val_frac, args.seed)
    train_pairs = [all_pairs[int(i)] for i in tr_i]
    val_pairs = [all_pairs[int(i)] for i in va_i]
    test_pairs = [all_pairs[int(i)] for i in te_i]
    print(
        f"[pfmn chunk_parent] samples total={len(all_pairs)} "
        f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)} "
        f"split_seed={args.seed} pack_lru={pack_lru}",
        flush=True,
    )
    if len(all_pairs) > 5000:
        print(
            f"[pfmn] WARNING: {len(all_pairs)} samples across {len(chunks)} chunks "
            f"(mean {len(all_pairs)/max(len(chunks),1):.0f}/chunk). "
            f"Folder name may imply ~50/chunk — check CSVs / sample_id coverage. "
            f"Using pack-local batches + lazy LRU to limit I/O thrashing.",
            flush=True,
        )

    norms = _fit_norm_stats(
        packs,
        train_pairs,
        max_samples=int(getattr(args, "norm_max_samples", 4096)),
        seed=int(args.seed),
    )
    meta0 = packs.peek_meta(0)
    node_dim = int(meta0["node_dim"])
    edge_dim = int(meta0["edge_dim"])
    state_dim = int(meta0["state_dim"])
    n_bus = int(meta0["n_bus"])
    n_edge = int(meta0["n_edge"])
    print(
        f"[pfmn] graph n_bus={n_bus} n_edge={n_edge} "
        f"node_dim={node_dim} edge_dim={edge_dim} state_dim={state_dim} "
        f"hidden={args.hidden} layers={args.layers} lambda_sub={args.lambda_sub}",
        flush=True,
    )

    ds_train = PfmnOracleDataset(packs, train_pairs, **norms)
    ds_val = PfmnOracleDataset(packs, val_pairs, **norms)
    ds_test = PfmnOracleDataset(packs, test_pairs, **norms)

    nw = int(args.num_workers)
    # Lazy pack store is process-local; workers would each reload caches → RAM blowup.
    if nw > 0 and pack_lru < len(packs_paths):
        print(
            f"[pfmn] forcing num_workers=0 (was {nw}) with pack_lru={pack_lru} "
            f"to avoid per-worker cache copies",
            flush=True,
        )
        nw = 0
    use_persistent = bool(getattr(args, "persistent_workers", False)) and nw > 0
    pack_locality = bool(getattr(args, "pack_locality", True))
    bs = int(args.batch_size)
    base_kw: dict = dict(
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    if nw > 0:
        base_kw["persistent_workers"] = use_persistent
        base_kw["prefetch_factor"] = 2 if use_persistent else 1

    train_batch_sampler: PackAwareBatchSampler | None = None
    if pack_locality:
        train_batch_sampler = PackAwareBatchSampler(
            train_pairs, bs, shuffle=True, seed=int(args.seed)
        )
        val_batch_sampler = PackAwareBatchSampler(
            val_pairs, bs, shuffle=False, seed=int(args.seed)
        )
        test_batch_sampler = PackAwareBatchSampler(
            test_pairs, bs, shuffle=False, seed=int(args.seed)
        )
        train_loader = DataLoader(ds_train, batch_sampler=train_batch_sampler, **base_kw)
        val_loader = DataLoader(ds_val, batch_sampler=val_batch_sampler, **base_kw)
        test_loader = DataLoader(ds_test, batch_sampler=test_batch_sampler, **base_kw)
    else:
        train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, **base_kw)
        val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, **base_kw)
        test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, **base_kw)
    print(
        f"[pfmn] dataloader num_workers={nw} persistent_workers={use_persistent} "
        f"pin_memory={base_kw['pin_memory']} batch_size={bs} "
        f"grad_accum={args.grad_accum} pack_locality={pack_locality}",
        flush=True,
    )

    model = PowerFlowMultiNet(
        node_dim,
        edge_dim,
        state_dim,
        hidden=int(args.hidden),
        num_layers=int(args.layers),
        dropout=float(args.dropout),
        predict_substation=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    milestones = _multistep_milestones(int(args.epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=float(args.lr_gamma))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    use_amp = device.type == "cuda" and not args.no_amp

    accum = max(1, int(args.grad_accum))
    effective_bs = int(args.batch_size) * accum
    print(
        f"[pfmn] batch_size={args.batch_size} grad_accum={accum} effective_batch~={effective_bs} "
        f"amp={use_amp} MultiStepLR milestones={milestones} gamma={args.lr_gamma}",
        flush=True,
    )

    best_val = float("inf")
    best_es = float("inf")
    best_epoch = 0
    best_ckpt_epoch = 0
    best_val_r2 = float("nan")
    bad = 0
    history: list[dict] = []
    train_pool_epoch_history: list[dict] = []
    last_val_met: dict[str, float] | None = None
    t0 = time.time()
    min_delta = float(args.min_delta)
    log_every = int(args.log_every)
    no_early_stop = bool(args.no_early_stop)
    lam_sub = float(args.lambda_sub)

    manifest = {
        "task": "PowerFlowMultiNet oracle chunk_parent",
        "model": "PowerFlowMultiNet — oracle device states",
        "paper": "arXiv:2403.00892v3 PowerFlowMultiNet",
        "not_da_gps": True,
        "oracle_device_states": True,
        "chunk_parent": str(chunk_parent),
        "chunks": [c.name for c in chunks],
        "n_chunks": len(chunks),
        "chunk_tensor_cache_dir": str(cache_dir),
        "cache_schema": "pfmn_oracle_v2",
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "lr_schedule": {
            "type": "MultiStepLR",
            "milestones": milestones,
            "gamma": float(args.lr_gamma),
            "note": "implementation assumption — paper states Adam lr=0.001 without schedule details",
        },
        "batch_size": int(args.batch_size),
        "grad_accum": accum,
        "effective_batch": effective_bs,
        "lambda_sub": lam_sub,
        "dropout": float(args.dropout),
        "n_bus": n_bus,
        "n_edge": n_edge,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "state_dim": state_dim,
        "reg_cols": packs.get(0).get("reg_cols", []),
        "cap_cols": packs.get(0).get("cap_cols", []),
        "hyperparameters": vars(args),
        "manifest_stage": "pre_train",
        "paper_fidelity": {
            "matches": [
                "physical buses as nodes; parallel phase edges (multigraph)",
                "node features P,Q per phase + phase masks + source + bus caps",
                "edge features phase / type / tap / switch_closed",
                "capacitor (and switch) states via separate state MLP — oracle OpenDSS inputs",
                "targets: bus Vmag/Vang and substation P/Q per phase",
                "GENConv DeeperGCN: powermean, learn_p, msg_norm, learn_msg_scale, residual res+",
                "joint MSE on voltage + substation heads (default lambda_sub=1)",
                "Adam lr=0.001; epochs default 50; effective batch ≈128",
            ],
            "implementation_choices": [
                "hidden=128, L=12 unified across ieee34/906/8500 (paper silent on exact L/hidden)",
                "MultiStepLR at 50%/80% of max epochs, gamma=0.1 (not specified in paper)",
                "80/10/10 split on pooled chunk scenarios (paper used 8k/2k)",
                "DeepGCNLayer first block=plain, subsequent=res+",
                "source bus = physical bus with smallest node_idx",
            ],
            "gaps_vs_paper": [
                "8500 secondary / split-phase not modeled beyond A/B/C edges in edge CSV",
                "dataset scale differs from paper's 8k/2k synthetic set",
                "exact paper residual-block hyperparameters may differ from torch_geometric defaults",
            ],
        },
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _manifest_text = json.dumps(manifest, indent=2, default=str)
    (out_dir / "run_manifest.json").write_text(_manifest_text, encoding="utf-8")
    (out_dir / "pfmn_run_manifest.json").write_text(_manifest_text, encoding="utf-8")
    print(f"Wrote run manifest: {out_dir / 'run_manifest.json'}", flush=True)

    def _bundle(epoch: int, *, best: bool = False) -> dict:
        return {
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "epoch": int(epoch),
            "best": bool(best),
            "best_val": float(best_val) if best_val < float("inf") else None,
            "best_epoch": int(best_ckpt_epoch),
            "best_val_r2_mean": float(best_val_r2) if best_val_r2 == best_val_r2 else None,
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "state_dim": state_dim,
            "dropout": float(args.dropout),
            "n_bus": n_bus,
            "n_edge": n_edge,
            "lambda_sub": lam_sub,
            "norms": {k: v.detach().cpu().clone() for k, v in norms.items()},
            "args": vars(args),
            "manifest": manifest,
            "model_name": "powerflowmultinet_oracle",
            "chunk_parent": str(chunk_parent),
            "chunk_folders": [c.name for c in chunks],
            "cache_schema": "pfmn_oracle_v2",
        }

    def _save_training_last(epoch: int, *, reason: str = "") -> Path:
        ck = out_dir / "training_last.pt"
        _atomic_torch_save(
            {
                **_bundle(epoch, best=False),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "bad": int(bad),
                "checkpoint_type": "training_last",
            },
            ck,
        )
        tag = f" ({reason})" if reason else ""
        print(f"  periodic checkpoint{tag} -> {ck}", flush=True)
        return ck

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        running_v = 0.0
        running_s = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        packs.reset_stats()
        n_train_batches = len(train_loader)
        log_steps = int(getattr(args, "log_steps", 50))
        t_epoch0 = time.time()
        print(
            f"[pfmn] epoch {epoch}/{int(args.epochs)} start "
            f"({n_train_batches} microbatches, train_n={len(train_pairs)})",
            flush=True,
        )
        it = train_loader
        if tqdm is not None and args.show_tqdm:
            it = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(it, start=1):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_v, pred_sub = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.device_state, batch.batch
                )
                loss, loss_v, loss_s = _joint_loss(pred_v, pred_sub, batch, lambda_sub=lam_sub)
                loss = loss / float(accum)
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * float(accum)
            running_v += float(loss_v.item())
            running_s += float(loss_s.item())
            n_seen += 1
            if log_steps > 0 and (step == 1 or step % log_steps == 0 or step == n_train_batches):
                avg = running / max(n_seen, 1)
                hit = packs.n_hits
                load = packs.n_loads
                tot = max(1, hit + load)
                print(
                    f"[pfmn] epoch {epoch} step {step}/{n_train_batches} "
                    f"loss={avg:.5f} volt={running_v/max(n_seen,1):.5f} "
                    f"sub={running_s/max(n_seen,1):.5f} "
                    f"pack_hit={hit/(tot):.2f} loads={load} "
                    f"elapsed={time.time()-t_epoch0:.0f}s",
                    flush=True,
                )

        tr_loss = running / max(n_seen, 1)
        tr_volt = running_v / max(n_seen, 1)
        tr_sub = running_s / max(n_seen, 1)
        sch.step()
        history.append(
            {
                "epoch": int(epoch),
                "train_tot": tr_loss,
                "train_volt": tr_volt,
                "train_sub": tr_sub,
                "train_loss": tr_loss,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

        do_eval = (
            (epoch == 1)
            or (epoch % int(args.eval_every) == 0)
            or (epoch == int(args.epochs))
        )
        # Defer test until a regular eval cadence (not forced on epoch 1 alone).
        do_test = (epoch % int(args.eval_every) == 0) or (epoch == int(args.epochs))
        do_log = (
            do_eval
            or (log_every > 0 and (epoch == 1 or epoch % log_every == 0))
            or (epoch == int(args.epochs))
        )

        val_met: dict[str, float] | None = None
        test_met: dict[str, float] | None = None
        if do_eval:
            print(f"[pfmn] epoch {epoch} eval val (n={len(val_pairs)}) ...", flush=True)
            val_met = evaluate(model, val_loader, device, norms, lambda_sub=lam_sub)
            if do_test:
                print(f"[pfmn] epoch {epoch} eval test (n={len(test_pairs)}) ...", flush=True)
                test_met = evaluate(model, test_loader, device, norms, lambda_sub=lam_sub)
            last_val_met = val_met
            history[-1].update(
                {
                    "val": val_met,
                    "test": test_met,
                    "val_tot": float(val_met["loss_total"]),
                    "val_volt": float(val_met["loss_volt"]),
                    "val_sub": float(val_met["loss_sub"]),
                    "val_r2_mean": float(val_met["r2_vmag_mean"]),
                }
            )
            train_pool_epoch_history.append(
                {"epoch": int(epoch), "val_metrics": val_met, "test_metrics": test_met}
            )

            score = float(val_met["loss_total"])
            if score < best_val:
                best_val = score
                best_ckpt_epoch = epoch
                best_val_r2 = float(val_met["r2_vmag_mean"])
                _atomic_torch_save(_bundle(epoch, best=True), out_dir / "pfmn_oracle_best.pt")
                print(f"Saved {out_dir / 'pfmn_oracle_best.pt'}", flush=True)

            # Calendar-epoch early stop (min_delta), matching DA-GPS
            if best_es == float("inf") or (best_es - score) >= min_delta:
                best_es = score
                best_epoch = epoch

        # Always refresh from calendar epoch so log_every lines between evals
        # are truthful (bad used to stay frozen at 0 until the next eval).
        if best_epoch > 0:
            bad = int(epoch - best_epoch)

        if do_log:
            print(
                _format_epoch_log(
                    epoch=epoch,
                    epochs=int(args.epochs),
                    train_tot=tr_loss,
                    train_volt=tr_volt,
                    train_sub=tr_sub,
                    val_met=val_met if do_eval else None,
                    best_val=best_val,
                    best_epoch=best_ckpt_epoch if best_ckpt_epoch > 0 else best_epoch,
                    bad=bad,
                    patience=int(args.patience),
                    min_delta=min_delta,
                    lr=float(opt.param_groups[0]["lr"]),
                ),
                flush=True,
            )

        if do_eval and val_met is not None:
            if test_met is not None:
                _print_eval_section("train_pool_eval", f"epoch {epoch}", val_met, test_met)
            else:
                print(f"\n=== train_pool_eval (epoch {epoch}, val-only) ===", flush=True)
                _print_metrics_line("Val ", val_met)

        _ce = int(args.checkpoint_every)
        if _ce > 0 and (epoch % _ce == 0 or epoch == int(args.epochs)):
            _save_training_last(epoch)

        if do_eval and not no_early_stop and bad >= int(args.patience):
            print(
                f"[pfmn chunk_parent] early stop at epoch {epoch}, "
                f"best={best_val:.6f} @ epoch {best_ckpt_epoch} "
                f"(epochs_since_best={bad}, patience={int(args.patience)}, "
                f"min_delta={min_delta:g})",
                flush=True,
            )
            if _ce > 0:
                _save_training_last(epoch, reason="early stop")
            break

        if should_interactive_pause(epoch, args):
            _choice = ask_continue_or_stop(
                out_dir=out_dir,
                epoch=epoch,
                epochs=int(args.epochs),
                best_val=float(best_val) if best_val < float("inf") else None,
                best_epoch=int(best_ckpt_epoch),
            )
            if _choice == "stop":
                print(
                    f"[pfmn chunk_parent] interactive stop at epoch {epoch}, "
                    f"best={best_val:.6f} @ epoch {best_ckpt_epoch}",
                    flush=True,
                )
                if _ce > 0:
                    _save_training_last(epoch, reason="interactive stop")
                break

    best_path = out_dir / "pfmn_oracle_best.pt"
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    val_met = evaluate(model, val_loader, device, norms, lambda_sub=lam_sub)
    test_met = evaluate(model, test_loader, device, norms, lambda_sub=lam_sub)
    _print_eval_section("train_pool_eval", f"best epoch {best_ckpt_epoch}", val_met, test_met)

    if not (out_dir / "training_last.pt").is_file():
        _save_training_last(int(best_ckpt_epoch) if best_ckpt_epoch > 0 else int(args.epochs))

    train_seconds = time.time() - t0
    report = {
        "task": "PowerFlowMultiNet oracle chunk_parent",
        "model": "PowerFlowMultiNet — oracle device states",
        "paper": "arXiv:2403.00892v3 PowerFlowMultiNet",
        "not_da_gps": True,
        "oracle_device_states": True,
        "chunk_parent": str(chunk_parent),
        "chunks": [c.name for c in chunks],
        "n_chunks": len(chunks),
        "chunk_tensor_cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
        "hyperparameters": vars(args),
        "best_epoch": int(best_ckpt_epoch),
        "best_val": float(best_val) if best_val < float("inf") else None,
        "best_val_loss": float(best_val) if best_val < float("inf") else None,
        "best_val_r2_mean": float(best_val_r2) if best_val_r2 == best_val_r2 else None,
        "val_metrics": val_met,
        "test_metrics": test_met,
        "train_pool_eval": {
            "label": "train_pool_eval",
            "chunk_parent": str(chunk_parent),
            "chunks": [c.name for c in chunks],
            "split_seed": int(args.seed),
            "n_chunks": len(chunks),
            "best_epoch": int(best_ckpt_epoch),
            "final_val_metrics": val_met,
            "final_test_metrics": test_met,
            "epoch_history": train_pool_epoch_history,
        },
        "history": history,
        "train_seconds": train_seconds,
        "elapsed_sec": train_seconds,
        "checkpoint": str(best_path.resolve()) if best_path.is_file() else None,
        "checkpoint_last": str((out_dir / "training_last.pt").resolve()),
        "manifest": manifest,
    }
    report_path = out_dir / "pfmn_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(
        f"Val  |V| MAE={val_met['mae_vmag_pu']:.6f}  angle MAE={val_met['mae_angle_deg']:.6f}  "
        f"sub P MAE={val_met['mae_sub_p']:.6f}  sub Q MAE={val_met['mae_sub_q']:.6f}  "
        f"Re/Im MSE(nrm)={val_met.get('mse_ri_normalized', val_met['mse_ri']):.6f}  "
        f"r2_mean={val_met['r2_vmag_mean']:.6f}",
        flush=True,
    )
    print(
        f"Test |V| MAE={test_met['mae_vmag_pu']:.6f}  angle MAE={test_met['mae_angle_deg']:.6f}  "
        f"sub P MAE={test_met['mae_sub_p']:.6f}  sub Q MAE={test_met['mae_sub_q']:.6f}  "
        f"Re/Im MSE(nrm)={test_met.get('mse_ri_normalized', test_met['mse_ri']):.6f}  "
        f"r2_mean={test_met['r2_vmag_mean']:.6f}  time={train_seconds:.1f}s",
        flush=True,
    )
    if best_path.is_file():
        print(f"Saved {best_path}", flush=True)
    print(f"Run dir: {out_dir}", flush=True)
    print(f"Checkpoint (best): {best_path}", flush=True)
    print(f"Checkpoint (last): {out_dir / 'training_last.pt'}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PowerFlowMultiNet oracle-device-state baseline")
    p.add_argument("--chunk_parent", type=str, required=True)
    p.add_argument("--chunk_subdir_glob", type=str, default="run_*")
    p.add_argument("--nodes_csv", type=str, default="gnn_node_features_and_targets_mvagg.csv")
    p.add_argument("--edge_catalog_csv", type=str, default="gnn_edges_phase_static.csv")
    p.add_argument("--meta_csv", type=str, default="gnn_sample_meta.csv")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument(
        "--dedupe_sample_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep first occurrence of each sample_id across chunks (fixes ~40x inflation "
        "when each chunk CSV embeds the full scenario set).",
    )
    p.add_argument(
        "--pack_locality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Batch within one chunk pack (avoids PackLRU thrashing under shuffle).",
    )
    p.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="Print mid-epoch heartbeat every N microbatches (0=off).",
    )
    p.add_argument(
        "--pack_lru",
        type=int,
        default=2,
        help="Max chunk caches held in RAM (lazy load). Keep small on 8500/Colab.",
    )
    p.add_argument(
        "--norm_max_samples",
        type=int,
        default=4096,
        help="Cap train samples used for mean/std (streamed; avoids OOM).",
    )
    p.add_argument("--epochs", type=int, default=50, help="Max epochs (launcher default 50)")
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Per-step graphs; launcher sets feeder-aware (ieee34=64, 906=32, 8500=16)",
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Toward effective batch ≈128 (launcher: ieee34×2, 906×4, 8500×8)",
    )
    p.add_argument("--hidden", type=int, default=128, help="Implementation choice (paper silent)")
    p.add_argument("--layers", type=int, default=12, help="Unified L=12 across feeders")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_gamma", type=float, default=0.1, help="MultiStepLR gamma")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20, help="Early-stop patience (calendar epochs)")
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1, help="Epoch summary line frequency")
    add_interactive_pause_args(p)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument(
        "--lambda_sub",
        type=float,
        default=1.0,
        help="Weight on substation P/Q MSE (paper joint training; 0=volt-only)",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between epochs (extra RAM; off for large feeders).",
    )
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--show_tqdm", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    # Accept legacy underscore form from stale Colab imports of the launcher.
    raw = list(sys.argv[1:] if argv is None else argv)
    raw = [
        "--no-persistent-workers" if a == "--no_persistent_workers" else a for a in raw
    ]
    args = build_argparser().parse_args(raw)
    train(args)


if __name__ == "__main__":
    main()
