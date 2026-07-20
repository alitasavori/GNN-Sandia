"""Train PowerFlowMultiNet — oracle device states (GENConv baseline).

Not DA-GPS: taps/caps are inputs; targets are Vmag/Vang (and optional substation P/Q).

Artifacts (per OUT_DIR):
  - pfmn_oracle_best.pt
  - training_last.pt
  - pfmn_report.json
  - run_manifest.json

Default loss is volt-only (``--lambda_sub 0``). Paper epochs=1000; Colab default 200
via launcher (``--epochs`` exposed).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


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
    return cache_dir / f"{chunk_dir.name}__pfmn_oracle.pt"


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
    torch.save(pack, cp)
    print(f"[pfmn cache] wrote {cp}", flush=True)
    return pack


class PfmnData(Data):
    """Stack graph-level tensors on a new batch dim (device_state, y_sub)."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("device_state", "y_sub"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class PfmnOracleDataset(Dataset):
    def __init__(
        self,
        packs: list[dict],
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
        p = self.packs[pi]
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


def _fit_norm_stats(packs: list[dict], train_pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    xs = []
    ys = []
    eas = []
    subs = []
    for pi, si in train_pairs:
        p = packs[pi]
        xs.append(p["x"][si])
        ys.append(p["y_voltage"][si])
        subs.append(p["y_substation"][si])
        ea = materialize_edge_attr(p["edge_attr_static"], p["edge_tap_reg_idx"], p["reg_taps"][si])
        eas.append(ea)
    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)
    e_cat = torch.cat(eas, dim=0)
    sub_cat = torch.stack(subs, dim=0)

    x_mean = torch.zeros(NODE_FEAT_DIM)
    x_std = torch.ones(NODE_FEAT_DIM)
    for j in NODE_CONT_IDX:
        x_mean[j] = x_cat[:, j].mean()
        x_std[j] = x_cat[:, j].std(unbiased=False).clamp_min(1e-6)

    y_mean = y_cat.mean(dim=0)
    y_std = y_cat.std(dim=0, unbiased=False).clamp_min(1e-6)

    e_mean = torch.zeros(EDGE_FEAT_DIM)
    e_std = torch.ones(EDGE_FEAT_DIM)
    for j in EDGE_CONT_IDX:
        e_mean[j] = e_cat[:, j].mean()
        e_std[j] = e_cat[:, j].std(unbiased=False).clamp_min(1e-6)

    sub_mean = sub_cat.mean(dim=0)
    sub_std = sub_cat.std(dim=0, unbiased=False).clamp_min(1e-6)
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
    n_batches = 0
    # Accumulators for physical metrics
    abs_vmag = []
    abs_ang = []
    mse_re = []
    mse_im = []
    # R2 over flattened masked mag
    y_true_mag = []
    y_pred_mag = []

    y_mean = norms["y_mean"].to(device)
    y_std = norms["y_std"].to(device)
    sub_mean = norms["sub_mean"].to(device)
    sub_std = norms["sub_std"].to(device)

    for batch in loader:
        batch = batch.to(device)
        pred_v, pred_sub = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.device_state, batch.batch
        )
        loss_v = _masked_mse(pred_v, batch.y, batch.y_mask)
        loss = loss_v
        if lambda_sub > 0 and pred_sub is not None:
            # batch.y_sub is [B,6] after PyG stacking
            ys = batch.y_sub
            if ys.dim() == 1:
                ys = ys.view(-1, 6)
            loss = loss + float(lambda_sub) * F.mse_loss(pred_sub, ys)
        tot_loss += float(loss.item())
        volt_loss += float(loss_v.item())
        n_batches += 1

        pred_phys = _denorm(pred_v, y_mean, y_std)
        true_phys = _denorm(batch.y, y_mean, y_std)
        m = batch.y_mask > 0.5
        # mag channels 0,2,4 ; angle 1,3,5
        for ph in range(3):
            mm = m[:, 2 * ph]
            ma = m[:, 2 * ph + 1]
            if mm.any():
                pv = pred_phys[mm, 2 * ph]
                tv = true_phys[mm, 2 * ph]
                abs_vmag.append((pv - tv).abs())
                y_pred_mag.append(pv)
                y_true_mag.append(tv)
                # reconstruct re/im
                pa = pred_phys[ma, 2 * ph + 1] if ma.any() else None
                ta = true_phys[ma, 2 * ph + 1] if ma.any() else None
                if pa is not None and ta is not None and pa.numel() == pv.numel():
                    pr = pv * torch.cos(pa)
                    pi_ = pv * torch.sin(pa)
                    tr = tv * torch.cos(ta)
                    ti = tv * torch.sin(ta)
                    mse_re.append((pr - tr).pow(2))
                    mse_im.append((pi_ - ti).pow(2))
            if ma.any():
                # wrap-aware angle error
                d = pred_phys[ma, 2 * ph + 1] - true_phys[ma, 2 * ph + 1]
                d = (d + math.pi) % (2 * math.pi) - math.pi
                abs_ang.append(d.abs() * (180.0 / math.pi))

    def _cat_mean(xs: list[torch.Tensor]) -> float:
        if not xs:
            return float("nan")
        return float(torch.cat(xs).mean().item())

    mae_vmag = _cat_mean(abs_vmag)
    mae_ang = _cat_mean(abs_ang)
    mse_ri = float("nan")
    if mse_re and mse_im:
        mse_ri = float(0.5 * (torch.cat(mse_re).mean() + torch.cat(mse_im).mean()).item())

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
        "mae_vmag_pu": mae_vmag,
        "mae_angle_deg": mae_ang,
        "mse_ri": mse_ri,
        "r2_vmag_mean": r2,
    }


def _print_metrics_line(tag: str, met: dict[str, float]) -> None:
    print(
        f"{tag} |V| MAE={met['mae_vmag_pu']:.6f}  angle MAE={met['mae_angle_deg']:.6f}  "
        f"Re/Im MSE={met['mse_ri']:.6f}  r2_mean={met['r2_vmag_mean']:.6f}  "
        f"tot={met['loss_total']:.6f}  volt={met['loss_volt']:.6f}",
        flush=True,
    )


def _print_eval_section(label: str, phase: str, val_met: dict, test_met: dict) -> None:
    print(f"\n=== {label} ({phase}) ===", flush=True)
    _print_metrics_line("Val ", val_met)
    _print_metrics_line("Test", test_met)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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

    packs: list[dict] = []
    all_pairs: list[tuple[int, int]] = []
    for ci, ch in enumerate(chunks):
        pack = _load_or_build_chunk(
            ch,
            cache_dir,
            nodes_csv=args.nodes_csv,
            edges_csv=args.edge_catalog_csv,
            meta_csv=args.meta_csv,
            rebuild=bool(args.rebuild_cache),
        )
        packs.append(pack)
        n_s = int(pack["x"].shape[0])
        all_pairs.extend((ci, s) for s in range(n_s))

    # Optional subsample
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
        f"split_seed={args.seed}",
        flush=True,
    )

    norms = _fit_norm_stats(packs, train_pairs)
    node_dim = int(packs[0]["node_dim"])
    edge_dim = int(packs[0]["edge_dim"])
    state_dim = int(packs[0]["state_dim"])
    n_bus = int(packs[0]["x"].shape[1])
    n_edge = int(packs[0]["edge_index"].shape[1])
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
    loader_kw = dict(
        batch_size=int(args.batch_size),
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(ds_train, shuffle=True, **loader_kw)
    val_loader = DataLoader(ds_val, shuffle=False, **loader_kw)
    test_loader = DataLoader(ds_test, shuffle=False, **loader_kw)

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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    use_amp = device.type == "cuda" and not args.no_amp

    accum = max(1, int(args.grad_accum))
    effective_bs = int(args.batch_size) * accum
    print(
        f"[pfmn] batch_size={args.batch_size} grad_accum={accum} effective_batch~={effective_bs} amp={use_amp}",
        flush=True,
    )

    best_val = float("inf")
    best_epoch = 0
    bad = 0
    history: list[dict] = []
    t0 = time.time()

    manifest = {
        "model": "PowerFlowMultiNet — oracle device states",
        "not_da_gps": True,
        "chunk_parent": str(chunk_parent),
        "chunks": [c.name for c in chunks],
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "grad_accum": accum,
        "lambda_sub": float(args.lambda_sub),
        "dropout": float(args.dropout),
        "n_bus": n_bus,
        "n_edge": n_edge,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "state_dim": state_dim,
        "reg_cols": packs[0].get("reg_cols", []),
        "cap_cols": packs[0].get("cap_cols", []),
        "implementation_notes": {
            "hidden_layers": "implementation choice (paper does not clearly report hidden/L)",
            "epochs_default": "200 for Colab practicality; paper cites 1000 — override with --epochs",
            "edge_attrs": "paper-style minimum: phase one-hot, type one-hot, tap, switch_closed",
            "8500_secondary": "A/B/C multigraph from existing edge CSV only; no extra secondary graph",
            "net_injection": "P = p_load_kw - p_pv_kw, Q = q_load_kvar",
            "source_bus": "bus with smallest node_idx",
            "volt_only": "lambda_sub=0 by default; substation head still constructed",
        },
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _bundle(epoch: int, *, best: bool = False) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best": bool(best),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "state_dim": state_dim,
            "dropout": float(args.dropout),
            "norms": {k: v.cpu() for k, v in norms.items()},
            "manifest": manifest,
            "model_name": "powerflowmultinet_oracle",
        }

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        running_v = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        it = train_loader
        if tqdm is not None and args.show_tqdm:
            it = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(it, start=1):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_v, pred_sub = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.device_state, batch.batch
                )
                loss_v = _masked_mse(pred_v, batch.y, batch.y_mask)
                loss = loss_v
                if float(args.lambda_sub) > 0 and pred_sub is not None:
                    ys = batch.y_sub
                    if ys.dim() == 1:
                        ys = ys.view(-1, 6)
                    loss = loss + float(args.lambda_sub) * F.mse_loss(pred_sub, ys)
                loss = loss / float(accum)
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * float(accum)
            running_v += float(loss_v.item())
            n_seen += 1

        tr_loss = running / max(n_seen, 1)
        tr_volt = running_v / max(n_seen, 1)
        print(
            f"[pfmn chunk_parent] epoch {epoch}/{args.epochs} "
            f"loss={tr_loss:.6f} volt={tr_volt:.6f} lr={opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        do_eval = (epoch == 1) or (epoch % int(args.eval_every) == 0) or (epoch == int(args.epochs))
        if do_eval:
            val_met = evaluate(model, val_loader, device, norms, lambda_sub=float(args.lambda_sub))
            test_met = evaluate(model, test_loader, device, norms, lambda_sub=float(args.lambda_sub))
            _print_eval_section("train_pool_eval", f"epoch {epoch}", val_met, test_met)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val": val_met, "test": test_met})
            score = float(val_met["loss_total"])
            if score < best_val - 1e-8:
                best_val = score
                best_epoch = epoch
                bad = 0
                _atomic_torch_save(_bundle(epoch, best=True), out_dir / "pfmn_oracle_best.pt")
                print(f"Saved {out_dir / 'pfmn_oracle_best.pt'}", flush=True)
            else:
                bad += 1

        if epoch % int(args.checkpoint_every) == 0 or epoch == int(args.epochs):
            _atomic_torch_save(
                {
                    **_bundle(epoch, best=False),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                out_dir / "training_last.pt",
            )

        if bad >= int(args.patience):
            print(f"[pfmn] early stop at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Final / best eval
    best_path = out_dir / "pfmn_oracle_best.pt"
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    val_met = evaluate(model, val_loader, device, norms, lambda_sub=float(args.lambda_sub))
    test_met = evaluate(model, test_loader, device, norms, lambda_sub=float(args.lambda_sub))
    _print_eval_section("train_pool_eval", f"best epoch {best_epoch}", val_met, test_met)

    report = {
        "model": "PowerFlowMultiNet — oracle device states",
        "out_dir": str(out_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_sec": time.time() - t0,
        "val_metrics": val_met,
        "test_metrics": test_met,
        "history": history,
        "manifest": manifest,
    }
    (out_dir / "pfmn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out_dir / 'pfmn_report.json'}", flush=True)
    print(f"Checkpoint (best): {best_path}", flush=True)
    print(f"Checkpoint (last): {out_dir / 'training_last.pt'}", flush=True)
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
    p.add_argument("--epochs", type=int, default=200, help="Paper cites 1000; default 200 for Colab")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=16, help="Toward effective batch ≈128")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--lambda_sub", type=float, default=0.0, help="0 = volt-only baseline")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--show_tqdm", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
