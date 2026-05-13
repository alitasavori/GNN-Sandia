"""
Extract per-layer node→token attention from a trained DA-GPS checkpoint.

Uses ``DAGPSModel.forward_node_to_token_attention``: **first** cross-attn (token→node)
and **second** cross-attn (node→token). Token order: ``cap_target_cols``, ``reg_target_cols``,
``sys_0`` … ``sys_{K-1}``.

Example (Colab paths):
  python extract_da_gps_attention.py \\
    --ckpt /content/drive/.../da_gps_multitask_best.pt \\
    --run_dir /content/drive/.../da_gps_chunked_l4_mvagg_20260510_134709 \\
    --cache_pt /content/drive/.../cache/.../run_001_...__full.pt \\
    --edges_csv /content/drive/.../run_001_.../gnn_edges_phase_static.csv \\
    --sample_idx 0
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from train_homo_gine_global_localres_pq_loadonly import _load_compacted_edges


def _dagps_model_cls():
    """Fresh ``DAGPSModel`` class (reload train script — notebooks cache old 4-return forward)."""
    import importlib

    import train_da_gps_multitask_complex_voltage_gine as tdg

    importlib.reload(tdg)
    return tdg.DAGPSModel


def _node_order_from_ntl(node_to_local: dict[str, int]) -> list[str]:
    inv: list[str | None] = [None] * len(node_to_local)
    for name, i in node_to_local.items():
        inv[int(i)] = str(name)
    if any(x is None for x in inv):
        raise ValueError("node_to_local indices are not a dense 0..N-1 range")
    return [str(x) for x in inv]


def _token_names(cap_cols: list[str], reg_cols: list[str], n_system: int) -> list[str]:
    return list(cap_cols) + list(reg_cols) + [f"sys_{i}" for i in range(int(n_system))]


def _infer_node_in_edge_dim(state_dict: dict, *, hidden: int, node_emb_dim: int) -> tuple[int, int]:
    w_node = state_dict["node_in.0.weight"]
    node_in_dim = int(w_node.shape[1]) - int(node_emb_dim)
    w_msg = state_dict["blocks.0.mpnn.msg.0.weight"]
    edge_dim = int(w_msg.shape[1]) - 2 * int(hidden)
    if node_in_dim < 1 or edge_dim < 1:
        raise ValueError(f"Bad inferred dims: node_in_dim={node_in_dim}, edge_dim={edge_dim}")
    return node_in_dim, edge_dim


def _build_model(ckpt: dict, sd: dict, *, num_edges: int, dropout: float):
    DAGPSModel = _dagps_model_cls()
    hidden = int(ckpt["hidden"])
    node_emb_dim = int(ckpt["node_emb_dim"])
    edge_emb_dim = int(ckpt["edge_emb_dim"])
    node_in_dim, edge_dim = _infer_node_in_edge_dim(sd, hidden=hidden, node_emb_dim=node_emb_dim)
    return DAGPSModel(
        n_nodes=int(ckpt["n_nodes"]),
        num_edges=int(num_edges),
        hidden=hidden,
        heads=int(ckpt["heads"]),
        n_layers=int(ckpt["layers"]),
        n_cap=int(ckpt["n_cap"]),
        n_reg=int(ckpt["n_reg"]),
        n_system=int(ckpt["n_system_tokens"]),
        node_in_dim=node_in_dim,
        node_emb_dim=node_emb_dim,
        edge_emb_dim=edge_emb_dim,
        edge_dim=edge_dim,
        dropout=float(dropout),
        gradient_checkpointing=False,
        per_node_heads=bool(ckpt.get("per_node_heads", False)),
        per_device_cap_head=bool(ckpt.get("per_device_cap_head", False)),
        per_device_reg_head=bool(ckpt.get("per_device_reg_head", False)),
        n_pv_aux=int(ckpt.get("n_pv_aux", 0)),
    )


def run_attention_extract(
    ckpt_path: Path | str,
    run_dir: Path | str,
    cache_pt: Path | str,
    edges_csv: Path | str,
    *,
    sample_idx: int = 0,
    sample_id: int | None = None,
    out_dir: Path | str | None = None,
    device: str | None = None,
    dropout: float = 0.0,
    head_mean: bool = True,
    save_outputs: bool = True,
) -> dict:
    """Run full attention extract and write ``attention.pt``, ``manifest.json``, ``attention_mean_heads.npz``.

    Saves both cross-attentions: ``mean_heads`` / ``probs_nt`` = node→token (L,N,T),
    ``mean_heads_tn`` / ``probs_tn`` = token→node (L,T,N).

    Set ``save_outputs=False`` when looping over many samples (e.g. averaging attention only).

    Returns a dict with ``mean_heads``, ``mean_heads_tn``, ``manifest``, etc.
    """
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = Path(run_dir).resolve()
    if out_dir is None or str(out_dir).strip() == "":
        out_dir = run_dir / "attention_extract"
    out_dir = Path(out_dir).resolve()
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    dev_str = (device or ("cuda" if torch.cuda.is_available() else "cpu")).strip().lower()
    if dev_str == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Device 'cuda' requested but CUDA is not available (no GPU driver?); using CPU.",
            UserWarning,
            stacklevel=2,
        )
        dev_str = "cpu"
    dev = torch.device(dev_str)

    pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pack["model_state_dict"]

    cache_pt = Path(cache_pt).resolve()
    z = torch.load(cache_pt, map_location="cpu", weights_only=False)
    node_to_local = z["node_to_local"]
    x = z["x"].float()
    sids = z["sample_ids"]
    if isinstance(sids, torch.Tensor):
        sids_list = [int(t) for t in sids.tolist()]
    else:
        sids_list = [int(t) for t in list(sids)]

    if sample_id is not None and int(sample_id) >= 0:
        try:
            si = sids_list.index(int(sample_id))
        except ValueError as ex:
            raise ValueError(f"sample_id={sample_id} not in cache (len={len(sids_list)})") from ex
    else:
        si = int(sample_idx)
    if si < 0 or si >= x.shape[0]:
        raise IndexError(f"sample index {si} out of range [0, {x.shape[0]})")

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=True).float()
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=True).float()
    if x_mean.shape != (1, x.shape[2]) or x_std.shape != (1, x.shape[2]):
        raise ValueError(f"x_mean/x_std shape {x_mean.shape} vs x feature dim {x.shape[2]}")

    node_order = _node_order_from_ntl(node_to_local)
    edge_index, edge_attr = _load_compacted_edges(Path(edges_csv).resolve(), node_to_local)
    num_edges_train = int(edge_index.shape[1])
    model = _build_model(pack, sd, num_edges=num_edges_train, dropout=float(dropout))
    model.load_state_dict(sd, strict=True)
    model.to(dev)
    model.eval()

    x_row = x[si : si + 1]
    x_n = ((x_row - x_mean) / x_std).to(dtype=torch.float32)

    data = Data(
        x=x_n.view(-1, x_n.size(-1)),
        edge_index=edge_index.to(dev),
        edge_attr=edge_attr.to(dev),
    )
    data.num_graphs = 1

    with torch.no_grad():
        _attn = model.forward_node_to_token_attention(data)
    if len(_attn) == 4:
        raise ValueError(
            "forward_node_to_token_attention returned 4 values (stale DA-GPS code in this process). "
            "Restart the Jupyter kernel, or run before extract:\n"
            "  import importlib, train_da_gps_multitask_complex_voltage_gine as m; importlib.reload(m)"
        )
    if len(_attn) != 6:
        raise ValueError(f"expected 6 return values from forward_node_to_token_attention, got {len(_attn)}")
    layer_probs_nt, layer_probs_tn, volt, cap_logits, reg_pred, pv_pred = _attn

    cap_cols = list(pack["cap_target_cols"])
    reg_cols = list(pack["reg_target_cols"])
    n_sys = int(pack["n_system_tokens"])
    tokens = _token_names(cap_cols, reg_cols, n_sys)

    probs_stacked_nt = torch.stack(layer_probs_nt, dim=0).float().cpu()
    probs_stacked_tn = torch.stack(layer_probs_tn, dim=0).float().cpu()
    payload: dict = {
        "layer_probs_nt": probs_stacked_nt,
        "layer_probs_tn": probs_stacked_tn,
        "layer_probs": probs_stacked_nt,
        "sample_id": int(sids_list[si]),
        "sample_idx_in_cache": int(si),
        "cache_pt": str(cache_pt),
        "edges_csv": str(Path(edges_csv).resolve()),
        "ckpt": str(ckpt_path),
        "volt": volt.cpu(),
        "cap_logits": cap_logits.cpu(),
        "reg_pred": reg_pred.cpu(),
    }
    mean_heads_nt = probs_stacked_nt.mean(dim=2)
    mean_heads_tn = probs_stacked_tn.mean(dim=2)
    if mean_heads_nt.dim() == 4 and int(mean_heads_nt.size(1)) == 1:
        mean_heads_nt = mean_heads_nt[:, 0, :, :]
    if mean_heads_tn.dim() == 4 and int(mean_heads_tn.size(1)) == 1:
        mean_heads_tn = mean_heads_tn[:, 0, :, :]
    if head_mean:
        payload["layer_probs_head_mean_nt"] = mean_heads_nt
        payload["layer_probs_head_mean_tn"] = mean_heads_tn
        payload["layer_probs_head_mean"] = mean_heads_nt

    if save_outputs:
        torch.save(payload, out_dir / "attention.pt")

    manifest = {
        "tensor_shapes": {
            "layer_probs_nt": list(probs_stacked_nt.shape),
            "layer_probs_nt_legend": "[layers, batch, heads, nodes, tokens] node→token",
            "layer_probs_tn": list(probs_stacked_tn.shape),
            "layer_probs_tn_legend": "[layers, batch, heads, tokens, nodes] token→node",
            "layer_probs_head_mean_nt": list(mean_heads_nt.shape),
            "layer_probs_head_mean_tn": list(mean_heads_tn.shape),
        },
        "n_layers": int(probs_stacked_nt.shape[0]),
        "n_heads": int(probs_stacked_nt.shape[2]),
        "n_nodes": int(probs_stacked_nt.shape[3]),
        "n_tokens": int(probs_stacked_nt.shape[4]),
        "token_names": tokens,
        "reg_target_cols": reg_cols,
        "cap_target_cols": cap_cols,
        "node_names": node_order,
        "sample_id": int(sids_list[si]),
    }
    if save_outputs:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mean_heads = mean_heads_nt.numpy().astype(np.float32)
    mean_heads_tn_np = mean_heads_tn.numpy().astype(np.float32)
    if mean_heads.ndim != 3 or mean_heads_tn_np.ndim != 3:
        raise RuntimeError(
            f"internal bug: expected (L,N,T) and (L,T,N); got nt={mean_heads.shape} tn={mean_heads_tn_np.shape}"
        )
    if save_outputs:
        np.savez_compressed(
            out_dir / "attention_mean_heads.npz",
            probs_nt=mean_heads,
            probs_tn=mean_heads_tn_np,
            probs=mean_heads,
            sample_id=np.int64(sids_list[si]),
        )

    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "mean_heads": mean_heads,
        "mean_heads_tn": mean_heads_tn_np,
        "reg_target_cols": reg_cols,
        "cap_target_cols": cap_cols,
        "n_cap": len(cap_cols),
        "sample_id": int(sids_list[si]),
        "sample_idx_in_cache": int(si),
    }


@torch.no_grad()
def eval_aux_per_device_on_cache_indices(
    ckpt_path: Path | str,
    run_dir: Path | str,
    cache_pt: Path | str,
    edges_csv: Path | str,
    sample_indices: list[int],
    *,
    device: str | None = None,
    dropout: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run ``model.forward`` on selected cache rows; per-regulator MAE/MSE in **tap pu**; per-cap BCE + accuracy.

    Targets come from the DA cache tensor ``y_reg`` (raw pu) and ``y_cap`` (0/1), same as training.
    Predictions use ``reg_mean.pt`` / ``reg_std.pt`` from ``run_dir`` to denormalize regulator outputs.
    """
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = Path(run_dir).resolve()
    cache_pt = Path(cache_pt).resolve()
    edges_csv = Path(edges_csv).resolve()

    dev_str = (device or ("cuda" if torch.cuda.is_available() else "cpu")).strip().lower()
    if dev_str == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; using CPU.", UserWarning, stacklevel=2)
        dev_str = "cpu"
    dev = torch.device(dev_str)

    pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pack["model_state_dict"]
    reg_cols = list(pack["reg_target_cols"])
    cap_cols = list(pack["cap_target_cols"])

    z = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if "y_reg" not in z or "y_cap" not in z:
        raise KeyError(f"{cache_pt} must contain 'y_reg' and 'y_cap' (DA multitask cache).")
    x = z["x"].float()
    y_cap_all = z["y_cap"].float()
    y_reg_all = z["y_reg"].float()
    node_to_local = z["node_to_local"]

    for si in sample_indices:
        if int(si) < 0 or int(si) >= int(x.shape[0]):
            raise IndexError(f"sample_idx {si} out of range for cache batch [0, {int(x.shape[0])})")

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=True).float()
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=True).float()
    reg_mean = torch.load(run_dir / "reg_mean.pt", map_location="cpu", weights_only=True).float()
    reg_std = torch.load(run_dir / "reg_std.pt", map_location="cpu", weights_only=True).float()

    edge_index, edge_attr = _load_compacted_edges(edges_csv, node_to_local)
    num_edges_train = int(edge_index.shape[1])
    model = _build_model(pack, sd, num_edges=num_edges_train, dropout=float(dropout))
    model.load_state_dict(sd, strict=True)
    model.to(dev)
    model.eval()

    reg_preds: list[torch.Tensor] = []
    cap_logits_l: list[torch.Tensor] = []
    for si in sample_indices:
        si = int(si)
        x_row = x[si : si + 1]
        x_n = ((x_row - x_mean) / x_std).to(dtype=torch.float32)
        data = Data(
            x=x_n.view(-1, x_n.size(-1)),
            edge_index=edge_index.to(dev),
            edge_attr=edge_attr.to(dev),
        )
        data.num_graphs = 1
        _v, c_log, r_p, _pv = model(data)
        reg_preds.append(r_p.float().cpu())
        cap_logits_l.append(c_log.float().cpu())

    reg_pred_n = torch.cat(reg_preds, dim=0)
    cap_log = torch.cat(cap_logits_l, dim=0)
    idx_t = torch.tensor(sample_indices, dtype=torch.long)
    y_reg = y_reg_all.index_select(0, idx_t)
    y_cap = y_cap_all.index_select(0, idx_t)

    rp_denorm = reg_pred_n * reg_std + reg_mean
    reg_rows: list[dict] = []
    for j, name in enumerate(reg_cols):
        pred_j = rp_denorm[:, j]
        tgt_j = y_reg[:, j]
        err = pred_j - tgt_j
        reg_rows.append(
            {
                "reg_col": name,
                "mae_tap_pu": float(err.abs().mean().item()),
                "mse_tap_pu": float((err * err).mean().item()),
                "rmse_tap_pu": float(torch.sqrt((err * err).mean()).item()),
            }
        )

    cap_rows: list[dict] = []
    probs = torch.sigmoid(cap_log)
    for j, name in enumerate(cap_cols):
        logits_j = cap_log[:, j]
        tgt_j = y_cap[:, j]
        bce_j = F.binary_cross_entropy_with_logits(logits_j, tgt_j).item()
        pred_on = (probs[:, j] >= 0.5).float()
        acc_j = float((pred_on == tgt_j).float().mean().item())
        cap_rows.append({"cap_col": name, "bce": float(bce_j), "accuracy_thresh0p5": acc_j})

    meta = {
        "n_evaluated": len(sample_indices),
        "sample_indices": [int(s) for s in sample_indices],
        "cache_pt": str(cache_pt),
        "reg_mse_tap_pu_all": float(F.mse_loss(rp_denorm, y_reg.to(rp_denorm.dtype)).item()),
        "cap_bce_all": float(F.binary_cross_entropy_with_logits(cap_log, y_cap.to(cap_log.dtype)).item()),
    }
    return pd.DataFrame(reg_rows), pd.DataFrame(cap_rows), meta


@torch.no_grad()
def eval_voltage_per_node_errors_on_cache_indices(
    ckpt_path: Path | str,
    run_dir: Path | str,
    cache_pt: Path | str,
    edges_csv: Path | str,
    sample_indices: list[int],
    *,
    device: str | None = None,
    dropout: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Per-bus voltage error **ranking** over selected cache rows (same denorm as training).

    For each graph index ``si``, runs ``model.forward``, denormalizes ``volt`` with ``y_mean.pt`` /
    ``y_std.pt``, compares to cache ``y_ri[si]`` (rectangular **I/Q** in physical units).

    For each node :math:`n`, aggregates over samples:

    - ``mean_mae_vmag_pu``: mean of :math:`| |V|_{pred} - |V|_{true} |`
    - ``mean_mae_angle_deg``: mean of smallest-magnitude angle difference in degrees
    - ``rmse_vmag_pu``: root mean square of :math:`|V|_{pred} - |V|_{true}|` over samples
    - ``std_vmag_true_pu``: population std of true :math:`|V|` across the same cache rows (pu);
      near-zero where :math:`|V|` is almost constant.
    - ``r2_vmag``: R² of :math:`|V|` **across the evaluated cache rows** (one value per bus); finite only
      where the sample variance of true :math:`|V|` exceeds ``1e-10`` pu² (otherwise ``NaN``).

    - ``rank_vmag``: 1 = largest mean |V| MAE (table sorted by this, worst MAE first).
    - ``rank_r2_worst``: 1 = lowest finite ``r2_vmag`` (worst R²); NaN ``r2_vmag`` ranked last.

    Returned rows are ordered by ``mean_mae_vmag_pu`` (worst MAE first). For a view sorted by
    worst R² first, sort by ``r2_vmag`` ascending (``na_position='last'``); the attention
    notebook prints that table (no separate CSV).

    Note: loads the model and runs one forward per sample (same cost order as aux eval);
    call once and cache if you also run ``eval_aux_per_device_on_cache_indices`` separately.
    """
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = Path(run_dir).resolve()
    cache_pt = Path(cache_pt).resolve()
    edges_csv = Path(edges_csv).resolve()

    dev_str = (device or ("cuda" if torch.cuda.is_available() else "cpu")).strip().lower()
    if dev_str == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; using CPU.", UserWarning, stacklevel=2)
        dev_str = "cpu"
    dev = torch.device(dev_str)

    pack = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pack["model_state_dict"]

    z = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if "y_ri" not in z:
        raise KeyError(f"{cache_pt} must contain 'y_ri' (per-node complex voltage targets).")
    x = z["x"].float()
    y_ri_all = z["y_ri"].float()
    node_to_local = z["node_to_local"]

    for si in sample_indices:
        if int(si) < 0 or int(si) >= int(x.shape[0]):
            raise IndexError(f"sample_idx {si} out of range for cache batch [0, {int(x.shape[0])})")

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=True).float()
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=True).float()
    y_mean = torch.load(run_dir / "y_mean.pt", map_location="cpu", weights_only=True).float()
    y_std = torch.load(run_dir / "y_std.pt", map_location="cpu", weights_only=True).float()

    node_names = _node_order_from_ntl(node_to_local)
    n_nodes = len(node_names)
    if int(y_ri_all.shape[1]) != n_nodes or int(y_ri_all.shape[2]) != 2:
        raise ValueError(f"y_ri shape {tuple(y_ri_all.shape)} vs n_nodes={n_nodes}")

    edge_index, edge_attr = _load_compacted_edges(edges_csv, node_to_local)
    num_edges_train = int(edge_index.shape[1])
    model = _build_model(pack, sd, num_edges=num_edges_train, dropout=float(dropout))
    model.load_state_dict(sd, strict=True)
    model.to(dev)
    model.eval()

    y_mean_d = y_mean.to(dev)
    y_std_d = y_std.to(dev)

    sum_mae_v = torch.zeros(n_nodes, dtype=torch.float64)
    sum_mae_a = torch.zeros(n_nodes, dtype=torch.float64)
    sum_sq_v = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_m = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_m2 = torch.zeros(n_nodes, dtype=torch.float64)
    n_s = 0
    for si in sample_indices:
        si = int(si)
        x_row = x[si : si + 1]
        x_n = ((x_row - x_mean) / x_std).to(dtype=torch.float32)
        data = Data(
            x=x_n.view(-1, x_n.size(-1)),
            edge_index=edge_index.to(dev),
            edge_attr=edge_attr.to(dev),
        )
        data.num_graphs = 1
        volt_n, _, _, _ = model(data)
        v_flat = volt_n.view(1, -1)
        pred_flat = v_flat * y_std_d + y_mean_d
        pred = pred_flat.view(n_nodes, 2)
        true = y_ri_all[si].to(dev)
        pre, pim = pred[:, 0], pred[:, 1]
        tre, tim = true[:, 0], true[:, 1]
        pred_mag = torch.sqrt(pre * pre + pim * pim + 1e-12)
        true_mag = torch.sqrt(tre * tre + tim * tim + 1e-12)
        pred_ang = torch.atan2(pim, pre)
        true_ang = torch.atan2(tim, tre)
        d_ang = pred_ang - true_ang
        d_ang = (d_ang + math.pi) % (2.0 * math.pi) - math.pi
        ang_err_deg = torch.rad2deg(d_ang).abs()
        vmag_err = pred_mag - true_mag
        tm = true_mag.double().cpu()
        sum_true_m += tm
        sum_true_m2 += tm * tm
        sum_mae_v += vmag_err.abs().double().cpu()
        sum_mae_a += ang_err_deg.double().cpu()
        sum_sq_v += (vmag_err * vmag_err).double().cpu()
        n_s += 1

    mean_mae_v = (sum_mae_v / float(n_s)).numpy()
    mean_mae_a = (sum_mae_a / float(n_s)).numpy()
    rmse_v = torch.sqrt(sum_sq_v / float(n_s)).numpy()
    mean_t = sum_true_m / float(n_s)
    var_t = sum_true_m2 / float(n_s) - mean_t * mean_t
    mse_v = (sum_sq_v / float(n_s)).numpy()
    var_np = var_t.numpy()
    std_vmag = np.sqrt(np.maximum(var_np, 0.0))
    r2_v = np.full(n_nodes, np.nan, dtype=np.float64)
    mask = var_np > 1e-10
    r2_v[mask] = 1.0 - mse_v[mask] / var_np[mask]

    rows = [
        {
            "node": node_names[i],
            "mean_mae_vmag_pu": float(mean_mae_v[i]),
            "mean_mae_angle_deg": float(mean_mae_a[i]),
            "rmse_vmag_pu": float(rmse_v[i]),
            "std_vmag_true_pu": float(std_vmag[i]),
            "r2_vmag": float(r2_v[i]) if np.isfinite(r2_v[i]) else float("nan"),
        }
        for i in range(n_nodes)
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_mae_vmag_pu", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_vmag", np.arange(1, len(df) + 1, dtype=np.int32))
    # rank 1 = lowest finite R² (worst fit); NaN R² (near-constant |V| across samples) ranked last
    df["rank_r2_worst"] = (
        df["r2_vmag"].rank(method="min", ascending=True, na_option="bottom").astype(np.int64)
    )
    r2_fin = r2_v[np.isfinite(r2_v)]
    _finite_r2 = df[np.isfinite(df["r2_vmag"])]
    if len(_finite_r2):
        _iwr = int(_finite_r2["r2_vmag"].idxmin())
        worst_r2_node = str(df.loc[_iwr, "node"])
        worst_r2_vmag = float(df.loc[_iwr, "r2_vmag"])
    else:
        worst_r2_node = ""
        worst_r2_vmag = float("nan")
    meta = {
        "n_nodes": int(n_nodes),
        "n_samples": int(n_s),
        "sample_indices": [int(s) for s in sample_indices],
        "cache_pt": str(cache_pt),
        "worst_node": str(df.iloc[0]["node"]) if len(df) else "",
        "worst_mean_mae_vmag_pu": float(df.iloc[0]["mean_mae_vmag_pu"]) if len(df) else float("nan"),
        "r2_vmag_mean": float(np.mean(r2_fin)) if len(r2_fin) else float("nan"),
        "r2_vmag_median": float(np.median(r2_fin)) if len(r2_fin) else float("nan"),
        "r2_vmag_min": float(np.min(r2_fin)) if len(r2_fin) else float("nan"),
        "r2_vmag_max": float(np.max(r2_fin)) if len(r2_fin) else float("nan"),
        "n_nodes_r2_vmag_finite": int(len(r2_fin)),
        "worst_r2_node": worst_r2_node,
        "worst_r2_vmag": worst_r2_vmag,
        "node_names": list(node_names),
        "reg_target_cols": list(pack["reg_target_cols"]),
    }
    return df, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DA-GPS node→token attention weights.")
    p.add_argument("--ckpt", type=str, required=True, help="da_gps_multitask_best.pt")
    p.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Directory with x_mean.pt, x_std.pt (default: parent of --ckpt).",
    )
    p.add_argument("--cache_pt", type=str, required=True, help="Chunk DA cache .pt (x, sample_ids, node_to_local, …).")
    p.add_argument("--edges_csv", type=str, required=True, help="gnn_edges_phase_static.csv for same chunk topology.")
    p.add_argument("--sample_idx", type=int, default=0, help="Index into cache tensor batch dimension.")
    p.add_argument(
        "--sample_id",
        type=int,
        default=-1,
        help="If >= 0, pick row with this sample_id instead of --sample_idx.",
    )
    p.add_argument("--dropout", type=float, default=0.0, help="Must match train if you rely on norm scales (use 0 for eval).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="", help="Output directory (default: run_dir/attention_extract).")
    p.add_argument(
        "--head_mean",
        action="store_true",
        help="Also store probs averaged over heads: (n_layers, n_nodes, n_tokens).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else ckpt_path.parent
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (run_dir / "attention_extract")
    sid = int(args.sample_id) if int(args.sample_id) >= 0 else None
    try:
        run_attention_extract(
            ckpt_path,
            run_dir,
            args.cache_pt,
            args.edges_csv,
            sample_idx=int(args.sample_idx),
            sample_id=sid,
            out_dir=out_dir,
            device=str(args.device),
            dropout=float(args.dropout),
            head_mean=bool(args.head_mean),
        )
    except (ValueError, IndexError) as ex:
        raise SystemExit(str(ex)) from ex
    print(f"Wrote {out_dir / 'attention.pt'}", flush=True)
    print(f"Wrote {out_dir / 'manifest.json'}", flush=True)
    print(f"Wrote {out_dir / 'attention_mean_heads.npz'} (probs_nt, probs_tn)", flush=True)


if __name__ == "__main__":
    main()
