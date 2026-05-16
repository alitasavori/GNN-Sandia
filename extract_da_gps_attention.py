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
import importlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def _dagps_model_cls():
    """Fresh **GINE** ``DAGPSModel`` (``train_da_gps_multitask_complex_voltage_gine``)."""
    import importlib

    import train_da_gps_multitask_complex_voltage_gine as tdg

    importlib.reload(tdg)
    return tdg.DAGPSModel


def _dagps_model_cls_legacy():
    """Fresh **legacy EdgeAttn** ``DAGPSModel`` (``train_da_gps_multitask_complex_voltage``)."""
    import importlib

    import train_da_gps_multitask_complex_voltage as tdg

    importlib.reload(tdg)
    return tdg.DAGPSModel


def _state_dict_is_gine_da_gps(sd: dict) -> bool:
    """``True`` if weights use PyG ``GINEConv`` (``…mpnn.conv…``); else EdgeAttn MPNN (``…mpnn.msg…``)."""
    return any(str(k).startswith("blocks.0.mpnn.conv.") for k in sd)


def _train_da_gps_targets_module(sd: dict):
    """Training script module for ``TARGET_*_COLS`` (GINE vs legacy); reload for notebooks."""
    if _state_dict_is_gine_da_gps(sd):
        mod = importlib.import_module("train_da_gps_multitask_complex_voltage_gine")
    else:
        mod = importlib.import_module("train_da_gps_multitask_complex_voltage")
    importlib.reload(mod)
    return mod


def _node_order_from_ntl(node_to_local: dict[str, int]) -> list[str]:
    inv: list[str | None] = [None] * len(node_to_local)
    for name, i in node_to_local.items():
        inv[int(i)] = str(name)
    if any(x is None for x in inv):
        raise ValueError("node_to_local indices are not a dense 0..N-1 range")
    return [str(x) for x in inv]


def _assert_x_zscore_shapes(
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    *,
    run_dir: Path,
    cache_pt: Path,
) -> None:
    """``x`` ends with ``[..., n_feats]``; ``x_mean`` / ``x_std`` from ``run_dir`` must be ``(1, n_feats)``."""
    nfe = int(x.shape[-1])
    if x_mean.dim() != 2 or x_mean.shape[0] != 1 or x_std.dim() != 2 or x_std.shape[0] != 1:
        raise ValueError(f"x_mean/x_std must be shape (1, n_feats); got {tuple(x_mean.shape)}, {tuple(x_std.shape)}")
    nmean = int(x_mean.shape[-1])
    nstdw = int(x_std.shape[-1])
    if nmean != nfe or nstdw != nfe:
        raise ValueError(
            f"Cache vs RUN_DIR node-feature mismatch: cache ``x`` has {nfe} features per node, but "
            f"``{run_dir / 'x_mean.pt'}`` / ``x_std.pt`` have width {nmean} / {nstdw}. "
            f"Use a ``CACHE_PT`` tensor cache built for this training run (same BESS / ``__nobess__`` / meta-aux "
            f"column set as ``RUN_DIR``), or set ``RUN_DIR`` to the run that produced this cache. cache_pt={cache_pt}"
        )


def _token_names(cap_cols: list[str], reg_cols: list[str], n_system: int) -> list[str]:
    return list(cap_cols) + list(reg_cols) + [f"sys_{i}" for i in range(int(n_system))]


# Order for expanding static edge CSVs beyond (R_full, X_full) when checkpoints use more raw edge dims.
_EDGE_ATTR_COL_PRIORITY: tuple[str, ...] = (
    "R_full",
    "X_full",
    "length",
    "phase",
    "C_full",
    "nph_line",
    "R_per_len",
    "X_per_len",
    "C_per_len",
)


def _pick_edge_attr_columns(edge_csv: Path, raw_edge_dim: int) -> tuple[str, ...]:
    """Choose ``raw_edge_dim`` numeric columns present in ``gnn_edges_phase_static.csv``-style files."""
    p = Path(edge_csv).resolve()
    if int(raw_edge_dim) < 1:
        raise ValueError(f"raw_edge_dim must be >= 1, got {raw_edge_dim}")
    head = pd.read_csv(p, nrows=1)
    cols = set(head.columns)
    picked: list[str] = []
    for c in _EDGE_ATTR_COL_PRIORITY:
        if c in cols and c not in picked:
            picked.append(c)
        if len(picked) >= int(raw_edge_dim):
            return tuple(picked[: int(raw_edge_dim)])
    raise ValueError(
        f"Cannot build edge_attr with raw_edge_dim={raw_edge_dim} from {p}: "
        f"found priority columns {picked!r} among {sorted(cols)}."
    )


def _load_named_bidir_edges(
    edge_csv: Path,
    node_to_local: dict[str, int],
    attr_cols: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional edges (same convention as ``_load_compacted_edges``) with explicit feature columns."""
    p = Path(edge_csv).resolve()
    df = pd.read_csv(p)
    for c in ("from_node", "to_node", *attr_cols):
        if c not in df.columns:
            raise ValueError(f"{p} missing column {c!r} (need from_node, to_node, and {attr_cols!r})")
    src: list[int] = []
    dst: list[int] = []
    feat_rows: list[list[float]] = []
    for _, r in df.iterrows():
        u = str(r["from_node"]).strip()
        v = str(r["to_node"]).strip()
        if u not in node_to_local or v not in node_to_local:
            continue
        iu, iv = node_to_local[u], node_to_local[v]
        feats = [float(r[c]) for c in attr_cols]
        src.extend([iu, iv])
        dst.extend([iv, iu])
        feat_rows.extend([feats, feats])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.asarray(feat_rows, dtype=np.float32))
    return edge_index, edge_attr


def _load_eval_edges(
    edge_csv: Path,
    node_to_local: dict[str, int],
    *,
    raw_edge_dim: int,
    cache_z: dict | None,
    sd: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Topology for eval: prefer cache ``edge_index``/``edge_attr`` if they match checkpoint raw width."""
    n_loc = len(node_to_local)
    eemb_w = sd.get("edge_emb.weight")
    n_emb_edges = int(eemb_w.shape[0]) if isinstance(eemb_w, torch.Tensor) else None

    if cache_z is not None:
        ei = cache_z.get("edge_index")
        ea = cache_z.get("edge_attr")
        if isinstance(ei, torch.Tensor) and isinstance(ea, torch.Tensor) and ea.dim() == 2:
            if int(ea.size(1)) == int(raw_edge_dim) and ei.numel() > 0:
                mx = int(ei.max().item())
                if 0 <= mx < n_loc:
                    ne = int(ei.shape[1])
                    if n_emb_edges is None or ne == n_emb_edges:
                        return ei.long().cpu(), ea.float().cpu()
                    warnings.warn(
                        f"Ignoring cache edge_index: num_edges={ne} != checkpoint edge_emb rows={n_emb_edges}; "
                        f"reloading from {edge_csv}.",
                        UserWarning,
                        stacklevel=2,
                    )

    attr_cols = _pick_edge_attr_columns(edge_csv, int(raw_edge_dim))
    ei, ea = _load_named_bidir_edges(edge_csv, node_to_local, attr_cols)
    if n_emb_edges is not None and int(ei.shape[1]) != n_emb_edges:
        raise RuntimeError(
            f"Edge count mismatch: built {int(ei.shape[1])} directed edges from {edge_csv} but "
            f"checkpoint edge_emb has {n_emb_edges} rows. Use the same edge CSV / node map as training."
        )
    return ei, ea


def _infer_node_in_edge_dim(
    state_dict: dict, *, hidden: int, node_emb_dim: int, edge_emb_dim: int = 0
) -> tuple[int, int]:
    """Recover ``node_in_dim`` and raw ``edge_dim`` (excluding learned edge-id embedding)."""
    w_node = state_dict["node_in.0.weight"]
    node_in_dim = int(w_node.shape[1]) - int(node_emb_dim)
    eemb = int(edge_emb_dim)

    w_gine = state_dict.get("blocks.0.mpnn.conv.lin.weight")
    if w_gine is not None:
        # PyG GINEConv: ``lin: edge_dim -> nn.in_channels``; DAGPS passes ``edge_dim + edge_emb_dim``.
        eff_edge = int(w_gine.shape[1])
        edge_dim = eff_edge - eemb
    else:
        w_msg = state_dict.get("blocks.0.mpnn.msg.0.weight")
        if w_msg is None:
            raise KeyError(
                "Cannot infer edge dim: expected ``blocks.0.mpnn.conv.lin.weight`` (GINE) "
                "or ``blocks.0.mpnn.msg.0.weight`` (legacy EdgeAttn MPNN)."
            )
        # Legacy EdgeAttn MPNN: message input is ``[h_src || h_dst || edge_attr_cat]`` where
        # ``edge_attr_cat`` has width ``raw_edge_dim + edge_emb_dim`` (see ``DAGPSModel``).
        eff_edge = int(w_msg.shape[1]) - 2 * int(hidden)
        edge_dim = eff_edge - eemb

    if node_in_dim < 1 or edge_dim < 1:
        raise ValueError(f"Bad inferred dims: node_in_dim={node_in_dim}, edge_dim={edge_dim}")
    return node_in_dim, edge_dim


def _read_hyperparameters(run_dir: Path) -> dict:
    p = Path(run_dir) / "da_gps_report.json"
    if not p.is_file():
        return {}
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    hp = r.get("hyperparameters")
    return hp if isinstance(hp, dict) else {}


def _max_block_index(sd: dict) -> int:
    mx = -1
    for k in sd:
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def _infer_heads_from_state(hidden: int, hp: dict) -> int:
    h = int(hp.get("heads", 0) or 0)
    if 0 < h <= hidden and hidden % h == 0:
        return h
    for cand in (8, 4, 2, 1, 16, 32, 6, 3, 12):
        if cand <= hidden and hidden % cand == 0:
            return cand
    return 1


def _pack_has_arch_lists(pack: dict) -> bool:
    rc = pack.get("reg_target_cols")
    cc = pack.get("cap_target_cols")
    return (
        isinstance(rc, (list, tuple))
        and len(rc) > 0
        and isinstance(cc, (list, tuple))
        and len(cc) > 0
    )


def _pack_complete_for_build(pack: dict) -> bool:
    if not _pack_has_arch_lists(pack):
        return False
    try:
        nn = int(pack.get("n_nodes", 0) or 0)
        hi = int(pack.get("hidden", 0) or 0)
        la = int(pack.get("layers", 0) or 0)
        ns = pack.get("n_system_tokens")
        if ns is None:
            return False
        ns_i = int(ns)
    except (TypeError, ValueError):
        return False
    return nn > 0 and hi > 0 and la > 0 and ns_i >= 0


def _merge_metadata_from_best_pt(pack: dict, ckpt_path: Path, run_dir: Path) -> None:
    """Copy all keys except ``model_state_dict`` from ``da_gps_multitask_best.pt`` when the loaded file is periodic."""
    best = (Path(run_dir) / "da_gps_multitask_best.pt").resolve()
    cur = Path(ckpt_path).resolve()
    if not best.is_file() or best == cur:
        return
    mp = torch.load(best, map_location="cpu", weights_only=False)
    for k, v in mp.items():
        if k == "model_state_dict":
            continue
        if k not in pack or pack[k] is None:
            pack[k] = v
            continue
        if k in ("reg_target_cols", "cap_target_cols", "pv_target_cols", "meta_aux_target_cols"):
            cur_v = pack.get(k)
            if not cur_v:
                pack[k] = v


def _sync_edge_emb_dim_from_state_dict(pack: dict, sd: dict) -> None:
    """Align ``pack['edge_emb_dim']`` with ``edge_emb.weight`` (saved metadata is often 0 or missing).

    Legacy / GINE DA-GPS both use ``nn.Embedding(num_edges, edge_emb_dim)`` when ``edge_emb_dim > 0``.
    If this disagrees with ``blocks.*.mpnn`` input width, eval loads too few raw edge columns (e.g. 5 vs 11).
    """
    w = sd.get("edge_emb.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[1]) > 0:
        pack["edge_emb_dim"] = int(w.shape[1])
    else:
        pack["edge_emb_dim"] = 0


def augment_da_gps_pack_for_eval(
    pack: dict,
    sd: dict,
    ckpt_path: Path | str,
    run_dir: Path | str,
    *,
    node_in_dim_hint: int | None = None,
) -> None:
    """Periodic ``training_last.pt`` checkpoints omit architecture metadata.

    If ``reg_target_cols`` / ``cap_target_cols`` (etc.) are missing, merge non-weight keys from
    ``run_dir/da_gps_multitask_best.pt`` when present; otherwise infer from ``state_dict`` and
    ``train_da_gps_multitask_complex_voltage_gine`` / ``train_da_gps_multitask_complex_voltage`` defaults
    (and optional ``da_gps_report.json``), chosen from checkpoint **weights** (GINE vs legacy EdgeAttn).

    Mutates ``pack`` in place. Does not replace ``model_state_dict`` (caller's weights stay in ``sd``).
    """
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = Path(run_dir).resolve()
    _sync_edge_emb_dim_from_state_dict(pack, sd)
    if _pack_complete_for_build(pack):
        return

    _merge_metadata_from_best_pt(pack, ckpt_path, run_dir)
    _sync_edge_emb_dim_from_state_dict(pack, sd)
    if _pack_complete_for_build(pack):
        return

    tdg = _train_da_gps_targets_module(sd)

    hp = _read_hyperparameters(run_dir)

    if not _pack_has_arch_lists(pack):
        pack["reg_target_cols"] = list(tdg.TARGET_REG_COLS)
        pack["cap_target_cols"] = list(tdg.TARGET_CAP_COLS)

    reg_cols = list(pack["reg_target_cols"])
    cap_cols = list(pack["cap_target_cols"])
    pack.setdefault("n_reg", len(reg_cols))
    pack.setdefault("n_cap", len(cap_cols))

    tlat = sd.get("token_latent")
    if tlat is None:
        raise KeyError("state_dict missing token_latent (not a DA-GPS checkpoint?)")
    g_tok, hidden = int(tlat.shape[0]), int(tlat.shape[1])
    pack.setdefault("hidden", hidden)

    n_cap = int(pack["n_cap"])
    n_reg = int(pack["n_reg"])
    n_sys = g_tok - n_cap - n_reg
    if n_sys < 0:
        raise ValueError(
            f"token_latent rows={g_tok} incompatible with n_cap={n_cap}, n_reg={n_reg} from target column lists"
        )
    pack.setdefault("n_system_tokens", n_sys)

    if sd.get("pv_W") is not None:
        pack["n_pv_aux"] = int(sd["pv_W"].shape[0])
    else:
        pack.setdefault("n_pv_aux", int(hp.get("n_pv_aux", 0) or 0))
    if int(pack.get("n_pv_aux", 0) or 0) > int(pack["n_system_tokens"]):
        warnings.warn(
            f"Inferred n_pv_aux={pack.get('n_pv_aux')} > n_system_tokens={pack.get('n_system_tokens')}; "
            "if eval fails, use da_gps_multitask_best.pt or wait for da_gps_report.json.",
            UserWarning,
            stacklevel=2,
        )

    mx = _max_block_index(sd)
    pack.setdefault("layers", mx + 1 if mx >= 0 else int(hp.get("layers", 1)))
    pack.setdefault("heads", _infer_heads_from_state(int(pack["hidden"]), hp))

    w_node = sd["node_in.0.weight"]
    w1 = int(w_node.shape[1])
    if pack.get("node_emb_dim") is None:
        if node_in_dim_hint is not None and 0 < int(node_in_dim_hint) <= w1:
            pack["node_emb_dim"] = w1 - int(node_in_dim_hint)
        else:
            pack["node_emb_dim"] = int(hp.get("node_emb_dim", 0) or 0)
    else:
        pack["node_emb_dim"] = int(pack["node_emb_dim"])

    _sync_edge_emb_dim_from_state_dict(pack, sd)

    if sd.get("volt_W") is not None:
        pack["per_node_heads"] = True
        pack["n_nodes"] = int(sd["volt_W"].shape[0])
    else:
        pack.setdefault("per_node_heads", bool(hp.get("per_node_heads", False)))
        nw = sd.get("node_emb.weight")
        if nw is not None:
            pack["n_nodes"] = int(nw.shape[0])
        else:
            pack.setdefault("n_nodes", int(hp.get("n_nodes", 0) or 0))
        if int(pack.get("n_nodes", 0) or 0) <= 0:
            raise ValueError(
                "Cannot infer n_nodes (no volt_W / node_emb in state_dict and no hyperparameters.n_nodes in report)"
            )

    pack.setdefault("per_device_cap_head", "cap_W" in sd and sd["cap_W"] is not None)
    pack.setdefault("per_device_reg_head", "reg_W" in sd and sd["reg_W"] is not None)

    n_pv = int(pack.get("n_pv_aux", 0) or 0)
    if n_pv > 0 and not pack.get("meta_aux_target_cols") and not pack.get("pv_target_cols"):
        aux_s = hp.get("aux_meta_cols")
        if isinstance(aux_s, str) and aux_s.strip():
            cols = [c.strip() for c in aux_s.split(",") if c.strip()]
            pack["meta_aux_target_cols"] = cols[:n_pv]
            pack["pv_target_cols"] = list(pack["meta_aux_target_cols"])
        else:
            pack.setdefault("meta_aux_target_cols", [])
            pack.setdefault("pv_target_cols", [])

    _infer_node_in_edge_dim(
        sd,
        hidden=int(pack["hidden"]),
        node_emb_dim=int(pack["node_emb_dim"]),
        edge_emb_dim=int(pack["edge_emb_dim"]),
    )


def _build_model(ckpt: dict, sd: dict, *, num_edges: int, dropout: float):
    gine = _state_dict_is_gine_da_gps(sd)
    DAGPSModel = _dagps_model_cls() if gine else _dagps_model_cls_legacy()
    hidden = int(ckpt["hidden"])
    node_emb_dim = int(ckpt["node_emb_dim"])
    edge_emb_dim = int(ckpt["edge_emb_dim"])
    node_in_dim, edge_dim = _infer_node_in_edge_dim(
        sd, hidden=hidden, node_emb_dim=node_emb_dim, edge_emb_dim=edge_emb_dim
    )
    kw: dict = dict(
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
    )
    if gine:
        kw["n_pv_aux"] = int(ckpt.get("n_pv_aux", 0))
    return DAGPSModel(**kw)


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
    _assert_x_zscore_shapes(x, x_mean, x_std, run_dir=run_dir, cache_pt=cache_pt)

    augment_da_gps_pack_for_eval(
        pack, sd, ckpt_path, run_dir, node_in_dim_hint=int(x.shape[-1])
    )

    node_order = _node_order_from_ntl(node_to_local)
    hidden = int(pack["hidden"])
    node_emb_dim = int(pack["node_emb_dim"])
    edge_emb_dim = int(pack["edge_emb_dim"])
    _, raw_edge_dim = _infer_node_in_edge_dim(
        sd, hidden=hidden, node_emb_dim=node_emb_dim, edge_emb_dim=edge_emb_dim
    )
    edge_index, edge_attr = _load_eval_edges(
        Path(edges_csv).resolve(),
        node_to_local,
        raw_edge_dim=int(raw_edge_dim),
        cache_z=z,
        sd=sd,
    )
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
            "Restart the Jupyter kernel, or reload the matching train script before extract:\n"
            "  import importlib, train_da_gps_multitask_complex_voltage_gine as m; importlib.reload(m)\n"
            "  # legacy EdgeAttn: import train_da_gps_multitask_complex_voltage as m; importlib.reload(m)"
        )
    if len(_attn) == 5:
        layer_probs_nt, layer_probs_tn, volt, cap_logits, reg_pred = _attn
    elif len(_attn) == 6:
        layer_probs_nt, layer_probs_tn, volt, cap_logits, reg_pred, _pv_pred = _attn
    else:
        raise ValueError(
            f"expected 5 (legacy EdgeAttn DA-GPS) or 6 (GINE + meta-aux) return values from "
            f"forward_node_to_token_attention, got {len(_attn)}"
        )

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
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Run ``model.forward`` on selected cache rows; per-regulator MAE/MSE in **tap pu**; per-cap BCE + accuracy.

    Returns ``(reg_df, cap_df, meta, meta_aux_df)``. ``meta_aux_df`` is empty unless the checkpoint
    has ``n_pv_aux > 0``, the cache includes ``y_pv``, and ``pv_mean.pt`` / ``pv_std.pt`` exist in ``run_dir``.
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
    if "y_reg" not in z or "y_cap" not in z:
        raise KeyError(f"{cache_pt} must contain 'y_reg' and 'y_cap' (DA multitask cache).")
    x = z["x"].float()
    y_cap_all = z["y_cap"].float()
    y_reg_all = z["y_reg"].float()
    y_pv_all = z.get("y_pv")
    node_to_local = z["node_to_local"]

    for si in sample_indices:
        if int(si) < 0 or int(si) >= int(x.shape[0]):
            raise IndexError(f"sample_idx {si} out of range for cache batch [0, {int(x.shape[0])})")

    x_mean = torch.load(run_dir / "x_mean.pt", map_location="cpu", weights_only=True).float()
    x_std = torch.load(run_dir / "x_std.pt", map_location="cpu", weights_only=True).float()
    _assert_x_zscore_shapes(x, x_mean, x_std, run_dir=run_dir, cache_pt=cache_pt)

    augment_da_gps_pack_for_eval(
        pack, sd, ckpt_path, run_dir, node_in_dim_hint=int(x.shape[-1])
    )
    reg_cols = list(pack["reg_target_cols"])
    cap_cols = list(pack["cap_target_cols"])

    reg_mean = torch.load(run_dir / "reg_mean.pt", map_location="cpu", weights_only=True).float()
    reg_std = torch.load(run_dir / "reg_std.pt", map_location="cpu", weights_only=True).float()

    hidden = int(pack["hidden"])
    node_emb_dim = int(pack["node_emb_dim"])
    edge_emb_dim = int(pack["edge_emb_dim"])
    _, raw_edge_dim = _infer_node_in_edge_dim(
        sd, hidden=hidden, node_emb_dim=node_emb_dim, edge_emb_dim=edge_emb_dim
    )
    edge_index, edge_attr = _load_eval_edges(
        edges_csv,
        node_to_local,
        raw_edge_dim=int(raw_edge_dim),
        cache_z=z,
        sd=sd,
    )
    num_edges_train = int(edge_index.shape[1])
    model = _build_model(pack, sd, num_edges=num_edges_train, dropout=float(dropout))
    model.load_state_dict(sd, strict=True)
    model.to(dev)
    model.eval()

    n_pv_aux = int(pack.get("n_pv_aux", 0) or 0)
    meta_aux_cols = list(pack.get("meta_aux_target_cols") or pack.get("pv_target_cols") or [])

    reg_preds: list[torch.Tensor] = []
    cap_logits_l: list[torch.Tensor] = []
    pv_preds: list[torch.Tensor] = []
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
        _fwd = model(data)
        if len(_fwd) == 3:
            _v, c_log, r_p = _fwd
        elif len(_fwd) == 4:
            _v, c_log, r_p, _pv_p = _fwd
        else:
            raise ValueError(f"model.forward returned {len(_fwd)} values (expected 3 or 4)")
        reg_preds.append(r_p.float().cpu())
        cap_logits_l.append(c_log.float().cpu())
        if n_pv_aux > 0:
            if len(_fwd) != 4:
                raise RuntimeError("n_pv_aux>0 but legacy model.forward has no pv head (expected GINE ckpt)")
            pv_preds.append(_fwd[3].float().cpu())

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
        abs_e = err.abs()
        reg_rows.append(
            {
                "reg_col": name,
                "mae_tap_pu": float(abs_e.mean().item()),
                "mse_tap_pu": float((err * err).mean().item()),
                "rmse_tap_pu": float(torch.sqrt((err * err).mean()).item()),
                "frac_abs_err_le_0p01_tap_pu": float((abs_e <= 0.01).float().mean().item()),
                "frac_abs_err_le_0p02_tap_pu": float((abs_e <= 0.02).float().mean().item()),
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
        cap_rows.append({"cap_col": name, "bce": float(bce_j), "accuracy": acc_j})

    meta = {
        "n_evaluated": len(sample_indices),
        "sample_indices": [int(s) for s in sample_indices],
        "cache_pt": str(cache_pt),
        "reg_mse_tap_pu_all": float(F.mse_loss(rp_denorm, y_reg.to(rp_denorm.dtype)).item()),
        "cap_bce_all": float(F.binary_cross_entropy_with_logits(cap_log, y_cap.to(cap_log.dtype)).item()),
    }

    meta_aux_rows: list[dict] = []
    if (
        n_pv_aux > 0
        and len(pv_preds) == len(sample_indices)
        and y_pv_all is not None
        and len(meta_aux_cols) == n_pv_aux
    ):
        pm_path = run_dir / "pv_mean.pt"
        ps_path = run_dir / "pv_std.pt"
        if pm_path.is_file() and ps_path.is_file():
            pv_mean = torch.load(pm_path, map_location="cpu", weights_only=True).float()
            pv_std = torch.load(ps_path, map_location="cpu", weights_only=True).float()
            pv_pred_n = torch.cat(pv_preds, dim=0)
            y_pv = y_pv_all.index_select(0, idx_t).to(dtype=torch.float32)
            if pv_pred_n.shape == y_pv.shape:
                pv_pred_raw = pv_pred_n * pv_std + pv_mean
                y_pv_n = (y_pv - pv_mean) / pv_std
                meta["pv_mse_nrm_all"] = float(F.mse_loss(pv_pred_n, y_pv_n.to(pv_pred_n.dtype)).item())
                meta["pv_mse_raw_all"] = float(F.mse_loss(pv_pred_raw, y_pv.to(pv_pred_raw.dtype)).item())
                for j, name in enumerate(meta_aux_cols):
                    pred_nj = pv_pred_n[:, j]
                    tgt_nj = y_pv_n[:, j]
                    pred_rj = pv_pred_raw[:, j]
                    tgt_rj = y_pv[:, j]
                    err_n = pred_nj - tgt_nj
                    err_r = pred_rj - tgt_rj
                    abs_r = err_r.abs()
                    yv = tgt_rj.double()
                    vy = yv - yv.mean()
                    var_y = float((vy * vy).mean().item())
                    r2_raw = float("nan")
                    if var_y > 1e-12:
                        mse_r = float((err_r * err_r).mean().item())
                        r2_raw = 1.0 - mse_r / var_y
                    meta_aux_rows.append(
                        {
                            "meta_col": str(name),
                            "mse_nrm": float((err_n * err_n).mean().item()),
                            "rmse_nrm": float(torch.sqrt((err_n * err_n).mean()).item()),
                            "mae_nrm": float(err_n.abs().mean().item()),
                            "mse_raw": float((err_r * err_r).mean().item()),
                            "rmse_raw": float(torch.sqrt((err_r * err_r).mean()).item()),
                            "mae_raw": float(abs_r.mean().item()),
                            "r2_raw_across_samples": float(r2_raw),
                            "frac_rel_err_le_5pct": float(
                                (abs_r / (tgt_rj.abs().clamp(min=1e-6))).le(0.05).float().mean().item()
                            ),
                        }
                    )
            else:
                warnings.warn(
                    f"meta aux shape mismatch: pv_pred {tuple(pv_pred_n.shape)} vs y_pv {tuple(y_pv.shape)}; "
                    "skipping meta_aux metrics.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"Checkpoint has n_pv_aux={n_pv_aux} but {pm_path} or {ps_path} missing; skipping meta_aux metrics.",
                UserWarning,
                stacklevel=2,
            )
    elif n_pv_aux > 0 and y_pv_all is None:
        warnings.warn(
            f"Checkpoint has n_pv_aux={n_pv_aux} but cache has no 'y_pv'; skipping meta_aux metrics.",
            UserWarning,
            stacklevel=2,
        )

    meta_aux_df = pd.DataFrame(meta_aux_rows)
    return pd.DataFrame(reg_rows), pd.DataFrame(cap_rows), meta, meta_aux_df


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
    - ``r2_vmag``: R² of :math:`|V|` **across the evaluated cache rows only** (one value per bus). Uses the
      same definition as training validation: :math:`1 - \\mathrm{MSE}/\\mathrm{Var}` where MSE is the mean
      squared error of **magnitude** :math:`|\\hat V|-|V|` and Var is the **population** variance of true
      :math:`|V|` over those same rows. ``NaN`` when :math:`\\mathrm{Var}(|V|) \\le 10^{-10}` pu² (near-constant
      true magnitude across rows). With few rows (e.g. 10), Var is often tiny for stiff buses, so R² is
      **legitimately** very negative or volatile even when MAE looks moderate — this is not a sign error in code.

    - ``rank_vmag``: 1 = largest mean |V| MAE (table sorted by this, worst MAE first).
    - ``rank_r2_worst``: 1 = lowest finite ``r2_vmag`` (worst R²); NaN ``r2_vmag`` ranked last.

    The returned ``meta`` includes pooled globals over all node×cache-row pairs (same definitions as
    ``run_da_gps_daily_opendss_compare`` overall metrics): ``mae_global_vmag_pu``, ``rmse_global_vmag_pu``,
    ``r2_global_vmag_pu``, ``mae_global_angle_deg``, ``r2_global_vang_deg_naive``. Legacy keys
    ``mean_mae_vmag_pu_global`` / ``mean_mae_angle_deg_global`` equal the |V|/angle MAE globals (mean over
    all pairs, same as training ``_metrics_voltage`` on these rows).

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
    _assert_x_zscore_shapes(x, x_mean, x_std, run_dir=run_dir, cache_pt=cache_pt)

    augment_da_gps_pack_for_eval(
        pack, sd, ckpt_path, run_dir, node_in_dim_hint=int(x.shape[-1])
    )

    y_mean = torch.load(run_dir / "y_mean.pt", map_location="cpu", weights_only=True).float()
    y_std = torch.load(run_dir / "y_std.pt", map_location="cpu", weights_only=True).float()

    node_names = _node_order_from_ntl(node_to_local)
    n_nodes = len(node_names)
    if int(y_ri_all.shape[1]) != n_nodes or int(y_ri_all.shape[2]) != 2:
        raise ValueError(f"y_ri shape {tuple(y_ri_all.shape)} vs n_nodes={n_nodes}")

    hidden = int(pack["hidden"])
    node_emb_dim = int(pack["node_emb_dim"])
    edge_emb_dim = int(pack["edge_emb_dim"])
    _, raw_edge_dim = _infer_node_in_edge_dim(
        sd, hidden=hidden, node_emb_dim=node_emb_dim, edge_emb_dim=edge_emb_dim
    )
    edge_index, edge_attr = _load_eval_edges(
        edges_csv,
        node_to_local,
        raw_edge_dim=int(raw_edge_dim),
        cache_z=z,
        sd=sd,
    )
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
    sum_sq_ang_deg = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_m = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_m2 = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_ang_deg = torch.zeros(n_nodes, dtype=torch.float64)
    sum_true_ang2_deg = torch.zeros(n_nodes, dtype=torch.float64)
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
        _out = model(data)
        volt_n = _out[0]
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
        d_ang_deg = torch.rad2deg(d_ang)
        ang_err_deg = d_ang_deg.abs()
        vmag_err = pred_mag - true_mag
        tm = true_mag.double().cpu()
        ta_deg = torch.rad2deg(true_ang).double().cpu()
        sum_true_m += tm
        sum_true_m2 += tm * tm
        sum_true_ang_deg += ta_deg
        sum_true_ang2_deg += ta_deg * ta_deg
        sum_mae_v += vmag_err.abs().double().cpu()
        sum_mae_a += ang_err_deg.double().cpu()
        sum_sq_v += (vmag_err * vmag_err).double().cpu()
        sum_sq_ang_deg += (d_ang_deg * d_ang_deg).double().cpu()
        n_s += 1

    mean_mae_v = (sum_mae_v / float(n_s)).numpy()
    mean_mae_a = (sum_mae_a / float(n_s)).numpy()
    rmse_v = torch.sqrt(sum_sq_v / float(n_s)).numpy()
    # Same as training ``_metrics_voltage`` ``mae_vmag_pu`` / ``mae_angle_deg`` over these rows:
    # mean of per-node mean abs error equals mean over all (node, sample) pairs.
    mean_mae_vmag_pu_global = float(np.mean(mean_mae_v))
    mean_mae_angle_deg_global = float(np.mean(mean_mae_a))
    n_pairs = int(n_nodes) * int(n_s)
    if n_pairs > 0:
        ss_res_v_global = float(sum_sq_v.sum().item())
        sum_t_v = float(sum_true_m.sum().item())
        sum_t2_v = float(sum_true_m2.sum().item())
        mae_global_vmag_pu = float(sum_mae_v.sum().item()) / float(n_pairs)
        mse_global_vmag_pu = ss_res_v_global / float(n_pairs)
        rmse_global_vmag_pu = float(np.sqrt(mse_global_vmag_pu))
        y_mean_v_global = sum_t_v / float(n_pairs)
        ss_tot_v_global = sum_t2_v - float(n_pairs) * y_mean_v_global * y_mean_v_global
        if ss_tot_v_global > 1e-30 and np.isfinite(ss_tot_v_global):
            r2_global_vmag_pu = float(1.0 - ss_res_v_global / ss_tot_v_global)
        else:
            r2_global_vmag_pu = float("nan")

        ss_res_ang_global = float(sum_sq_ang_deg.sum().item())
        sum_t_a = float(sum_true_ang_deg.sum().item())
        sum_t2_a = float(sum_true_ang2_deg.sum().item())
        mae_global_angle_deg = float(sum_mae_a.sum().item()) / float(n_pairs)
        y_mean_a_global = sum_t_a / float(n_pairs)
        ss_tot_ang_global = sum_t2_a - float(n_pairs) * y_mean_a_global * y_mean_a_global
        if ss_tot_ang_global > 1e-30 and np.isfinite(ss_tot_ang_global):
            r2_global_vang_deg_naive = float(1.0 - ss_res_ang_global / ss_tot_ang_global)
        else:
            r2_global_vang_deg_naive = float("nan")
        n_points_vmag_finite_overlap = n_pairs
        n_points_angle_finite_overlap = n_pairs
    else:
        mae_global_vmag_pu = rmse_global_vmag_pu = r2_global_vmag_pu = float("nan")
        mae_global_angle_deg = r2_global_vang_deg_naive = float("nan")
        n_points_vmag_finite_overlap = 0
        n_points_angle_finite_overlap = 0
    mean_t = sum_true_m / float(n_s)
    var_t = sum_true_m2 / float(n_s) - mean_t * mean_t
    mse_v = (sum_sq_v / float(n_s)).numpy()
    var_np = var_t.numpy()
    std_vmag = np.sqrt(np.maximum(var_np, 0.0))
    # Match train_da_gps_multitask_complex_voltage_gine validation: r2 = 1 - mse / var_true (population var).
    # Use a small floor only for masking "undefined" R² (near-constant |V| across rows), not the 1e-8 clamp
    # used inside training (which never emits NaN per node).
    _var_floor_r2_mask = 1e-10
    r2_v = np.full(n_nodes, np.nan, dtype=np.float64)
    mask = var_np > _var_floor_r2_mask
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
        "mean_mae_vmag_pu_global": mean_mae_vmag_pu_global,
        "mean_mae_angle_deg_global": mean_mae_angle_deg_global,
        "mae_global_vmag_pu": mae_global_vmag_pu,
        "rmse_global_vmag_pu": rmse_global_vmag_pu,
        "r2_global_vmag_pu": r2_global_vmag_pu,
        "n_points_vmag_finite_overlap": int(n_points_vmag_finite_overlap),
        "mae_global_angle_deg": mae_global_angle_deg,
        "r2_global_vang_deg_naive": r2_global_vang_deg_naive,
        "n_points_angle_finite_overlap": int(n_points_angle_finite_overlap),
        "r2_global_vmag_pu_definition": (
            "pooled over all (node, cache row): 1 - sum((|V_pred|-|V_true|)^2) / sum((|V_true|-mean)^2); "
            "same as run_da_gps_daily_opendss_compare overall |V| R²."
        ),
        "r2_global_vang_deg_naive_definition": (
            "pooled over all (node, cache row): circular wrapped pred-true angle residual in deg; "
            "R² vs linear variance of true angle in deg (naive, same as daily compare)."
        ),
        "r2_vmag_definition": (
            "per bus: 1 - mean_s((|V_pred,s|-|V_true,s|)^2) / var_s(|V_true,s|) over evaluated cache rows s; "
            f"NaN if var_s(|V_true|) <= {_var_floor_r2_mask:g} pu^2. Same structure as training val R², "
            "but training aggregates many val graphs — few rows here makes R² noisy / often negative."
        ),
        "r2_vmag_var_floor_pu2": float(_var_floor_r2_mask),
        "r2_vmag_small_n_warning": bool(int(n_s) < 30),
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
