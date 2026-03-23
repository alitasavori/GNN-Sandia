"""
Inspect a PF-identity GNN checkpoint and report key dimensions.

Usage (PowerShell):
  python inspect_checkpoint_dims.py "C:\path\to\best.pt"

It prints:
  - N (nodes), E (edges)
  - node_in_dim (input feature dim per node)
  - edge_in_dim (edge feature dim; usually 2 for [R, X])
  - out_dim (output dim per node; usually 1 for vmag)
  - node_emb_dim / edge_emb_dim
  - h_dim, num_layers
  - dataset + target_col if present
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python inspect_checkpoint_dims.py "path/to/best.pt"')

    ckpt_path = Path(sys.argv[1]).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    def g(key: str, default=None):
        return cfg.get(key, default)

    print("Checkpoint:", str(ckpt_path))
    print("---- config ----")
    for k in [
        "dataset",
        "target_col",
        "N",
        "E",
        "node_in_dim",
        "edge_in_dim",
        "out_dim",
        "node_emb_dim",
        "edge_emb_dim",
        "h_dim",
        "num_layers",
        "use_norm",
        "use_phase_onehot",
    ]:
        if k in cfg:
            print(f"{k:>16s} = {g(k)}")

    print("---- tensors ----")
    for k in ["edge_index", "edge_attr", "edge_id"]:
        t = ckpt.get(k, None)
        if isinstance(t, torch.Tensor):
            print(f"{k:>16s}: shape={tuple(t.shape)} dtype={t.dtype}")
        else:
            print(f"{k:>16s}: (missing or not a Tensor)")

    # Helpful interpretation
    n = int(g("N", -1))
    din = int(g("node_in_dim", -1))
    dout = int(g("out_dim", -1))
    if n > 0 and din > 0 and dout > 0:
        print("---- interpretation ----")
        print(f"Per timestep, model input is x with shape ({n}, {din}).")
        print(f"Per timestep, model output is yhat with shape ({n}, {dout}).")


if __name__ == "__main__":
    main()

