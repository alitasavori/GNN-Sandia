"""
Practical inference optimizations for an existing trained .pt PF-identity GNN.

This does NOT change architecture and does NOT require retraining.

It demonstrates:
  - Loading a checkpoint
  - Moving static graph tensors to GPU once
  - Batched inference across many timesteps
  - Optional torch.compile (PyTorch 2) for GPU speedups

Run examples (PowerShell):
  python optimize_inference_existing_pt.py --ckpt "gnn2_architecture_search/original/best.pt" --device cuda --compile
  python optimize_inference_existing_pt.py --ckpt "gnn2_architecture_search/original/best.pt" --device cuda

Notes:
  - torch.compile requires PyTorch 2.x. If unavailable, the script will warn and continue.
  - ONNX export is not included here because torch_geometric MessagePassing often does not export cleanly.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from run_gnn3_overlay_7 import load_model_for_inference


def _make_batched_graphs(X_list: list[np.ndarray], static, device: torch.device) -> Batch:
    edge_index = static["edge_index"].to(device)
    edge_attr = static["edge_attr"].to(device)
    edge_id = static["edge_id"].to(device)

    data_list = [
        Data(
            x=torch.tensor(X, dtype=torch.float32, device=device),
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_id=edge_id,
            num_nodes=int(static["N"]),
        )
        for X in X_list
    ]
    return Batch.from_data_list(data_list)


@torch.no_grad()
def benchmark_forward(model, batch: Batch, device: torch.device, iters: int = 10) -> float:
    # Warmup
    for _ in range(3):
        _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    ap.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    ap.add_argument("--nsteps", type=int, default=288, help="How many timesteps to batch")
    ap.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, static = load_model_for_inference(args.ckpt, device=device)
    cfg = static["config"]
    n = int(cfg["N"])
    din = int(cfg["node_in_dim"])

    # Fake feature tensors (replace with real X per timestep in your pipeline)
    rng = np.random.default_rng(0)
    X_list = [rng.standard_normal((n, din), dtype=np.float32) for _ in range(args.nsteps)]

    batch = _make_batched_graphs(X_list, static, device=device)

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[attr-defined]
        else:
            print("torch.compile not available in this PyTorch; skipping.")

    sec = benchmark_forward(model, batch, device=device, iters=args.iters)
    print("Device:", device)
    print("Batched graphs:", args.nsteps, "| nodes:", n, "| node_in_dim:", din)
    print("Avg forward time per batched call: %.6f s" % sec)
    print("Implied per-step (divide by nsteps): %.6f ms/step" % (1000.0 * sec / max(1, args.nsteps)))


if __name__ == "__main__":
    main()

