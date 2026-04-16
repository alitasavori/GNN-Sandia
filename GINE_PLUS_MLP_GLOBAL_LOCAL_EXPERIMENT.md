# GINE + MLP With Local And Global Paths

This experiment keeps the successful `GINE + MLP` setup but decomposes the final voltage prediction into:

- a **local path** that predicts per-node complex voltage directly from each node's GNN embedding
- a **global correction path** that uses the same `RealMLP` head as the MLP-only baseline on the flattened 2D node states

Final prediction:

`V_pred = V_local + DeltaV_global`

## Why try this?

- The local branch can model node-wise effects that are easy to read off from local graph context.
- The global branch can model long-range, feeder-wide coupling and coordinated corrections across many nodes.
- This can be easier to optimize than forcing one global MLP to explain both local and system-level structure by itself.

## When is this kind of approach common?

This is common in problems where targets have both local and global structure, for example:

- power-system state prediction and correction
- image restoration or super-resolution (`coarse/global + residual/local`)
- weather, CFD, and PDE surrogate models
- mesh/graph simulators with local dynamics plus long-range constraints
- sequence models with token-local paths plus global context correction

## Colab execution cell

```python
import os, sys, subprocess

REPO = "/content/GNN-Sandia"
os.chdir(REPO)
os.environ["PYTHONUNBUFFERED"] = "1"

cmd = [
    sys.executable, "-u", "train_gine_plus_mlp_global_local_complex_voltage.py",
    "--data_root", "/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg",
    "--nodes_csv", "Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    "--edge_catalog_csv", "Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
    "--out_dir", "/content/gine_plus_mlp_global_local_complex_8500",
    "--epochs", "200",
    "--batch_size", "16",
    "--hidden_gnn", "64",
    "--layers", "3",
    "--state_dim", "2",
    "--hidden_mlp", "1024",
    "--node_emb_dim", "16",
    "--edge_emb_dim", "8",
    "--lr", "5e-4",
    "--weight_decay", "1e-5",
    "--patience", "30",
    "--seed", "42",
    "--train_frac", "0.8",
    "--val_frac", "0.1",
    "--sample_frac", "1.0",
    "--cache_tensor", "/content/preloaded_gine_plus_mlp_global_local.pt",
    "--num_workers", "2",
    "--disable_dropout",
]

print(" ".join(cmd))
with subprocess.Popen(
    cmd,
    cwd=REPO,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
) as p:
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()

if rc != 0:
    raise subprocess.CalledProcessError(rc, cmd)
```

## Outputs

Written to `--out_dir`:

- `gine_plus_mlp_global_local_best.pt`
- `x_mean.pt`, `x_std.pt`
- `y_mean.pt`, `y_std.pt`
- `gine_plus_mlp_global_local_report.json`

## Suggested comparison

Compare:

- `gine_plus_mlp_report.json`
- `gine_plus_mlp_global_local_report.json`

using:

- `test_mae_vmag`
- `test_rmse_vmag`
- `test_mae_angle_deg`
- `test_rmse_angle_deg`

The main question is whether adding the local branch improves the already-strong `GINE + MLP` baseline.
