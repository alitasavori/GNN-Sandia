# GNN+MLP Comparison (No Local/Global Split)

This experiment compares GNN message-passing choices while keeping the same flattened MLP head:

- `gine + mlp`
- `sage + mlp`
- `gcn + mlp`

## Architecture

For each model:

1. GNN encoder consumes node inputs `[p_load_kw, q_load_kvar]` (and graph edges).
2. Encoder outputs a 2D state per node.
3. Flatten all node states to `[2N]`.
4. Feed flattened states into the same `RealMLP` head (`Linear-ReLU-Linear-ReLU-Linear`).
5. Predict flattened complex voltage `[V_re, V_im]` for all nodes.

No local/global decomposition is used.

## Colab execution cell

```python
import os, sys, subprocess

REPO = "/content/GNN-Sandia"
os.chdir(REPO)
os.environ["PYTHONUNBUFFERED"] = "1"

cmd = [
    sys.executable, "-u", "train_gnn_plus_mlp_compare_complex_voltage.py",
    "--data_root", "/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg",
    "--nodes_csv", "Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    "--edge_catalog_csv", "Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
    "--out_dir", "/content/gnn_plus_mlp_compare_complex_8500",
    "--models", "gine,sage,gcn",
    "--epochs", "200",
    "--batch_size", "16",
    "--hidden_gnn", "64",
    "--layers", "3",
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
    "--cache_tensor", "/content/preloaded_gnn_plus_mlp_compare.pt",
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

- `gine_gnn_plus_mlp_best.pt`
- `sage_gnn_plus_mlp_best.pt`
- `gcn_gnn_plus_mlp_best.pt`
- `x_mean.pt`, `x_std.pt`, `y_mean.pt`, `y_std.pt`
- `gnn_plus_mlp_compare_report.json`

The report contains all three model metrics for direct ranking and comparison.
