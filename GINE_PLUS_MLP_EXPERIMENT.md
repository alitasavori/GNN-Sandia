# GINE + MLP Complex Voltage Training

This experiment adds graph inductive bias before the same MLP head used in the MLP-only baseline.

## Model

- Input node features: `p_load_kw`, `q_load_kvar`
- GNN trunk: GINE over fixed feeder graph (`R_full`, `X_full` edge attributes)
- Per-node state: exactly 2 values per node (`state_dim=2`)
- Head: flatten all node states to `[2N]` and pass through the same real MLP architecture as baseline
  - `Linear -> ReLU -> Linear -> ReLU -> Linear`
- Output: flattened complex voltage `[V_re, V_im]` for all nodes (`[2N]`)

No local/global decomposition is used in this script.

## Training objective

- Main loss: normalized MSE between predicted and target flattened complex voltage.
- Evaluation metrics (same protocol as MLP-only comparison):
  - `mae_vmag_pu`, `rmse_vmag_pu`
  - `mae_angle_deg`, `rmse_angle_deg`

## Run command (Colab style)

```python
import os, sys, subprocess

REPO = "/content/GNN-Sandia"
os.chdir(REPO)
os.environ["PYTHONUNBUFFERED"] = "1"

cmd = [
    sys.executable, "-u", "train_gine_plus_mlp_complex_voltage.py",
    "--data_root", "/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg",
    "--nodes_csv", "Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    "--edge_catalog_csv", "Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
    "--out_dir", "/content/gine_plus_mlp_complex_8500",
    "--epochs", "200",
    "--batch_size", "16",
    "--hidden_gnn", "64",
    "--layers", "3",
    "--state_dim", "2",
    "--hidden_mlp", "1024",
    "--node_emb_dim", "16",
    "--edge_emb_dim", "8",
    "--lr", "1e-3",
    "--weight_decay", "1e-5",
    "--patience", "30",
    "--seed", "42",
    "--sample_frac", "1.0",
    "--cache_tensor", "/content/preloaded_gine_plus_mlp.pt",
    "--num_workers", "2",
    "--disable_dropout",
]

print(" ".join(cmd))
subprocess.run(cmd, cwd=REPO, check=True)
```

## Outputs

Written to `--out_dir`:

- `gine_plus_mlp_best.pt`
- `x_mean.pt`, `x_std.pt`
- `y_mean.pt`, `y_std.pt`
- `gine_plus_mlp_report.json`

Use these alongside the MLP-only report to compare whether graph inductive bias improves accuracy.
