# GINE+MLP With Auxiliary Heads (No Local/Global Voltage Split)

This experiment keeps the `train_gine_plus_mlp_complex_voltage.py` trunk and adds aux heads.

## Model

- Main trunk: `GINE -> 2D state per node -> flatten -> RealMLP -> [V_re, V_im]`
- No local/global voltage decomposition.
- Aux heads are attached to the **GINE+MLP voltage output vector**.

Aux targets (from `gnn_sample_meta.csv`):

- 12 regulator classification tasks
- 10 capacitor-step classification tasks

## Loss

- Main voltage loss: normalized MSE on flattened `[V_re, V_im]`
- Aux loss: mean CE over reg heads and cap heads
- Total:
  - `L = L_voltage + λ_reg * scale(epoch) * L_reg + λ_cap * scale(epoch) * L_cap`

Aux schedule (same idea as `train_homo_gine_global_localres_pq_aux.py`):

- warmup epochs: aux scale = 0
- ramp epochs: aux scale increases linearly to 1

## Colab execution cell

```python
import os, sys, subprocess, datetime

REPO = "/content/GNN-Sandia"
os.chdir(REPO)
os.environ["PYTHONUNBUFFERED"] = "1"

tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/content/gine_plus_mlp_aux_complex_fresh_{tag}"

cmd = [
    sys.executable, "-u", "train_gine_plus_mlp_aux_complex_voltage.py",
    "--data_root", "/content/drive/MyDrive/datasets_gnn2/loadtype_8500_dailyagg",
    "--nodes_csv", "Heterogenous GNN dataset/nodes/hetero_mv_nodes_load_transformer_reg_tap_only.csv",
    "--edge_catalog_csv", "Heterogenous GNN dataset/edges/hetero_mv_line_edges_load_only_compacted.csv",
    "--meta_csv", "gnn_sample_meta.csv",
    "--out_dir", out_dir,
    "--epochs", "200",
    "--batch_size", "16",
    "--hidden_gnn", "64",
    "--layers", "3",
    "--state_dim", "2",
    "--hidden_mlp", "1024",
    "--aux_hidden", "512",
    "--node_emb_dim", "16",
    "--edge_emb_dim", "8",
    "--lr", "5e-4",
    "--weight_decay", "1e-5",
    "--patience", "30",
    "--seed", "42",
    "--train_frac", "0.8",
    "--val_frac", "0.1",
    "--sample_frac", "1.0",
    "--cache_tensor", "/content/preloaded_gine_plus_mlp_aux.pt",
    "--num_workers", "2",
    "--lambda_reg", "0.02",
    "--lambda_cap", "0.01",
    "--aux_warmup_epochs", "30",
    "--aux_ramp_epochs", "20",
    "--log_every", "1",
    "--disable_dropout",
]

print(" ".join(cmd))
# Stream stdout line-by-line so Colab shows each epoch as it runs (subprocess.run buffers until exit).
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
print("Saved run bundle:", out_dir)
```

## Outputs

Written to `--out_dir`:

- `gine_plus_mlp_aux_best.pt`
- `x_mean.pt`, `x_std.pt`, `y_mean.pt`, `y_std.pt`
- `gine_plus_mlp_aux_report.json`

## Compare vs non-aux baseline

Compare:

- `gine_plus_mlp_report.json` (no aux)
- `gine_plus_mlp_aux_report.json` (with aux)

Primary voltage metrics:

- `test_mae_vmag`
- `test_rmse_vmag`
- `test_mae_angle_deg`
- `test_rmse_angle_deg`
