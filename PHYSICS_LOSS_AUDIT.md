# Physics-Informed Loss Audit (DA-GPS GINE)

OpenDSS labels are ground truth. Physics should be a differentiable surrogate close to OpenDSS that does **not** hurt |V| MAE.

## 1. Implementation summary (correct)

| Item | Behavior |
|------|----------|
| **flow_relative** (default) | `Huber_pu(S(Y,V_label) − S(Y,V_pred))`; `V_label` detached; **P_inj unused** |
| **absolute** | `Huber_pu(P_inj − S(Y,V_pred))` with feature injections |
| **hybrid** | `flow_relative + α·absolute` (α=`--pf_absolute_weight`, default 0.1) |
| **S(Y,V)** | `S = V·conj(YV)`; V in **volts** (pu × per-bus `v_scale_volts`); Y in **Siemens**; P/Q in kW/kvar then ÷ `S_base` for Huber |
| **Y base** | Line `R_full`/`X_full` from edges; regulator xfmr branches skipped in base, re-stamped from catalog |
| **Controls** | Reg tap + cap shunt `jB` stamped into Y; caps **not** in `Q_inj` |
| **pf_use_label_controls** (default on) | Y stamped with **ground-truth** cap/reg targets |
| **pf_detach_controls** (default on in smoke) | `stop_grad` on tap/cap tensors used in Y |
| **P_inj** | `P_pv − P_load`, `Q = −Q_pv − Q_load` from denormalized features only |
| **Balance mask** | 185 refined nodes (`pf_balance_nodes_refined.csv`) on 3817-node chunk; slack excluded |
| **Units** | `S_base=5000 kVA`, `huber_delta_kw=10` → 0.002 pu; `V_base=12.47 kV` for Z metadata |
| **flow_relative scale** | `loss × 1/mean(y_std²)` to match normalized voltage MSE scale |
| **Curriculum** | `pf_weight_warmup_epochs` + `pf_weight_ramp_epochs` multiply effective weight |

**Verified on snapshot 0 (refined 185 nodes):**

- `flow_relative(V_label, V_label) = 0` exactly
- `absolute` at label V ≈ 0 (feature injections match linear Y@V on refined mask; median |r_py| ≈ 0.06 kW)

## 2. What is wrong / mismatched vs OpenDSS

### A. Surrogate vs OpenDSS (expected, not a code bug)

- **Linear Y-bus** vs **unbalanced Newton** OpenDSS: small residual at label V on refined nodes (median |ΔP| ~ 0.06 kW); larger on unrefined 1177-node list and interface buses.
- **Reg/cap in Y** are approximate tap/shunt stamps; orientation matches OpenDSS catalog (`from_node` = downstream tap winding).
- **No transformer magnetizing / no LV network** in MV subgraph.

### B. Training–validation parity bug (fixed)

`chunk_parent` validation called `_pf_loss_if_enabled` **without** `y_cap_label` / `y_reg_label` / `epoch`. With `--pf_use_label_controls`, train stamped Y from labels but val used **predicted** cap/reg → `train_pf` and `val_pf` measured different physics. Fixed in `train_da_gps_multitask_complex_voltage_gine.py` (val + test eval paths).

### C. What exacerbates |V| regression when physics turns on

1. **Curriculum cliff**: warmup=3 → epoch 4 first nonzero physics (×0.2 ramp); epochs 5–6 at ×0.4–×0.6 (`pf_wt` 0.008–0.012 on base 0.02). Best val_volt at epoch 3 (pre-physics) is **expected**.
2. **flow_relative volt scale** (`~1/0.02² ≈ 2500×`): amplifies gradients w.r.t. V even when raw pu Huber is tiny; physics gradients compete with `lambda_voltage=1` MSE.
3. **train_pf ≫ val_pf**: train mode + dropout; val eval; after fix, same label Y on both sides.
4. **Only 185/3817 nodes** in pf mask: gradients backprop through full graph but loss focuses on Y-closure hetero loads — can hurt tail nodes (`val_r2_min` volatile).
5. **Missing `q_pv_kvar` in `--node_feature_cols`** (smoke: `p_load_kw,q_load_kvar,p_pv_kw` only): `Q_pv=0` in absolute/hybrid; **no effect on flow_relative**.
6. **hybrid + wrong P_inj** would fight voltage; stick to **flow_relative** for label-safe physics.

### D. Epoch 5–6 collapse

**Expected under current hyperparams**, not a numerical bug: physics weight ramps while voltage was already near a good basin at epoch 3. `val_r2_min` tracks a single worst node across 3817 buses — noisy. `pf` improving while |V| worsens means the surrogate gradient is not aligned with label MSE at finite `V_pred`.

## 3. Recommended changes (ranked)

| Priority | Change | Rationale |
|----------|--------|-----------|
| 1 | Keep **flow_relative** + **pf_balance_nodes_refined.csv** (185) | P_inj cancels; smallest |r_py| at label V |
| 2 | **Longer warmup / slower ramp** (e.g. 8+8) or lower `--loss_power_balance_weight` (0.005–0.01) | Avoid epoch-4 shock |
| 3 | **`--lambda_voltage` 2–5** or Engage-style fine-tune with frozen backbone | Anchor |V| when adding physics |
| 4 | **`--pf_hard_node_topk 32`** | Physics only where V error is largest |
| 5 | Add **`q_pv_kvar`** to features if using hybrid/absolute | Correct Q_inj |
| 6 | OpenDSS-derived P_inj columns (future) | Only helps absolute/hybrid |
| 7 | **`--pf_auto_scale_volt`** with cap | Match batch-wise voltage scale (use carefully) |

## 4. Hyperparams (smoke reference)

```
--loss_power_balance_weight 0.02
--pf_loss_mode flow_relative
--pf_use_label_controls --pf_detach_controls
--pf_balance_node_list_csv colab_pf_data/pf_balance_nodes_refined.csv
--pf_weight_warmup_epochs 3 --pf_weight_ramp_epochs 2
--early_stop_on voltage
```

Effective physics weight: epoch 4 → 0.004, ep 5 → 0.008, ep 6 → 0.012, ep 7+ → 0.02.
