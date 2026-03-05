# GNN-Based Power Flow Surrogate for IEEE 34-Bus Feeder
## Presentation Material — Complete Recap

---

## 1. Project Overview

**Goal:** Replace iterative OpenDSS power-flow solves with a fast GNN surrogate for real-time voltage prediction on the IEEE 34-bus distribution feeder.

**Setup:**
- **Network:** IEEE 34-bus feeder (95 phase nodes, 184 edges)
- **Simulator:** OpenDSS (ground truth)
- **Framework:** PyTorch Geometric (GNN)
- **Time resolution:** 5-min intervals (288 steps/day)

---

## 2. Datasets (5 Types)

| # | Name | Output Dir | Samples | Features | Target | Purpose |
|---|------|------------|---------|----------|--------|--------|
| 1 | **Original** | `datasets_gnn2/original` | ~192k | p_load, q_load, p_pv (3) | vmag_pu | Baseline; raw P/Q per node |
| 2 | **Injection** | `datasets_gnn2/injection` | ~192k | p_inj, q_inj (2) | vmag_pu | Net injection (P_inj = P_PV − P_load) |
| 3 | **Load-type** | `datasets_gnn2/loadtype` | ~192k | 13 features* | vmag_pu | Load model breakdown (M1/M2/M4/M5), electrical distance |
| 4 | **Delta-V** | `datasets_gnn2/deltav` | 57,600 | 14 (Load-type + vmag_zero) | vmag_delta_pu | Voltage change due to PV |

*Load-type features: electrical_distance_ohm, m1_p, m1_q, m2_p, m2_q, m4_p, m4_q, m5_p, m5_q, q_cap, p_pv, p_sys_balance, q_sys_balance

---

## 3. Dataset Generation

### 3.1 Scenario Sampling (All Datasets)

- **Baseline:** P_load=1415 kW, Q_load=835 kvar, P_PV=1000 kW
- **Ranges:** P_load, Q_load ∈ [0.8–1.2]×baseline; P_PV ∈ [0.6–1.2]×baseline
- **24h profiles:** Load shape (mL[t]) and irradiance shape (mPV[t]) from CSV files

### 3.2 Time Sampling

- **Original, Injection, Load-type:** 200 scenarios × 960 snapshots/scenario via `select_times_three_profiles` (load, PV, net load bins)
- **Delta-V, Delta-V 5×:** 1000 scenarios × ~57 snapshots/scenario, same time sampling

### 3.3 Key Fix: Delta-V Train–Test Alignment

**Problem:** Original Delta-V used `mL=1.0` and random `mPV` in training. Test used full 24h profile (mL[t], mPV[t]) → train–test mismatch.

**Fix:** Delta-V now uses `mL[t]` and `mPV[t]` from the 24h profile at each timestep, matching test-time structure.

### 3.4 Generator Scripts

| Script | Dataset | Key Params |
|--------|---------|------------|
| `run_injection_dataset.py` | Injection | 200 scenarios, 960 snapshots |
| `run_loadtype_dataset.py` | Load-type | 200 scenarios, 960 snapshots |
| `run_deltav_dataset.py` | Delta-V | 1000 scenarios, 57,600 samples, 24h profile |

---

## 4. GNN Architecture

**Model:** PFIdentityGNN (message-passing GNN on phase graph)

- **Node embeddings:** Learnable per-node (identity)
- **Edge features:** R (resistance), X (reactance) from OpenDSS
- **Message passing:** EdgeIdentityMP (MLP on h_j + edge_attr + edge_emb)
- **Readout:** MLP → per-node voltage prediction

**Hyperparameters:** node_emb_dim, edge_emb_dim, h_dim, num_layers, use_norm, use_phase_onehot

---

## 5. Architecture Exploration (7 Blocks)

| Block | Script | Configs | Datasets | Best |
|-------|--------|---------|----------|------|
| 1 | gnn_final_exploration | 20 nominees | Original, Injection, Load-type | light_xwide + Load-type |
| 2 | gnn_further_exploration | 60 nominees | Same 3 | — |
| 3 | gnn_narrow_exploration | 10 promising | Same 3 | — |
| 4 | gnn_boost_exploration | 10 to beat Block 3 | Same 3 | — |
| 5 | gnn_refine_exploration | 8 from Block 4 | Same 3 | light_emb_h96_phase_onehot_depth3_h112 |
| 6 | gnn_deltav_exploration | 9 best Load-type | Delta-V | light_xwide_emb_phase_onehot |

**Exploration:** 30% data for exploration; 100% for final training.

---

## 6. Final 7 Models (Best per Dataset)

| Block | Dataset | Architecture | Target | Key Params |
|-------|---------|--------------|--------|------------|
| 1 | Original | medium | vmag_pu | h=32, 4L |
| 2 | Injection | deep | vmag_pu | h=64, 4L |
| 3 | Load-type | light_emb_h96 | vmag_pu | h=96, 2L |
| 4 | Load-type | light_emb_h96_phase_onehot_depth3 | vmag_pu | h=96, 3L, phase onehot |
| 5 | Load-type | light_emb_h96_phase_onehot_depth3_h112 | vmag_pu | h=112, 3L, phase onehot |
| 6 | Delta-V | light_xwide_emb_phase_onehot | vmag_delta_pu | h=128, 2L, phase onehot |
| 7 | Delta-V 5× | light_xwide_emb_phase_onehot | vmag_delta_pu | h=128, 2L, phase onehot |

**Training:** `run_gnn3_best7_train.py` — trains all 7; `run_gnn3_deltav_only_train.py` — trains only blocks 6 & 7 (for Delta-V refresh).

---

## 7. Evaluation & Analysis

### 7.1 24h Overlay & Metrics

- **Script:** `run_gnn3_timing_comparison.py`
- **Output:** Overlay plots (OpenDSS vs GNN), MAE/RMSE @ observed node (e.g., 816.1)
- **PV scenarios:** Run 3× with PV scale 0.7×, 1.0×, 1.3× (`%run run_gnn3_timing_comparison.py all`)

### 7.2 Error Analysis

- **Spatial:** Per-bus error breakdown
- **Temporal:** Per-timestep error to identify spikes
- **Slide title:** *"Error analysis: spatial and temporal breakdown"*

### 7.3 Key Results (from exploration)

- **Best full voltage (V):** Block 5 — light_emb_h96_phase_onehot_depth3_h112 + Load-type | MAE≈0.006 pu
- **Best delta-V:** Block 6/7 — light_xwide_emb_phase_onehot | MAE≈0.003–0.005 pu

---

## 8. Summary of Work Done

1. **5 datasets** — Original, Injection, Load-type, Delta-V, Delta-V 5×
2. **Dataset generation** — OpenDSS-based sampling with 24h profiles; Delta-V fix for train–test alignment
3. **Architecture exploration** — 7 blocks, 100+ configs across 4 exploration scripts
4. **7 best models** — One per dataset type, saved in `gnn3_best7_output/`
5. **Delta-V retrain** — Blocks 6 & 7 retrained on new 24h-profile datasets
6. **PV sensitivity** — Timing comparison at 3 PV penetration levels (0.7×, 1.0×, 1.3×)
7. **Error analysis** — Per-bus and per-time inspection to locate high-error regions

---

## 9. File Structure (Quick Reference)

```
GNN2/
├── run_injection_dataset.py      # Dataset 2
├── run_loadtype_dataset.py      # Dataset 3
├── run_deltav_dataset.py        # Dataset 4
<!-- run_deltav_5x_dataset.py (Delta-V 5×) removed -->
├── gnn_final_exploration.py     # Block 1
├── gnn_further_exploration.py   # Block 2
├── gnn_narrow_exploration.py    # Block 3
├── gnn_boost_exploration.py     # Block 4
├── gnn_refine_exploration.py    # Block 5
├── gnn_deltav_exploration.py    # Block 6
<!-- gnn_deltav_5x_exploration.py (Delta-V 5× exploration) removed -->
├── run_gnn3_best7_train.py      # Train all 7
├── run_gnn3_deltav_only_train.py# Train blocks 6 & 7 only
├── run_gnn3_timing_comparison.py # Overlay, MAE/RMSE, PV scenarios
├── run_gnn3_overlay_7.py        # Inference helpers
└── GNN3.ipynb                   # Orchestration
```

---

## 10. Suggested Slide Order

1. **Title** — GNN-based power flow surrogate for IEEE 34-bus feeder
2. **Problem** — Need fast voltage prediction for real-time applications
3. **Approach** — Message-passing GNN on phase graph; OpenDSS for data
4. **Datasets** — 5 types, features, targets (table)
5. **Dataset generation** — Scenario sampling, 24h profiles, Delta-V fix
6. **Architecture** — PFIdentityGNN, message passing, hyperparameters
7. **Exploration** — 7 blocks, 100+ configs
8. **Final models** — 7 best per dataset (table)
9. **Results** — MAE/RMSE, overlay plots
10. **Error analysis** — Spatial and temporal breakdown
11. **PV sensitivity** — 0.7×, 1.0×, 1.3× penetration
12. **Summary** — What we did, key findings
