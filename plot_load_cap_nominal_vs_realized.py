# -*- coding: utf-8 -*-
"""
Generate nominal vs realized figures for loads (one per load type M1/M2/M4/M5)
and for capacitor bank. Uses the same concepts as compare_nominal_vs_realized.py:
- Load nominal: set values from DSS (dss.Loads.kW, dss.Loads.kvar) after apply.
- Load realized: CktElement.TotalPowers() per Load element.
- Cap nominal: CAP_Q_KVAR[bus] per phase at each (bus, phase) node.
- Cap realized: from power-flow solution, aggregated to (bus, phase) per node.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("nominal_vs_realized_comparison")
load_df = pd.read_csv(base / "per_load_nominal_vs_realized.csv")
cap_df = pd.read_csv(base / "per_node_cap_nominal_vs_realized.csv")

# --- One figure per load type: P and Q nominal vs realized (same as code) ---
for load_type in ["M1", "M2", "M4", "M5"]:
    df = load_df[load_df["load_type"] == load_type]
    if df.empty:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # P: nominal (set kW) vs realized (TotalPowers P)
    ax[0].scatter(df["p_nominal_kw"], df["p_realized_kw"], s=14, alpha=0.7)
    plim = max(df["p_nominal_kw"].max(), df["p_realized_kw"].max()) * 1.05
    ax[0].plot([0, plim], [0, plim], "k--", lw=1)
    ax[0].set_xlim(0, plim)
    ax[0].set_ylim(0, plim)
    ax[0].set_xlabel("Nominal load P (kW) — set value")
    ax[0].set_ylabel("Realized load P (kW) — power-flow")
    ax[0].set_title(f"Load type {load_type} — active power")

    # Q: nominal (set kvar) vs realized (TotalPowers Q)
    ax[1].scatter(df["q_nominal_kvar"], df["q_realized_kvar"], s=14, alpha=0.7)
    qlim = max(df["q_nominal_kvar"].max(), df["q_realized_kvar"].max()) * 1.05
    ax[1].plot([0, qlim], [0, qlim], "k--", lw=1)
    ax[1].set_xlim(0, qlim)
    ax[1].set_ylim(0, qlim)
    ax[1].set_xlabel("Nominal load Q (kVAR) — set value")
    ax[1].set_ylabel("Realized load Q (kVAR) — power-flow")
    ax[1].set_title(f"Load type {load_type} — reactive power")

    plt.suptitle(f"Nominal vs realized — Load model {load_type} only", y=1.02)
    plt.tight_layout()
    out = base / f"load_{load_type}_nominal_vs_realized.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

# --- Capacitor bank only: Q nominal vs Q realized (per node = per bus-phase) ---
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(cap_df["q_nominal_kvar"], cap_df["q_realized_kvar"], s=24, alpha=0.7)
qmin = min(cap_df["q_nominal_kvar"].min(), cap_df["q_realized_kvar"].min())
qmax = max(cap_df["q_nominal_kvar"].max(), cap_df["q_realized_kvar"].max())
pad = max((qmax - qmin) * 0.05, 1.0)
ax.plot([qmin - pad, qmax + pad], [qmin - pad, qmax + pad], "k--", lw=1)
ax.set_xlim(qmin - pad, qmax + pad)
ax.set_ylim(qmin - pad, qmax + pad)
ax.set_xlabel("Nominal capacitor Q per node (kVAR) — CAP_Q_KVAR")
ax.set_ylabel("Realized capacitor Q per node (kVAR) — power-flow")
ax.set_title("Capacitor bank — reactive power (per bus-phase node)")
plt.tight_layout()
out_cap = base / "cap_nominal_vs_realized.png"
plt.savefig(out_cap, dpi=200)
plt.close()
print("Saved:", out_cap)
