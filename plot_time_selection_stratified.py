"""
Visualize stratified time-step selection for the injection dataset.

For each profile (load, PV, net): time indices are sorted by profile value and
split into B equal-population bins; anchors (min, max) are always included; from
each bin a random subset is drawn until the quota is filled. This script plots
the three profiles with bins and selected indices (anchors highlighted).

Run from repo root. Output: time_selection_stratified.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse selection logic from run_injection_dataset
import run_injection_dataset as inj

NPTS = 288
STEP_MIN = 5
# Use smaller K for clearer figure (fewer points)
K_TOTAL = 96
B = 10
SEED = 42

def read_profile(path, npts=NPTS):
    return inj.read_profile_csv_two_col_noheader(path, npts=npts, debug=False)

def equal_pop_bin_edges(profile, B):
    """Return value thresholds that define B equal-population bins (sorted by value)."""
    x = np.asarray(profile, dtype=float)
    T = len(x)
    order = np.argsort(x, kind="mergesort")
    sorted_vals = x[order]
    base = T // B
    extra = T % B
    edges = [sorted_vals[0]]
    start = 0
    for b in range(B - 1):
        size = base + (1 if b < extra else 0)
        start += size
        if start < T:
            edges.append(sorted_vals[start])
    edges.append(sorted_vals[-1])
    return np.array(edges)

def main():
    dss_path = os.path.join(os.path.dirname(__file__), "ieee34Mod1_with_loadshape.dss")
    if not os.path.isfile(dss_path):
        print("DSS not found; using synthetic profiles for demo.")
        t = np.linspace(0, 24, NPTS, endpoint=False)
        mL = 0.5 + 0.5 * np.sin(2 * np.pi * (t - 6) / 24)  # peak ~noon
        mL = np.clip(mL, 0.2, 1.0)
        mPV = np.maximum(0, np.sin(np.pi * (t - 6) / 12))  # day only
    else:
        csvL_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "5minDayShape")
        csvPV_token, _ = inj.find_loadshape_csv_in_dss(dss_path, "IrradShape")
        csvL_path = inj.resolve_csv_path(csvL_token, dss_path)
        csvPV_path = inj.resolve_csv_path(csvPV_token, dss_path)
        mL = read_profile(csvL_path) if os.path.isfile(csvL_path) else np.linspace(0.5, 1.0, NPTS)
        mPV = read_profile(csvPV_path) if os.path.isfile(csvPV_path) else np.maximum(0, np.sin(np.pi * (np.arange(NPTS) * 5 / 60 - 6) / 12))

    P_load, P_pv = 1415.2, 1000.0
    prof_load = mL.copy()
    prof_pv = mPV.copy()
    prof_net = P_load * mL - P_pv * mPV

    rng = np.random.default_rng(SEED)
    K_load, K_pv, K_net = inj.split_total_K_across_profiles(K_TOTAL)
    tL = inj.select_times_anchors_equalpop(prof_load, K_load, B=B, include_anchors=True, rng=rng)
    tP = inj.select_times_anchors_equalpop(prof_pv, K_pv, B=B, include_anchors=True, rng=np.random.default_rng(SEED + 1))
    tN = inj.select_times_anchors_equalpop(prof_net, K_net, B=B, include_anchors=True, rng=np.random.default_rng(SEED + 2))
    union = list(dict.fromkeys(tL + tP + tN))[:K_TOTAL]

    anchors_load = [int(np.argmin(prof_load)), int(np.argmax(prof_load))]
    anchors_pv  = [int(np.argmin(prof_pv)),  int(np.argmax(prof_pv))]
    anchors_net = [int(np.argmin(prof_net)), int(np.argmax(prof_net))]

    time_axis = np.arange(NPTS) * STEP_MIN / 60  # hours

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Stratified time-step selection: equal-population bins + anchors", fontsize=12)

    def plot_one(ax, profile, selected, anchors, title, ylabel):
        ax.plot(time_axis, profile, "k-", lw=1, alpha=0.8, label="Profile")
        # Equal-population bin shading (by value): alternating bands
        edges = equal_pop_bin_edges(profile, B)
        for b in range(B):
            lo, hi = edges[b], edges[b + 1]
            ax.axhspan(lo, hi, alpha=0.2, color="C0" if b % 2 == 0 else "C1")
        # Selected points: from bins
        sel_set = set(selected)
        anchor_set = set(anchors)
        from_bins = [t for t in selected if t not in anchor_set]
        ax.scatter(
            time_axis[from_bins], np.asarray(profile)[from_bins],
            c="C0", s=18, alpha=0.8, label=f"From bins (n={len(from_bins)})", zorder=3
        )
        ax.scatter(
            time_axis[list(anchor_set)], np.asarray(profile)[list(anchor_set)],
            c="red", s=80, marker="*", edgecolors="darkred", linewidths=0.5,
            label="Anchors (min, max)", zorder=4
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)

    plot_one(axes[0], prof_load, tL, anchors_load, "Load profile $m_L(t)$", "$m_L$")
    plot_one(axes[1], prof_pv,  tP, anchors_pv,  "PV profile $m_{PV}(t)$", "$m_{PV}$")
    plot_one(axes[2], prof_net, tN, anchors_net, "Net profile $P_{load}\\, m_L - P_{pv}\\, m_{PV}$", "Net (kW)")

    axes[2].set_xlabel("Time (hours)")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "time_selection_stratified.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # Optional: union timeline (which times made it to the final set)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 1.2))
    ax2.scatter(time_axis[union], np.ones(len(union)), c="C0", s=8, alpha=0.7)
    ax2.set_xlim(0, 24)
    ax2.set_xlabel("Time (hours)")
    ax2.set_yticks([])
    ax2.set_title(f"Final selected time indices (union, n={len(union)})")
    ax2.grid(True, alpha=0.3, axis="x")
    out2 = os.path.join(os.path.dirname(__file__), "time_selection_union.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

if __name__ == "__main__":
    main()
