# --- Offline physics verification (Windows local) ---
# Paste into nonunique.ipynb section 6, or run:  python notebook_section6_physics_verify_windows.py
import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# === Windows-local paths (edit if your layout differs) ===
REPO = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2")  # or None -> auto-find from cwd
PF_DATA_ROOT = None  # None -> REPO / "datasets_gnn2_from pc/loadtype_8500_dailyagg"
CHUNK_PARENT = None  # optional chunk fallback for CSV discovery

SAMPLE_ID = 0
USE_RANDOM_CONTROLS = False  # True: small tap noise + random cap flips
COMPARE_LEGACY = True        # legacy_pu vs physical side-by-side
RUN_OPENDSS_VERIFY = True    # False if opendssdirect not installed


def _find_repo(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if (p / "train_da_gps_multitask_complex_voltage_gine.py").is_file():
            return p
        raise FileNotFoundError(f"Trainer script not found under REPO={p}")
    for cand in (Path.cwd(), Path.cwd().parent):
        p = cand.resolve()
        if (p / "train_da_gps_multitask_complex_voltage_gine.py").is_file():
            return p
    raise FileNotFoundError(
        "Could not find GNN2 repo. Set REPO at top of cell or cd to repo first."
    )


REPO = _find_repo(REPO)
if PF_DATA_ROOT is None:
    PF_DATA_ROOT = REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
else:
    PF_DATA_ROOT = Path(PF_DATA_ROOT).expanduser().resolve()
if CHUNK_PARENT is not None:
    CHUNK_PARENT = Path(CHUNK_PARENT).expanduser().resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["GNN2_REPO_ROOT"] = str(REPO)  # overwrite stale Colab/content paths

from gnn2_pf_data_paths import resolve_pf_catalog_paths

_, _, PF_DATA_ROOT = resolve_pf_catalog_paths(
    repo=REPO,
    preferred_root=PF_DATA_ROOT if PF_DATA_ROOT.is_dir() else None,
    chunk_parent=CHUNK_PARENT if CHUNK_PARENT is not None and CHUNK_PARENT.is_dir() else None,
)

import gnn2_pf_physics_verify as pf_verify

importlib.reload(pf_verify)

print(f"Repo:              {REPO}")
print(f"PF_DATA_ROOT:      {PF_DATA_ROOT}")
print(f"sample_id:         {SAMPLE_ID}")
print(f"compare_legacy:    {COMPARE_LEGACY}")
print(f"opendss_available: {pf_verify.opendss_available()}")

snap = pf_verify.load_snapshot_state(
    SAMPLE_ID,
    repo=REPO,
    pf_data_root=PF_DATA_ROOT,
    chunk_parent=CHUNK_PARENT,
    use_physical_units=True,
)
if USE_RANDOM_CONTROLS:
    rng = np.random.default_rng(42)
    snap = pf_verify.apply_random_perturbation(
        snap, rng=rng, sigma_v_ri=0.0, sigma_tap=0.01, flip_cap_prob=0.15
    )
    print("Applied random control perturbation (label V unchanged)")

cmp = pf_verify.compare_legacy_physical_opendss(
    SAMPLE_ID,
    repo=REPO,
    pf_data_root=PF_DATA_ROOT,
    chunk_parent=CHUNK_PARENT,
    compare_legacy=COMPARE_LEGACY,
    run_opendss=RUN_OPENDSS_VERIFY,
    snap_physical=snap,
)
pf_verify.print_legacy_physical_opendss_report(cmp)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(
    cmp["physical_abs_dp_mv"],
    bins=80,
    log=True,
    color="steelblue",
    alpha=0.75,
    label="physical",
)
if COMPARE_LEGACY and "legacy_abs_dp_mv" in cmp:
    axes[0].hist(
        cmp["legacy_abs_dp_mv"],
        bins=80,
        log=True,
        color="coral",
        alpha=0.55,
        label="legacy_pu",
    )
axes[0].set_title("|P_inj - Re(VYV)| at label V (MV)")
axes[0].set_xlabel("|dP| [kW]")
axes[0].set_ylabel("count")
axes[0].legend()

if "residual_gap_physical_abs_dp_mv" in cmp:
    axes[1].hist(
        cmp["residual_gap_physical_abs_dp_mv"],
        bins=80,
        log=True,
        color="seagreen",
        alpha=0.85,
        label="physical gap",
    )
    if "residual_gap_legacy_abs_dp_mv" in cmp:
        axes[1].hist(
            cmp["residual_gap_legacy_abs_dp_mv"],
            bins=80,
            log=True,
            color="goldenrod",
            alpha=0.55,
            label="legacy gap",
        )
    axes[1].set_title("|r_py - r_dss| (MV, OpenDSS V)")
    axes[1].set_xlabel("|d residual| [kW]")
    axes[1].legend()
else:
    axes[1].hist(
        cmp["physical_abs_dq_mv"],
        bins=80,
        log=True,
        color="darkorange",
        alpha=0.85,
    )
    axes[1].set_title("|Q_inj - Im(VYV)| physical (MV)")
    axes[1].set_xlabel("|dQ| [kvar]")

plt.tight_layout()
plt.show()
