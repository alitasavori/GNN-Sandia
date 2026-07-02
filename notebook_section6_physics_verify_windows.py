# --- Offline physics verification: physical-units PyTorch vs OpenDSS (Windows local) ---
import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"C:\Users\alita\OneDrive\Desktop\GNN2")  # or None → auto-find from cwd
PF_DATA_ROOT = None  # None → REPO / "datasets_gnn2_from pc/loadtype_8500_dailyagg"
CHUNK_PARENT = None

SAMPLE_ID = 0
USE_RANDOM_CONTROLS = False
RUN_OPENDSS_VERIFY = True  # False if opendssdirect not installed


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
    raise FileNotFoundError("Set REPO at top of cell or cd to GNN2 repo first.")


REPO = _find_repo(REPO)
PF_DATA_ROOT = (
    REPO / "datasets_gnn2_from pc" / "loadtype_8500_dailyagg"
    if PF_DATA_ROOT is None
    else Path(PF_DATA_ROOT).expanduser().resolve()
)
if CHUNK_PARENT is not None:
    CHUNK_PARENT = Path(CHUNK_PARENT).expanduser().resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["GNN2_REPO_ROOT"] = str(REPO)

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
print(f"opendss_available: {pf_verify.opendss_available()}")

# Always physical-units path (same as training --pf_units physical)
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

cmp = pf_verify.compare_physical_opendss(
    snap,
    repo=REPO,
    run_opendss=RUN_OPENDSS_VERIFY,
)
pf_verify.print_physical_opendss_report(cmp)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(cmp["abs_r_py_mv"], bins=80, log=True, color="steelblue", alpha=0.85)
axes[0].set_title("|r_py| = |P_inj - Re(VYV)| physical PyTorch (MV)")
axes[0].set_xlabel("|ΔP| [kW]")
axes[0].set_ylabel("count")
if "residual_gap_abs_dp_mv" in cmp:
    axes[1].hist(cmp["residual_gap_abs_dp_mv"], bins=80, log=True, color="crimson", alpha=0.85)
    axes[1].set_title("|r_py - r_dss| backprop gap vs OpenDSS (MV)")
    axes[1].set_xlabel("|ΔP| [kW]")
else:
    axes[1].text(0.5, 0.5, cmp.get("opendss_skipped", "OpenDSS off"), ha="center", va="center")
    axes[1].set_axis_off()
plt.tight_layout()
plt.show()
