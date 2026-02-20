"""
Standalone script for GNN injection dataset generation.
Contains all helpers from the main dataset generation block (cell 3 of GNN2.ipynb).
Outputs to gnn_samples_inj_full/ with p_inj_kw, q_inj_kvar features.
"""
import os
import re
import numpy as np
import pandas as pd
import opendssdirect as dss

# ============================================================
# CONFIG
# ============================================================
DSS_FILE = "ieee34Mod1_with_loadshape.dss"
NPTS = 288
STEP_MIN = 5

VOLTAGE_BASES_KV = [69.0, 24.9, 4.16, 0.48]
VMAG_PU_MIN = 0.50
VMAG_PU_MAX = 1.50

BASELINE = dict(
    P_load_total_kw=1415.2,
    Q_load_total_kvar=835.2,
    P_pv_total_kw=1000.0,
    sigma_load=0.05,
    sigma_pv=0.05,
)

RANGES = dict(
    P_load_total_kw=(0.8, 1.2),
    Q_load_total_kvar=(0.8, 1.2),
    P_pv_total_kw=(0.6, 1.2),
    sigma_load=(0.5, 1.5),
    sigma_pv=(0.5, 1.5),
)

# ============================================================
# DEVICE-LEVEL P/Q SHARES
# ============================================================
DEVICE_P_SHARE = {
    "S890": 0.2544,
    "S844": 0.2289, "D842_844ra": 0.0025, "D844_846sb": 0.0071, "D844_846sc": 0.0057,
    "S860": 0.0339, "D834_860ra": 0.0045, "D834_860rb": 0.0057, "D834_860rc": 0.0311,
    "D860_836sa": 0.0085, "D860_836sb": 0.0028, "D860_836sc": 0.0119,
    "D858_834ra": 0.0011, "D858_834rb": 0.0042, "D858_834rc": 0.0037,
    "D834_860sa": 0.0045, "D834_860sb": 0.0057, "D834_860sc": 0.0311,
    "D818_820ra": 0.0096, "D820_822sa": 0.0382,
    "S848": 0.0339, "D846_848rb": 0.0065,
    "D820_822ra": 0.0382,
    "D860_836ra": 0.0085, "D860_836rb": 0.0028, "D860_836rc": 0.0119,
    "D836_840sa": 0.0051, "D836_840sb": 0.0062,
    "S830a": 0.0057, "S830b": 0.0057, "S830c": 0.0141, "D828_830ra": 0.0020,
    "S840": 0.0153, "D836_840ra": 0.0051, "D836_840rb": 0.0062,
    "D844_846rb": 0.0071, "D844_846rc": 0.0057, "D846_848sb": 0.0065,
    "D802_806sb": 0.0085, "D802_806sc": 0.0071,
    "D802_806rb": 0.0085, "D802_806rc": 0.0071,
    "D832_858ra": 0.0020, "D832_858rb": 0.0006, "D832_858rc": 0.0017,
    "D858_864sb": 0.0006,
    "D858_834sa": 0.0011, "D858_834sb": 0.0042, "D858_834sc": 0.0037,
    "D816_824rb": 0.0014, "D824_826sb": 0.0113, "D824_828sc": 0.0011,
    "D824_826rb": 0.0113,
    "D818_820sa": 0.0096,
    "D862_838rb": 0.0079,
    "D862_838sb": 0.0079,
    "D808_810rb": 0.0045,
    "D808_810sb": 0.0045,
    "D832_858sa": 0.0020, "D832_858sb": 0.0006, "D832_858sc": 0.0017,
    "D824_828rc": 0.0011, "D828_830sa": 0.0020,
    "D842_844sa": 0.0025,
    "D816_824sb": 0.0014,
    "D854_856sb": 0.0011, "D854_856rb": 0.0011,
    "D858_864rb": 0.0006,
}

DEVICE_Q_SHARE = {
    "S890": 0.2155,
    "S844": 0.3017, "D842_844ra": 0.0024, "D844_846sb": 0.0057, "D844_846sc": 0.0053,
    "S860": 0.0460, "D834_860ra": 0.0038, "D834_860rb": 0.0048, "D834_860rc": 0.0263,
    "D860_836sa": 0.0072, "D860_836sb": 0.0029, "D860_836sc": 0.0105,
    "D858_834ra": 0.0010, "D858_834rb": 0.0038, "D858_834rc": 0.0034,
    "D834_860sa": 0.0038, "D834_860sb": 0.0048, "D834_860sc": 0.0263,
    "D818_820ra": 0.0081, "D820_822sa": 0.0335,
    "S848": 0.0460, "D846_848rb": 0.0053,
    "D820_822ra": 0.0335,
    "D860_836ra": 0.0072, "D860_836rb": 0.0029, "D860_836rc": 0.0105,
    "D836_840sa": 0.0043, "D836_840sb": 0.0053,
    "S830a": 0.0048, "S830b": 0.0048, "S830c": 0.0096, "D828_830ra": 0.0014,
    "S840": 0.0201, "D836_840ra": 0.0043, "D836_840rb": 0.0053,
    "D844_846rb": 0.0057, "D844_846rc": 0.0053, "D846_848sb": 0.0053,
    "D802_806sb": 0.0072, "D802_806sc": 0.0067,
    "D802_806rb": 0.0072, "D802_806rc": 0.0067,
    "D832_858ra": 0.0014, "D832_858rb": 0.0005, "D832_858rc": 0.0014,
    "D858_864sb": 0.0005,
    "D858_834sa": 0.0010, "D858_834sb": 0.0038, "D858_834sc": 0.0034,
    "D816_824rb": 0.0010, "D824_826sb": 0.0096, "D824_828sc": 0.0010,
    "D824_826rb": 0.0096,
    "D818_820sa": 0.0081,
    "D862_838rb": 0.0067,
    "D862_838sb": 0.0067,
    "D808_810rb": 0.0038,
    "D808_810sb": 0.0038,
    "D832_858sa": 0.0014, "D832_858sb": 0.0005, "D832_858sc": 0.0014,
    "D824_828rc": 0.0010, "D828_830sa": 0.0014,
    "D842_844sa": 0.0024,
    "D816_824sb": 0.0010,
    "D854_856sb": 0.0010, "D854_856rb": 0.0010,
    "D858_864rb": 0.0005,
}

DEVICE_TO_BUSPH = {
    "S890": [("890", 1, 1/3), ("890", 2, 1/3), ("890", 3, 1/3)],
    "S844": [("844", 1, 1/3), ("844", 2, 1/3), ("844", 3, 1/3)],
    "S860": [("860", 1, 1/3), ("860", 2, 1/3), ("860", 3, 1/3)],
    "S848": [("848", 1, 1/3), ("848", 2, 1/3), ("848", 3, 1/3)],
    "S840": [("840", 1, 1/3), ("840", 2, 1/3), ("840", 3, 1/3)],
    "D842_844ra": [("844", 1, 1.0)],
    "D844_846sb": [("844", 2, 1.0)],
    "D844_846sc": [("844", 3, 1.0)],
    "D818_820ra": [("820", 1, 1.0)],
    "D820_822sa": [("820", 1, 1.0)],
    "D846_848rb": [("848", 2, 1.0)],
    "D820_822ra": [("822", 1, 1.0)],
    "D828_830ra": [("830", 1, 1.0)],
    "D844_846rb": [("846", 2, 1.0)],
    "D844_846rc": [("846", 3, 1.0)],
    "D846_848sb": [("846", 2, 1.0)],
    "D802_806sb": [("802", 2, 1.0)],
    "D802_806sc": [("802", 3, 1.0)],
    "D802_806rb": [("806", 2, 1.0)],
    "D802_806rc": [("806", 3, 1.0)],
    "D832_858ra": [("858", 1, 1.0)],
    "D832_858rb": [("858", 2, 1.0)],
    "D832_858rc": [("858", 3, 1.0)],
    "D858_864sb": [("858", 1, 1.0)],
    "D816_824rb": [("824", 2, 1.0)],
    "D824_826sb": [("824", 2, 1.0)],
    "D824_828sc": [("824", 3, 1.0)],
    "D824_826rb": [("826", 2, 1.0)],
    "D818_820sa": [("818", 1, 1.0)],
    "D862_838rb": [("838", 2, 1.0)],
    "D862_838sb": [("862", 2, 1.0)],
    "D808_810rb": [("810", 2, 1.0)],
    "D808_810sb": [("808", 2, 1.0)],
    "D832_858sa": [("832", 1, 1.0)],
    "D832_858sb": [("832", 2, 1.0)],
    "D832_858sc": [("832", 3, 1.0)],
    "D824_828rc": [("828", 3, 1.0)],
    "D828_830sa": [("828", 1, 1.0)],
    "D842_844sa": [("842", 1, 1.0)],
    "D816_824sb": [("816", 2, 1.0)],
    "D854_856sb": [("854", 2, 1.0)],
    "D854_856rb": [("856", 2, 1.0)],
    "D858_864rb": [("864", 1, 1.0)],
    "D834_860ra": [("860", 1, 0.5), ("860", 2, 0.5)],
    "D834_860rb": [("860", 2, 0.5), ("860", 3, 0.5)],
    "D834_860rc": [("860", 3, 0.5), ("860", 1, 0.5)],
    "D860_836sa": [("860", 1, 0.5), ("860", 2, 0.5)],
    "D860_836sb": [("860", 2, 0.5), ("860", 3, 0.5)],
    "D860_836sc": [("860", 3, 0.5), ("860", 1, 0.5)],
    "D858_834ra": [("834", 1, 0.5), ("834", 2, 0.5)],
    "D858_834rb": [("834", 2, 0.5), ("834", 3, 0.5)],
    "D858_834rc": [("834", 3, 0.5), ("834", 1, 0.5)],
    "D834_860sa": [("834", 1, 0.5), ("834", 2, 0.5)],
    "D834_860sb": [("834", 2, 0.5), ("834", 3, 0.5)],
    "D834_860sc": [("834", 3, 0.5), ("834", 1, 0.5)],
    "D860_836ra": [("836", 1, 0.5), ("836", 2, 0.5)],
    "D860_836rb": [("836", 2, 0.5), ("836", 3, 0.5)],
    "D860_836rc": [("836", 3, 0.5), ("836", 1, 0.5)],
    "D836_840sa": [("836", 1, 0.5), ("836", 2, 0.5)],
    "D836_840sb": [("836", 2, 0.5), ("836", 3, 0.5)],
    "S830a": [("830", 1, 0.5), ("830", 2, 0.5)],
    "S830b": [("830", 2, 0.5), ("830", 3, 0.5)],
    "S830c": [("830", 3, 0.5), ("830", 1, 0.5)],
    "D836_840ra": [("840", 1, 0.5), ("840", 2, 0.5)],
    "D836_840rb": [("840", 2, 0.5), ("840", 3, 0.5)],
    "D858_834sa": [("858", 1, 0.5), ("858", 2, 0.5)],
    "D858_834sb": [("858", 2, 0.5), ("858", 3, 0.5)],
    "D858_834sc": [("858", 3, 0.5), ("858", 1, 0.5)],
}

PV_PMMP_SHARE = {"pv840": 0.5, "pv860": 0.5}
PV_TO_BUSPH = {
    "pv840": [("840", 1, 1/3), ("840", 2, 1/3), ("840", 3, 1/3)],
    "pv860": [("860", 1, 1/3), ("860", 2, 1/3), ("860", 3, 1/3)],
}

# Output for injection dataset
OUT_DIR_INJ = "gnn_samples_inj_full"
os.makedirs(OUT_DIR_INJ, exist_ok=True)
EDGE_CSV_INJ = os.path.join(OUT_DIR_INJ, "gnn_edges_phase_static.csv")
NODE_CSV_INJ = os.path.join(OUT_DIR_INJ, "gnn_node_features_and_targets.csv")
SAMPLE_CSV_INJ = os.path.join(OUT_DIR_INJ, "gnn_sample_meta.csv")
NODE_INDEX_CSV_INJ = os.path.join(OUT_DIR_INJ, "gnn_node_index_master.csv")

# Capacitor Q per phase (kVAR)
CAP_Q_KVAR = {"844": 100.0, "848": 150.0}

# ============================================================
# DSS compile + time setup
# ============================================================
def _apply_voltage_bases():
    vb = ",".join(str(v) for v in VOLTAGE_BASES_KV)
    dss.Text.Command(f'set voltagebases="{vb}"')
    dss.Text.Command("calcvoltagebases")

def compile_once():
    dss.Basic.ClearAll()
    dss_path = os.path.abspath(DSS_FILE)
    if not os.path.exists(dss_path):
        raise FileNotFoundError(f"DSS file not found: {dss_path}")
    dss.Text.Command(f'compile "{dss_path}"')
    _apply_voltage_bases()
    return dss_path

def setup_daily():
    dss.Text.Command("set mode=daily")
    dss.Text.Command(f"set stepsize={STEP_MIN}m")
    dss.Text.Command("set number=1")
    dss.Text.Command("set hour=0")
    dss.Text.Command("set sec=0")

def set_time_index(t):
    total_sec = int(t * STEP_MIN * 60)
    dss.Solution.Hour(total_sec // 3600)
    dss.Solution.Seconds(total_sec % 3600)

# ============================================================
# Parse DSS to find csvfile for a Loadshape.<name>
# ============================================================
def _strip_quotes(s):
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def find_loadshape_csv_in_dss(dss_path, loadshape_name):
    txt = open(dss_path, "r", encoding="utf-8", errors="ignore").read()
    pat = re.compile(rf"(?im)^\s*new\s+loadshape\.{re.escape(loadshape_name)}\b.*?$", re.MULTILINE)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Could not find 'New Loadshape.{loadshape_name}' in DSS file.")
    line = m.group(0)
    m2 = re.search(r"(?i)\bcsvfile\s*=\s*([^\s]+)", line)
    if not m2:
        raise RuntimeError(f"Loadshape.{loadshape_name} found, but no csvfile=... on that line:\n{line}")
    csv_token = _strip_quotes(m2.group(1))
    return csv_token, line

def resolve_csv_path(csv_token, dss_path):
    csv_token = csv_token.replace("\\", "/")
    if os.path.isabs(csv_token):
        return csv_token
    base = os.path.dirname(dss_path)
    return os.path.abspath(os.path.join(base, csv_token))

# ============================================================
# Profile reader
# ============================================================
def read_profile_csv_two_col_noheader(csv_path, npts=NPTS, debug=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise RuntimeError(f"{csv_path} must have at least 2 columns (time,value). Got {df.shape[1]}")

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y)

    if debug and (~mask).any():
        bad_idx = np.where(~mask)[0]
        print(f"[WARN] {csv_path}: {len(bad_idx)} non-numeric rows. Showing first 5:")
        print(df.iloc[bad_idx[:5], :2])

    y = y[mask]
    if len(y) < npts:
        raise RuntimeError(f"CSV {csv_path} has only {len(y)} valid points < required {npts}")
    return y[:npts]

# ============================================================
# Time selection: Anchors + Equal-population (rank-based) bins
# ============================================================
def _equal_population_bins_indices(x, B):
    x = np.asarray(x, dtype=float)
    T = len(x)
    order = np.argsort(x, kind="mergesort")
    B = max(1, int(B))
    base = T // B
    extra = T % B
    sizes = [base + (1 if b < extra else 0) for b in range(B)]
    bins = []
    start = 0
    for b in range(B):
        end = start + sizes[b]
        bins.append(order[start:end].tolist())
        start = end
    return bins

def select_times_anchors_equalpop(profile, K, B=10, include_anchors=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    profile = np.asarray(profile, dtype=float)
    T = len(profile)
    if K <= 0:
        return []
    anchors = []
    if include_anchors:
        anchors = [int(np.argmin(profile)), int(np.argmax(profile))]
        anchors = list(dict.fromkeys(anchors))
    if len(anchors) >= K:
        return anchors[:K]
    bins = _equal_population_bins_indices(profile, B=B)
    sel = set(anchors)
    bins = [[t for t in bi if t not in sel] for bi in bins]
    remaining = K - len(anchors)
    base = remaining // max(1, int(B))
    extra = remaining % max(1, int(B))
    targets = [base + (1 if b < extra else 0) for b in range(max(1, int(B)))]
    selected = list(anchors)
    for b in range(len(bins)):
        take = targets[b] if b < len(targets) else 0
        if take <= 0:
            continue
        pool = bins[b]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(take, len(pool)), replace=False).tolist()
        selected += pick
        sel.update(pick)
    if len(selected) < K:
        pool = [t for t in range(T) if t not in sel]
        need = K - len(selected)
        if len(pool) > 0:
            selected += rng.choice(pool, size=min(need, len(pool)), replace=False).tolist()
    return selected[:K]

def split_total_K_across_profiles(K_total):
    K_total = int(K_total)
    base = K_total // 3
    extra = K_total % 3
    Ks = [base + (1 if i < extra else 0) for i in range(3)]
    return Ks[0], Ks[1], Ks[2]

def select_times_three_profiles(prof_load, prof_pv, prof_net, K_total, bins_by_profile, include_anchors=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    T = len(prof_load)
    assert len(prof_pv) == T and len(prof_net) == T
    K_load, K_pv, K_net = split_total_K_across_profiles(K_total)
    B_load = int(bins_by_profile.get("load", 10))
    B_pv = int(bins_by_profile.get("pv", 10))
    B_net = int(bins_by_profile.get("net", 10))
    tL = select_times_anchors_equalpop(prof_load, K_load, B=B_load, include_anchors=include_anchors, rng=rng)
    tP = select_times_anchors_equalpop(prof_pv, K_pv, B=B_pv, include_anchors=include_anchors, rng=rng)
    tN = select_times_anchors_equalpop(prof_net, K_net, B=B_net, include_anchors=include_anchors, rng=rng)
    anchors_all = set()
    if include_anchors:
        anchors_all |= {int(np.argmin(prof_load)), int(np.argmax(prof_load))}
        anchors_all |= {int(np.argmin(prof_pv)), int(np.argmax(prof_pv))}
        anchors_all |= {int(np.argmin(prof_net)), int(np.argmax(prof_net))}
    union = list(dict.fromkeys(tL + tP + tN))
    union_set = set(union)
    if len(union) > K_total:
        keep = list(sorted([t for t in anchors_all if t in union_set]))
        keep_set = set(keep)
        remaining_pool = [t for t in union if t not in keep_set]
        need = K_total - len(keep)
        if need < 0:
            return keep[:K_total]
        if need == 0:
            return keep
        pick = rng.choice(remaining_pool, size=need, replace=False).tolist()
        return (keep + pick)[:K_total]
    if len(union) < K_total:
        sel = set(union)
        pool = [t for t in range(T) if t not in sel]
        need = K_total - len(union)
        if len(pool) > 0:
            union += rng.choice(pool, size=min(need, len(pool)), replace=False).tolist()
        return union[:K_total]
    return union[:K_total]

# ============================================================
# Node list + bus phases
# ============================================================
def get_all_bus_phase_nodes():
    node_names_raw = list(dss.Circuit.AllNodeNames())
    node_names = []
    node_to_bus = {}
    node_to_phase = {}
    bus_to_phases = {}
    for n in node_names_raw:
        if "." not in n:
            continue
        bus, phs = n.split(".", 1)
        try:
            ph = int(phs)
        except Exception:
            continue
        if ph not in (1, 2, 3):
            continue
        name = f"{bus}.{ph}"
        if name in node_to_bus:
            continue
        node_names.append(name)
        node_to_bus[name] = bus
        node_to_phase[name] = ph
        bus_to_phases.setdefault(bus, set()).add(ph)
    bus_to_phases = {b: sorted(list(s)) for b, s in bus_to_phases.items()}
    return node_names, node_to_bus, node_to_phase, bus_to_phases

def parse_bus_spec(bus_full):
    parts = bus_full.split(".")
    bus = parts[0]
    phs = []
    for p in parts[1:]:
        try:
            ip = int(p)
            if ip in (1, 2, 3):
                phs.append(ip)
        except Exception:
            pass
    return bus, sorted(list(set(phs)))

# ============================================================
# Phase-edge extraction
# ============================================================
def list_to_sq(list_flat):
    arr = np.array(list_flat, dtype=float)
    k = int(np.sqrt(arr.size))
    return arr.reshape((k, k))

def pad_square(mat, n=3):
    mat = np.asarray(mat, dtype=float)
    out = np.zeros((n, n), dtype=float)
    k = mat.shape[0]
    out[:k, :k] = mat
    return out

def extract_static_phase_edges_to_csv(node_names_master, edge_csv_path, make_bidirectional=True):
    node_to_idx = {n: i for i, n in enumerate(node_names_master)}
    rows = []
    seen = set()

    def _add_edge(from_node, to_node, b_from, b_to, ph, elem_name, linecode, nph_line, length, r_per, x_per, c_per):
        if from_node not in node_to_idx or to_node not in node_to_idx:
            return
        u = int(node_to_idx[from_node])
        v = int(node_to_idx[to_node])
        key = (u, v, int(ph), elem_name)
        if key in seen:
            return
        seen.add(key)
        r_full = float(r_per * length)
        x_full = float(x_per * length)
        c_full = float(c_per * length)
        rows.append({
            "from_node": from_node, "to_node": to_node, "from_bus": b_from, "to_bus": b_to,
            "phase": int(ph), "line_name": elem_name, "linecode": linecode, "nph_line": int(nph_line),
            "length": float(length), "R_per_len": float(r_per), "X_per_len": float(x_per), "C_per_len": float(c_per),
            "R_full": r_full, "X_full": x_full, "C_full": c_full, "u_idx": u, "v_idx": v,
        })

    dss.Lines.First()
    while True:
        ln = dss.Lines.Name()
        dss.Circuit.SetActiveElement(f"Line.{ln}")
        busnames = dss.CktElement.BusNames()
        if len(busnames) < 2:
            if not dss.Lines.Next():
                break
            continue

        b1_full, b2_full = busnames[0], busnames[1]
        b1, phs1 = parse_bus_spec(b1_full)
        b2, phs2 = parse_bus_spec(b2_full)
        length = float(dss.Lines.Length())
        nph_line = int(dss.Lines.Phases())
        linecode = str(dss.Lines.LineCode()).strip()
        use_linecode = (linecode != "")
        if use_linecode:
            try:
                dss.LineCodes.Name(linecode)
                Rm = dss.LineCodes.Rmatrix()
                Xm = dss.LineCodes.Xmatrix()
                Cm = dss.LineCodes.Cmatrix()
                use_linecode = (len(Rm) > 0 and len(Xm) > 0)
            except Exception:
                use_linecode = False

        if use_linecode:
            Rraw = list_to_sq(Rm)
            Xraw = list_to_sq(Xm)
            Craw = list_to_sq(Cm) if len(Cm) > 0 else np.zeros_like(Rraw)
            kmat = Rraw.shape[0]
        else:
            r1 = float(dss.Lines.R1())
            x1 = float(dss.Lines.X1())
            Rraw = np.diag([r1, r1, r1])
            Xraw = np.diag([x1, x1, x1])
            Craw = np.zeros((3, 3), dtype=float)
            kmat = 3

        if len(phs1) == 0 or len(phs2) == 0:
            phs = list(range(1, min(3, nph_line) + 1))
        else:
            phs = sorted(list(set(phs1).intersection(set(phs2))))
            if len(phs) == 0:
                phs = sorted(list(set(phs1).union(set(phs2))))
        phs = [ph for ph in phs if ph in (1, 2, 3)]
        if len(phs) == 0:
            phs = list(range(1, min(3, nph_line) + 1))

        for ph in phs:
            if kmat >= 3:
                pos_local = ph - 1
            else:
                if ph not in phs:
                    continue
                pos_local = phs.index(ph)
            r_per = float(Rraw[pos_local, pos_local])
            x_per = float(Xraw[pos_local, pos_local])
            c_per = float(Craw[pos_local, pos_local])
            from_node = f"{b1}.{ph}"
            to_node = f"{b2}.{ph}"
            elem_name = f"Line.{ln}"
            _add_edge(from_node, to_node, b1, b2, ph, elem_name, linecode, nph_line, length, r_per, x_per, c_per)
            if make_bidirectional:
                _add_edge(to_node, from_node, b2, b1, ph, elem_name, linecode, nph_line, length, r_per, x_per, c_per)

        if not dss.Lines.Next():
            break

    dss.Transformers.First()
    while True:
        xname = dss.Transformers.Name()
        elem_name = f"Transformer.{xname}"
        dss.Circuit.SetActiveElement(elem_name)
        busnames = dss.CktElement.BusNames()
        if len(busnames) < 2:
            if not dss.Transformers.Next():
                break
            continue

        b1_full, b2_full = busnames[0], busnames[1]
        b1, phs1 = parse_bus_spec(b1_full)
        b2, phs2 = parse_bus_spec(b2_full)
        nph = int(dss.CktElement.NumPhases())
        xhl = float(dss.Transformers.Xhl())
        dss.Transformers.Wdg(1)
        kv1 = float(dss.Transformers.kV())
        kva1 = float(dss.Transformers.kVA())
        r1_pct = float(dss.Transformers.R())
        r2_pct = 0.0
        if dss.Transformers.NumWindings() >= 2:
            dss.Transformers.Wdg(2)
            r2_pct = float(dss.Transformers.R())
        r_total_pct = r1_pct + r2_pct
        if kv1 <= 0 or kva1 <= 0:
            if not dss.Transformers.Next():
                break
            continue
        z_base = (kv1 ** 2) / kva1
        r_ohms = (r_total_pct / 100.0) * z_base
        x_ohms = (xhl / 100.0) * z_base
        c_ohms = 0.0

        if len(phs1) == 0 or len(phs2) == 0:
            phs = list(range(1, min(3, nph) + 1))
        else:
            phs = sorted(list(set(phs1).intersection(set(phs2))))
            if len(phs) == 0:
                phs = sorted(list(set(phs1).union(set(phs2))))
        phs = [ph for ph in phs if ph in (1, 2, 3)]
        if len(phs) == 0:
            phs = list(range(1, min(3, nph) + 1))

        for ph in phs:
            from_node = f"{b1}.{ph}"
            to_node = f"{b2}.{ph}"
            _add_edge(from_node, to_node, b1, b2, ph, elem_name, "xfmr", nph, 1.0, r_ohms, x_ohms, c_ohms)
            if make_bidirectional:
                _add_edge(to_node, from_node, b2, b1, ph, elem_name, "xfmr", nph, 1.0, r_ohms, x_ohms, c_ohms)

        if not dss.Transformers.Next():
            break

    df_e = pd.DataFrame(rows)
    df_e.to_csv(edge_csv_path, index=False)
    print(f"[saved] phase-edge CSV -> {edge_csv_path} | rows={len(df_e)} | cols={df_e.shape[1]} | bidirectional={make_bidirectional}")
    return df_e

# ============================================================
# Voltage read (targets)
# ============================================================
def get_all_node_voltage_pu_and_angle_filtered(node_names_keep):
    buses = sorted(list({n.split(".")[0] for n in node_names_keep}))
    bus_cache = {}
    for bus in buses:
        dss.Circuit.SetActiveBus(bus)
        bus_nodes = list(dss.Bus.Nodes())
        if hasattr(dss.Bus, "puVmagAngle"):
            arr = list(dss.Bus.puVmagAngle())
            mags = arr[0::2]
            angs = arr[1::2]
            tmp = {int(nn): (float(m), float(a)) for nn, m, a in zip(bus_nodes, mags, angs)}
        else:
            arr = list(dss.Bus.VMagAngle())
            mags_v = arr[0::2]
            angs = arr[1::2]
            kvbase = float(dss.Bus.kVBase())
            if kvbase <= 0:
                tmp = {int(nn): (np.nan, np.nan) for nn in bus_nodes}
            else:
                vbase = kvbase * 1000.0
                tmp = {int(nn): (float(mv) / vbase, float(a)) for nn, mv, a in zip(bus_nodes, mags_v, angs)}
        bus_cache[bus] = {ph: tmp[ph] for ph in tmp.keys() if ph in (1, 2, 3)}

    vmag = []
    vang = []
    for n in node_names_keep:
        bus, phs = n.split(".")
        ph = int(phs)
        vm, va = bus_cache.get(bus, {}).get(ph, (np.nan, np.nan))
        vmag.append(float(vm))
        vang.append(float(va))
    return vmag, vang

# ============================================================
# Scenario sampling + noise
# ============================================================
def sample_scenario_from_baseline(baseline, ranges, rng):
    sc = dict(baseline)
    for k, (lo, hi) in ranges.items():
        f = float(rng.uniform(lo, hi))
        sc[k] = float(baseline[k]) * f
    sc["sigma_load"] = float(np.clip(sc["sigma_load"], 0.0, 0.5))
    sc["sigma_pv"] = float(np.clip(sc["sigma_pv"], 0.0, 0.5))
    return sc

def _noise_factor(rng, sigma):
    return max(0.0, 1.0 + float(rng.normal(0.0, sigma)))

# ============================================================
# Build device->DSS element mapping
# ============================================================
def _normalize_name(s: str) -> str:
    return str(s).strip().lower()

def build_load_device_maps(bus_to_phases):
    loads_dss = list(dss.Loads.AllNames())
    loads_dss_lut = {_normalize_name(n): n for n in loads_dss}
    dev_to_dss = {}
    missing = []
    for dev_key in DEVICE_P_SHARE.keys():
        k = _normalize_name(dev_key)
        if k in loads_dss_lut:
            dev_to_dss[k] = loads_dss_lut[k]
        else:
            missing.append(dev_key)
    if missing:
        print(f"[WARN] {len(missing)} load devices in DEVICE_P_SHARE not found in DSS Loads: {missing[:10]}{'...' if len(missing)>10 else ''}")
    dev_to_busph = {}
    for dev_key, lst in DEVICE_TO_BUSPH.items():
        dev_to_busph[_normalize_name(dev_key)] = [(str(b), int(ph), float(w)) for (b, ph, w) in lst]
    return loads_dss, dev_to_dss, dev_to_busph

def build_pv_device_maps():
    pv_dss = list(dss.PVsystems.AllNames())
    pv_dss_lut = {_normalize_name(n): n for n in pv_dss}
    pv_to_dss = {}
    missing = []
    for pv_key in PV_PMMP_SHARE.keys():
        k = _normalize_name(pv_key)
        if k in pv_dss_lut:
            pv_to_dss[k] = pv_dss_lut[k]
        else:
            missing.append(pv_key)
    if missing:
        print(f"[WARN] {len(missing)} PV devices in PV_PMMP_SHARE not found in DSS PVsystems: {missing}")
    pv_to_busph = {}
    for pv_key, lst in PV_TO_BUSPH.items():
        pv_to_busph[_normalize_name(pv_key)] = [(str(b), int(ph), float(w)) for (b, ph, w) in lst]
    return pv_dss, pv_to_dss, pv_to_busph

# ============================================================
# Apply ONE snapshot at time t
# ============================================================
def apply_snapshot_timeconditioned(
    P_load_total_kw, Q_load_total_kvar, P_pv_total_kw,
    mL_t, mPV_t,
    loads_dss, dev_to_dss_load, dev_to_busph_load,
    pv_dss, pv_to_dss, pv_to_busph,
    sigma_load, sigma_pv,
    rng
):
    P_load_t = float(P_load_total_kw) * float(mL_t)
    Q_load_t = float(Q_load_total_kvar) * float(mL_t)
    P_pv_t = float(P_pv_total_kw) * float(mPV_t)

    busphP_load = {}
    busphQ_load = {}

    for dev_key_raw in DEVICE_P_SHARE.keys():
        dev_key = _normalize_name(dev_key_raw)
        if dev_key not in dev_to_dss_load or dev_key not in dev_to_busph_load:
            continue
        p0 = P_load_t * float(DEVICE_P_SHARE.get(dev_key_raw, 0.0))
        q0 = Q_load_t * float(DEVICE_Q_SHARE.get(dev_key_raw, 0.0))
        fp = _noise_factor(rng, sigma_load)
        fq = _noise_factor(rng, sigma_load)
        p_set = float(p0 * fp)
        q_set = float(q0 * fq)
        ln = dev_to_dss_load[dev_key]
        dss.Loads.Name(ln)
        dss.Loads.kW(p_set)
        dss.Loads.kvar(q_set)
        for (bus, ph, w) in dev_to_busph_load[dev_key]:
            busphP_load[(bus, ph)] = busphP_load.get((bus, ph), 0.0) + p_set * w
            busphQ_load[(bus, ph)] = busphQ_load.get((bus, ph), 0.0) + q_set * w

    busphP_pv = {}
    busphQ_pv = {}
    for pv_key_raw in PV_PMMP_SHARE.keys():
        pv_key = _normalize_name(pv_key_raw)
        if pv_key not in pv_to_dss or pv_key not in pv_to_busph:
            continue
        # Pmpp is scaled by irradiance shape in OpenDSS; set base so output = P_pv_t * share * noise
        pmpp0 = P_pv_total_kw * float(PV_PMMP_SHARE.get(pv_key_raw, 0.0))
        f = _noise_factor(rng, sigma_pv)
        pmpp_set = float(pmpp0 * f)
        pvname = pv_to_dss[pv_key]
        dss.PVsystems.Name(pvname)
        dss.PVsystems.Pmpp(pmpp_set)
        p_nominal = pmpp_set * float(mPV_t)  # expected output at time t
        for (bus, ph, w) in pv_to_busph[pv_key]:
            busphP_pv[(bus, ph)] = busphP_pv.get((bus, ph), 0.0) + p_nominal * w
            busphQ_pv[(bus, ph)] = busphQ_pv.get((bus, ph), 0.0) + 0.0

    totals = dict(
        P_load_time_kw=P_load_t,
        Q_load_time_kvar=Q_load_t,
        P_pv_time_kw=P_pv_t,
        p_load_kw_set_total=float(sum(busphP_load.values())),
        q_load_kvar_set_total=float(sum(busphQ_load.values())),
        p_pv_pmpp_kw_set_total=float(sum(busphP_pv.values())),
    )
    return totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv

# ============================================================
# Injection dataset generation (P_inj, Q_inj with upstream grid + capacitors)
# ============================================================
def generate_gnn_snapshot_dataset_injection(
    n_scenarios=200,
    k_snapshots_per_scenario_total=960,
    bins_by_profile=None,
    include_anchors=True,
    master_seed=20260130,
    loadshape_name="5minDayShape",
    irradshape_name="IrradShape",
):
    if bins_by_profile is None:
        bins_by_profile = {"load": 10, "pv": 10, "net": 10}

    dss_path = compile_once()
    setup_daily()

    node_names_master, _, _, bus_to_phases_master = get_all_bus_phase_nodes()
    node_to_idx_master = {n: i for i, n in enumerate(node_names_master)}

    pd.DataFrame({"node": node_names_master, "node_idx": np.arange(len(node_names_master), dtype=int)}).to_csv(NODE_INDEX_CSV_INJ, index=False)
    print(f"[saved] master node index -> {NODE_INDEX_CSV_INJ} | N_nodes={len(node_names_master)}")

    df_edges = extract_static_phase_edges_to_csv(node_names_master=node_names_master, edge_csv_path=EDGE_CSV_INJ)

    csvL_token, lineL = find_loadshape_csv_in_dss(dss_path, loadshape_name)
    csvPV_token, linePV = find_loadshape_csv_in_dss(dss_path, irradshape_name)
    csvL = resolve_csv_path(csvL_token, dss_path)
    csvPV = resolve_csv_path(csvPV_token, dss_path)
    print("Loadshape line:", lineL)
    print("Irradshape line:", linePV)
    print("Resolved load CSV:", csvL)
    print("Resolved irrad CSV:", csvPV)
    mL = read_profile_csv_two_col_noheader(csvL, npts=NPTS, debug=False)
    mPV = read_profile_csv_two_col_noheader(csvPV, npts=NPTS, debug=False)

    rng_master = np.random.default_rng(master_seed)
    rows_sample = []
    rows_node = []
    sample_id = 0
    kept = 0
    skipped_nonconv = 0
    skipped_badV = 0

    for s in range(n_scenarios):
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{dss_path}"')
        _apply_voltage_bases()
        setup_daily()

        node_names_s, _, _, bus_to_phases = get_all_bus_phase_nodes()
        if len(node_names_s) != len(node_names_master):
            raise RuntimeError(f"Scenario {s}: node count mismatch")
        loads_dss, dev_to_dss_load, dev_to_busph_load = build_load_device_maps(bus_to_phases)
        pv_dss, pv_to_dss, pv_to_busph = build_pv_device_maps()

        sc = sample_scenario_from_baseline(BASELINE, RANGES, rng_master)
        P_load, Q_load, P_pv = sc["P_load_total_kw"], sc["Q_load_total_kvar"], sc["P_pv_total_kw"]
        sigL, sigPV = sc["sigma_load"], sc["sigma_pv"]
        prof_load, prof_pv = mL, mPV
        prof_net = (P_load * mL) - (P_pv * mPV)

        rng_times = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        times = select_times_three_profiles(
            prof_load=prof_load, prof_pv=prof_pv, prof_net=prof_net,
            K_total=k_snapshots_per_scenario_total, bins_by_profile=bins_by_profile,
            include_anchors=include_anchors, rng=rng_times
        )
        times = [int(t) for t in times]
        rng_solve = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        for t in times:
            set_time_index(t)
            totals, busphP_load, busphQ_load, busphP_pv, busphQ_pv = apply_snapshot_timeconditioned(
                P_load_total_kw=P_load, Q_load_total_kvar=Q_load, P_pv_total_kw=P_pv,
                mL_t=float(mL[t]), mPV_t=float(mPV[t]),
                loads_dss=loads_dss, dev_to_dss_load=dev_to_dss_load, dev_to_busph_load=dev_to_busph_load,
                pv_dss=pv_dss, pv_to_dss=pv_to_dss, pv_to_busph=pv_to_busph,
                sigma_load=sigL, sigma_pv=sigPV, rng=rng_solve
            )

            dss.Solution.Solve()
            if not dss.Solution.Converged():
                skipped_nonconv += 1
                continue

            vmag_m, vang_m = get_all_node_voltage_pu_and_angle_filtered(node_names_master)
            vmag_arr = np.asarray(vmag_m, dtype=float)
            if (not np.isfinite(vmag_arr).all()) or (vmag_arr.min() < VMAG_PU_MIN) or (vmag_arr.max() > VMAG_PU_MAX):
                skipped_badV += 1
                continue

            vdict_m = {n: (float(vm), float(va)) for n, vm, va in zip(node_names_master, vmag_m, vang_m)}

            # Upstream grid P and Q (TotalPower returns negative when circuit draws; we want positive = injection)
            pwr = dss.Circuit.TotalPower()
            P_grid = -float(pwr[0])
            Q_grid = -float(pwr[1])
            P_grid_per_ph = P_grid / 3.0
            Q_grid_per_ph = Q_grid / 3.0

            rows_sample.append({
                "sample_id": sample_id, "scenario_id": s, "t_index": t, "t_minutes": t * STEP_MIN,
                "P_load_total_kw": float(P_load), "Q_load_total_kvar": float(Q_load), "P_pv_total_kw": float(P_pv),
                "sigma_load": float(sigL), "sigma_pv": float(sigPV),
                "m_loadshape": float(mL[t]), "m_irradshape": float(mPV[t]),
                "P_load_time_kw": float(totals["P_load_time_kw"]), "Q_load_time_kvar": float(totals["Q_load_time_kvar"]),
                "P_pv_time_kw": float(totals["P_pv_time_kw"]),
                "p_load_kw_set_total": float(totals["p_load_kw_set_total"]),
                "q_load_kvar_set_total": float(totals["q_load_kvar_set_total"]),
                "p_pv_pmpp_kw_set_total": float(totals["p_pv_pmpp_kw_set_total"]),
                "prof_load": float(prof_load[t]), "prof_pv": float(prof_pv[t]), "prof_net": float(prof_net[t]),
                "P_grid_kw": P_grid, "Q_grid_kvar": Q_grid,
            })

            for n in node_names_master:
                bus, phs = n.split(".")
                ph = int(phs)
                p_load_node = float(busphP_load.get((bus, ph), 0.0))
                q_load_node = float(busphQ_load.get((bus, ph), 0.0))
                p_pv_node = float(busphP_pv.get((bus, ph), 0.0))
                vm, va = vdict_m.get(n, (np.nan, np.nan))

                if bus == "sourcebus":
                    p_inj = P_grid_per_ph
                    q_inj = Q_grid_per_ph
                else:
                    p_inj = p_pv_node - p_load_node
                    q_inj = -q_load_node + float(CAP_Q_KVAR.get(bus, 0.0))

                rows_node.append({
                    "sample_id": sample_id, "node": n, "node_idx": int(node_to_idx_master[n]),
                    "bus": bus, "phase": int(ph),
                    "p_inj_kw": p_inj, "q_inj_kvar": q_inj,
                    "vmag_pu": float(vm), "vang_deg": float(va),
                })

            sample_id += 1
            kept += 1

        print(f"[scenario {s+1}/{n_scenarios}] kept={kept} skip_nonconv={skipped_nonconv} skip_badV={skipped_badV} Pload={P_load:.1f} Qload={Q_load:.1f} Ppv={P_pv:.1f}")

    df_sample = pd.DataFrame(rows_sample)
    df_node = pd.DataFrame(rows_node)
    df_sample.to_csv(SAMPLE_CSV_INJ, index=False)
    df_node.to_csv(NODE_CSV_INJ, index=False)

    print(f"\n[INJECTION DATASET] Saved to {OUT_DIR_INJ}/")
    print(f"  {NODE_CSV_INJ} | samples={df_sample['sample_id'].nunique()} | node-rows={len(df_node)}")
    print(f"  Skipped: nonconv={skipped_nonconv} badV={skipped_badV}")
    return df_edges, df_sample, df_node


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    df_edges_inj, df_samples_inj, df_nodes_inj = generate_gnn_snapshot_dataset_injection(
        n_scenarios=200,
        k_snapshots_per_scenario_total=960,
        bins_by_profile={"load": 10, "pv": 10, "net": 10},
        include_anchors=True,
        master_seed=20260130,
    )
