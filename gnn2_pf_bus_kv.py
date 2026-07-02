"""Per-bus OpenDSS kVBase (line-to-neutral) for physical-units physics loss.

OpenDSS convention (matches dataset ``vmag_pu`` export):
- ``dss.Bus.kVBase()`` returns **line-to-neutral** kV for the active bus.
- ``vmag_pu`` in hetero/dailyagg CSVs is ``|V| / (kVBase_LN * 1000)`` volts.
- Physical nodal power: ``S_VA = V_volts * conj(Y_siemens @ V_volts)``,
  ``P_kW = Re(S_VA) / 1000``, ``Q_kvar = Im(S_VA) / 1000``.

Cached under the PF data root as ``bus_kv_base_by_node.csv`` to avoid recompiling
OpenDSS on every training start.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
BUS_KV_CACHE_NAME = "bus_kv_base_by_node.csv"
DEFAULT_KV_FALLBACK_LN = 7.1996  # 12.47 kV LL / sqrt(3) when bus lookup fails


def node_bus_name(node: str) -> str:
    """OpenDSS bus name (no phase suffix) from a phased node label."""
    return str(node).strip().lower().split(".")[0]


def kv_ln_to_volts(kv_base_ln: float) -> float:
    return float(kv_base_ln) * 1000.0


def resolve_opendss_master(repo: Path | None = None) -> Path:
    """Return the PV/unbalanced master used for dailyagg generation."""
    repo = (repo or REPO).resolve()
    candidates = [
        repo / "8500 nodes with solar unbalanced" / "Master-PV2MW-inv.dss",
        repo / "8500-node" / "Master.dss",
        repo / "8500-node" / "Master-unbal.dss",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "OpenDSS master not found. Expected one of:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def _compile_opendss_master(master_dss: Path) -> None:
    import opendssdirect as dss

    master_dss = master_dss.resolve()
    if not master_dss.is_file():
        raise FileNotFoundError(f"Missing DSS master: {master_dss}")
    grid_dir = master_dss.parent
    prev = os.getcwd()
    try:
        os.chdir(grid_dir)
        dss.Basic.ClearAll()
        dss.Text.Command(f'redirect "{master_dss}"')
        dss.Text.Command("set mode=daily")
    finally:
        os.chdir(prev)


def _bus_kv_map_all_buses(master_dss: Path) -> dict[str, float]:
    """``bus_name.lower() -> kVBase LN`` for every bus in the compiled circuit."""
    import opendssdirect as dss

    _compile_opendss_master(master_dss)
    out: dict[str, float] = {}
    for b in dss.Circuit.AllBusNames():
        bus = str(b).strip().lower()
        try:
            dss.Circuit.SetActiveBus(bus)
            out[bus] = float(dss.Bus.kVBase())
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"No buses read from OpenDSS master {master_dss}")
    return out


def load_bus_kv_base_map(
    master_dss: Path,
    node_index_csv: Path,
    *,
    mapping_csv: Path | None = None,
) -> dict[str, float]:
    """``phased_node.lower() -> kVBase LN`` for nodes in the index CSV.

    Looks up ``node_bus_name(node)`` in the OpenDSS bus map. Optional
    ``mv_x_sx_node_mapping_8500.csv`` aliases uppercase MV labels to SX bus names
    when the direct bus name is absent.
    """
    import pandas as pd

    bus_kv = _bus_kv_map_all_buses(master_dss)
    alias: dict[str, str] = {}
    if mapping_csv is not None and mapping_csv.is_file():
        mdf = pd.read_csv(mapping_csv)
        mv_col = next((c for c in mdf.columns if str(c).strip().lower() == "mv_node"), None)
        sx_cols = [c for c in mdf.columns if str(c).strip().lower().startswith("sx_node")]
        if mv_col and sx_cols:
            for _, row in mdf.iterrows():
                mv = str(row[mv_col]).strip().lower()
                bus_mv = node_bus_name(mv)
                for sc in sx_cols:
                    sx = str(row[sc]).strip().lower()
                    if sx and sx != "nan":
                        alias[bus_mv] = node_bus_name(sx)
                        break

    idx = pd.read_csv(node_index_csv, usecols=["node"])
    out: dict[str, float] = {}
    missing: list[str] = []
    for node in idx["node"].astype(str):
        key = str(node).strip().lower()
        bus = node_bus_name(key)
        kv = bus_kv.get(bus)
        if kv is None and bus in alias:
            kv = bus_kv.get(alias[bus])
        if kv is None or not np.isfinite(kv) or kv <= 0:
            missing.append(key)
            kv = DEFAULT_KV_FALLBACK_LN
        out[key] = float(kv)
    if missing:
        print(
            f"WARNING: kVBase missing for {len(missing)} node(s); using "
            f"{DEFAULT_KV_FALLBACK_LN} kV LN fallback. First few: {missing[:5]}",
            flush=True,
        )
    return out


def kv_base_ln_v_array(
    node_to_local: dict[str, int],
    kv_by_node: dict[str, float],
    n_nodes: int,
) -> np.ndarray:
    """Per local index line-to-neutral base voltage in volts."""
    arr = np.full(int(n_nodes), kv_ln_to_volts(DEFAULT_KV_FALLBACK_LN), dtype=np.float64)
    for node, li in node_to_local.items():
        kv = kv_by_node.get(str(node).strip().lower())
        if kv is not None and np.isfinite(kv) and kv > 0:
            arr[int(li)] = kv_ln_to_volts(kv)
    return arr


def write_bus_kv_cache(
    cache_csv: Path,
    node_to_local: dict[str, int],
    kv_by_node: dict[str, float],
    v_scale: np.ndarray,
) -> None:
    import pandas as pd

    rows = []
    for node, li in sorted(node_to_local.items(), key=lambda kv: int(kv[1])):
        key = str(node).strip().lower()
        kv_ln = float(kv_by_node.get(key, DEFAULT_KV_FALLBACK_LN))
        rows.append(
            {
                "node": key,
                "node_idx": int(li),
                "bus": node_bus_name(key),
                "kv_base_ln": kv_ln,
                "v_base_ln_volts": float(v_scale[int(li)]),
            }
        )
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache_csv, index=False)


def read_bus_kv_cache(cache_csv: Path) -> tuple[dict[str, float], np.ndarray]:
    import pandas as pd

    df = pd.read_csv(cache_csv)
    kv_by_node = {
        str(r["node"]).strip().lower(): float(r["kv_base_ln"]) for _, r in df.iterrows()
    }
    n_nodes = int(df["node_idx"].max()) + 1
    v_scale = np.full(n_nodes, kv_ln_to_volts(DEFAULT_KV_FALLBACK_LN), dtype=np.float64)
    for _, r in df.iterrows():
        li = int(r["node_idx"])
        v_scale[li] = float(r["v_base_ln_volts"])
    return kv_by_node, v_scale


def load_or_build_bus_kv_tensors(
    *,
    repo: Path,
    data_root: Path,
    node_to_local: dict[str, int],
    n_nodes: int,
    node_index_csv: Path | None = None,
    master_dss: Path | None = None,
    cache_csv: Path | None = None,
    mapping_csv: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, float], Path]:
    """Return ``(v_scale_volts[n_nodes], kv_by_node, cache_path)``."""
    data_root = data_root.resolve()
    cache_path = (cache_csv or (data_root / BUS_KV_CACHE_NAME)).resolve()
    idx_csv = (node_index_csv or (data_root / "gnn_node_index_master.csv")).resolve()
    if cache_path.is_file() and not force_rebuild:
        kv_by_node, v_scale = read_bus_kv_cache(cache_path)
        if v_scale.shape[0] >= int(n_nodes):
            return v_scale[: int(n_nodes)].copy(), kv_by_node, cache_path
    master = master_dss or resolve_opendss_master(repo)
    map_csv = mapping_csv
    if map_csv is None:
        cand = data_root / "mv_x_sx_node_mapping_8500.csv"
        if cand.is_file():
            map_csv = cand
    kv_by_node = load_bus_kv_base_map(master, idx_csv, mapping_csv=map_csv)
    v_scale = kv_base_ln_v_array(node_to_local, kv_by_node, n_nodes)
    write_bus_kv_cache(cache_path, node_to_local, kv_by_node, v_scale)
    print(f"Wrote per-bus kVBase cache: {cache_path}", flush=True)
    return v_scale, kv_by_node, cache_path
