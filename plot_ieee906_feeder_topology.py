#!/usr/bin/env python3
"""IEEE 906 European LV Test Case geographic topology plot.

Default style:

  - Geographic PES coordinates from ``Buscoords.txt`` + ``Lines.txt``
  - Line thickness ∝ **subtree-max** conductor ampacity (capacity rating).
    LVTestCase sizes TR1→LINE1 as ``4c_70`` while thicker cables exist
    downstream; display ampacity is the max in each radial subtree so the
    feeder from the substation stays thick. LineCode.txt has no NormAmps —
    size-based table.
  - All buses visible: solid grey zero-injection dots, solid black load dots
  - Substation / transformer icon from ``outputs/Icons/substation.svg``
    (scale with ``icon_size_scale`` / ``device_icon_zooms['substation']``)

Optional ``taper_line_width=True`` switches to 8500-like **loading**/power taper.

Example::

  python plot_ieee906_feeder_topology.py \\
      --out-dir outputs/ieee906_topology_paper
"""
from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import plot_ieee8500_feeder_topology as p8500

REPO = Path(__file__).resolve().parent
DEFAULT_DSS_DIR = (
    REPO
    / "906 bus system"
    / "OpenDSS-master"
    / "OpenDSS-master"
    / "Distrib"
    / "IEEETestCases"
    / "LVTestCase"
)
DEFAULT_OUT_DIR = REPO / "outputs" / "ieee906_topology_paper"

# Geographic layout palette aligned with IEEE 8500 paper plot
# (``plot_ieee8500_feeder_topology.PAPER_BACKBONE_COLOR`` / bus / load markers).
PLAIN_LINE_COLOR = "#163d63"  # == p8500.PAPER_BACKBONE_COLOR
# Used only as fallback when a linecode has no ampacity entry.
PLAIN_LINE_WIDTH = 3.0
PLAIN_LINE_ALPHA = 1.0
# Ampacity → display width (pt). Keeps nodes readable above thick cables.
AMPACITY_THICKNESS_MIN = 1.15
AMPACITY_THICKNESS_MAX = 5.5
# Zero-injection / junction buses: solid grey disks (no halo).
PLAIN_JUNCTION_COLOR = "#8a8a8a"
PLAIN_JUNCTION_SIZE = 55.0 * 0.75  # scatter ``s`` (points^2); 0.75× prior size
# Injection / load buses: same marker, black fill (no outline).
PLAIN_LOAD_FACE = "#000000"
PLAIN_LOAD_EDGE = "none"
PLAIN_LOAD_SIZE = 95.0
PLAIN_SOURCE_COLOR = "#000000"

# European LV: Master.dss circuit-plot comment uses Max=30.
PAPER_POWER_MAX_DEFAULT = 30.0
# Service laterals (2-core / customer drops) stay thinner than tapered mains.
LATERAL_LINECODE_PREFIXES = ("2c_",)
# Conductor R1 (ohm/km) from LineCode.txt — lower R ⇒ thicker cable.
LINECODE_R1: dict[str, float] = {
    "4c_.35": 0.089,
    "4c_185": 0.166,
    "4c_.1": 0.274,
    "4c_95_sac_xc": 0.322,
    "4c_70": 0.446,
    "4c_.06": 0.469,
    "35_sac_xsc": 0.868,
    "2c_16": 1.15,
    "2c_.0225": 1.257,
    "2c_.007": 3.97,
}
# Estimated continuous ampacity (A). LVTestCase LineCode.txt has no NormAmps;
# values follow typical XLPE Al buried ratings by cross-section in the code name.
LINECODE_AMPACITY_A: dict[str, float] = {
    "4c_.35": 430.0,  # ~350 mm²
    "4c_185": 310.0,  # 185 mm²
    "4c_.1": 220.0,  # ~100 mm²
    "4c_95_sac_xc": 205.0,  # 95 mm²
    "4c_70": 170.0,  # 70 mm²
    "4c_.06": 155.0,  # ~60 mm²
    "35_sac_xsc": 115.0,  # 35 mm²
    "2c_16": 75.0,  # 16 mm²
    "2c_.0225": 28.0,  # ~2.25 mm² service
    "2c_.007": 15.0,  # ~0.7 mm² service
}
# Fixed width floor/ceiling by linecode class when not using power taper alone.
LINECODE_LW_SCALE: dict[str, float] = {
    "4c_.35": 1.00,
    "4c_185": 0.92,
    "4c_.1": 0.82,
    "4c_95_sac_xc": 0.75,
    "4c_70": 0.68,
    "4c_.06": 0.62,
    "35_sac_xsc": 0.48,
    "2c_16": 0.38,
    "2c_.0225": 0.32,
    "2c_.007": 0.28,
}


def _base_bus(bus: str | None) -> str | None:
    return p8500.base_bus(bus)


def _get_param(command: str, key: str) -> str | None:
    return p8500.get_param(command, key)


def _parse_object_name(command: str, objtype: str) -> str | None:
    return p8500.parse_object_name(command, objtype)


def load_buscoords_txt(dss_dir: Path) -> dict[str, tuple[float, float]]:
    """Load ``Buscoords.txt`` (or ``Buscoords.dss``) with bus, x, y columns."""
    candidates = [
        dss_dir / "Buscoords.txt",
        dss_dir / "buscoords.txt",
        dss_dir / "BusCoords.txt",
        dss_dir / "Buscoords.dss",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No Buscoords.txt under {dss_dir}")
    coords: dict[str, tuple[float, float]] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("!") or line.startswith("//"):
            continue
        line = re.split(r"\s!|^!|//", line, maxsplit=1)[0].strip().replace(",", " ")
        parts = line.split()
        if len(parts) < 3:
            continue
        bus = _base_bus(parts[0])
        try:
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if bus:
            coords[p8500.canonical_bus(bus)] = (x, y)
    if not coords:
        raise RuntimeError(f"No coordinates parsed from {path}")
    return coords


def _read_new_commands(path: Path, objtype: str) -> list[str]:
    if not path.is_file():
        return []
    text = p8500.strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    cmds = p8500.join_continuations(text)
    prefix = f"new {objtype}."
    return [c for c in cmds if c.lower().startswith(prefix)]


def parse_linecodes(dss_dir: Path) -> dict[str, float]:
    """Return linecode → R1 (ohm/km). Falls back to built-in table."""
    r1 = dict(LINECODE_R1)
    for cmd in _read_new_commands(dss_dir / "LineCode.txt", "linecode"):
        name = (_parse_object_name(cmd, "linecode") or "").lower()
        if not name:
            continue
        raw = _get_param(cmd, "r1")
        if raw is None:
            continue
        try:
            r1[name] = float(raw)
        except ValueError:
            continue
    return r1


def parse_lines(dss_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cmd in _read_new_commands(dss_dir / "Lines.txt", "line"):
        name = _parse_object_name(cmd, "line")
        b1 = _base_bus(_get_param(cmd, "bus1"))
        b2 = _base_bus(_get_param(cmd, "bus2"))
        if not name or not b1 or not b2:
            continue
        phases_raw = _get_param(cmd, "phases") or "3"
        try:
            phases = int(float(phases_raw))
        except ValueError:
            phases = 3
        lc = (_get_param(cmd, "linecode") or "").lower()
        length_raw = _get_param(cmd, "length")
        try:
            length = float(length_raw) if length_raw is not None else None
        except ValueError:
            length = None
        units = (_get_param(cmd, "units") or "m").lower()
        is_lateral = any(lc.startswith(p) for p in LATERAL_LINECODE_PREFIXES)
        rows.append(
            {
                "name": name,
                "bus1": p8500.canonical_bus(b1),
                "bus2": p8500.canonical_bus(b2),
                "phases": phases,
                "linecode": lc,
                "length": length,
                "units": units,
                "is_lateral": is_lateral,
            }
        )
    return rows


def parse_loads(dss_dir: Path) -> set[str]:
    buses: set[str] = set()
    for cmd in _read_new_commands(dss_dir / "Loads.txt", "load"):
        b1 = _base_bus(_get_param(cmd, "bus1"))
        if b1:
            buses.add(p8500.canonical_bus(b1))
    return buses


def _iter_pvsystem_commands(dss_dir: Path) -> list[str]:
    """Collect ``New PVSystem.*`` from stock files and PV add-on DSS scripts."""
    paths: list[Path] = [
        dss_dir / "PVSystems.txt",
        dss_dir / "Generators.txt",
    ]
    # Volt-Var / DER add-ons (e.g. LVTestCase_PV_voltvar/PV_voltvar_906.dss)
    paths.extend(sorted(dss_dir.glob("*PV*.dss")))
    paths.extend(sorted(dss_dir.glob("*pv*.dss")))
    seen: set[Path] = set()
    cmds: list[str] = []
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        cmds.extend(_read_new_commands(path, "pvsystem"))
    return cmds


def parse_devices(dss_dir: Path) -> dict[str, Any]:
    """Inventory devices present in the LVTestCase folder."""
    transformers: list[dict[str, Any]] = []
    for cmd in _read_new_commands(dss_dir / "Transformers.txt", "transformer"):
        name = _parse_object_name(cmd, "transformer") or ""
        buses_raw = _get_param(cmd, "buses")
        bus_list = p8500.parse_bus_list(buses_raw) if buses_raw else []
        bus_bases = [_base_bus(b) for b in bus_list]
        bus_bases = [p8500.canonical_bus(b) for b in bus_bases if b]
        sub = (_get_param(cmd, "sub") or "").lower() in {"y", "yes", "true"}
        transformers.append({"name": name, "buses": bus_bases, "sub": sub, "cmd": cmd})

    caps = _read_new_commands(dss_dir / "Capacitors.txt", "capacitor")
    regs = []
    for path_name in ("RegControls.txt", "RegControl.txt", "Controls.txt"):
        regs.extend(_read_new_commands(dss_dir / path_name, "regcontrol"))
    pvs = _iter_pvsystem_commands(dss_dir)
    # Generators.txt in stock LVTestCase only edits Vsource — treat as no PV.
    gens = [
        c
        for c in _read_new_commands(dss_dir / "Generators.txt", "generator")
        if "pv" in c.lower() or "solar" in c.lower()
    ]
    storage = _read_new_commands(dss_dir / "Storage.txt", "storage")

    pv_buses: dict[str, str] = {}
    for cmd in list(pvs) + list(gens):
        obj = "pvsystem" if "pvsystem." in cmd.lower() else "generator"
        name = _parse_object_name(cmd, obj)
        b = _base_bus(_get_param(cmd, "bus1")) or _base_bus(_get_param(cmd, "bus"))
        if name and b:
            pv_buses[name] = p8500.canonical_bus(b)

    sub_buses: list[str] = []
    for xf in transformers:
        if xf["sub"] and xf["buses"]:
            # Prefer LV secondary (second bus) for icon placement on the LV map.
            sub_buses.append(xf["buses"][-1] if len(xf["buses"]) >= 2 else xf["buses"][0])
        elif xf["buses"] and not sub_buses:
            sub_buses.append(xf["buses"][-1] if len(xf["buses"]) >= 2 else xf["buses"][0])

    return {
        "transformers": transformers,
        "substation_buses": sub_buses,
        "pv_buses": pv_buses,
        "n_capacitors": len(caps),
        "n_regulators": len(regs),
        "n_pv": len(pv_buses),
        "n_storage": len(storage),
        "has_capacitor": bool(caps),
        "has_regulator": bool(regs),
        "has_pv": bool(pv_buses),
        "has_storage": bool(storage),
        "has_substation": bool(sub_buses or transformers),
    }


def solve_lvtestcase_line_powers_kw(
    dss_dir: Path,
    *,
    master_name: str = "Master_snapshot.dss",
) -> dict[str, float]:
    """Solve LVTestCase snapshot and return |P| kW per Line."""
    try:
        import opendssdirect as dss
    except ImportError as exc:
        raise RuntimeError("opendssdirect is required for OpenDSS line powers") from exc

    master_path = dss_dir / master_name
    if not master_path.is_file():
        # Fall back to Master.dss but force snapshot after compile.
        master_name = "Master.dss"
        master_path = dss_dir / master_name
    if not master_path.is_file():
        raise FileNotFoundError(f"OpenDSS master not found under {dss_dir}")

    prev_cwd = os.getcwd()
    try:
        os.chdir(dss_dir)
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{master_name}"')
        if master_name.lower() != "master_snapshot.dss":
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Set ControlMode=OFF")
            dss.Text.Command("Set Number=1")
            dss.Text.Command("Set Hour=0")
            dss.Text.Command("Set Sec=0")
            dss.Text.Command("Set LoadMult=1")
            dss.Text.Command("solve")
        powers: dict[str, float] = {}
        dss.Lines.First()
        while True:
            name = dss.Lines.Name().lower()
            dss.Circuit.SetActiveElement(f"Line.{dss.Lines.Name()}")
            pwr = dss.CktElement.TotalPowers()
            powers[name] = abs(float(pwr[0])) if pwr else 0.0
            if not dss.Lines.Next():
                break
        return powers
    finally:
        os.chdir(prev_cwd)


def _linecode_lw_scale(linecode: str) -> float:
    key = linecode.lower()
    if key in LINECODE_LW_SCALE:
        return LINECODE_LW_SCALE[key]
    # Unknown: scale from R1 if known, else mid.
    r1 = LINECODE_R1.get(key)
    if r1 is None:
        return 0.55
    # Map R1 in [0.089, 3.97] → scale in [1.0, 0.28]
    r_lo, r_hi = 0.089, 3.97
    t = (math.log(max(r1, r_lo)) - math.log(r_lo)) / (math.log(r_hi) - math.log(r_lo))
    return 1.0 - 0.72 * min(1.0, max(0.0, t))


def _linecode_ampacity_a(linecode: str | None) -> float:
    """Estimated ampacity (A) for a LVTestCase linecode (capacity, not loading)."""
    key = (linecode or "").strip().lower()
    if key in LINECODE_AMPACITY_A:
        return float(LINECODE_AMPACITY_A[key])
    # Fallback: larger ampacity for lower R1 (fatter conductor).
    r1 = LINECODE_R1.get(key)
    if r1 is not None and r1 > 0:
        # Map R1≈0.089→430 A, R1≈3.97→15 A (same span as table extremes).
        r_lo, r_hi = 0.089, 3.97
        a_hi, a_lo = 430.0, 15.0
        t = (math.log(max(r1, r_lo)) - math.log(r_lo)) / (math.log(r_hi) - math.log(r_lo))
        t = min(1.0, max(0.0, t))
        return a_hi + (a_lo - a_hi) * t
    return 100.0


def _linewidth_from_ampacity(
    linecode: str | None,
    *,
    thickness_min: float = AMPACITY_THICKNESS_MIN,
    thickness_max: float = AMPACITY_THICKNESS_MAX,
    ampacity_min: float | None = None,
    ampacity_max: float | None = None,
    ampacity_a: float | None = None,
) -> float:
    """Line width ∝ conductor ampacity (rating), not operating loading.

    Pass ``ampacity_a`` to override the linecode table (e.g. subtree-max ampacity).
    """
    a = float(ampacity_a) if ampacity_a is not None else _linecode_ampacity_a(linecode)
    a_min = float(ampacity_min) if ampacity_min is not None else min(LINECODE_AMPACITY_A.values())
    a_max = float(ampacity_max) if ampacity_max is not None else max(LINECODE_AMPACITY_A.values())
    if a_max <= a_min:
        return 0.5 * (thickness_min + thickness_max)
    t = (a - a_min) / (a_max - a_min)
    t = min(1.0, max(0.0, t))
    return float(thickness_min + (thickness_max - thickness_min) * t)


def compute_line_subtree_max_ampacity(
    lines: list[dict[str, Any]],
    root_bus: str,
) -> dict[str, float]:
    """Per-line display ampacity = max(own, max ampacity in the radial subtree).

    LVTestCase sizes the first feeder out of TR1 as ``4c_70``, while thicker
    ``4c_185`` / ``4c_.35`` segments appear further downstream. Local ampacity
    alone therefore draws the substation feeder thinner than some later mains.
    Using the subtree maximum restores the expected thick root corridor while
    keeping leaf laterals thin.
    """
    root = p8500.canonical_bus(root_bus)
    adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    own_a: dict[str, float] = {}
    for row in lines:
        name = str(row["name"]).lower()
        b1, b2 = str(row["bus1"]), str(row["bus2"])
        a = _linecode_ampacity_a(row.get("linecode"))
        own_a[name] = a
        adj[b1].append((b2, name, a))
        adj[b2].append((b1, name, a))

    parent: dict[str, str | None] = {root: None}
    # child_bus -> (parent_bus, line_name, own_ampacity)
    oriented: dict[str, tuple[str, str, float]] = {}
    q: deque[str] = deque([root])
    while q:
        u = q.popleft()
        for v, name, a in adj.get(u, []):
            if v in parent:
                continue
            parent[v] = u
            oriented[v] = (u, name, a)
            q.append(v)

    children: dict[str, list[str]] = defaultdict(list)
    for child, (par, _name, _a) in oriented.items():
        children[par].append(child)

    subtree_max: dict[str, float] = {}

    def _dfs(u: str) -> float:
        best = 0.0
        for v in children.get(u, []):
            _par, name, a = oriented[v]
            down = _dfs(v)
            val = max(a, down)
            subtree_max[name] = val
            if val > best:
                best = val
        return best

    _dfs(root)

    # Lines not reached from root (disconnected) keep own ampacity.
    out = dict(own_a)
    out.update(subtree_max)
    return out


def _linewidth_combined(
    *,
    quantity: float,
    quantity_max: float,
    linecode: str,
    is_lateral: bool,
    thickness_max: float,
    thickness_min: float,
) -> float:
    """OpenDSS-style taper modulated by conductor class; laterals stay thin."""
    if is_lateral:
        scale = _linecode_lw_scale(linecode)
        return max(0.35, min(1.15, thickness_min + (1.15 - thickness_min) * scale))
    power_lw = p8500.linewidth_opendss(
        quantity,
        quantity_max=quantity_max,
        thickness_max=thickness_max,
        thickness_min=thickness_min,
    )
    scale = _linecode_lw_scale(linecode)
    # Blend power taper with conductor class so thick cables near source stay bold.
    return max(thickness_min, power_lw * (0.55 + 0.45 * scale))


def plot_ieee906_feeder_topology(
    *,
    dss_dir: Path | str = DEFAULT_DSS_DIR,
    out_dir: Path | str | None = None,
    out_basename: str = "ieee906_lvtestcase_topology",
    style: Literal["paper", "draft"] = "paper",
    show_loads: bool = True,
    show_junctions: bool = True,
    show_source_marker: bool = True,
    show_device_icons: bool = True,
    taper_line_width: bool = False,
    use_opendss_power: bool = True,
    master_name: str = "Master_snapshot.dss",
    label_font_size: int = 9,
    figsize: tuple[float, float] | None = None,
    png_dpi: int = 600,
    device_icon_paths: dict[str, str | Path] | None = None,
    device_icon_zooms: dict[str, float] | None = None,
    icon_size_scale: float = 2.0,
    svg_icon_raster_scale: float = p8500.SVG_ICON_RASTER_SCALE,
    svg_savefig_dpi: int = p8500.SVG_SAVEFIG_DPI_DEFAULT,
    vector_svg_icons: bool | None = None,
    svg_to_pdf: bool | None = None,
    reload_icons: bool = True,
    visio_friendly_svg: bool = False,
    visio_preserve_appearance: bool = True,
    show_title: bool = False,
    show_legend: bool = False,
    crop_margins: bool = True,
    show: bool = False,
) -> dict[str, Path]:
    """Parse LVTestCase and save a paper-ready geographic topology figure."""
    if reload_icons:
        p8500.clear_icon_caches()

    dss_dir = Path(dss_dir).resolve()
    out_root = Path(out_dir).resolve() if out_dir is not None else DEFAULT_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    coords = load_buscoords_txt(dss_dir)
    lines = parse_lines(dss_dir)
    load_buses = parse_loads(dss_dir)
    devices = parse_devices(dss_dir)
    linecodes_r1 = parse_linecodes(dss_dir)
    for row in lines:
        if row["linecode"] and row["linecode"] not in LINECODE_R1 and row["linecode"] in linecodes_r1:
            # Keep local table in sync for unknown codes discovered in LineCode.txt
            LINECODE_R1[row["linecode"]] = linecodes_r1[row["linecode"]]

    # Ensure substation bus is plottable: LVTestCase SourceBus has no coords.
    # Plain geographic plot marked source bus "1"; prefer that when present.
    sub_bus = None
    for cand in ["1"] + list(devices["substation_buses"]):
        if cand in coords:
            sub_bus = cand
            break
    if sub_bus is None and coords:
        # Closest to LINE1 bus1 if present
        for row in lines:
            if row["name"].lower() == "line1" and row["bus1"] in coords:
                sub_bus = row["bus1"]
                break
        if sub_bus is None:
            sub_bus = next(iter(coords))

    edges = [(r["bus1"], r["bus2"]) for r in lines if r["bus1"] in coords and r["bus2"] in coords]
    lc_counts = Counter(r["linecode"] or "?" for r in lines)
    n_lateral = sum(1 for r in lines if r["is_lateral"])
    n_main = len(lines) - n_lateral

    load_with_xy = {b for b in load_buses if b in coords}
    junction_buses = sorted(b for b in coords if b not in load_with_xy)
    # Keep source/substation out of the junction layer so the icon/square is clear.
    if sub_bus is not None:
        junction_buses = [b for b in junction_buses if b != sub_bus]

    print("Parsed IEEE 906 LVTestCase model:")
    print(f"  DSS folder:             {dss_dir}")
    print(f"  Style:                  {style}")
    print(f"  Bus coordinates:        {len(coords):,}")
    print(f"  Line segments:          {len(lines):,}  (drawable {len(edges):,})")
    print(f"    main / trunk cables:  {n_main:,}")
    print(f"    service laterals:     {n_lateral:,}  (2c_* linecodes)")
    print(f"  Linecodes:              {dict(lc_counts)}")
    print(f"  Load buses:             {len(load_buses):,}  (with coords: {len(load_with_xy):,})")
    print(f"  Junction buses:         {len(junction_buses):,}")
    print(f"  Transformers:           {len(devices['transformers']):,}")
    for xf in devices["transformers"]:
        print(f"    - {xf['name']} buses={xf['buses']} sub={xf['sub']}")
    print(f"  Substation icon bus:    {sub_bus}")
    print(f"  Regulators:             {devices['n_regulators']}")
    print(f"  Capacitors:             {devices['n_capacitors']}")
    print(f"  PV / storage:           {devices['n_pv']} / {devices['n_storage']}")
    for pv_name, pv_bus in sorted((devices.get("pv_buses") or {}).items()):
        print(f"    - PVSystem.{pv_name} @ bus {pv_bus}")
    print(f"  Line-width mode:        {'power taper' if taper_line_width else 'ampacity (capacity)'}")
    print("  Monitored-bus icons:    disabled (none drawn)")

    # --- line quantities for taper (optional) ---
    line_qty: dict[str, float] = {}
    line_display_amps: dict[str, float] = {}
    quantity_max = PAPER_POWER_MAX_DEFAULT
    if taper_line_width:
        if use_opendss_power:
            try:
                powers = solve_lvtestcase_line_powers_kw(dss_dir, master_name=master_name)
                line_qty = {k.lower(): float(v) for k, v in powers.items()}
                pvals = sorted(v for v in line_qty.values() if v > 0)
                if pvals:
                    quantity_max = PAPER_POWER_MAX_DEFAULT
                    print("  Line-width rule:        OpenDSS daisy (|kW|/Max)")
                    print(
                        f"    solved |P| range:     {pvals[0]:.3f} .. {pvals[-1]:.3f} kW  "
                        f"(Max={quantity_max:.0f})"
                    )
            except Exception as exc:
                print(f"  OpenDSS power solve:    unavailable ({exc})")
        if not line_qty:
            edge_qty = p8500.compute_edge_downstream_quantity(edges, sub_bus or "1")
            quantity_max = p8500._quantity_max(edge_qty) or float(len(coords))
            for row in lines:
                key = p8500._norm_edge(row["bus1"], row["bus2"])
                line_qty[row["name"].lower()] = float(edge_qty.get(key, 1.0))
            qvals = sorted(line_qty.values())
            print("  Line-width rule:        subtree fallback (|qty|/Max)")
            if qvals:
                print(f"    proxy range:          {qvals[0]:.0f} .. {qvals[-1]:.0f}  (Max={quantity_max:.0f})")
    else:
        amps_own = [_linecode_ampacity_a(r["linecode"]) for r in lines]
        line_display_amps = compute_line_subtree_max_ampacity(lines, sub_bus or "1")
        amps_disp = [line_display_amps.get(r["name"].lower(), _linecode_ampacity_a(r["linecode"])) for r in lines]
        print(
            f"  Line-width rule:        subtree-max ampacity "
            f"(own {min(amps_own):.0f}–{max(amps_own):.0f} A → "
            f"display {min(amps_disp):.0f}–{max(amps_disp):.0f} A → "
            f"{AMPACITY_THICKNESS_MIN}–{AMPACITY_THICKNESS_MAX} pt)"
        )
        print(
            "  Note:                  LVTestCase LINE1 is 4c_70 while thicker 4c_185/"
            "4c_.35 exist downstream; width uses max ampacity in each radial subtree "
            "so the TR1 feeder corridor stays thick."
        )
        print("                         LineCode.txt has no NormAmps; size-based ampacity table.")

    all_xy = np.array(list(coords.values()))
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    dx, dy = float(xmax - xmin), float(ymax - ymin)
    span = max(dx, dy, 1.0)

    if figsize is None:
        if crop_margins and taper_line_width:
            figsize = p8500._figsize_for_data_aspect(dx, dy, style=style)
        else:
            figsize = (10.5, 9.5)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=figsize)

    use_taper = taper_line_width and bool(line_qty)
    thickness_max = p8500.PAPER_THICKNESS_MAX
    thickness_min = p8500.PAPER_THICKNESS_MIN

    # Draw laterals first, then mains (mains on top) when tapering; else natural order.
    if use_taper:
        draw_order = sorted(lines, key=lambda r: (0 if r["is_lateral"] else 1, r["name"]))
    else:
        draw_order = list(lines)
    segs: list[list[tuple[float, float]]] = []
    lws: list[float] = []
    for row in draw_order:
        b1, b2 = row["bus1"], row["bus2"]
        if b1 not in coords or b2 not in coords:
            continue
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        segs.append([(x1, y1), (x2, y2)])
        if use_taper:
            q = line_qty.get(row["name"].lower(), 0.0)
            lws.append(
                _linewidth_combined(
                    quantity=q,
                    quantity_max=quantity_max,
                    linecode=row["linecode"],
                    is_lateral=row["is_lateral"],
                    thickness_max=thickness_max,
                    thickness_min=thickness_min,
                )
            )
        else:
            disp_a = line_display_amps.get(
                row["name"].lower(),
                _linecode_ampacity_a(row["linecode"]),
            )
            lws.append(
                _linewidth_from_ampacity(
                    row["linecode"],
                    ampacity_a=disp_a,
                )
            )
    if style == "paper":
        line_color = p8500.PAPER_BACKBONE_COLOR
        line_alpha = 1.0
    elif use_taper:
        line_color = "#333333"
        line_alpha = 0.85
    else:
        line_color = PLAIN_LINE_COLOR
        line_alpha = PLAIN_LINE_ALPHA

    ax.add_collection(
        LineCollection(
            segs,
            linewidths=lws,
            colors=line_color,
            alpha=line_alpha,
            zorder=2,
            capstyle="round",
            joinstyle="round",
        )
    )

    # --- node markers (solid disks; sit above LineCollection) ---
    if show_junctions and junction_buses:
        jxy = np.array([coords[b] for b in junction_buses], dtype=float)
        ax.scatter(
            jxy[:, 0],
            jxy[:, 1],
            s=PLAIN_JUNCTION_SIZE,
            facecolors=PLAIN_JUNCTION_COLOR,
            edgecolors="none",
            linewidths=0,
            marker="o",
            zorder=5,
            alpha=1.0,
        )

    if show_loads and load_with_xy:
        load_xy = np.array([coords[b] for b in sorted(load_with_xy)], dtype=float)
        ax.scatter(
            load_xy[:, 0],
            load_xy[:, 1],
            s=28 if (use_taper and style == "paper") else PLAIN_LOAD_SIZE,
            facecolors=PLAIN_LOAD_FACE,
            edgecolors="none",
            linewidths=0,
            marker="o",
            zorder=6,
            alpha=1.0,
        )

    # --- device icons (only devices that exist) ---
    # Paper base is uniform; ``device_icon_zooms`` overrides absolute base zoom per kind
    # (e.g. larger transformer). ``icon_size_scale`` then scales everything.
    icon_zooms = {
        key: float(p8500.PAPER_UNIFORM_ICON_ZOOM if style == "paper" else zoom)
        for key, zoom in p8500.DEFAULT_DEVICE_ICON_ZOOM.items()
    }
    if device_icon_zooms:
        for key, zoom in device_icon_zooms.items():
            icon_zooms[str(key)] = float(zoom)
    for key in list(icon_zooms):
        icon_zooms[key] = float(icon_zooms[key]) * float(icon_size_scale)

    icon_paths = dict(device_icon_paths or {})
    icon_placements: list[dict[str, Any]] = []
    placed_substation_icon = False

    def _device_icon(device_key: str) -> str | Path | None:
        return p8500._resolve_device_icon_path(device_key, icon_paths)

    if show_device_icons and devices["has_substation"] and sub_bus is not None and sub_bus in coords:
        sub_icon = _device_icon("substation")
        if sub_icon is not None:
            sx, sy = coords[sub_bus]
            icon_placements.append(
                p8500._icon_placement(
                    x=float(sx),
                    y=float(sy),
                    icon_path=sub_icon,
                    zoom=icon_zooms["substation"],
                    zorder=12,
                    kind="substation",
                    bus=str(sub_bus),
                )
            )
            placed_substation_icon = True
        elif show_source_marker:
            sx, sy = coords[sub_bus]
            ax.scatter([sx], [sy], s=90, marker="s", c=PLAIN_SOURCE_COLOR, zorder=12)
    elif show_source_marker and sub_bus is not None and sub_bus in coords:
        sx, sy = coords[sub_bus]
        ax.scatter([sx], [sy], s=90, marker="s", c=PLAIN_SOURCE_COLOR, zorder=12)

    # Caps / regs / PV: only if present (stock LVTestCase has none; PV copy has PV906).
    def _add_device_icons(
        points: dict[str, tuple[float, float]],
        *,
        device_key: str,
        kind: str,
        zorder: int,
        keep_on_map: bool = False,
    ) -> None:
        icon_path = _device_icon(device_key)
        # Controllable Volt-Var / InvControl DER: allow controllable_pv path, fall back to pv.
        if icon_path is None and device_key == "controllable_pv":
            icon_path = _device_icon("pv")
        if icon_path is None:
            return
        zoom = float(
            icon_zooms.get(
                device_key,
                icon_zooms.get("pv", p8500.PAPER_UNIFORM_ICON_ZOOM * float(icon_size_scale)),
            )
        )
        for name, (x, y) in points.items():
            placement = p8500._icon_placement(
                x=float(x),
                y=float(y),
                icon_path=icon_path,
                zoom=zoom,
                zorder=zorder,
                kind=kind,
                bus=str(name),
            )
            if keep_on_map:
                placement["keep_on_map"] = True
            icon_placements.append(placement)

    if show_device_icons and devices["has_capacitor"]:
        pass
    if show_device_icons and devices["has_regulator"]:
        pass
    if show_device_icons and devices["has_pv"]:
        pv_buses = devices.get("pv_buses") or {}
        pv_points = {
            name: coords[b] for name, b in pv_buses.items() if b in coords
        }
        missing = sorted(name for name, b in pv_buses.items() if b not in coords)
        if missing:
            print(f"  PV icons skipped (no coords): {', '.join(missing)}")
        if pv_points:
            # Prefer an explicit controllable_pv path (InvControl Volt-Var), else pv,
            # else default paper controllable asset when available.
            if "controllable_pv" in icon_paths:
                pv_key = "controllable_pv"
            elif "pv" in icon_paths:
                pv_key = "pv"
            elif _device_icon("controllable_pv") is not None:
                pv_key = "controllable_pv"
            else:
                pv_key = "pv"
            _add_device_icons(
                pv_points,
                device_key=pv_key,
                kind="pv",
                zorder=10,
                keep_on_map=True,
            )
            print(
                f"  PV icons placed:         {len(pv_points)} "
                f"({', '.join(sorted(pv_points))}; key={pv_key})"
            )

    flat_segs = p8500._flatten_segments(segs)
    use_vector_svg = p8500._resolve_vector_svg_icons_flag(
        vector_svg_icons,
        *[p["icon_path"] for p in icon_placements],
    )
    use_svg_to_pdf = p8500._resolve_svg_to_pdf_flag(svg_to_pdf, use_vector_svg=use_vector_svg)

    vector_icon_placements: list[dict[str, Any]] = []
    if icon_placements:
        # Coords are meters; allow generous snap so the substation icon stays.
        on_net_max = max(25.0, 0.08 * span)
        p8500._place_icon_placements_stacked(
            ax,
            icon_placements,
            dx=dx,
            dy=dy,
            flat_segs=flat_segs,
            raster_scale=svg_icon_raster_scale,
            on_network_max_ft=on_net_max,
            vector_svg=use_vector_svg,
            vector_placements_out=vector_icon_placements if use_vector_svg else None,
        )

    legend_items: list[Any] = []
    if show_legend:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color=line_color,
                lw=1.2,
                alpha=line_alpha,
                label=f"Lines ({len(segs)})",
            )
        )
        if show_junctions and junction_buses:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=PLAIN_JUNCTION_COLOR,
                    markeredgecolor=PLAIN_JUNCTION_COLOR,
                    markersize=5,
                    label=f"Zero-injection buses ({len(junction_buses)})",
                )
            )
        if show_loads and load_with_xy:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=PLAIN_LOAD_FACE,
                    markeredgecolor=PLAIN_LOAD_FACE,
                    markersize=8,
                    label=f"Injection / load buses ({len(load_with_xy)})",
                )
            )
        if placed_substation_icon:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=p8500.PAPER_BACKBONE_COLOR,
                    markersize=8,
                    label="Substation (TR1)",
                )
            )
        elif show_source_marker and sub_bus is not None:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=PLAIN_SOURCE_COLOR,
                    markeredgecolor="w",
                    markersize=8,
                    label=f"Source bus {sub_bus}",
                )
            )

    if show_title:
        ax.set_title(
            "IEEE 906-bus European LV Test Feeder (geographic)",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

    ax.set_aspect("equal", adjustable="box")
    data_pad_frac = p8500.TIGHT_DATA_PAD_FRAC if crop_margins else p8500.LOOSE_DATA_PAD_FRAC
    pad_x, pad_y = data_pad_frac * dx, data_pad_frac * dy
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    if crop_margins and not show_title and not show_legend:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if show_legend and legend_items:
        ax.legend(
            handles=legend_items,
            loc="lower left",
            frameon=True,
            framealpha=0.8,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=9,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    pdf_path = out_root / f"{out_basename}.pdf"
    svg_path = out_root / f"{out_basename}.svg"
    png_path = out_root / f"{out_basename}.png"
    save_kwargs: dict[str, Any] = {
        "bbox_inches": "tight",
        "facecolor": "white",
        "pad_inches": p8500.TIGHT_SAVEFIG_PAD_INCHES if crop_margins else float(plt.rcParams["savefig.pad_inches"]),
    }

    if use_vector_svg and vector_icon_placements:
        fig.savefig(svg_path, dpi=int(svg_savefig_dpi), **save_kwargs)
        n_embedded = p8500._embed_vector_icons_in_svg(
            svg_path,
            vector_icon_placements,
            fig=fig,
            ax=ax,
            pad_inches=save_kwargs.get("pad_inches"),
        )
        print(f"  Vector SVG icons:       {n_embedded} nested groups embedded")
        for placement in vector_icon_placements:
            p8500._place_bus_icon(
                ax,
                float(placement["x"]),
                float(placement["y"]),
                placement["icon_path"],
                zoom=float(placement["zoom"]),
                zorder=int(placement["zorder"]),
                raster_scale=svg_icon_raster_scale,
            )
        fig.savefig(png_path, dpi=int(png_dpi), **save_kwargs)
        if not use_svg_to_pdf:
            fig.savefig(pdf_path, **save_kwargs)
    else:
        fig.savefig(pdf_path, **save_kwargs)
        fig.savefig(svg_path, dpi=int(svg_savefig_dpi), **save_kwargs)
        fig.savefig(png_path, dpi=int(png_dpi), **save_kwargs)

    paths: dict[str, Path] = {"pdf": pdf_path, "svg": svg_path, "png": png_path}
    if crop_margins and p8500._crop_svg_viewbox_to_content(svg_path):
        print("  SVG viewBox:            cropped to axes content")
    if visio_friendly_svg:
        visio_stats = p8500.postprocess_svg_for_visio(
            svg_path,
            preserve_appearance=visio_preserve_appearance,
        )
        print(
            f"  Visio-friendly SVG:     "
            f"{visio_stats['path_count_before']:,} -> {visio_stats['path_count_after']:,} paths"
        )

    if use_svg_to_pdf:
        try:
            backend = p8500._export_pdf_from_svg(svg_path, pdf_path)
            n_icons = p8500._count_vector_device_icons_in_svg(svg_path)
            icon_note = f", {n_icons} vector icon group(s)" if n_icons else ""
            print(f"  PDF from SVG ({backend}{icon_note}): {pdf_path}")
        except RuntimeError as exc:
            if use_vector_svg and vector_icon_placements:
                fig.savefig(pdf_path, **save_kwargs)
                print(f"  PDF fallback (matplotlib): {pdf_path}")
                print(f"  Warning: {exc}")
            else:
                raise

    if use_vector_svg or use_svg_to_pdf:
        png_backend = p8500.overwrite_png_from_vector(
            png_path,
            pdf_path=pdf_path,
            svg_path=svg_path,
            png_dpi=int(png_dpi),
        )
        if png_backend:
            print(f"  PNG from vector ({png_backend}): {png_path}")
        else:
            print(
                "  PNG from vector skipped: pymupdf unavailable or rasterize failed "
                f"(kept matplotlib PNG @ {int(png_dpi)} dpi)"
            )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\nSaved figures:")
    for k, p in paths.items():
        print(f"  {k.upper()}: {p}")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot IEEE 906 LVTestCase feeder topology (plain geographic + substation icon)."
    )
    p.add_argument("--dss-dir", type=Path, default=DEFAULT_DSS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out-basename", default="ieee906_lvtestcase_topology")
    p.add_argument("--style", choices=("paper", "draft"), default="paper")
    p.add_argument("--no-loads", action="store_true")
    p.add_argument(
        "--no-junctions",
        action="store_true",
        help="Hide junction bus dots (paper-style sparse nodes).",
    )
    p.add_argument("--no-icons", action="store_true")
    p.add_argument(
        "--taper",
        action="store_true",
        help="Enable hierarchical / power-based line-width taper (8500-like).",
    )
    p.add_argument("--no-opendss", action="store_true")
    p.add_argument("--master", default="Master_snapshot.dss")
    p.add_argument("--png-dpi", type=int, default=600)
    p.add_argument("--icon-size-scale", type=float, default=2.0)
    p.add_argument("--title", action="store_true", help="Show figure title")
    p.add_argument("--legend", action="store_true", help="Show legend")
    p.add_argument("--no-vector-svg-icons", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    plot_ieee906_feeder_topology(
        dss_dir=args.dss_dir,
        out_dir=args.out_dir,
        out_basename=args.out_basename,
        style=args.style,
        show_loads=not args.no_loads,
        show_junctions=not args.no_junctions,
        show_device_icons=not args.no_icons,
        taper_line_width=bool(args.taper),
        use_opendss_power=not args.no_opendss,
        master_name=args.master,
        png_dpi=args.png_dpi,
        icon_size_scale=args.icon_size_scale,
        vector_svg_icons=False if args.no_vector_svg_icons else None,
        show_title=bool(args.title),
        show_legend=bool(args.legend),
        show=args.show,
    )


if __name__ == "__main__":
    main()
