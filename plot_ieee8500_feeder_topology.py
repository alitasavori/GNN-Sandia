#!/usr/bin/env python3
"""Publication-style IEEE 8500-node feeder topology from OpenDSS.

Line styling follows OpenDSS ``Linecode`` / source file:
  - **Paper style** — all MV lines in one blue; width follows the OpenDSS ``Plot`` rule
    (``quantity=Power``, linear ``|P|/Max`` scaling per LINE segment). Without a solved
    case we solve the feeder in OpenDSS and use ``|kW|`` per ``Line`` (EPRI daisy plot).
    If OpenDSS is unavailable, we fall back to downstream subtree size on the connected
    MV graph after coordinate backfill.
  - **Draft style** — raw 3-phase vs 1-phase layers (no backfill / trunk merge).
  - **Triplex (LV) lines are not drawn** (no coordinates in standard ``Buscoords.dss``;
    use ``--show-triplex`` only if you add service-bus coords and explicitly want them).

Example::

  python plot_ieee8500_feeder_topology.py --style paper --out-dir outputs/ieee8500_topology

Custom device icons (PNG or SVG). For ``.svg`` icons the exported ``.svg`` figure
embeds the icon paths as nested vector groups (``vector_svg_icons=True``, default).
The ``.pdf`` is then generated from that finalized SVG (``svg_to_pdf=True``,
default when vector icons are enabled) so PDF and SVG match. The publication
``.png`` defaults to ``PAPER_PNG_DPI_DEFAULT`` (2400); after vector SVG + PDF-from-SVG,
that PNG is overwritten by rasterizing the finalized PDF (preferred) or SVG with
PyMuPDF at exact ``png_dpi/72`` zoom — **no** lower-DPI retry unless
``allow_png_dpi_fallback=True``. For rasterized SVG icon loading on Windows,
install ``pip install pycairo`` (preferred) or
``pip install svglib reportlab pillow pymupdf`` as a cairo-free fallback.

For Microsoft Visio, enable ``visio_friendly_svg=True`` to post-process the saved
SVG with **appearance-preserving** cleanup by default (metadata/DOCTYPE removal,
inline strokes, empty-group pruning; clip-paths and per-segment line widths are
kept). Set ``visio_preserve_appearance=False`` for aggressive flattening (merge
constant-width line batches, strip clip-paths, drop raster legend thumbs) when
you need a smaller file and accept possible visual drift. Optional
``visio_emf_export=True`` writes ``.emf`` via Inkscape CLI when installed::

  plot_ieee8500_feeder_topology(
      device_icon_paths={
          "substation": "icons/substation.png",
          "regulator": "icons/regulator.png",
          "capacitor": "icons/capacitor.png",
          "pv": "icons/pv.png",
      },
      highlight_bus_groups=[{
          "buses": ["m1026767"],
          "legend": "Monitored bus",
          "icon_path": "outputs/Icons/monitored bus.svg",
          "size": 24,
      }],
      svg_icon_raster_scale=32.0,
      svg_savefig_dpi=600,
      vector_svg_icons=True,
      visio_friendly_svg=True,
      visio_preserve_appearance=True,
      visio_emf_export=True,
      icon_size_scale=1.4,
      reload_icons=True,
      show_legend=False,
  )
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib.image as mpimg
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    DrawingArea,
    HPacker,
    OffsetImage,
    TextArea,
    VPacker,
)

REPO = Path(__file__).resolve().parent
DEFAULT_DSS_DIR = REPO / "8500 nodes with solar unbalanced"

LineTier = Literal["primary_3ph", "lateral_1ph", "triplex"]

TIER_STYLE: dict[str, dict[str, float | str]] = {
    "triplex": {"color": "#cccccc", "lw": 0.18, "alpha": 0.45, "zorder": 0},
    "lateral_1ph": {"color": "#7a7a7a", "lw": 0.35, "alpha": 0.72, "zorder": 1},
    "primary_3ph": {"color": "#111111", "lw": 1.05, "alpha": 0.95, "zorder": 2},
}

# OpenDSS circuit-plot style (Plot type=circuit quantity=Power Max=... thickness=...).
# Line width = thickness * min(1, |quantity| / Max) per LINE object — see Plot help.
# EPRI daisy/circuit plots use one dark blue + ``thickness * |P|/Max`` for every Line.
PAPER_BACKBONE_COLOR = "#163d63"
PAPER_THICKNESS_MAX = 7.0  # OpenDSS default ``thickness`` (screen pixels)
PAPER_THICKNESS_MIN = 0.50  # floor for power-tapered primary lines
PAPER_LATERAL_FIXED_LW = 0.60  # service laterals: constant width (not power-tapered)
PAPER_POWER_MAX_DEFAULT = 2000.0  # ``plot daisy power max=2000``
PAPER_POWER_GAMMA = 1.0  # strict EPRI linear scaling
# Ampacity mode: match loading/daisy visual envelope (same pt floor/ceiling +
# linear OpenDSS map). Discrete WireData steps are radially smoothed and blended
# with downstream subtree size so the taper looks continuous like |P|.
AMPACITY_THICKNESS_MIN = PAPER_THICKNESS_MIN  # 0.50 — same as loading
AMPACITY_THICKNESS_MAX = PAPER_THICKNESS_MAX  # 7.0 — same as loading
AMPACITY_GAMMA = PAPER_POWER_GAMMA  # 1.0 linear
AMPACITY_SCALE: Literal["linear", "log"] = "linear"
AMPACITY_QMAX_PERCENTILE = 100.0
AMPACITY_SMOOTH_BLEND = 0.35  # 1=raw ampacity, 0=fully inherit parent display
AMPACITY_DOWNSTREAM_WEIGHT = 0.55  # blend continuous subtree size (loading-like)
AMPACITY_DEFAULT_A = 200.0  # fallback when a line has no wire/geometry match
LineWidthMode = Literal["power", "ampacity", "uniform"]
AmpacityScale = Literal["linear", "log"]
# Publication PNG DPI (matplotlib savefig + pymupdf PDF/SVG→PNG overwrite).
# 2400 ≈ 27.6k×20.4k px at 11.5×8.5 in — practical ceiling before RAM/time blow up;
# bump only if you have headroom (3000 is usually fine; 3600+ often OOMs).
PAPER_PNG_DPI_DEFAULT = 2400
# Minimum PyMuPDF Matrix zoom when overwriting PNG from finalized PDF/SVG
# (effective DPI ≈ zoom×72). At png_dpi=2400 the derived zoom (~33.3) dominates.
PAPER_PNG_FROM_VECTOR_ZOOM_DEFAULT = 4.0

# OffsetImage zoom factors (tune after exporting PNG/SVG icons from PowerPoint).
# Zoom is normalized to ICON_REFERENCE_PX so 185 px and 330 px SVG exports share
# the same on-map size at the same zoom value (~zoom * REF display points).
ICON_REFERENCE_PX = 64.0
DEFAULT_DEVICE_ICON_ZOOM: dict[str, float] = {
    "substation": 0.15,
    "regulator": 0.085,
    "capacitor": 0.08,
    "pv": 0.09,
    "storage": 0.085,
}
# Paper / PowerPoint reference: one zoom for every square device icon on the map.
PAPER_UNIFORM_ICON_ZOOM = 0.16
PAPER_LEGEND_ICON_THUMB_PX = 48
PAPER_LEGEND_ICON_ZOOM = 0.55  # OffsetImage zoom for legend thumbnails (points)
PAPER_MONITORED_BUS_SCATTER_SIZE = 12.0  # scatter ``s`` in points^2 (not data coords)
CIRCLED_STAR_MARKER_ALIASES = frozenset({"circled_star", "star_in_circle"})
# Font size = marker diameter (pt) * CIRCLED_STAR_FONT_FRAC so the * ink fills ~65-70% of the circle.
CIRCLED_STAR_FONT_FRAC = 1.25
CIRCLED_STAR_FONT_FAMILY = "DejaVu Sans"
DEFAULT_HIGHLIGHT_ICON_ZOOM = 0.16
ICON_STACK_STEP_FRAC = 0.0045  # diagonal offset per stacked icon (fraction of map span)
ICON_STACK_DER_COLLOC_STEP_MULT = 3.0  # fallback mult on frac-based step (non-directional spread)
ICON_STACK_DER_OFFSET_FT = 450.0  # anchor offset (ft) for co-located der_highlight icons (~1 icon width on 8500 map)
DEFAULT_PAPER_ICON_DIR = REPO / "outputs" / "Icons"
DEFAULT_PAPER_ICON_FILES: dict[str, str] = {
    "substation": "substation.svg",
    "regulator": "voltage regulator.svg",
    "capacitor": "capacitor bank.svg",
    "pv": "autonomous pv.svg",
    "autonomous_pv": "autonomous pv.svg",
    "storage": "controlable battery.svg",
    "controllable_pv": "controlable pv.svg",
    "monitored": "monitored bus.svg",
    "monitored_bus": "monitored bus.svg",
}
ICON_STACK_CLUSTER_FT = 55.0  # buses within this distance share one stack
MONITORED_ICON_SUPPRESS_FT = 30.0  # drop monitored icons within this distance of any other icon
MONITORED_ICON_SUPPRESS_DISPLAY_PAD_PX = 36.0  # extra display px for stacked DER offsets
MONITORED_ICON_FOOTPRINT_AXES_PT = 450.0  # typical axes width in pt for footprint scaling
DER_HIGHLIGHT_BUS_SUPPRESS_KEYS = frozenset({"m1069224", "m1108263"})
ICON_ON_NETWORK_MAX_FT = 40.0  # drop icons farther than this from a drawn MV segment
ICON_STACK_PRIORITY: dict[str, int] = {
    "highlight": -1,  # monitored / highlight buses — bottom of co-located stacks
    "der_highlight": 3,  # controllable DER icons from highlight_bus_groups
    "capacitor": 0,
    "regulator": 1,
    "pv": 2,
    "storage": 2,
    "substation": 4,
}
# Supersample PowerPoint-exported SVG icons for crisp PDF/PNG/SVG raster embeds.
SVG_ICON_RASTER_SCALE = 32.0
SVG_ICON_RASTER_MAX_PX = 2048  # cap per-axis output (330 px SVGs × 32 would OOM)
SVG_SAVEFIG_DPI_DEFAULT = 600
TIGHT_SAVEFIG_PAD_INCHES = 0.01
TIGHT_DATA_PAD_FRAC = 0.005
LOOSE_DATA_PAD_FRAC = 0.03


def _ieee_tight_bbox_inches(
    fig: plt.Figure,
    *,
    crop_margins: bool,
    want_cbar: bool,
    cbar_label: str = "",
):
    """Return ``(bbox_inches, pad_inches)`` for IEEE-tight export.

    Colorbar case: ``get_tightbbox`` already covers the rotated label when
    ``clip_on=False``; only add a tiny safety margin on the right, plus a bit
    of top/bottom so end tick numbers are not clipped.
    """
    from matplotlib.transforms import Bbox

    base_pad = TIGHT_SAVEFIG_PAD_INCHES if crop_margins else float(plt.rcParams["savefig.pad_inches"])
    if not want_cbar:
        return "tight", base_pad

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    if bbox is None:
        return "tight", base_pad

    del cbar_label  # label extent comes from tightbbox; avoid length-based overpad
    extra_right = 0.03  # small safety only (not proportional to label length)
    extra_tb = 0.08  # end tick labels (top/bottom digits)
    bbox2 = Bbox.from_extents(
        float(bbox.x0) - base_pad,
        float(bbox.y0) - base_pad - extra_tb,
        float(bbox.x1) + base_pad + extra_right,
        float(bbox.y1) + base_pad + extra_tb,
    )
    return bbox2, 0.0
# Nearest-neighbor keeps sharp edges when OffsetImage downscales high-res rasters.
MAP_ICON_INTERPOLATION = "nearest"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
_ICON_IMAGE_CACHE: dict[tuple[str, float, float], tuple[np.ndarray, float, float]] = {}
_SVG_ICON_FRAGMENT_CACHE: dict[tuple[str, float], tuple[list[Any], list[Any], float, float]] = {}


def _icon_path_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def clear_icon_caches() -> None:
    """Drop in-memory icon raster / parsed-SVG caches (e.g. after editing ``outputs/Icons``)."""
    _ICON_IMAGE_CACHE.clear()
    _SVG_ICON_FRAGMENT_CACHE.clear()


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.split(r"\s!|^!|//", line, maxsplit=1)[0]
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def join_continuations(text: str) -> list[str]:
    commands: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("~"):
            current += " " + line[1:].strip()
        else:
            if current:
                commands.append(current)
            current = line
    if current:
        commands.append(current)
    return commands


def canonical_bus(bus: str) -> str:
    """Normalize OpenDSS bus aliases (``D5710794_3_INT`` vs ``D5710794-3_INT``)."""
    bus = bus.lower()
    return re.sub(r"_(\d+)_int$", r"-\1_int", bus)


def base_bus(bus: str | None) -> str | None:
    if bus is None:
        return None
    bus = bus.strip().strip("[](),")
    bus = bus.replace('"', "").replace("'", "")
    parts = bus.split(".")
    if len(parts) > 1 and all(p.isdigit() for p in parts[1:]):
        return canonical_bus(parts[0])
    return canonical_bus(bus)


def get_param(command: str, key: str) -> str | None:
    pattern = (
        rf"(?i)(?:^|\s){re.escape(key)}\s*=\s*"
        r"(\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"|'[^']*'|[^\s]+)"
    )
    m = re.search(pattern, command)
    return m.group(1).strip() if m else None


def is_opendss_enabled(command: str, *, default: bool = True) -> bool:
    """OpenDSS ``enabled=`` flag (default True when omitted)."""
    raw = get_param(command, "enabled")
    if raw is None:
        return bool(default)
    tok = raw.strip().strip("\"'").lower()
    if tok in ("false", "no", "n", "0"):
        return False
    if tok in ("true", "yes", "y", "1"):
        return True
    return bool(default)


def parse_bus_list(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip().strip("[]()")
    value = value.replace(",", " ")
    return [base_bus(v) for v in value.split() if v.strip()]


def parse_object_name(command: str, objtype: str) -> str | None:
    pattern = rf"(?i)\b(?:new|edit)\s+{objtype}\.([^\s]+)"
    m = re.search(pattern, command)
    return m.group(1).lower() if m else None


def read_all_dss_commands(dss_dir: Path) -> list[tuple[str, str]]:
    if not dss_dir.is_dir():
        raise FileNotFoundError(f"DSS_DIR does not exist: {dss_dir}")
    commands: list[tuple[str, str]] = []
    for file in sorted(dss_dir.rglob("*.dss")):
        try:
            text = file.read_text(errors="ignore")
        except OSError:
            continue
        text = strip_comments(text)
        for cmd in join_continuations(text):
            commands.append((file.name.lower(), cmd))
    return commands


def load_buscoords(dss_dir: Path) -> dict[str, tuple[float, float]]:
    candidates = list(dss_dir.rglob("*bus*coord*.dss")) + list(dss_dir.rglob("*BusCoords*.dss"))
    coords: dict[str, tuple[float, float]] = {}
    for file in candidates:
        try:
            text = file.read_text(errors="ignore")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("!") or line.startswith("//"):
                continue
            line = re.split(r"\s!|^!|//", line, maxsplit=1)[0].strip()
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) < 3:
                continue
            bus = base_bus(parts[0])
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            if bus:
                coords[bus] = (x, y)
    if not coords:
        raise RuntimeError(f"No bus coordinates found under {dss_dir}. Expected Buscoords.dss.")
    merged: dict[str, tuple[float, float]] = {}
    for bus, xy in coords.items():
        key = canonical_bus(bus)
        if key not in merged:
            merged[key] = xy
    return merged


def classify_line_tier(fname: str, cmd: str) -> LineTier:
    fname_l = fname.lower()
    cmd_l = cmd.lower()
    obj = (parse_object_name(cmd, "line") or "").lower()
    if "triplex" in fname_l or obj.startswith("tpx") or "triplex" in cmd_l:
        return "triplex"
    lc = (get_param(cmd, "linecode") or "").lower()
    bus1_raw = get_param(cmd, "bus1") or ""
    bus2_raw = get_param(cmd, "bus2") or ""
    has_phase = bool(re.search(r"\.\d", bus1_raw) or re.search(r"\.\d", bus2_raw))
    if "hvmv" in obj or "sub_connector" in obj:
        return "primary_3ph"
    if lc.startswith("3ph") or lc.startswith("3p_") or "3ph_" in lc:
        return "primary_3ph"
    if lc.startswith("1ph") or lc.startswith("1p_") or "1ph" in lc or has_phase:
        return "lateral_1ph"
    if not has_phase and lc:
        return "primary_3ph"
    return "lateral_1ph"


def _norm_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _unique_line_commands(commands: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One entry per OpenDSS Line object (Lines.dss beats Lines-Geometry.dss)."""
    best: dict[str, tuple[int, str, str]] = {}
    for fname, cmd in commands:
        if not re.search(r"(?i)\b(?:new|edit)\s+line\.", cmd):
            continue
        name = parse_object_name(cmd, "line")
        if not name:
            continue
        key = name.lower()
        if fname == "lines.dss":
            pri = 0
        elif "lines-geometry" in fname:
            pri = 1
        else:
            pri = 2
        if key not in best or pri < best[key][0]:
            best[key] = (pri, fname, cmd)
    return [(fname, cmd) for _, fname, cmd in best.values()]


def collect_mv_line_edges(
    commands: list[tuple[str, str]],
    *,
    include_disabled: bool = True,
) -> list[tuple[str, str, LineTier]]:
    """All non-triplex MV line endpoints (coords not required)."""
    edges: list[tuple[str, str, LineTier]] = []
    for fname, cmd in _unique_line_commands(commands):
        if not include_disabled and not is_opendss_enabled(cmd):
            continue
        b1 = base_bus(get_param(cmd, "bus1"))
        b2 = base_bus(get_param(cmd, "bus2"))
        if not b1 or not b2:
            continue
        tier = classify_line_tier(fname, cmd)
        if tier == "triplex":
            continue
        edges.append((b1, b2, tier))
    return edges


def _parse_line_length_ft(cmd: str) -> float | None:
    raw = get_param(cmd, "length")
    if raw is None:
        return None
    try:
        length = float(raw)
    except ValueError:
        return None
    units = (get_param(cmd, "units") or "km").lower()
    if units in {"km", "kft"}:
        return length * 3280.84
    if units in {"m", "meter", "meters"}:
        return length * 3.28084
    if units in {"mi", "mile", "miles"}:
        return length * 5280.0
    if units in {"ft", "feet", "len"}:
        return length
    return length * 3280.84


def backfill_missing_bus_coords(
    coords: dict[str, tuple[float, float]],
    commands: list[tuple[str, str]],
    *,
    max_passes: int = 64,
    include_disabled: bool = True,
) -> tuple[dict[str, tuple[float, float]], int]:
    """Interpolate coordinates for sectionalizing / internal buses missing from Buscoords."""
    out = dict(coords)
    line_rows: list[tuple[str, str, float]] = []
    for fname, cmd in _unique_line_commands(commands):
        if not include_disabled and not is_opendss_enabled(cmd):
            continue
        b1 = base_bus(get_param(cmd, "bus1"))
        b2 = base_bus(get_param(cmd, "bus2"))
        if not b1 or not b2:
            continue
        if classify_line_tier(fname, cmd) == "triplex":
            continue
        length = _parse_line_length_ft(cmd) or 1.0
        line_rows.append((b1, b2, length))

    pairs = [(a, b) for a, b, _ in line_rows]

    for _ in range(max_passes):
        changed = False
        missing_adj: dict[str, set[str]] = defaultdict(set)
        for a, b in pairs:
            if a in out and b not in out:
                missing_adj[b].add(a)
            if b in out and a not in out:
                missing_adj[a].add(b)

        for bus, nbrs in missing_adj.items():
            if bus in out:
                continue
            known = [n for n in nbrs if n in out]
            if len(known) >= 2:
                xs = [out[n][0] for n in known]
                ys = [out[n][1] for n in known]
                out[bus] = (sum(xs) / len(known), sum(ys) / len(known))
                changed = True

        if not changed:
            break

    adj_w: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a, b, length in line_rows:
        adj_w[a].append((b, length))
        adj_w[b].append((a, length))

    for anchor in list(out):
        for n0, _ in adj_w[anchor]:
            if n0 in out:
                continue
            chain = [anchor, n0]
            prev, cur = anchor, n0
            while cur not in out:
                others = list({nb for nb, _ in adj_w[cur] if nb != prev})
                if len(others) != 1:
                    break
                nxt = others[0]
                chain.append(nxt)
                prev, cur = cur, nxt
            if cur not in out or len(chain) < 3:
                continue
            start, end = chain[0], chain[-1]
            seg_lens: list[float] = []
            for u, v in zip(chain, chain[1:]):
                seg_len = next((l for nb, l in adj_w[u] if nb == v), 1.0)
                seg_lens.append(max(seg_len, 1e-3))
            total = sum(seg_lens)
            sx, sy = out[start]
            ex, ey = out[end]
            dist = 0.0
            for u, v, seg_len in zip(chain, chain[1:], seg_lens):
                if v in out:
                    continue
                dist += seg_len
                t = dist / total
                out[v] = (sx + t * (ex - sx), sy + t * (ey - sy))

    return out, len(out) - len(coords)


def identify_trunk_lateral_edges(
    primary_edges: list[tuple[str, str]],
    lateral_edges: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    """1-phase edges on degree-2 chains linking two 3-phase (primary) buses."""
    primary_nodes: set[str] = set()
    for a, b in primary_edges:
        primary_nodes.add(a)
        primary_nodes.add(b)

    adj_lat: dict[str, set[str]] = defaultdict(set)
    for a, b in lateral_edges:
        adj_lat[a].add(b)
        adj_lat[b].add(a)

    trunk: set[tuple[str, str]] = set()
    for p in primary_nodes:
        for n0 in adj_lat[p]:
            prev, cur = p, n0
            path = [_norm_edge(p, n0)]
            while True:
                if cur in primary_nodes and cur != p:
                    trunk.update(path)
                    break
                if cur in primary_nodes:
                    break
                nbrs = adj_lat[cur]
                if len(nbrs) >= 3 or len(nbrs) <= 1:
                    break
                nxt = (nbrs - {prev}).pop()
                path.append(_norm_edge(cur, nxt))
                prev, cur = cur, nxt
    return trunk


def _dedupe_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for a, b in edges:
        key = _norm_edge(a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def split_mv_backbone_and_service_laterals(
    line_edges_by_tier: dict[LineTier, list[tuple[str, str]]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Paper-style MV backbone vs service laterals (peel 1ph service taps)."""
    primary = _dedupe_edges(line_edges_by_tier["primary_3ph"])
    lateral = _dedupe_edges(line_edges_by_tier["lateral_1ph"])
    primary_nodes = {b for e in primary for b in e}

    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in primary + lateral:
        adj[a].add(b)
        adj[b].add(a)

    changed = True
    while changed:
        changed = False
        for node in list(adj):
            if node in primary_nodes or len(adj[node]) > 1:
                continue
            (nbr,) = tuple(adj[node])
            adj[nbr].remove(node)
            del adj[node]
            changed = True

    backbone = _dedupe_edges([(n, m) for n in adj for m in adj[n] if n < m])
    backbone_set = set(backbone)
    service = [e for e in lateral if _norm_edge(*e) not in backbone_set]
    return backbone, service


def _snap_coords_for_display(
    coords: dict[str, tuple[float, float]],
    snap_buses: set[str],
    *,
    snap_ft: float = 22.0,
) -> dict[str, tuple[float, float]]:
    """Merge nearly coincident backbone buses for cleaner paper rendering."""
    out = dict(coords)
    buses = sorted(snap_buses)
    if len(buses) < 2:
        return out

    parent = {b: b for b in buses}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(buses):
        x1, y1 = coords[a]
        for b in buses[i + 1 :]:
            x2, y2 = coords[b]
            if math.hypot(x2 - x1, y2 - y1) <= snap_ft:
                union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for b in buses:
        groups[find(b)].append(b)
    for members in groups.values():
        xs = [coords[m][0] for m in members]
        ys = [coords[m][1] for m in members]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        for m in members:
            out[m] = (cx, cy)
    return out


def parse_opendss_model(
    commands: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
    *,
    include_disabled_lines: bool = True,
) -> dict[str, object]:
    line_edges_by_tier: dict[LineTier, list[tuple[str, str]]] = {
        "primary_3ph": [],
        "lateral_1ph": [],
        "triplex": [],
    }
    transformer_edges: list[tuple[str, str]] = []
    transformer_buses: dict[str, list[str]] = {}
    capacitor_buses: dict[str, str] = {}
    pv_buses: dict[str, str] = {}
    storage_buses: dict[str, str] = {}
    regcontrol_to_transformer: dict[str, str] = {}
    capcontrol_to_capacitor: dict[str, str] = {}
    load_buses: set[str] = set()
    n_disabled_lines_skipped = 0

    for fname, cmd in _unique_line_commands(commands):
        if not include_disabled_lines and not is_opendss_enabled(cmd):
            n_disabled_lines_skipped += 1
            continue
        b1 = base_bus(get_param(cmd, "bus1"))
        b2 = base_bus(get_param(cmd, "bus2"))
        if b1 in coords and b2 in coords:
            tier = classify_line_tier(fname, cmd)
            if tier == "triplex":
                continue
            line_edges_by_tier[tier].append((b1, b2))

    for fname, cmd in commands:
        if re.search(r"(?i)\b(?:new|edit)\s+transformer\.", cmd):
            name = parse_object_name(cmd, "transformer")
            buses = parse_bus_list(get_param(cmd, "buses"))
            if len(buses) < 2:
                b1 = base_bus(get_param(cmd, "bus1")) or base_bus(get_param(cmd, "bus"))
                b2 = base_bus(get_param(cmd, "bus2"))
                buses = [b for b in (b1, b2) if b]
            buses = [b for b in buses if b in coords]
            if name and buses:
                transformer_buses[name] = buses
            if len(buses) >= 2:
                transformer_edges.append((buses[0], buses[1]))

        if re.search(r"(?i)\b(?:new|edit)\s+capacitor\.", cmd):
            name = parse_object_name(cmd, "capacitor")
            b = base_bus(get_param(cmd, "bus1")) or base_bus(get_param(cmd, "bus"))
            if name and b in coords:
                capacitor_buses[name] = b

        if re.search(r"(?i)\b(?:new|edit)\s+regcontrol\.", cmd):
            name = parse_object_name(cmd, "regcontrol")
            xf = get_param(cmd, "transformer")
            if name and xf:
                regcontrol_to_transformer[name] = xf.lower().replace("transformer.", "")

        if re.search(r"(?i)\b(?:new|edit)\s+capcontrol\.", cmd):
            name = parse_object_name(cmd, "capcontrol")
            cap = get_param(cmd, "capacitor")
            if name and cap:
                capcontrol_to_capacitor[name] = cap.lower().replace("capacitor.", "")

        if re.search(r"(?i)\b(?:new|edit)\s+pvsystem\.", cmd):
            name = parse_object_name(cmd, "pvsystem")
            b = base_bus(get_param(cmd, "bus1")) or base_bus(get_param(cmd, "bus"))
            if name and b in coords:
                pv_buses[name] = b

        if re.search(r"(?i)\b(?:new|edit)\s+storage\.", cmd):
            name = parse_object_name(cmd, "storage")
            b = base_bus(get_param(cmd, "bus1")) or base_bus(get_param(cmd, "bus"))
            if name and b in coords:
                storage_buses[name] = b

        if re.search(r"(?i)\b(?:new|edit)\s+load\.", cmd):
            b = base_bus(get_param(cmd, "bus1"))
            if b in coords:
                load_buses.add(b)

    regulator_points: dict[str, tuple[float, float]] = {}
    for reg_name, xf_name in regcontrol_to_transformer.items():
        buses = [b for b in transformer_buses.get(xf_name, []) if b in coords]
        if len(buses) >= 2:
            x1, y1 = coords[buses[0]]
            x2, y2 = coords[buses[1]]
            regulator_points[reg_name] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        elif len(buses) == 1:
            regulator_points[reg_name] = coords[buses[0]]

    capcontrol_points = {
        name: coords[b]
        for name, cap_name in capcontrol_to_capacitor.items()
        if (b := capacitor_buses.get(cap_name)) in coords
    }
    capacitor_points = {name: coords[b] for name, b in capacitor_buses.items() if b in coords}
    transformer_points: dict[str, tuple[float, float]] = {}
    for name, buses in transformer_buses.items():
        valid = [coords[b] for b in buses if b in coords]
        if valid:
            xs, ys = zip(*valid)
            transformer_points[name] = (float(np.mean(xs)), float(np.mean(ys)))

    pv_points = {name: coords[b] for name, b in pv_buses.items() if b in coords}
    storage_points = {name: coords[b] for name, b in storage_buses.items() if b in coords}

    return {
        "line_edges_by_tier": line_edges_by_tier,
        "transformer_edges": transformer_edges,
        "transformer_points": transformer_points,
        "regulator_points": regulator_points,
        "capacitor_points": capacitor_points,
        "capcontrol_points": capcontrol_points,
        "pv_points": pv_points,
        "storage_points": storage_points,
        "load_buses": load_buses,
        "n_disabled_lines_skipped": n_disabled_lines_skipped,
    }


def resolve_source_bus(coords: dict[str, tuple[float, float]]) -> str:
    for preferred in ("hvmv_sub_48332", "sourcebus"):
        if preferred in coords:
            return preferred
    for bus in coords:
        bl = bus.lower()
        if "hvmv_sub" in bl and "regxfmr" not in bl and "_hsb" not in bl and "_lsb" not in bl:
            return bus
    source_candidates = [
        b for b in coords if "source" in b or "hvmv_sub" in b or b.endswith("sub")
    ]
    return source_candidates[0] if source_candidates else max(coords, key=lambda b: coords[b][0])


def collect_line_records(
    commands: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
    *,
    include_disabled: bool = True,
) -> list[tuple[str, str, str, LineTier]]:
    """One drawable segment per unique OpenDSS Line object."""
    rows: list[tuple[str, str, str, LineTier]] = []
    for fname, cmd in _unique_line_commands(commands):
        if not include_disabled and not is_opendss_enabled(cmd):
            continue
        name = parse_object_name(cmd, "line")
        b1 = base_bus(get_param(cmd, "bus1"))
        b2 = base_bus(get_param(cmd, "bus2"))
        if not name or not b1 or not b2 or b1 not in coords or b2 not in coords:
            continue
        tier = classify_line_tier(fname, cmd)
        if tier == "triplex":
            continue
        rows.append((name, b1, b2, tier))
    return rows


def solve_opendss_line_powers_kw(
    dss_dir: Path,
    *,
    master_name: str = "Master-PV2MW-inv.dss",
    solve_hour: float | None = None,
    solve_sec: float = 0.0,
) -> dict[str, float]:
    """Solve IEEE 8500 in OpenDSS and return ``|P|`` in kW for each Line.

    If ``solve_hour`` is set, runs a daily-mode snapshot at that time so
    ``Daily=IrradDay001`` scales PV (e.g. hour=3 → zero irradiance on
    ``irr_day_001.csv``). Loads stay at rated values unless the master
    attaches a load daily shape separately.
    """
    try:
        import opendssdirect as dss
    except ImportError as exc:
        raise RuntimeError("opendssdirect is required for OpenDSS line powers") from exc

    master_path = dss_dir / master_name
    if not master_path.is_file():
        raise FileNotFoundError(f"OpenDSS master not found: {master_path}")

    prev_cwd = os.getcwd()
    try:
        os.chdir(dss_dir)
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{master_name}"')
        dss.Text.Command("Set maxcontroliter=50")
        if solve_hour is not None:
            dss.Text.Command("Set mode=daily")
            dss.Text.Command(f"Set hour={float(solve_hour)} sec={float(solve_sec)}")
        try:
            dss.Solution.Solve()
        except Exception:
            if not dss.Solution.Converged():
                raise
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


def compute_edge_downstream_quantity(
    edges: list[tuple[str, str]],
    source_bus: str,
) -> dict[tuple[str, str], float]:
    """Fallback ``Power`` proxy: downstream subtree size on a radial MV tree."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    if source_bus not in adj:
        return {}

    parent: dict[str, str | None] = {source_bus: None}
    children: dict[str, list[str]] = defaultdict(list)
    depth: dict[str, int] = {source_bus: 0}
    queue: list[str] = [source_bus]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if v in parent:
                continue
            parent[v] = u
            depth[v] = depth[u] + 1
            children[u].append(v)
            queue.append(v)

    subtree: dict[str, float] = {}

    def dfs(u: str) -> float:
        total = 1.0
        for child in children[u]:
            total += dfs(child)
        subtree[u] = total
        return total

    dfs(source_bus)

    edge_qty: dict[tuple[str, str], float] = {}
    for a, b in edges:
        key = _norm_edge(a, b)
        if parent.get(b) == a:
            edge_qty[key] = subtree[b]
        elif parent.get(a) == b:
            edge_qty[key] = subtree[a]
        elif depth.get(a, 0) >= depth.get(b, 0):
            edge_qty[key] = subtree.get(a, 1.0)
        else:
            edge_qty[key] = subtree.get(b, 1.0)
    return edge_qty


def _connected_components(
    edges: list[tuple[str, str]],
) -> list[set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        components.append(comp)
    return components


def resolve_main_feeder_root(
    edges: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
    substation_bus: str,
) -> str:
    """Root for quantity taper: main MV component bus nearest the substation."""
    components = _connected_components(edges)
    if not components:
        return substation_bus
    main_comp = max(components, key=len)
    if substation_bus in main_comp:
        return substation_bus
    if substation_bus not in coords:
        return next(iter(main_comp))
    sx, sy = coords[substation_bus]
    best_bus = substation_bus
    best_dist = float("inf")
    for bus in main_comp:
        if bus not in coords:
            continue
        x, y = coords[bus]
        dist = math.hypot(x - sx, y - sy)
        if dist < best_dist:
            best_dist = dist
            best_bus = bus
    return best_bus


def build_line_power_quantities(
    line_records: list[tuple[str, str, str, LineTier]],
    line_powers_kw: dict[str, float],
) -> dict[str, float]:
    """Map each OpenDSS Line name to ``|P|`` in kW."""
    return {name: line_powers_kw.get(name.lower(), 0.0) for name, _, _, _ in line_records}


def enrich_backbone_lateral_powers(
    line_records: list[tuple[str, str, str, LineTier]],
    line_powers_kw: dict[str, float],
    backbone_buses: set[str],
) -> tuple[dict[str, float], int]:
    """Keep all lines; give backbone 1ph glue segments the local 3ph |P| for width."""
    bus_primary_max: dict[str, float] = defaultdict(float)
    for name, b1, b2, tier in line_records:
        if tier != "primary_3ph":
            continue
        p = abs(line_powers_kw.get(name.lower(), 0.0))
        bus_primary_max[b1] = max(bus_primary_max[b1], p)
        bus_primary_max[b2] = max(bus_primary_max[b2], p)

    display: dict[str, float] = {}
    n_glue = 0
    for name, b1, b2, tier in line_records:
        key = name.lower()
        p = abs(line_powers_kw.get(key, 0.0))
        if tier == "lateral_1ph" and b1 in backbone_buses and b2 in backbone_buses:
            inherited = max(bus_primary_max[b1], bus_primary_max[b2])
            if inherited > p:
                p = inherited
                n_glue += 1
        display[key] = p
    return display, n_glue


def _is_backbone_glue_lateral(b1: str, b2: str, backbone_buses: set[str]) -> bool:
    return b1 in backbone_buses and b2 in backbone_buses


def compute_plot_edge_quantities(
    edges: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
    substation_bus: str,
) -> dict[tuple[str, str], float]:
    """Fallback edge quantities from downstream subtree size."""
    return compute_edge_downstream_quantity(edges, substation_bus)


def _quantity_max(
    quantities: dict[tuple[str, str], float],
    *,
    default: float = PAPER_POWER_MAX_DEFAULT,
) -> float:
    """OpenDSS ``Max``: value that maps to full ``thickness`` (largest branch quantity)."""
    if not quantities:
        return default
    return max(1.0, max(quantities.values()))


def linewidth_opendss(
    quantity: float,
    *,
    quantity_max: float,
    thickness_max: float = PAPER_THICKNESS_MAX,
    thickness_min: float = PAPER_THICKNESS_MIN,
    gamma: float = PAPER_POWER_GAMMA,
) -> float:
    """Map quantity to width (OpenDSS ``thickness * |qty| / Max``)."""
    if quantity_max <= 0.0:
        return thickness_min
    scaled = min(1.0, max(0.0, quantity) / quantity_max)
    if gamma != 1.0:
        scaled = scaled**gamma
    return thickness_min + (thickness_max - thickness_min) * scaled


def linewidth_ampacity(
    amps: float,
    *,
    amp_min: float,
    amp_max: float,
    thickness_min: float = AMPACITY_THICKNESS_MIN,
    thickness_max: float = AMPACITY_THICKNESS_MAX,
    gamma: float = AMPACITY_GAMMA,
    scale: AmpacityScale = AMPACITY_SCALE,
) -> float:
    """Map conductor ampacity (A) → display linewidth (pt).

    ``scale="log"`` equalizes discrete WireData steps; ``gamma>1`` keeps most
    laterals near ``thickness_min`` so only large conductors read as thick.
    """
    a = max(float(amps), 1e-9)
    lo = max(float(amp_min), 1e-9)
    hi = max(float(amp_max), lo * 1.01)
    if scale == "log":
        u = (math.log(a) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        u = (a - lo) / (hi - lo)
    u = min(1.0, max(0.0, u))
    if gamma != 1.0:
        u = u**float(gamma)
    return float(thickness_min + (thickness_max - thickness_min) * u)


def load_wiredata_normamps(dss_dir: Path | str) -> dict[str, float]:
    """Parse ``WireData.dss`` → lowercase wire name → Normamps (A)."""
    path = Path(dss_dir) / "WireData.dss"
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        low = line.lower()
        if low.startswith("new wiredata."):
            current = parse_object_name(line, "wiredata")
            if current:
                current = current.lower()
        if current is None:
            continue
        m = re.search(r"normamps\s*=\s*([0-9.+-eE]+)", line, flags=re.I)
        if m:
            try:
                out[current] = float(m.group(1))
            except ValueError:
                pass
    return out


def load_linegeometry_ampacity(
    dss_dir: Path | str,
    wire_amps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Ampacity per LineGeometry = min Normamps among phase conductors.

    Neutral (cond > nphases) is ignored. Keys are lowercase geometry names
    (usually identical to Linecode= on IEEE 8500 Lines.dss).
    """
    path = Path(dss_dir) / "LineGeometry.dss"
    if not path.is_file():
        return {}
    if wire_amps is None:
        wire_amps = load_wiredata_normamps(dss_dir)
    geom_amps: dict[str, float] = {}
    name: str | None = None
    nphases = 3
    phase_amps: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        low = line.lower()
        if low.startswith("new linegeometry."):
            if name and phase_amps:
                geom_amps[name] = float(min(phase_amps))
            name = parse_object_name(line, "linegeometry")
            name = name.lower() if name else None
            nphases = 3
            phase_amps = []
            np_m = re.search(r"nphases\s*=\s*(\d+)", line, flags=re.I)
            if np_m:
                nphases = int(np_m.group(1))
            continue
        if name is None:
            continue
        np_m = re.search(r"nphases\s*=\s*(\d+)", line, flags=re.I)
        if np_m:
            nphases = int(np_m.group(1))
        cond_m = re.search(r"cond\s*=\s*(\d+)", line, flags=re.I)
        wire_m = re.search(r"wire\s*=\s*([^\s]+)", line, flags=re.I)
        if not (cond_m and wire_m):
            continue
        cond_i = int(cond_m.group(1))
        if cond_i > nphases:
            continue  # skip neutral
        wkey = wire_m.group(1).strip().lower()
        if wkey in wire_amps:
            phase_amps.append(float(wire_amps[wkey]))
    if name and phase_amps:
        geom_amps[name] = float(min(phase_amps))
    return geom_amps


def collect_line_linecodes(commands: list[tuple[str, str]]) -> dict[str, str]:
    """Map lowercase Line name → lowercase Linecode (or empty)."""
    out: dict[str, str] = {}
    for _fname, cmd in _unique_line_commands(commands):
        name = parse_object_name(cmd, "line")
        if not name:
            continue
        lc = get_param(cmd, "linecode") or get_param(cmd, "geometry") or ""
        out[name.lower()] = lc.strip().lower()
    return out


def build_line_ampacity_qty(
    line_records: list[tuple[str, str, str, LineTier]],
    line_linecodes: dict[str, str],
    geom_amps: dict[str, float],
    *,
    default_a: float = AMPACITY_DEFAULT_A,
) -> dict[str, float]:
    """Per-line ampacity (A) keyed by lowercase line name."""
    qty: dict[str, float] = {}
    for name, _b1, _b2, _tier in line_records:
        lc = line_linecodes.get(name.lower(), "")
        a = geom_amps.get(lc, default_a)
        qty[name.lower()] = float(a)
    return qty


def smooth_line_qty_radial(
    line_records: list[tuple[str, str, str, LineTier]],
    raw_qty: dict[str, float],
    root_bus: str,
    *,
    blend: float = AMPACITY_SMOOTH_BLEND,
) -> dict[str, float]:
    """Exponentially smooth per-line quantities along the radial tree from ``root_bus``.

    ``disp_child = blend * raw_child + (1 - blend) * disp_parent``.
    Low ``blend`` softens abrupt WireData / ampacity jumps (loading-like taper).
    """
    root = canonical_bus(root_bus) if root_bus else ""
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, b1, b2, _tier in line_records:
        key = name.lower()
        adj[b1].append((b2, key))
        adj[b2].append((b1, key))

    if not root or root not in adj:
        return dict(raw_qty)

    blend = float(min(1.0, max(0.0, blend)))
    parent: dict[str, str | None] = {root: None}
    child_line: dict[str, str] = {}
    queue = [root]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, name in adj[u]:
            if v in parent:
                continue
            parent[v] = u
            child_line[v] = name
            queue.append(v)

    children: dict[str, list[str]] = defaultdict(list)
    for child, par in parent.items():
        if par is not None:
            children[par].append(child)

    disp: dict[str, float] = {}
    for v in children.get(root, []):
        name = child_line[v]
        disp[name] = float(raw_qty.get(name, 0.0))

    for v in queue:
        if v == root or v not in child_line:
            continue
        name = child_line[v]
        raw = float(raw_qty.get(name, 0.0))
        par = parent[v]
        if par is None or par == root:
            disp[name] = raw
            continue
        parent_name = child_line.get(par)
        parent_disp = disp.get(
            parent_name,
            float(raw_qty.get(parent_name or "", raw)),
        )
        disp[name] = blend * raw + (1.0 - blend) * float(parent_disp)

    out = dict(raw_qty)
    out.update(disp)
    return out


def blend_ampacity_qty_with_downstream(
    line_records: list[tuple[str, str, str, LineTier]],
    amp_qty: dict[str, float],
    edges: list[tuple[str, str]],
    root_bus: str,
    *,
    downstream_weight: float = AMPACITY_DOWNSTREAM_WEIGHT,
) -> dict[str, float]:
    """Mix ampacity with continuous downstream subtree size (loading-like).

    Resulting quantities stay in ampacity-like units; map with
    ``linewidth_opendss`` using Max = max(qty) — same formula as power daisy.
    """
    w = float(min(1.0, max(0.0, downstream_weight)))
    down = compute_edge_downstream_quantity(edges, root_bus)
    if not down or w <= 0.0:
        return dict(amp_qty)

    amp_vals = [v for v in amp_qty.values() if v > 0]
    down_vals = [v for v in down.values() if v > 0]
    if not amp_vals or not down_vals:
        return dict(amp_qty)
    a_max = max(amp_vals)
    d_max = max(down_vals)
    if a_max <= 0 or d_max <= 0:
        return dict(amp_qty)

    out: dict[str, float] = {}
    for name, b1, b2, _tier in line_records:
        key = name.lower()
        a = float(amp_qty.get(key, 0.0))
        d = float(down.get(_norm_edge(b1, b2), 0.0))
        a_n = a / a_max
        d_n = d / d_max
        out[key] = a_max * ((1.0 - w) * a_n + w * d_n)
    return out


def _line_segments(
    line_records: list[tuple[str, str, str, LineTier]],
    coords: dict[str, tuple[float, float]],
    *,
    min_length: float = 3.0,
    line_qty: dict[str, float] | None = None,
    quantity_max: float | None = None,
    thickness_max: float = PAPER_THICKNESS_MAX,
    thickness_min: float = PAPER_THICKNESS_MIN,
    gamma: float = PAPER_POWER_GAMMA,
    linewidth_mapper: Any | None = None,
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """One segment per OpenDSS Line object (EPRI daisy rendering)."""
    segs: list[list[tuple[float, float]]] = []
    lws: list[float] = []
    for name, b1, b2, tier in line_records:
        if b1 not in coords or b2 not in coords:
            continue
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        if math.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        segs.append([(x1, y1), (x2, y2)])
        if linewidth_mapper is not None and line_qty is not None:
            q = line_qty.get(name.lower(), line_qty.get(name, 0.0))
            lws.append(float(linewidth_mapper(q)))
        elif line_qty is not None and quantity_max is not None:
            q = line_qty.get(name.lower(), line_qty.get(name, 0.0))
            lws.append(
                linewidth_opendss(
                    q,
                    quantity_max=quantity_max,
                    thickness_max=thickness_max,
                    thickness_min=thickness_min,
                    gamma=gamma,
                )
            )
    return segs, lws


def _line_segments_mixed_laterals(
    line_records: list[tuple[str, str, str, LineTier]],
    coords: dict[str, tuple[float, float]],
    backbone_buses: set[str],
    *,
    min_length: float = 3.0,
    line_qty: dict[str, float] | None = None,
    quantity_max: float | None = None,
    thickness_max: float = PAPER_THICKNESS_MAX,
    thickness_min: float = PAPER_THICKNESS_MIN,
    gamma: float = PAPER_POWER_GAMMA,
    lateral_fixed_lw: float | None = PAPER_LATERAL_FIXED_LW,
    linewidth_mapper: Any | None = None,
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """Service laterals fixed thin (unless ``lateral_fixed_lw is None``); backbone 1ph uses qty."""
    segs: list[list[tuple[float, float]]] = []
    lws: list[float] = []
    for name, b1, b2, _tier in line_records:
        if b1 not in coords or b2 not in coords:
            continue
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        if math.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        segs.append([(x1, y1), (x2, y2)])
        use_qty = line_qty and (
            linewidth_mapper is not None
            or (
                quantity_max
                and (lateral_fixed_lw is None or _is_backbone_glue_lateral(b1, b2, backbone_buses))
            )
        )
        if use_qty and linewidth_mapper is not None:
            q = line_qty.get(name.lower(), line_qty.get(name, 0.0))
            lws.append(float(linewidth_mapper(q)))
        elif use_qty:
            q = line_qty.get(name.lower(), line_qty.get(name, 0.0))
            lws.append(
                linewidth_opendss(
                    q,
                    quantity_max=quantity_max,
                    thickness_max=thickness_max,
                    thickness_min=thickness_min,
                    gamma=gamma,
                )
            )
        else:
            lws.append(float(PAPER_LATERAL_FIXED_LW if lateral_fixed_lw is None else lateral_fixed_lw))
    return segs, lws

def _edge_segments(
    edges: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
    *,
    min_length: float = 3.0,
    edge_qty: dict[tuple[str, str], float] | None = None,
    quantity_max: float | None = None,
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """One segment per edge; constant width per segment (OpenDSS LINE rendering)."""
    segs: list[list[tuple[float, float]]] = []
    lws: list[float] = []
    for b1, b2 in edges:
        if b1 not in coords or b2 not in coords:
            continue
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        if math.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        segs.append([(x1, y1), (x2, y2)])
        if edge_qty is not None and quantity_max is not None:
            q = edge_qty.get(_norm_edge(b1, b2), 1.0)
            lws.append(linewidth_opendss(q, quantity_max=quantity_max))
    return segs, lws


def _flatten_segments(
    segs: list[list[tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for seg in segs:
        if len(seg) == 2:
            out.append((seg[0], seg[1]))
    return out


def _dist_point_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    qx, qy, dist = _project_point_segment(px, py, x1, y1, x2, y2)
    return dist


def _project_point_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float]:
    """Return projection ``(qx, qy, distance)`` of ``(px, py)`` onto segment ``(x1,y1)-(x2,y2)``."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c2 = vx * vx + vy * vy
    if c2 <= 1e-18:
        return x1, y1, math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c2))
    qx, qy = x1 + t * vx, y1 + t * vy
    return qx, qy, math.hypot(px - qx, py - qy)


def _min_dist_to_segments(
    px: float,
    py: float,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    if not flat_segs:
        return 1e9
    return min(
        _dist_point_segment(px, py, a[0], a[1], b[0], b[1])
        for a, b in flat_segs
    )


def _snap_xy_to_network(
    x: float,
    y: float,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    max_dist_ft: float = ICON_ON_NETWORK_MAX_FT,
) -> tuple[float, float] | None:
    """Snap ``(x, y)`` onto the nearest drawn MV segment; ``None`` if too far away."""
    if not flat_segs:
        return None
    best_q: tuple[float, float] | None = None
    best_d = float("inf")
    for (x1, y1), (x2, y2) in flat_segs:
        qx, qy, dist = _project_point_segment(x, y, x1, y1, x2, y2)
        if dist < best_d:
            best_d = dist
            best_q = (qx, qy)
    if best_q is None or best_d > max_dist_ft:
        return None
    return best_q


def _dedupe_device_points_by_location(
    points: dict[str, tuple[float, float]],
    *,
    tol_ft: float = ICON_STACK_CLUSTER_FT,
) -> dict[str, tuple[float, float]]:
    """One marker per physical location (merge A/B/C phase banks at the same bus)."""
    if not points:
        return {}
    items = sorted(points.items(), key=lambda kv: kv[0])
    kept: list[tuple[str, tuple[float, float]]] = []
    for name, xy in items:
        merged = False
        for i, (_prev_name, prev_xy) in enumerate(kept):
            if math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]) <= tol_ft:
                xs = (xy[0] + prev_xy[0]) / 2.0
                ys = (xy[1] + prev_xy[1]) / 2.0
                kept[i] = (name, (xs, ys))
                merged = True
                break
        if not merged:
            kept.append((name, xy))
    return dict(kept)


def _choose_label_xy(
    anchor_x: float,
    anchor_y: float,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    placed: list[tuple[float, float]],
    dx: float,
    dy: float,
    *,
    min_line_clear: float,
    min_label_sep: float,
) -> tuple[float, float]:
    """Pick label position in data coords, away from lines and other labels."""
    best_xy = (anchor_x + 0.10 * dx, anchor_y + 0.10 * dy)
    best_score = -1.0
    span = max(dx, dy)
    for k in range(32):
        ang = 2.0 * math.pi * k / 32.0
        for r in (0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.22):
            lx = anchor_x + r * dx * math.cos(ang)
            ly = anchor_y + r * dy * math.sin(ang)
            d_line = _min_dist_to_segments(lx, ly, flat_segs)
            if placed:
                d_lab = min(math.hypot(lx - px, ly - py) for px, py in placed)
            else:
                d_lab = span
            if d_line < min_line_clear or d_lab < min_label_sep:
                continue
            score = d_line + 0.6 * d_lab
            if score > best_score:
                best_score = score
                best_xy = (lx, ly)
    return best_xy


def _estimate_legend_box(dx: float, dy: float, n_items: int) -> tuple[float, float]:
    """Rough legend width/height in data coordinates."""
    width = max(0.13 * dx, 3800.0)
    height = max((0.020 * n_items + 0.035) * dy, 0.10 * dy)
    return width, height


def _choose_legend_anchor(
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    box_w: float,
    box_h: float,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    avoid_points: list[tuple[float, float]],
    min_line_clear: float,
    point_margin: float,
) -> tuple[float, float]:
    """Place legend in the clearest empty region that does not overlap the grid."""
    dx, dy = xmax - xmin, ymax - ymin
    x_lo = xmin + 0.02 * dx
    x_hi = xmax - box_w - 0.02 * dx
    y_lo = ymin + box_h + 0.02 * dy
    y_hi = ymax - 0.01 * dy
    best_clear = -1.0
    best_ul = (x_lo, y_hi)

    def box_clear(lx: float, top_y: float) -> float:
        xs = np.linspace(lx, lx + box_w, 6)
        ys = np.linspace(top_y - box_h, top_y, 6)
        min_d = min(_min_dist_to_segments(float(x), float(y), flat_segs) for x in xs for y in ys)
        for px, py in avoid_points:
            if (
                lx - point_margin <= px <= lx + box_w + point_margin
                and top_y - box_h - point_margin <= py <= top_y + point_margin
            ):
                min_d = min(min_d, 0.0)
        return min_d

    for lx_frac in np.linspace(0.02, 0.98, 22):
        for ty_frac in np.linspace(0.08, 0.98, 24):
            lx = xmin + float(lx_frac) * dx
            if lx > x_hi:
                lx = x_hi
            if lx < x_lo:
                continue
            top_y = ymin + float(ty_frac) * dy
            if top_y > y_hi or top_y - box_h < y_lo:
                continue
            clear = box_clear(lx, top_y)
            if clear < min_line_clear:
                continue
            if clear > best_clear:
                best_clear = clear
                best_ul = (lx, top_y)
    return best_ul


def _annotate_clear(
    ax: plt.Axes,
    *,
    anchor: tuple[float, float],
    label: str,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    placed: list[tuple[float, float]],
    dx: float,
    dy: float,
    label_font_size: int,
    min_line_clear: float,
    min_label_sep: float,
) -> None:
    x, y = anchor
    lx, ly = _choose_label_xy(
        x,
        y,
        flat_segs,
        placed,
        dx,
        dy,
        min_line_clear=min_line_clear,
        min_label_sep=min_label_sep,
    )
    placed.append((lx, ly))
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(lx, ly),
        textcoords="data",
        fontsize=label_font_size,
        fontstyle="italic",
        fontweight="bold",
        color="#0b3c7c",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color="#888888",
            lw=0.7,
            shrinkA=2,
            shrinkB=2,
            connectionstyle="arc3,rad=0.15",
        ),
        zorder=25,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.92),
    )


def _reg_group_label(reg_name: str) -> str:
    m = re.match(r"vreg(\d+)", reg_name.lower())
    if m:
        return f"Reg {m.group(1)}"
    if "feeder" in reg_name.lower():
        return "Substation LTC"
    return reg_name


def resolve_bus_in_coords(
    bus: str,
    coords: dict[str, tuple[float, float]],
    *,
    bus_aliases: dict[str, str] | None = None,
) -> str | None:
    """Map a bus name (optionally phase-suffixed) to a coordinate key."""
    key = base_bus(bus)
    if key is None:
        return None
    if key in coords:
        return key
    if bus_aliases:
        alias = bus_aliases.get(key) or bus_aliases.get(bus.lower())
        if alias:
            alias_key = base_bus(alias)
            if alias_key in coords:
                return alias_key
    return None


def _resolve_highlight_group_buses(
    buses: list[str],
    coords: dict[str, tuple[float, float]],
    *,
    bus_aliases: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return unique coordinate keys found and raw bus names still missing."""
    found: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for raw in buses:
        key = resolve_bus_in_coords(raw, coords, bus_aliases=bus_aliases)
        if key is None:
            missing.append(raw)
            continue
        if key not in seen:
            seen.add(key)
            found.append(key)
    return found, missing


class _IconLegendHandle:
    """Legend handle that carries a raster icon path."""

    def __init__(
        self,
        path: str | Path,
        *,
        zoom: float = 0.12,
        label: str = "",
        raster_scale: float = SVG_ICON_RASTER_SCALE,
    ):
        self.path = Path(path)
        self.zoom = float(zoom)
        self.raster_scale = float(raster_scale)
        self._label = str(label)

    def get_label(self) -> str:
        return self._label


def _is_circled_star_marker(group: dict[str, Any]) -> bool:
    """True when a highlight group should draw a star inside a circle."""
    marker = str(group.get("marker", "o")).lower()
    marker_style = str(group.get("marker_style", "")).lower()
    if marker in CIRCLED_STAR_MARKER_ALIASES or marker_style in CIRCLED_STAR_MARKER_ALIASES:
        return True
    inner = group.get("inner_marker")
    if inner in ("*", "star"):
        return True
    return False


def _circled_star_marker_diameter_pt(size: float) -> float:
    """Scatter circle diameter in points for matplotlib ``s`` (area in pt²)."""
    return 2.0 * math.sqrt(max(float(size), 1e-9) / math.pi)


def _circled_star_fontsize(size: float, *, fontscale: float = 1.0) -> float:
    """Font size so the * glyph visually fills ~65-70% of the scatter circle."""
    diameter_pt = _circled_star_marker_diameter_pt(size)
    return max(4.0, diameter_pt * CIRCLED_STAR_FONT_FRAC * float(fontscale))


def _plot_circled_star_markers(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    star_color: str,
    size: float,
    edgecolor: str,
    linewidth: float,
    zorder: int,
    alpha: float,
    star_fontscale: float = 1.0,
) -> None:
    """Filled circle with a typographic asterisk centered inside."""
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        c=color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        marker="o",
        zorder=zorder,
        alpha=alpha,
    )
    star_fontsize = _circled_star_fontsize(size, fontscale=star_fontscale)
    for x, y in xy:
        ax.text(
            float(x),
            float(y),
            "*",
            ha="center",
            va="center",
            fontsize=star_fontsize,
            fontfamily=CIRCLED_STAR_FONT_FAMILY,
            color=star_color,
            alpha=alpha,
            zorder=zorder + 1,
            clip_on=True,
        )


class _CircledStarLegendHandle:
    """Legend handle for monitored-bus circled-star markers."""

    def __init__(
        self,
        *,
        color: str,
        edgecolor: str,
        star_color: str,
        markersize: float,
        scatter_size: float,
        star_fontscale: float = 1.0,
        label: str = "",
    ):
        self.color = str(color)
        self.edgecolor = str(edgecolor)
        self.star_color = str(star_color)
        self.markersize = float(markersize)
        self.scatter_size = float(scatter_size)
        self.star_fontscale = float(star_fontscale)
        self._label = str(label)

    def get_label(self) -> str:
        return self._label


class _HandlerCircledStarLegend(HandlerBase):
    """Render circled-star swatches in the legend."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        da = DrawingArea(handlebox.width, handlebox.height, 0, 0, clip=False)
        cx = handlebox.width / 2.0
        cy = handlebox.height / 2.0
        radius = min(handlebox.width, handlebox.height) * 0.28
        circle = Circle(
            (cx, cy),
            radius,
            facecolor=orig_handle.color,
            edgecolor=orig_handle.edgecolor,
            linewidth=0.6,
            zorder=1,
        )
        da.add_artist(circle)
        legend_diameter = 2.0 * radius
        map_fs = _circled_star_fontsize(
            orig_handle.scatter_size,
            fontscale=orig_handle.star_fontscale,
        )
        map_diameter = _circled_star_marker_diameter_pt(orig_handle.scatter_size)
        star_fs = max(4.0, map_fs * (legend_diameter / max(map_diameter, 1e-9)))
        star_text = plt.Text(
            cx,
            cy,
            "*",
            ha="center",
            va="center",
            fontsize=star_fs,
            fontfamily=CIRCLED_STAR_FONT_FAMILY,
            color=orig_handle.star_color,
            zorder=2,
        )
        da.add_artist(star_text)
        handlebox.add_artist(da)
        return handlebox


class _HandlerIconLegend(HandlerBase):
    """Render legend swatches from PNG/SVG raster icons (non-paper_reference fallback)."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        thumb = _icon_legend_thumbnail(
            orig_handle.path,
            raster_scale=min(orig_handle.raster_scale, 3.0),
        )
        sh, sw = thumb.shape[:2]
        target = min(handlebox.width, handlebox.height) * 0.85
        zoom = target / max(sh, sw, 1)
        da = DrawingArea(handlebox.width, handlebox.height, 0, 0, clip=False)
        oi = OffsetImage(thumb, zoom=zoom)
        w_disp, h_disp = sw * zoom, sh * zoom
        oi.set_offset(
            (
                (handlebox.width - w_disp) / 2.0,
                (handlebox.height - h_disp) / 2.0,
            )
        )
        da.add_artist(oi)
        handlebox.add_artist(da)
        return handlebox


def _icon_legend_thumbnail(
    icon_path: str | Path,
    *,
    raster_scale: float = 2.0,
    px: int = PAPER_LEGEND_ICON_THUMB_PX,
) -> np.ndarray:
    """Small RGBA thumbnail for legend rows (AnchoredOffsetbox / handler)."""
    img, _, _ = _load_icon_image(icon_path, raster_scale=raster_scale)
    h, w = img.shape[:2]
    step_y = max(1, h // px)
    step_x = max(1, w // px)
    return img[::step_y, ::step_x, :][:px, :px, :]


def _default_paper_icon_path(device_key: str) -> Path | None:
    fname = DEFAULT_PAPER_ICON_FILES.get(device_key)
    if not fname:
        return None
    path = DEFAULT_PAPER_ICON_DIR / fname
    return path if path.is_file() else None


def _resolve_device_icon_path(
    device_key: str,
    icon_paths: dict[str, str | Path],
) -> str | Path | None:
    if device_key in icon_paths and icon_paths[device_key] is not None:
        return icon_paths[device_key]
    return _default_paper_icon_path(device_key)


def _resolve_highlight_group_icon_path(group: dict[str, Any]) -> Path | None:
    """Resolve highlight-group icon: explicit path, catalog key, or monitored SVG fallback."""
    raw = group.get("icon_path")
    if raw is not None:
        if isinstance(raw, str) and raw in DEFAULT_PAPER_ICON_FILES:
            resolved = _default_paper_icon_path(raw)
            if resolved is not None:
                return resolved
        return _resolve_icon_path(raw)
    if group.get("use_monitored_icon", False) or _is_circled_star_marker(group):
        return _default_paper_icon_path("monitored")
    return None


def _resolve_paper_pv_legend_icon(
    icon_paths: dict[str, str | Path],
    highlight_icon_placements: list[dict[str, Any]],
) -> str | Path | None:
    for key in ("pv", "autonomous_pv", "autonomous pv"):
        resolved = _resolve_device_icon_path(key, icon_paths)
        if resolved is not None:
            return resolved
    for placement in highlight_icon_placements:
        kind = str(placement.get("kind", ""))
        if kind in {"pv", "highlight"} and placement.get("icon_path") is not None:
            return placement["icon_path"]
    return _default_paper_icon_path("pv")


def _paper_legend_line_row(label: str, *, fontsize: int) -> HPacker:
    line_da = DrawingArea(40, 16, 0, 0)
    line_da.add_artist(
        Line2D(
            [2, 34],
            [8, 8],
            color=PAPER_BACKBONE_COLOR,
            lw=3,
            transform=line_da.get_transform(),
            solid_capstyle="round",
        )
    )
    return HPacker(
        children=[line_da, TextArea(label, textprops={"fontsize": fontsize})],
        align="center",
        sep=6,
    )


def _paper_legend_icon_row(
    label: str,
    icon_path: str | Path,
    *,
    fontsize: int,
    icon_zoom: float = PAPER_LEGEND_ICON_ZOOM,
    raster_scale: float = 2.0,
) -> HPacker:
    thumb = _icon_legend_thumbnail(icon_path, raster_scale=raster_scale)
    oi = OffsetImage(thumb, zoom=icon_zoom)
    return HPacker(
        children=[oi, TextArea(label, textprops={"fontsize": fontsize})],
        align="center",
        sep=6,
    )


def _add_paper_reference_legend(
    ax: plt.Axes,
    *,
    icon_paths: dict[str, str | Path],
    highlight_icon_placements: list[dict[str, Any]],
    fontsize: int = 9,
    icon_zoom: float = PAPER_LEGEND_ICON_ZOOM,
    raster_scale: float = 2.0,
) -> None:
    """Legend with SVG icon thumbnails (matplotlib legend handlers miss raster icons)."""
    rows: list[HPacker] = [
        _paper_legend_line_row(
            "Distribution lines (width $\\propto$ loading)",
            fontsize=fontsize,
        ),
    ]
    device_rows: list[tuple[str, str]] = [
        ("substation", "Substation"),
        ("regulator", "Voltage regulator"),
        ("capacitor", "Capacitor bank"),
    ]
    for device_key, label in device_rows:
        icon_path = _resolve_device_icon_path(device_key, icon_paths)
        if icon_path is None:
            continue
        rows.append(
            _paper_legend_icon_row(
                label,
                icon_path,
                fontsize=fontsize,
                icon_zoom=icon_zoom,
                raster_scale=raster_scale,
            )
        )
    pv_icon = _resolve_paper_pv_legend_icon(icon_paths, highlight_icon_placements)
    if pv_icon is not None:
        rows.append(
            _paper_legend_icon_row(
                "PV connection point",
                pv_icon,
                fontsize=fontsize,
                icon_zoom=icon_zoom,
                raster_scale=raster_scale,
            )
        )
    box = VPacker(children=rows, align="left", sep=4)
    anchored = AnchoredOffsetbox(
        loc="upper right",
        child=box,
        pad=0.4,
        frameon=True,
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=ax.transAxes,
        borderpad=0.5,
    )
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("#666666")
    anchored.patch.set_alpha(0.97)
    ax.add_artist(anchored)


def _resolve_icon_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    repo_path = (REPO / p).resolve()
    if repo_path.is_file():
        return repo_path
    raise FileNotFoundError(f"Icon not found: {path}")


def _parse_svg_transform(transform: str | None) -> tuple[float, float, float, float, float, float]:
    """Return affine matrix (a, b, c, d, e, f) for SVG transform attribute."""
    if not transform:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    m = re.search(
        r"matrix\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)",
        transform,
    )
    if m:
        return tuple(float(v) for v in m.groups())  # type: ignore[return-value]
    m = re.search(r"translate\(\s*([-\d.eE+]+)(?:\s+([-\d.eE+]+))?\s*\)", transform)
    if m:
        tx = float(m.group(1))
        ty = float(m.group(2) or 0.0)
        return (1.0, 0.0, 0.0, 1.0, tx, ty)
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose_svg_transform(
    outer: tuple[float, float, float, float, float, float],
    inner: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_svg_transform(
    x: float,
    y: float,
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def _iter_svg_paths(
    elem: Any,
    parent_xform: tuple[float, float, float, float, float, float],
) -> list[tuple[str, str | None, str | None, float, tuple[float, float, float, float, float, float]]]:
    tag = elem.tag.split("}")[-1]
    xform = _compose_svg_transform(parent_xform, _parse_svg_transform(elem.attrib.get("transform")))
    rows: list[tuple[str, str | None, str | None, float, tuple[float, float, float, float, float, float]]] = []
    if tag == "path" and elem.attrib.get("d"):
        stroke_w = float(elem.attrib.get("stroke-width", 1.0))
        rows.append(
            (
                elem.attrib["d"],
                elem.attrib.get("fill"),
                elem.attrib.get("stroke"),
                stroke_w,
                xform,
            )
        )
    for child in elem:
        rows.extend(_iter_svg_paths(child, xform))
    return rows


def _tokenize_svg_path_d(d: str) -> list[str]:
    return re.findall(
        r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
        d.replace(",", " "),
    )


def _parse_svg_color(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None or value.lower() in {"none", "transparent"}:
        return None
    value = value.strip()
    if value.startswith("#") and len(value) in {4, 7, 9}:
        if len(value) == 4:
            r = int(value[1] * 2, 16) / 255.0
            g = int(value[2] * 2, 16) / 255.0
            b = int(value[3] * 2, 16) / 255.0
        else:
            r = int(value[1:3], 16) / 255.0
            g = int(value[3:5], 16) / 255.0
            b = int(value[5:7], 16) / 255.0
        return (r, g, b, 1.0)
    return (0.0, 0.0, 0.0, 1.0)


def _draw_svg_path_on_cairo(
    ctx: Any,
    d: str,
    *,
    fill: str | None,
    stroke: str | None,
    stroke_width: float,
    matrix: tuple[float, float, float, float, float, float],
) -> None:
    import cairo
    tokens = _tokenize_svg_path_d(d)
    if not tokens:
        return
    idx = 0
    cmd = tokens[idx]
    idx += 1
    cx = cy = 0.0
    subx = suby = 0.0

    def _read() -> float:
        nonlocal idx
        val = float(tokens[idx])
        idx += 1
        return val

    def _point(x: float, y: float) -> tuple[float, float]:
        return _apply_svg_transform(x, y, matrix)

    ctx.new_path()
    while True:
        if cmd in "Mm":
            rel = cmd.islower()
            if idx >= len(tokens):
                break
            cx = _read() + (cx if rel else 0.0)
            cy = _read() + (cy if rel else 0.0)
            px, py = _point(cx, cy)
            ctx.move_to(px, py)
            subx, suby = cx, cy
            cmd = "L" if cmd == "M" else "l"
            continue
        if cmd in "Ll":
            rel = cmd.islower()
            while idx < len(tokens) and tokens[idx] not in "MmLlHhVvCcSsQqTtAaZz":
                cx = _read() + (cx if rel else 0.0)
                cy = _read() + (cy if rel else 0.0)
                px, py = _point(cx, cy)
                ctx.line_to(px, py)
            if idx < len(tokens):
                cmd = tokens[idx]
                idx += 1
            else:
                break
            continue
        if cmd in "Hh":
            rel = cmd.islower()
            while idx < len(tokens) and tokens[idx] not in "MmLlHhVvCcSsQqTtAaZz":
                cx = _read() + (cx if rel else 0.0)
                px, py = _point(cx, cy)
                ctx.line_to(px, py)
            if idx < len(tokens):
                cmd = tokens[idx]
                idx += 1
            else:
                break
            continue
        if cmd in "Vv":
            rel = cmd.islower()
            while idx < len(tokens) and tokens[idx] not in "MmLlHhVvCcSsQqTtAaZz":
                cy = _read() + (cy if rel else 0.0)
                px, py = _point(cx, cy)
                ctx.line_to(px, py)
            if idx < len(tokens):
                cmd = tokens[idx]
                idx += 1
            else:
                break
            continue
        if cmd in "Cc":
            rel = cmd.islower()
            while idx < len(tokens) and tokens[idx] not in "MmLlHhVvCcSsQqTtAaZz":
                x1 = _read() + (cx if rel else 0.0)
                y1 = _read() + (cy if rel else 0.0)
                x2 = _read() + (cx if rel else 0.0)
                y2 = _read() + (cy if rel else 0.0)
                cx = _read() + (cx if rel else 0.0)
                cy = _read() + (cy if rel else 0.0)
                p1 = _point(x1, y1)
                p2 = _point(x2, y2)
                p3 = _point(cx, cy)
                ctx.curve_to(*p1, *p2, *p3)
            if idx < len(tokens):
                cmd = tokens[idx]
                idx += 1
            else:
                break
            continue
        if cmd in "Zz":
            ctx.close_path()
            cx, cy = subx, suby
            if idx < len(tokens):
                cmd = tokens[idx]
                idx += 1
            else:
                break
            continue
        if idx < len(tokens):
            cmd = tokens[idx]
            idx += 1
        else:
            break

    fill_rgba = _parse_svg_color(fill)
    stroke_rgba = _parse_svg_color(stroke)
    if fill_rgba is not None:
        ctx.set_source_rgba(*fill_rgba)
        if stroke_rgba is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if stroke_rgba is not None:
        ctx.set_source_rgba(*stroke_rgba)
        ctx.set_line_width(stroke_width)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()


def _svg_natural_size(path: Path) -> tuple[float, float]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass
    width = float(str(root.attrib.get("width", "64")).replace("px", ""))
    height = float(str(root.attrib.get("height", "64")).replace("px", ""))
    return width, height


def _svg_local_tag(elem: Any) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _is_svg_icon_path(icon_path: str | Path) -> bool:
    return _resolve_icon_path(icon_path).suffix.lower() == ".svg"


def _resolve_vector_svg_icons_flag(
    vector_svg_icons: bool | None,
    *icon_paths: str | Path | None,
) -> bool:
    """Auto-enable vector SVG embed when any icon path is ``.svg``."""
    if vector_svg_icons is False:
        return False
    if vector_svg_icons is True:
        return True
    return any(p is not None and _is_svg_icon_path(p) for p in icon_paths)


def _icon_display_size_points(
    zoom: float,
    natural_w: float,
    natural_h: float,
) -> tuple[float, float]:
    """Map icon zoom to on-map width/height in matplotlib SVG points."""
    max_nat = max(float(natural_w), float(natural_h), 1.0)
    base = float(zoom) * ICON_REFERENCE_PX
    return base * (float(natural_w) / max_nat), base * (float(natural_h) / max_nat)


def _data_to_axes_svg_center_pt(
    fig: plt.Figure,
    ax: plt.Axes,
    x_data: float,
    y_data: float,
    *,
    pad_inches: float | None = None,
) -> tuple[float, float]:
    """Map data coords to SVG ``axes_1`` center points (scatter / PathCollection convention).

    Matplotlib's ``get_tightbbox`` is in figure inches while ``transData`` is in display
    pixels; mix them only after ``transformed(fig.dpi_scale_trans)``. Saved SVG coords
    also include ``savefig.pad_inches`` padding around the tight crop.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if pad_inches is None:
        pad_inches = float(plt.rcParams["savefig.pad_inches"])
    pad_pt = float(pad_inches) * 72.0
    tight_px = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    x_disp, y_disp = ax.transData.transform((float(x_data), float(y_data)))
    dpi = float(fig.dpi)
    cx_pt = (x_disp - tight_px.x0) / dpi * 72.0 + pad_pt
    cy_pt = (tight_px.y1 - y_disp) / dpi * 72.0 + pad_pt
    return cx_pt, cy_pt


def _remap_svg_id_refs(elem: Any, id_map: dict[str, str]) -> None:
    for attr_name, value in list(elem.attrib.items()):
        if "url(#" in value:
            m = re.search(r"url\(#([^)]+)\)", value)
            if m and m.group(1) in id_map:
                elem.set(attr_name, value.replace(f"#{m.group(1)}", f"#{id_map[m.group(1)]}"))
        if value.startswith("#") and value[1:] in id_map:
            elem.set(attr_name, f"#{id_map[value[1:]]}")
    for child in elem:
        _remap_svg_id_refs(child, id_map)


def _prepare_svg_icon_fragment(
    icon_path: str | Path,
) -> tuple[list[Any], list[Any], float, float]:
    """Parse an icon SVG into embeddable ``(defs, body, width, height)`` fragments."""
    import xml.etree.ElementTree as ET

    path = _resolve_icon_path(icon_path)
    mtime = _icon_path_mtime(path)
    cache_key = (str(path), mtime)
    if cache_key in _SVG_ICON_FRAGMENT_CACHE:
        defs_elems, body_elems, nat_w, nat_h = _SVG_ICON_FRAGMENT_CACHE[cache_key]
        return (
            [copy.deepcopy(e) for e in defs_elems],
            [copy.deepcopy(e) for e in body_elems],
            nat_w,
            nat_h,
        )

    id_prefix = f"icon_{abs(hash(cache_key[0])) & 0xFFFFFF:06x}"
    root = ET.parse(path).getroot()
    nat_w, nat_h = _svg_natural_size(path)
    id_map: dict[str, str] = {}
    defs_elems: list[Any] = []
    body_elems: list[Any] = []
    staged: list[tuple[str, Any]] = []

    for child in root:
        tag = _svg_local_tag(child)
        elem = copy.deepcopy(child)
        if tag == "defs":
            for def_child in elem:
                staged.append(("def", def_child))
        else:
            staged.append(("body", elem))

    def _assign_ids(elem: Any) -> None:
        old_id = elem.attrib.get("id")
        if old_id:
            new_id = f"{id_prefix}_{old_id}"
            id_map[old_id] = new_id
            elem.set("id", new_id)
        for sub in elem:
            _assign_ids(sub)

    for _kind, elem in staged:
        _assign_ids(elem)

    for kind, elem in staged:
        _remap_svg_id_refs(elem, id_map)
        if kind == "def":
            defs_elems.append(elem)
        else:
            body_elems.append(elem)

    _SVG_ICON_FRAGMENT_CACHE[cache_key] = (
        [copy.deepcopy(e) for e in defs_elems],
        [copy.deepcopy(e) for e in body_elems],
        nat_w,
        nat_h,
    )
    return defs_elems, body_elems, nat_w, nat_h


def _find_svg_defs_parent(root: Any) -> Any | None:
    for elem in root.iter():
        if _svg_local_tag(elem) == "defs":
            return elem
    return None


def _find_svg_axes_group(root: Any) -> Any | None:
    for elem in root.iter():
        if elem.attrib.get("id") == "axes_1":
            return elem
    return None


def _remove_raster_annotation_bboxes(axes_group: Any) -> int:
    removed = 0
    for child in list(axes_group):
        gid = child.attrib.get("id", "")
        if gid.startswith("AnnotationBbox_"):
            axes_group.remove(child)
            removed += 1
    return removed


def _embed_vector_icons_in_svg(
    svg_path: Path,
    placements: list[dict[str, Any]],
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    pad_inches: float | None = None,
) -> int:
    """Post-process a matplotlib SVG: inject nested vector icon groups."""
    import xml.etree.ElementTree as ET

    if not placements:
        return 0

    tree = ET.parse(svg_path)
    root = tree.getroot()
    defs_parent = _find_svg_defs_parent(root)
    axes_group = _find_svg_axes_group(root)
    if axes_group is None:
        raise RuntimeError(f"Could not find axes group in {svg_path}")

    icon_doc_cache: dict[str, tuple[list[Any], list[Any], float, float]] = {}
    defs_added: set[str] = set()
    embedded = 0

    for i, placement in enumerate(
        sorted(placements, key=lambda p: (int(p.get("zorder", 0)), _icon_stack_sort_key(p)))
    ):
        icon_path = placement["icon_path"]
        if not _is_svg_icon_path(icon_path):
            continue
        resolved = str(_resolve_icon_path(icon_path))
        if resolved not in icon_doc_cache:
            icon_doc_cache[resolved] = _prepare_svg_icon_fragment(icon_path)
        defs_elems, body_elems, nat_w, nat_h = icon_doc_cache[resolved]
        body_elems = [copy.deepcopy(e) for e in body_elems]

        if defs_parent is not None and resolved not in defs_added:
            for def_elem in defs_elems:
                defs_parent.append(copy.deepcopy(def_elem))
            defs_added.add(resolved)

        w_pt, h_pt = _icon_display_size_points(
            float(placement["zoom"]),
            nat_w,
            nat_h,
        )
        cx_pt, cy_pt = _data_to_axes_svg_center_pt(
            fig,
            ax,
            float(placement["x"]),
            float(placement["y"]),
            pad_inches=pad_inches,
        )
        scale = w_pt / max(nat_w, 1e-6)
        # Icon SVG paths use standard SVG coords (y down).  ``_data_to_axes_svg_center_pt``
        # already maps matplotlib data y-up into the saved SVG axes frame, so do not
        # flip y again here (scale(..., -scale) rendered PowerPoint icons upside down).
        transform = (
            f"translate({cx_pt:.6f},{cy_pt:.6f}) "
            f"scale({scale:.8f},{scale:.8f}) "
            f"translate({-nat_w / 2.0:.6f},{-nat_h / 2.0:.6f})"
        )

        bus_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(placement.get("bus") or f"icon{i}"))
        icon_group = ET.Element(f"{{{SVG_NS}}}g", {
            "id": f"vector_icon_{i}__{bus_slug}",
            "class": "vector-device-icon",
            "data-icon-bus": str(placement.get("bus") or ""),
            "data-icon-kind": str(placement.get("kind") or ""),
        })
        inner = ET.SubElement(icon_group, f"{{{SVG_NS}}}g", {"transform": transform})
        for body_elem in body_elems:
            inner.append(body_elem)
        axes_group.append(icon_group)
        embedded += 1

    _remove_raster_annotation_bboxes(axes_group)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    return embedded


def _svg_path_coord_pairs(d: str) -> list[tuple[float, float]]:
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _svg_icon_center_radius(transform: str) -> tuple[float, float, float] | None:
    m = re.search(
        r"translate\(([^,)]+),([^)]+)\)\s+scale\(([^,)]+),([^)]+)\)\s+translate\(([^,)]+),([^)]+)\)",
        transform.replace("\n", " "),
    )
    if not m:
        return None
    cx, cy = float(m.group(1)), float(m.group(2))
    sx, sy = abs(float(m.group(3))), abs(float(m.group(4)))
    half_w = abs(float(m.group(5))) * sx
    half_h = abs(float(m.group(6))) * sy
    return cx, cy, max(half_w, half_h, 4.0)


def _svg_axes_content_bbox(axes_group: Any) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []

    def _walk(elem: Any, *, skip_patch: bool = False) -> None:
        tag = _svg_local_tag(elem)
        if skip_patch and tag == "g" and elem.attrib.get("id", "").startswith("patch_"):
            return
        if tag == "path" and "d" in elem.attrib:
            for x, y in _svg_path_coord_pairs(elem.attrib["d"]):
                xs.append(x)
                ys.append(y)
        elif tag == "g":
            gid = elem.attrib.get("id", "")
            if "vector_icon" in gid:
                for child in elem:
                    icon_pt = _svg_icon_center_radius(child.attrib.get("transform", ""))
                    if icon_pt is not None:
                        cx, cy, radius = icon_pt
                        xs.extend((cx - radius, cx + radius))
                        ys.extend((cy - radius, cy + radius))
            for child in elem:
                _walk(child, skip_patch=skip_patch)

    _walk(axes_group, skip_patch=True)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _crop_svg_viewbox_to_content(svg_path: Path, *, margin_pt: float = 4.0) -> bool:
    """Tighten SVG ``viewBox`` / width / height to visible axes content."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()
    axes_group = _find_svg_axes_group(root)
    if axes_group is None:
        return False
    bbox = _svg_axes_content_bbox(axes_group)
    if bbox is None:
        return False
    min_x, min_y, max_x, max_y = bbox
    pad = max(float(margin_pt), 0.0)
    min_x -= pad
    min_y -= pad
    max_x += pad
    max_y += pad
    width = max(max_x - min_x, 1e-3)
    height = max(max_y - min_y, 1e-3)
    root.set("viewBox", f"{min_x:.6f} {min_y:.6f} {width:.6f} {height:.6f}")
    root.set("width", f"{width:.6f}pt")
    root.set("height", f"{height:.6f}pt")
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    return True


def _figsize_for_data_aspect(
    dx: float,
    dy: float,
    *,
    style: Literal["paper", "draft"],
) -> tuple[float, float]:
    data_ar = max(float(dx), 1e-6) / max(float(dy), 1e-6)
    base_h = 8.5 if style == "paper" else 9.0
    return base_h * data_ar, base_h


_VISIO_STYLE_PROPS = (
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-opacity",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "opacity",
)


def _parse_svg_inline_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in style.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _compact_svg_path_d(d: str) -> str:
    return re.sub(r"\s+", " ", d.strip())


def _svg_path_merge_key(elem: Any) -> tuple[tuple[str, str], ...]:
    style_bits = _parse_svg_inline_style(elem.attrib.get("style", ""))
    merged: dict[str, str] = {}
    for key in _VISIO_STYLE_PROPS:
        if key in elem.attrib:
            merged[key] = elem.attrib[key]
    for key, value in style_bits.items():
        if key in _VISIO_STYLE_PROPS or key == "style":
            merged[key] = value
    return tuple(sorted(merged.items()))


def _inline_svg_style_attrs(elem: Any) -> None:
    style = elem.attrib.pop("style", None)
    if not style:
        return
    for key, value in _parse_svg_inline_style(style).items():
        if key not in elem.attrib:
            elem.set(key, value)


def _inline_styles_recursive(root: Any) -> int:
    count = 0
    for elem in root.iter():
        if "style" in elem.attrib:
            _inline_svg_style_attrs(elem)
            count += 1
    return count


def _svg_path_stroke_width(elem: Any) -> str | None:
    """Return normalized stroke-width for a path (attribute or inline style)."""
    if "stroke-width" in elem.attrib:
        return elem.attrib["stroke-width"].strip()
    style_bits = _parse_svg_inline_style(elem.attrib.get("style", ""))
    return style_bits.get("stroke-width", "").strip() or None


def _linecollection_stroke_widths(parent: Any) -> set[str]:
    widths: set[str] = set()
    for child in parent:
        if _svg_local_tag(child) != "path":
            continue
        width = _svg_path_stroke_width(child)
        if width is not None:
            widths.add(width)
    return widths


def _merge_paths_in_parent(parent: Any) -> int:
    """Merge sibling ``path`` elements that share the same paint attributes."""
    import xml.etree.ElementTree as ET

    buckets: dict[tuple[tuple[str, str], ...], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    passthrough: list[Any] = []
    removed = 0

    for child in list(parent):
        if _svg_local_tag(child) != "path" or not child.attrib.get("d"):
            passthrough.append(child)
            continue
        _inline_svg_style_attrs(child)
        key = _svg_path_merge_key(child)
        attrs = {k: v for k, v in child.attrib.items() if k not in {"d", "id", "clip-path"}}
        buckets[key].append((child.attrib["d"], attrs))

    if not buckets or all(len(v) == 1 for v in buckets.values()):
        return 0

    for child in list(parent):
        parent.remove(child)

    for child in passthrough:
        parent.append(child)

    for _key, items in buckets.items():
        if len(items) < 2:
            for _d, attrs in items:
                path_elem = ET.Element(f"{{{SVG_NS}}}path", dict(attrs))
                path_elem.set("d", _d)
                parent.append(path_elem)
            continue
        merged_d = " ".join(_compact_svg_path_d(d) for d, _ in items)
        attrs = dict(items[0][1])
        attrs["d"] = merged_d
        parent.append(ET.Element(f"{{{SVG_NS}}}path", attrs))
        removed += max(0, len(items) - 1)
    return removed


def _merge_linecollection_paths(root: Any) -> tuple[int, int]:
    """Merge paths only in uniform-width ``LineCollection_*`` groups.

    Tapered MV lines export one path per segment with its own ``stroke-width``.
    Merging across segments — even within equal-width buckets — can change caps
    and Visio import behavior, so collections with more than one width are skipped.
    """
    removed = 0
    skipped_variable = 0
    for elem in root.iter():
        gid = elem.attrib.get("id", "")
        if not gid.startswith("LineCollection_"):
            continue
        if len(_linecollection_stroke_widths(elem)) > 1:
            skipped_variable += 1
            continue
        removed += _merge_paths_in_parent(elem)
    return removed, skipped_variable


def _strip_svg_clip_paths(root: Any) -> int:
    removed = 0
    for elem in root.iter():
        if elem.attrib.pop("clip-path", None) is not None:
            removed += 1

    for defs in [e for e in root.iter() if _svg_local_tag(e) == "defs"]:
        for child in list(defs):
            if _svg_local_tag(child) == "clipPath":
                defs.remove(child)
                removed += 1
    return removed


def _strip_svg_metadata(root: Any) -> int:
    removed = 0
    for child in list(root):
        if _svg_local_tag(child) == "metadata":
            root.remove(child)
            removed += 1
    return removed


def _strip_svg_raster_images(root: Any) -> int:
    removed = 0
    for parent in list(root.iter()):
        for child in list(parent):
            tag = _svg_local_tag(child)
            if tag == "image":
                parent.remove(child)
                removed += 1
            elif tag == "g" and child.attrib.get("id", "").startswith("AnnotationBbox_"):
                parent.remove(child)
                removed += 1
    return removed


def _collapse_vector_icon_groups(root: Any) -> int:
    """Flatten ``vector-device-icon`` wrappers to a single transform group."""
    collapsed = 0
    for icon_group in [e for e in root.iter() if "vector-device-icon" in e.attrib.get("class", "")]:
        children = [c for c in icon_group if _svg_local_tag(c) == "g"]
        if len(children) != 1:
            continue
        outer = children[0]
        transform = outer.attrib.get("transform", "")
        for child in list(outer):
            icon_group.append(child)
        icon_group.remove(outer)
        if transform:
            icon_group.set("transform", transform)
        collapsed += 1
    return collapsed


def _remove_empty_svg_groups(root: Any) -> int:
    removed = 0
    changed = True
    while changed:
        changed = False
        for parent in root.iter():
            for child in list(parent):
                if _svg_local_tag(child) != "g":
                    continue
                if len(child) == 0 and not child.attrib.get("id"):
                    parent.remove(child)
                    removed += 1
                    changed = True
    return removed


def _write_clean_svg(tree: Any, svg_path: Path) -> None:
    import xml.etree.ElementTree as ET

    root = tree.getroot()
    if "{" in root.tag:
        root.tag = _svg_local_tag(root)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def postprocess_svg_for_visio(
    svg_path: Path | str,
    *,
    preserve_appearance: bool = True,
    merge_line_paths: bool | None = None,
    strip_clip_paths: bool | None = None,
    strip_raster_images: bool | None = None,
    inline_styles: bool | None = None,
    simplify_icon_groups: bool | None = None,
    remove_metadata: bool | None = None,
) -> dict[str, Any]:
    """Post-process a matplotlib SVG for Microsoft Visio import.

    With ``preserve_appearance=True`` (default), only structural cleanup that
    should not change rendering: remove ``<metadata>`` and DOCTYPE, inline CSS
    ``style`` onto SVG attributes, and drop empty groups. Clip-paths, tapered
    per-segment stroke widths, vector icon nesting, and raster legend thumbs are
    left untouched.

    With ``preserve_appearance=False``, also strip clip-paths, flatten icon
    wrappers, flatten icon
    wrappers, drop raster ``<image>`` nodes, and merge paths inside
    uniform-width ``LineCollection_*`` groups only (tapered collections are
    skipped because each segment may have a different ``stroke-width``).

    Aggressive tradeoffs: merged segments lose per-segment IDs; clip-path
    removal may expose content outside the axes box; raster legend thumbs are
    dropped when ``strip_raster_images=True``.
    """
    import xml.etree.ElementTree as ET

    if preserve_appearance:
        merge_line_paths = False if merge_line_paths is None else merge_line_paths
        strip_clip_paths = False if strip_clip_paths is None else strip_clip_paths
        strip_raster_images = False if strip_raster_images is None else strip_raster_images
        inline_styles = True if inline_styles is None else inline_styles
        simplify_icon_groups = False if simplify_icon_groups is None else simplify_icon_groups
        remove_metadata = True if remove_metadata is None else remove_metadata
    else:
        merge_line_paths = True if merge_line_paths is None else merge_line_paths
        strip_clip_paths = True if strip_clip_paths is None else strip_clip_paths
        strip_raster_images = True if strip_raster_images is None else strip_raster_images
        inline_styles = True if inline_styles is None else inline_styles
        simplify_icon_groups = True if simplify_icon_groups is None else simplify_icon_groups
        remove_metadata = True if remove_metadata is None else remove_metadata

    path = Path(svg_path)
    size_before = path.stat().st_size
    tree = ET.parse(path)
    root = tree.getroot()
    path_count_before = sum(1 for e in root.iter() if _svg_local_tag(e) == "path")

    stats: dict[str, Any] = {
        "preserve_appearance": preserve_appearance,
        "path_count_before": path_count_before,
        "bytes_before": size_before,
    }

    if remove_metadata:
        stats["metadata_removed"] = _strip_svg_metadata(root)
    if strip_clip_paths:
        stats["clip_paths_stripped"] = _strip_svg_clip_paths(root)
    if strip_raster_images:
        stats["raster_nodes_removed"] = _strip_svg_raster_images(root)
    if inline_styles:
        stats["styles_inlined"] = _inline_styles_recursive(root)
    if simplify_icon_groups:
        stats["icon_groups_collapsed"] = _collapse_vector_icon_groups(root)
    if merge_line_paths:
        merged, skipped_variable = _merge_linecollection_paths(root)
        stats["paths_merged"] = merged
        stats["linecollections_skipped_variable_width"] = skipped_variable
    stats["empty_groups_removed"] = _remove_empty_svg_groups(root)

    _write_clean_svg(tree, path)

    tree_after = ET.parse(path)
    root_after = tree_after.getroot()
    path_count_after = sum(1 for e in root_after.iter() if _svg_local_tag(e) == "path")
    size_after = path.stat().st_size
    stats.update(
        {
            "path_count_after": path_count_after,
            "bytes_after": size_after,
            "path_reduction_pct": round(100.0 * (1.0 - path_count_after / max(path_count_before, 1)), 1),
            "size_reduction_pct": round(100.0 * (1.0 - size_after / max(size_before, 1)), 1),
        }
    )
    return stats


def _try_export_emf_from_svg(svg_path: Path, emf_path: Path) -> bool:
    """Export EMF via Inkscape CLI (Windows-friendly when Inkscape is installed)."""
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        return False
    emf_path.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [inkscape, str(svg_path), f"--export-filename={emf_path}", "--export-type=emf"],
        [inkscape, str(svg_path), f"--export-filename={emf_path}"],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if emf_path.is_file() and emf_path.stat().st_size > 0:
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def _resolve_svg_to_pdf_flag(
    svg_to_pdf: bool | None,
    *,
    use_vector_svg: bool,
) -> bool:
    """Default: derive PDF from finalized SVG when vector SVG icons are enabled."""
    if svg_to_pdf is False:
        return False
    if svg_to_pdf is True:
        return True
    return bool(use_vector_svg)


def _count_images_in_pdf(pdf_path: Path) -> int:
    """Count embedded raster images across all PDF pages (``-1`` if unknown)."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            return sum(len(doc[i].get_images(full=True)) for i in range(doc.page_count))
        finally:
            doc.close()
    except Exception:
        return -1


def _pdf_preserves_svg_icons(svg_path: Path, pdf_path: Path) -> bool:
    """Return whether a PDF export kept vector-device icons from the SVG."""
    expected = _count_vector_device_icons_in_svg(svg_path)
    if expected == 0:
        return True
    actual = _count_images_in_pdf(pdf_path)
    if actual < 0:
        return True
    return actual >= expected


def _accept_pdf_from_svg_backend(
    svg_path: Path,
    pdf_path: Path,
    backend: str,
) -> str | None:
    """Keep a non-empty PDF only when embedded device icons survived conversion."""
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        return None
    if _pdf_preserves_svg_icons(svg_path, pdf_path):
        return backend
    pdf_path.unlink(missing_ok=True)
    return None


def _export_pdf_from_svg(svg_path: Path, pdf_path: Path) -> str:
    """Convert finalized SVG to vector PDF (Inkscape, cairosvg, PyMuPDF, svglib).

    Device icons are embedded in the SVG as nested groups that reference PNG
    ``<image>`` nodes via ``<use>``.  ``svglib`` renders feeder lines but drops
    those images; on Windows use PyMuPDF ``convert_to_pdf`` (``doc.save`` on an
    SVG document raises ``AssertionError``).
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    inkscape = shutil.which("inkscape")
    if inkscape is not None:
        commands = [
            [inkscape, str(svg_path), f"--export-filename={pdf_path}", "--export-type=pdf"],
            [inkscape, str(svg_path), f"--export-filename={pdf_path}"],
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                accepted = _accept_pdf_from_svg_backend(svg_path, pdf_path, "inkscape")
                if accepted is not None:
                    return accepted
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

    try:
        import cairosvg

        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        accepted = _accept_pdf_from_svg_backend(svg_path, pdf_path, "cairosvg")
        if accepted is not None:
            return accepted
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open(stream=svg_path.read_bytes(), filetype="svg")
        try:
            if doc.page_count < 1:
                raise ValueError("SVG produced no pages")
            pdf_bytes = doc.convert_to_pdf()
        finally:
            doc.close()
        pdf_path.write_bytes(pdf_bytes)
        accepted = _accept_pdf_from_svg_backend(svg_path, pdf_path, "pymupdf")
        if accepted is not None:
            return accepted
    except Exception:
        pass

    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(str(svg_path))
        if drawing is None:
            raise ValueError("svg2rlg returned None")
        renderPDF.drawToFile(drawing, str(pdf_path))
        accepted = _accept_pdf_from_svg_backend(svg_path, pdf_path, "svglib")
        if accepted is not None:
            return accepted
    except Exception:
        pass

    raise RuntimeError(
        f"Could not convert {svg_path} to PDF with device icons preserved. "
        "Install Inkscape, or pip install cairosvg, or pip install pymupdf, "
        "or pip install svglib reportlab."
    )


def _png_vector_raster_zoom(
    png_dpi: int | float | None,
    *,
    zoom: float | None = None,
) -> float:
    """PyMuPDF Matrix zoom for PDF/SVG→PNG from ``png_dpi`` (``dpi/72``).

    No minimum-zoom floor: the caller's ``png_dpi`` is used exactly.
    """
    if zoom is not None:
        return max(float(zoom), 1.0)
    dpi = float(png_dpi) if png_dpi is not None else float(PAPER_PNG_DPI_DEFAULT)
    return max(dpi / 72.0, 1.0)


def overwrite_png_from_vector(
    png_path: Path | str,
    *,
    pdf_path: Path | str | None = None,
    svg_path: Path | str | None = None,
    png_dpi: int | float | None = PAPER_PNG_DPI_DEFAULT,
    zoom: float | None = None,
    allow_dpi_fallback: bool = False,
) -> str | None:
    """Overwrite ``png_path`` by rasterizing finalized PDF (preferred) or SVG via pymupdf.

    Uses ``Matrix(z, z)`` with ``z = png_dpi/72`` (or an explicit ``zoom``).
    By default there is **no** lower-DPI retry ladder — either the requested
    resolution succeeds or this returns ``None`` (unless
    ``allow_dpi_fallback=True``, which re-enables legacy 8→6→4 zoom retries).
    """
    try:
        import fitz
    except ImportError:
        return None

    png_path = Path(png_path)
    candidates: list[tuple[str, Path]] = []
    if pdf_path is not None:
        pdf_p = Path(pdf_path)
        if pdf_p.is_file() and pdf_p.stat().st_size > 0:
            candidates.append(("pdf", pdf_p))
    if svg_path is not None:
        svg_p = Path(svg_path)
        if svg_p.is_file() and svg_p.stat().st_size > 0:
            candidates.append(("svg", svg_p))
    if not candidates:
        return None

    target_z = _png_vector_raster_zoom(png_dpi, zoom=zoom)
    if allow_dpi_fallback:
        zoom_attempts: list[float] = []
        for z in (target_z, 8.0, 6.0, float(PAPER_PNG_FROM_VECTOR_ZOOM_DEFAULT)):
            z = float(z)
            if z >= 1.0 and z not in zoom_attempts:
                zoom_attempts.append(z)
    else:
        zoom_attempts = [float(target_z)]

    png_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None

    for kind, src in candidates:
        for z in zoom_attempts:
            try:
                if kind == "svg":
                    doc = fitz.open(stream=src.read_bytes(), filetype="svg")
                else:
                    doc = fitz.open(str(src))
                try:
                    if doc.page_count < 1:
                        raise ValueError(f"{src} produced no pages")
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
                    pix.save(str(png_path))
                finally:
                    doc.close()
                if png_path.is_file() and png_path.stat().st_size > 0:
                    if abs(z - target_z) > 1e-9:
                        return f"pymupdf-{kind}@zoom={z:g}(FALLBACK from {target_z:g})"
                    return f"pymupdf-{kind}@zoom={z:g}(dpi≈{z * 72:g})"
            except Exception as exc:
                last_err = exc
                continue
    if last_err is not None and not allow_dpi_fallback:
        print(
            f"  PNG @ requested dpi failed (no fallback): {type(last_err).__name__}: {last_err}"
        )
    return None


def _count_vector_device_icons_in_svg(svg_path: Path) -> int:
    """Count nested vector icon groups embedded by ``_embed_vector_icons_in_svg``."""
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    return text.count('class="vector-device-icon"')


def _icon_natural_extent(path: Path) -> float:
    """Largest pixel dimension of a PNG/JPG/SVG icon (for zoom normalization)."""
    if path.suffix.lower() == ".svg":
        width, height = _svg_natural_size(path)
        return max(width, height, 1.0)
    img = mpimg.imread(path)
    return float(max(img.shape[0], img.shape[1], 1))


def _effective_icon_zoom(zoom: float, zoom_corr: float, natural_px: float) -> float:
    """Map user zoom to OffsetImage zoom, correcting raster scale and SVG pixel size."""
    return zoom * zoom_corr * (ICON_REFERENCE_PX / max(natural_px, 1.0))


def _make_map_offset_image(img: np.ndarray, zoom: float) -> OffsetImage:
    """High-res map icon with nearest-neighbor downscale for crisp edges."""
    return OffsetImage(img, zoom=zoom, interpolation=MAP_ICON_INTERPOLATION)


def _rgba_uint8_to_float(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {arr.shape}")
    if arr.shape[2] == 4:
        return arr.astype(np.float32) / 255.0
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=2).astype(np.float32) / 255.0
    raise ValueError(f"unsupported channel count: {arr.shape[2]}")


def _rasterize_svg_icon_pycairo(path: Path, *, scale: float) -> np.ndarray:
    import cairo
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    width, height = _svg_natural_size(path)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, out_w, out_h)
    ctx = cairo.Context(surface)
    ctx.scale(out_w / width, out_h / height)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    for d, fill, stroke, stroke_w, matrix in _iter_svg_paths(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)):
        _draw_svg_path_on_cairo(
            ctx,
            d,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_w,
            matrix=matrix,
        )

    buf = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(out_h, out_w, 4)
    rgba = buf.astype(np.float32) / 255.0
    # Cairo ARGB32 byte order -> RGBA float for matplotlib.
    return rgba[:, :, [2, 1, 0, 3]]


def _rasterize_svg_icon_svglib(path: Path, *, scale: float) -> np.ndarray:
    from io import BytesIO

    from svglib.svglib import svg2rlg

    drawing = svg2rlg(str(path))
    if drawing is None:
        raise ValueError("svglib could not parse SVG")

    src_w = float(drawing.width or 64.0)
    src_h = float(drawing.height or 64.0)
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))

    try:
        from PIL import Image
        from reportlab.graphics import renderPM

        scaled = svg2rlg(str(path))
        if scaled is None:
            raise ValueError("svglib could not parse SVG")
        scaled.width = out_w
        scaled.height = out_h
        scaled.scale(out_w / src_w, out_h / src_h)
        png = renderPM.drawToString(scaled, fmt="PNG")
        return _rgba_uint8_to_float(np.asarray(Image.open(BytesIO(png)).convert("RGBA")))
    except Exception:
        pass

    # renderPM needs libcairo on Windows; PDF rasterization via pymupdf does not.
    try:
        import fitz
        from reportlab.graphics import renderPDF
    except ImportError as exc:
        raise ImportError(
            "svglib fallback needs pymupdf when renderPM is unavailable: pip install pymupdf"
        ) from exc

    buf = BytesIO()
    renderPDF.drawToFile(drawing, buf)
    page = fitz.open(stream=buf.getvalue(), filetype="pdf")[0]
    zoom_x = out_w / max(page.rect.width, 1e-6)
    zoom_y = out_h / max(page.rect.height, 1e-6)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=True)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return _rgba_uint8_to_float(arr)


def _rasterize_svg_icon_cairosvg(path: Path, *, scale: float) -> np.ndarray:
    from io import BytesIO

    import cairosvg
    from PIL import Image

    width, height = _svg_natural_size(path)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    png = cairosvg.svg2png(url=str(path), output_width=out_w, output_height=out_h)
    return _rgba_uint8_to_float(np.asarray(Image.open(BytesIO(png)).convert("RGBA")))


def _rasterize_svg_icon(path: Path, *, scale: float) -> np.ndarray:
    errors: list[str] = []
    for name, fn in (
        ("pycairo", _rasterize_svg_icon_pycairo),
        ("svglib", _rasterize_svg_icon_svglib),
        ("cairosvg", _rasterize_svg_icon_cairosvg),
    ):
        try:
            return fn(path, scale=scale)
        except ImportError as exc:
            errors.append(f"{name}: missing dependency ({exc})")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"Failed to rasterize SVG icon {path}.\n"
        + "\n".join(f"  - {line}" for line in errors)
        + "\nInstall one of:\n"
        "  pip install pycairo\n"
        "  pip install svglib reportlab pillow  "
        "(on Windows without libcairo, also: pip install pymupdf)\n"
        "  pip install cairosvg  (needs libcairo; pycairo wheel is easier on Windows)"
    )


def _effective_raster_scale(natural_px: float, raster_scale: float) -> float:
    """Clamp supersample factor so large PowerPoint SVGs stay within memory."""
    if natural_px <= 0:
        return raster_scale
    return min(float(raster_scale), SVG_ICON_RASTER_MAX_PX / natural_px)


def _load_icon_image(
    icon_path: str | Path,
    *,
    raster_scale: float = SVG_ICON_RASTER_SCALE,
) -> tuple[np.ndarray, float, float]:
    """Load PNG/JPG directly; rasterize SVG at ``raster_scale`` for sharp embeds."""
    path = _resolve_icon_path(icon_path)
    mtime = _icon_path_mtime(path)
    cache_key = (str(path), float(raster_scale), mtime)
    if cache_key in _ICON_IMAGE_CACHE:
        return _ICON_IMAGE_CACHE[cache_key]

    natural_px = _icon_natural_extent(path)
    if path.suffix.lower() == ".svg":
        eff_scale = _effective_raster_scale(natural_px, raster_scale)
        img = _rasterize_svg_icon(path, scale=eff_scale)
        raster_max = float(max(img.shape[0], img.shape[1], 1))
        zoom_corr = natural_px / raster_max
    else:
        img = mpimg.imread(path)
        zoom_corr = 1.0

    _ICON_IMAGE_CACHE[cache_key] = (img, zoom_corr, natural_px)
    return img, zoom_corr, natural_px


def _place_bus_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    icon_path: str | Path,
    *,
    zoom: float,
    zorder: int,
    raster_scale: float = SVG_ICON_RASTER_SCALE,
) -> None:
    img, zoom_corr, natural_px = _load_icon_image(icon_path, raster_scale=raster_scale)
    imagebox = _make_map_offset_image(
        img,
        zoom=_effective_icon_zoom(zoom, zoom_corr, natural_px),
    )
    ab = AnnotationBbox(
        imagebox,
        (x, y),
        frameon=False,
        pad=0.0,
        zorder=zorder,
    )
    ax.add_artist(ab)


def _place_bus_icons(
    ax: plt.Axes,
    xy: np.ndarray,
    icon_path: str | Path,
    *,
    zoom: float,
    zorder: int,
    raster_scale: float = SVG_ICON_RASTER_SCALE,
) -> None:
    for x, y in xy:
        _place_bus_icon(
            ax,
            float(x),
            float(y),
            icon_path,
            zoom=zoom,
            zorder=zorder,
            raster_scale=raster_scale,
        )


def _is_monitored_highlight_group(group: dict[str, Any]) -> bool:
    if group.get("use_monitored_icon", False) or _is_circled_star_marker(group):
        return True
    raw = group.get("icon_path")
    if raw is not None and "monitored" in str(raw).lower():
        return True
    return str(group.get("kind", "")) == "highlight"


def _is_monitored_icon_placement(placement: dict[str, Any]) -> bool:
    """True when a placement uses the monitored-bus artwork."""
    if str(placement.get("kind", "")) == "highlight":
        return True
    raw = placement.get("icon_path")
    return raw is not None and "monitored" in str(raw).lower()


def _is_bottom_stack_icon(placement: dict[str, Any]) -> bool:
    """True when an icon should stay at the exact bus anchor (under other icons)."""
    stack_priority = placement.get("stack_priority")
    if stack_priority == "bottom":
        return True
    if stack_priority == "top":
        return False
    if _is_monitored_icon_placement(placement):
        return True
    kind = str(placement.get("kind", ""))
    if kind == "highlight":
        return True
    return ICON_STACK_PRIORITY.get(kind, 1) < 0


def _icon_stack_sort_key(placement: dict[str, Any]) -> tuple[int, int]:
    """Lower priority sorts first (bottom/back layer in a co-located stack)."""
    stack_priority = placement.get("stack_priority")
    if stack_priority == "bottom":
        pri = -2
    elif stack_priority == "top":
        pri = 10_000
    else:
        pri = ICON_STACK_PRIORITY.get(str(placement.get("kind", "")), 1)
    return (pri, int(placement.get("zorder", 0)))


def _placement_bus_key(placement: dict[str, Any]) -> str | None:
    """Phase-stripped bus id for co-location matching."""
    bus = str(placement.get("bus") or "").strip()
    if not bus:
        return None
    return base_bus(bus)


def _placements_share_bus_or_location(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    tol_ft: float = ICON_STACK_CLUSTER_FT,
) -> bool:
    """True when two placements are the same bus or within stack cluster distance."""
    key_a = _placement_bus_key(a)
    key_b = _placement_bus_key(b)
    if key_a and key_b and key_a == key_b:
        return True
    ax, ay = float(a["x"]), float(a["y"])
    bx, by = float(b["x"]), float(b["y"])
    return math.hypot(ax - bx, ay - by) <= tol_ft


def _icon_footprint_radius_display(
    placement: dict[str, Any],
    *,
    dpi: float,
) -> float:
    """Icon radius in matplotlib display pixels."""
    zoom = float(placement.get("zoom", 0.1))
    icon_path = placement.get("icon_path")
    nat_w, nat_h = 120.0, 105.0
    if icon_path is not None:
        try:
            nat_w, nat_h = _svg_natural_size(icon_path)
        except (OSError, ValueError, RuntimeError):
            pass
    w_pt, h_pt = _icon_display_size_points(zoom, nat_w, nat_h)
    radius_pt = max(w_pt, h_pt) / 2.0
    return radius_pt * float(dpi) / 72.0


def _icon_footprint_radius_ft(
    placement: dict[str, Any],
    *,
    map_span_ft: float,
) -> float:
    """Approximate icon radius in data feet for overlap suppression."""
    zoom = float(placement.get("zoom", 0.1))
    icon_path = placement.get("icon_path")
    nat_w, nat_h = 120.0, 105.0
    if icon_path is not None:
        try:
            nat_w, nat_h = _svg_natural_size(icon_path)
        except (OSError, ValueError, RuntimeError):
            pass
    w_pt, h_pt = _icon_display_size_points(zoom, nat_w, nat_h)
    radius_pt = max(w_pt, h_pt) / 2.0
    span = max(float(map_span_ft), 1.0)
    axes_pt = max(MONITORED_ICON_FOOTPRINT_AXES_PT, 1.0)
    return radius_pt * (span / axes_pt)


def _placement_display_distance(
    ax: plt.Axes,
    a: dict[str, Any],
    b: dict[str, Any],
) -> float:
    p1 = ax.transData.transform((float(a["x"]), float(a["y"])))
    p2 = ax.transData.transform((float(b["x"]), float(b["y"])))
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _monitored_icon_should_suppress(
    monitored: dict[str, Any],
    other: dict[str, Any],
    *,
    tol_ft: float,
    map_span_ft: float | None,
    der_keys: set[str],
    ax: plt.Axes | None = None,
) -> bool:
    """True when a monitored icon should yield to another placement."""
    mon_key = _placement_bus_key(monitored)
    other_key = _placement_bus_key(other)
    if mon_key and mon_key in der_keys:
        return True
    if mon_key and other_key and mon_key == other_key:
        return True

    if ax is not None:
        dist = _placement_display_distance(ax, monitored, other)
        dpi = float(ax.figure.dpi)
        r_mon = _icon_footprint_radius_display(monitored, dpi=dpi)
        r_other = _icon_footprint_radius_display(other, dpi=dpi)
        return dist <= r_mon + r_other + MONITORED_ICON_SUPPRESS_DISPLAY_PAD_PX

    dist = math.hypot(
        float(monitored["x"]) - float(other["x"]),
        float(monitored["y"]) - float(other["y"]),
    )
    threshold = float(tol_ft)
    if map_span_ft is not None:
        r_mon = _icon_footprint_radius_ft(monitored, map_span_ft=map_span_ft)
        r_other = _icon_footprint_radius_ft(other, map_span_ft=map_span_ft)
        threshold = max(threshold, r_mon + r_other)
    return dist <= threshold


def _der_highlight_suppress_bus_keys(
    placements: list[dict[str, Any]],
    *,
    extra_keys: Iterable[str] | None = None,
) -> set[str]:
    """Bus keys where monitored-bus icons must never appear (ISO-DSO DER sites)."""
    keys: set[str] = set(DER_HIGHLIGHT_BUS_SUPPRESS_KEYS)
    if extra_keys:
        for raw in extra_keys:
            key = base_bus(str(raw))
            if key:
                keys.add(key)
    for placement in placements:
        kind = str(placement.get("kind", ""))
        path = str(placement.get("icon_path", "")).lower()
        if kind == "der_highlight" or any(
            tok in path
            for tok in ("controlable", "controllable", "battery", "autonomous pv")
        ):
            key = _placement_bus_key(placement)
            if key:
                keys.add(key)
    return keys


def _suppress_monitored_icon_placements(
    placements: list[dict[str, Any]],
    *,
    tol_ft: float = MONITORED_ICON_SUPPRESS_FT,
    der_bus_keys: Iterable[str] | None = None,
    map_span_ft: float | None = None,
    ax: plt.Axes | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop monitored-bus icons co-located with any other icon or explicit DER buses."""
    explicit_der_keys = _der_highlight_suppress_bus_keys(placements, extra_keys=der_bus_keys)
    monitored: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for placement in placements:
        if _is_monitored_icon_placement(placement):
            monitored.append(placement)
        else:
            other.append(placement)
    if not monitored:
        return placements, []

    kept: list[dict[str, Any]] = []
    skipped_tags: list[str] = []
    for placement in monitored:
        if any(
            _monitored_icon_should_suppress(
                placement,
                other_icon,
                tol_ft=tol_ft,
                map_span_ft=map_span_ft,
                der_keys=explicit_der_keys,
                ax=ax,
            )
            for other_icon in other
        ):
            tag = placement.get("bus") or placement.get("label") or _placement_bus_key(placement) or "monitored"
            skipped_tags.append(str(tag))
            continue
        kept.append(placement)

    if not skipped_tags:
        return placements, []
    return other + kept, skipped_tags


def _drop_bottom_stack_icons_when_coclustered(
    placements: list[dict[str, Any]],
    *,
    tol_ft: float = ICON_STACK_CLUSTER_FT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Omit bottom-stack icons when any other icon shares the same bus/location."""
    bottom: list[dict[str, Any]] = []
    upper: list[dict[str, Any]] = []
    for p in placements:
        if _is_bottom_stack_icon(p) and not _is_monitored_icon_placement(p):
            bottom.append(p)
        else:
            upper.append(p)
    if not bottom or not upper:
        return placements, []

    kept_bottom: list[dict[str, Any]] = []
    skipped_tags: list[str] = []
    for p in bottom:
        if any(
            _placements_share_bus_or_location(p, other, tol_ft=tol_ft)
            for other in upper
        ):
            tag = p.get("bus") or p.get("label") or p.get("kind", "icon")
            skipped_tags.append(str(tag))
        else:
            kept_bottom.append(p)

    if not skipped_tags:
        return placements, []

    return upper + kept_bottom, skipped_tags


_STACK_DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "up_left": (-1.0, 1.0),
    "up_right": (1.0, 1.0),
    "down_left": (-1.0, -1.0),
    "down_right": (1.0, -1.0),
}


def _normalize_stack_direction(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in _STACK_DIRECTION_VECTORS else None


def _infer_der_stack_direction(placement: dict[str, Any]) -> str | None:
    explicit = _normalize_stack_direction(placement.get("stack_direction"))
    if explicit is not None:
        return explicit
    path = str(placement.get("icon_path", "")).lower()
    if "battery" in path or "storage" in path:
        return "up_left"
    if "pv" in path:
        return "down_right"
    return None


def _infer_cluster_stack_direction(
    placement: dict[str, Any],
    *,
    der_collocated: bool,
) -> str | None:
    if placement.get("stack_direction") is not None:
        return _normalize_stack_direction(placement.get("stack_direction"))
    kind = str(placement.get("kind", ""))
    if der_collocated and kind == "der_highlight":
        return _infer_der_stack_direction(placement)
    return None


def _placement_stack_offset_ft(
    placement: dict[str, Any],
    *,
    base_step: float,
    wide_offset: bool,
) -> float:
    custom = placement.get("stack_offset_ft")
    if custom is not None:
        return float(custom)
    if wide_offset:
        return ICON_STACK_DER_OFFSET_FT
    return base_step


def _icon_placement(
    *,
    x: float,
    y: float,
    icon_path: str | Path,
    zoom: float,
    zorder: int,
    kind: str,
    bus: str = "",
    label: str = "",
    stack_priority: str | None = None,
    stack_offset_ft: float | None = None,
    stack_direction: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "x": float(x),
        "y": float(y),
        "icon_path": icon_path,
        "zoom": float(zoom),
        "zorder": int(zorder),
        "kind": kind,
        "bus": bus,
        "label": label,
    }
    if stack_priority is not None:
        out["stack_priority"] = str(stack_priority)
    if stack_offset_ft is not None:
        out["stack_offset_ft"] = float(stack_offset_ft)
    if stack_direction is not None:
        out["stack_direction"] = str(stack_direction)
    return out


def _cluster_icon_placements(
    placements: list[dict[str, Any]],
    *,
    tol_ft: float = ICON_STACK_CLUSTER_FT,
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    anchors: list[tuple[float, float]] = []
    for placement in placements:
        x, y = placement["x"], placement["y"]
        assigned = False
        for i, (ax_, ay) in enumerate(anchors):
            if math.hypot(x - ax_, y - ay) <= tol_ft:
                clusters[i].append(placement)
                n = len(clusters[i])
                anchors[i] = (
                    (ax_ * (n - 1) + x) / n,
                    (ay * (n - 1) + y) / n,
                )
                assigned = True
                break
        if not assigned:
            clusters.append([placement])
            anchors.append((x, y))
    return clusters


def _resolve_cluster_icon_placements(
    cluster: list[dict[str, Any]],
    *,
    step: float,
) -> list[dict[str, Any]]:
    """Compute final stacked coordinates for one icon cluster."""
    cluster = sorted(cluster, key=_icon_stack_sort_key)
    anchor_x = float(np.mean([p["x"] for p in cluster]))
    anchor_y = float(np.mean([p["y"] for p in cluster]))
    bottom = [p for p in cluster if _is_bottom_stack_icon(p)]
    upper = [p for p in cluster if not _is_bottom_stack_icon(p)]
    upper.sort(key=_icon_stack_sort_key)
    cluster_base_z = min(int(p["zorder"]) for p in cluster)
    layer = 0
    resolved: list[dict[str, Any]] = []

    for p in bottom:
        placed = dict(p)
        placed["x"] = anchor_x
        placed["y"] = anchor_y
        placed["zorder"] = cluster_base_z + layer
        resolved.append(placed)
        layer += 1

    if not upper:
        return resolved

    der_upper = [p for p in upper if str(p.get("kind", "")) == "der_highlight"]
    der_collocated = len(der_upper) >= 2
    wide_offset = der_collocated
    directions = [
        _infer_cluster_stack_direction(
            p,
            der_collocated=der_collocated,
        )
        for p in upper
    ]
    use_directional = any(directions)

    if use_directional:
        for p, dir_key in zip(upper, directions):
            if dir_key is None:
                dir_key = "up_right"
            vx, vy = _STACK_DIRECTION_VECTORS[dir_key]
            offset = _placement_stack_offset_ft(
                p,
                base_step=step,
                wide_offset=wide_offset,
            )
            placed = dict(p)
            placed["x"] = anchor_x + vx * offset
            placed["y"] = anchor_y + vy * offset
            placed["zorder"] = cluster_base_z + layer
            resolved.append(placed)
            layer += 1
        return resolved

    n_upper = len(upper)
    eff_step = ICON_STACK_DER_OFFSET_FT if wide_offset else step
    for i, p in enumerate(upper):
        if bottom:
            ox = (i + 1) * eff_step
            oy = (i + 1) * eff_step
        else:
            ox = (i - (n_upper - 1) / 2.0) * eff_step
            oy = (i - (n_upper - 1) / 2.0) * eff_step
        placed = dict(p)
        placed["x"] = anchor_x + ox
        placed["y"] = anchor_y + oy
        placed["zorder"] = cluster_base_z + layer
        resolved.append(placed)
        layer += 1
    return resolved


def _place_icon_placements_stacked(
    ax: plt.Axes,
    placements: list[dict[str, Any]],
    *,
    dx: float,
    dy: float,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    raster_scale: float = SVG_ICON_RASTER_SCALE,
    stack_step_frac: float = ICON_STACK_STEP_FRAC,
    on_network_max_ft: float = ICON_ON_NETWORK_MAX_FT,
    vector_svg: bool = False,
    vector_placements_out: list[dict[str, Any]] | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Place icons with diagonal stacking at coincident / nearby buses."""
    valid: list[dict[str, Any]] = []
    skipped: list[str] = []
    for p in placements:
        # Highlight/monitoring buses use exact bus coordinates (same as scatter markers).
        # Substation + Substation LTC stay pinned (LTC sits ~40 ft from source — at the
        # on-network cutoff — and was getting dropped / buried under the transformer).
        if str(p.get("kind", "")) == "highlight" or p.get("keep_on_map"):
            valid.append(dict(p))
            continue
        snapped = _snap_xy_to_network(
            p["x"],
            p["y"],
            flat_segs,
            max_dist_ft=on_network_max_ft,
        )
        if snapped is None:
            tag = p.get("bus") or p.get("label") or p.get("kind", "icon")
            skipped.append(str(tag))
            continue
        q = dict(p)
        q["x"], q["y"] = snapped
        valid.append(q)

    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = " ..." if len(skipped) > 8 else ""
        print(f"  Skipped {len(skipped)} off-network icon(s): {preview}{suffix}")

    step = stack_step_frac * max(dx, dy)
    avoid_xy: list[tuple[float, float]] = []

    def _emit_placement(placed: dict[str, Any], *, zorder: int) -> None:
        placed = dict(placed)
        placed["zorder"] = int(zorder)
        x = float(placed["x"])
        y = float(placed["y"])
        if vector_svg and _is_svg_icon_path(placed["icon_path"]):
            if vector_placements_out is not None:
                vector_placements_out.append(placed)
        else:
            _place_bus_icon(
                ax,
                x,
                y,
                placed["icon_path"],
                zoom=float(placed["zoom"]),
                zorder=int(zorder),
                raster_scale=raster_scale,
            )
        avoid_xy.append((x, y))

    resolved_all: list[dict[str, Any]] = []
    for cluster in _cluster_icon_placements(valid):
        resolved_all.extend(_resolve_cluster_icon_placements(cluster, step=step))

    ax.figure.canvas.draw()
    resolved_all, post_suppressed = _suppress_monitored_icon_placements(
        resolved_all,
        map_span_ft=max(dx, dy),
        ax=ax,
    )
    if post_suppressed:
        preview = ", ".join(post_suppressed[:8])
        suffix = " ..." if len(post_suppressed) > 8 else ""
        print(
            f"  Suppressed {len(post_suppressed)} monitored-bus icon(s) "
            f"after stack layout: {preview}{suffix}"
        )

    for placed in resolved_all:
        _emit_placement(placed, zorder=int(placed["zorder"]))
    return avoid_xy, skipped


def _scatter_or_icons(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    icon_path: str | Path | None,
    icon_zoom: float,
    zorder: int,
    scatter_kwargs: dict[str, Any],
    raster_scale: float = SVG_ICON_RASTER_SCALE,
) -> None:
    if icon_path is not None:
        _place_bus_icons(
            ax,
            xy,
            icon_path,
            zoom=icon_zoom,
            zorder=zorder,
            raster_scale=raster_scale,
        )
        return
    ax.scatter(xy[:, 0], xy[:, 1], zorder=zorder, **scatter_kwargs)


def _plot_highlight_bus_groups(
    ax: plt.Axes,
    *,
    groups: list[dict[str, Any]],
    coords: dict[str, tuple[float, float]],
    bus_aliases: dict[str, str] | None,
    flat_segs: list[tuple[tuple[float, float], tuple[float, float]]],
    dx: float,
    dy: float,
    label_font_size: int,
    min_line_clear: float,
    min_label_sep: float,
    annotate_labels: bool,
    raster_scale: float = SVG_ICON_RASTER_SCALE,
    legend_mode: str = "full",
    paper_icon_zoom: float = PAPER_UNIFORM_ICON_ZOOM,
) -> tuple[list[Line2D], list[tuple[float, float]], dict[str, str], list[dict[str, Any]]]:
    """Scatter highlight groups; return icon placements for stacked drawing."""
    legend_items: list[Line2D] = []
    avoid_xy: list[tuple[float, float]] = []
    annotations: dict[str, str] = {}
    placed_ann: list[tuple[float, float]] = []
    icon_placements: list[dict[str, Any]] = []

    for group in groups:
        buses = group.get("buses") or []
        if not buses:
            continue
        if _is_monitored_highlight_group(group):
            der_keys = _der_highlight_suppress_bus_keys(
                [],
                extra_keys=DER_HIGHLIGHT_BUS_SUPPRESS_KEYS,
            )
            buses = [
                raw
                for raw in buses
                if base_bus(str(raw)) not in der_keys
            ]
            if not buses:
                continue
        resolved, missing = _resolve_highlight_group_buses(
            buses,
            coords,
            bus_aliases=bus_aliases,
        )
        if missing:
            print(
                f"  Highlight group {group.get('legend', '(unnamed)')}: "
                f"{len(missing)} bus(es) without coordinates: {', '.join(missing[:8])}"
                + (" ..." if len(missing) > 8 else "")
            )
        if not resolved:
            continue

        color = str(group.get("color", "#d62728"))
        marker = str(group.get("marker", "o"))
        size = float(group.get("size", 36.0))
        edgecolor = str(group.get("edgecolor", "#1a1a1a"))
        linewidth = float(group.get("linewidth", 0.45))
        zorder = int(group.get("zorder", 14))
        alpha = float(group.get("alpha", 0.95))
        legend = group.get("legend")
        icon_path = _resolve_highlight_group_icon_path(group)
        icon_zoom = float(group.get("icon_zoom", paper_icon_zoom))
        stack_priority = group.get("stack_priority")
        group_kind = str(group.get("kind", "")).strip()
        if not group_kind:
            group_kind = "highlight" if _is_monitored_highlight_group(group) else "der_highlight"
        if stack_priority is None:
            stack_priority = "bottom" if group_kind == "highlight" else "top"
        include_in_legend = group.get("include_in_legend", True)
        if legend_mode == "paper_reference":
            include_in_legend = False
            if icon_path is not None:
                icon_zoom = paper_icon_zoom
            elif icon_path is None:
                size = min(size, PAPER_MONITORED_BUS_SCATTER_SIZE)
                linewidth = min(linewidth, 0.35)

        xy = np.array([coords[b] for b in resolved])
        if icon_path is not None:
            for bus, (x, y) in zip(resolved, xy):
                icon_placements.append(
                    _icon_placement(
                        x=float(x),
                        y=float(y),
                        icon_path=icon_path,
                        zoom=icon_zoom,
                        zorder=zorder,
                        kind=group_kind,
                        bus=bus,
                        label=str(legend or ""),
                        stack_priority=str(stack_priority),
                        stack_offset_ft=group.get("stack_offset_ft"),
                        stack_direction=group.get("stack_direction"),
                    )
                )
        else:
            star_color = str(group.get("star_color", edgecolor))
            star_fontscale = float(group.get("star_fontscale", 1.0))
            if _is_circled_star_marker(group):
                _plot_circled_star_markers(
                    ax,
                    xy,
                    color=color,
                    star_color=star_color,
                    size=size,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=zorder,
                    alpha=alpha,
                    star_fontscale=star_fontscale,
                )
            else:
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=size,
                    c=color,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    marker=marker,
                    zorder=zorder,
                    alpha=alpha,
                )
            avoid_xy.extend((float(x), float(y)) for x, y in xy)

        if legend and include_in_legend:
            if icon_path is not None:
                legend_items.append(
                    _IconLegendHandle(
                        icon_path,
                        zoom=icon_zoom * 1.35,
                        label=str(legend),
                        raster_scale=raster_scale,
                    )
                )
            else:
                if _is_circled_star_marker(group):
                    legend_items.append(
                        _CircledStarLegendHandle(
                            color=color,
                            edgecolor=edgecolor,
                            star_color=str(group.get("star_color", edgecolor)),
                            markersize=max(5.0, min(12.0, size ** 0.5)),
                            scatter_size=size,
                            star_fontscale=star_fontscale,
                            label=str(legend),
                        )
                    )
                else:
                    legend_items.append(
                        Line2D(
                            [0],
                            [0],
                            marker=marker,
                            color="w",
                            label=str(legend),
                            markerfacecolor=color,
                            markeredgecolor=edgecolor,
                            markersize=max(5.0, min(12.0, size ** 0.5)),
                        )
                    )

        labels = group.get("labels") or {}
        for raw_bus, text in labels.items():
            key = resolve_bus_in_coords(raw_bus, coords, bus_aliases=bus_aliases)
            if key is None or not text:
                continue
            if key in annotations:
                if text not in annotations[key].split(" / "):
                    annotations[key] = f"{annotations[key]} / {text}"
            else:
                annotations[key] = str(text)

    if annotate_labels:
        for key in sorted(annotations):
            x, y = coords[key]
            _annotate_clear(
                ax,
                anchor=(x, y),
                label=annotations[key],
                flat_segs=flat_segs,
                placed=placed_ann,
                dx=dx,
                dy=dy,
                label_font_size=label_font_size,
                min_line_clear=min_line_clear,
                min_label_sep=min_label_sep,
            )

    return legend_items, avoid_xy, annotations, icon_placements


def _dedupe_regulator_points(
    regulator_points: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """One marker per regulator bank (merge A/B/C phase controls)."""
    buckets: dict[str, list[tuple[float, float]]] = {}
    for name, xy in regulator_points.items():
        buckets.setdefault(_reg_group_label(name), []).append(xy)
    out: dict[str, tuple[float, float]] = {}
    for lab, pts in buckets.items():
        xs, ys = zip(*pts)
        out[lab] = (float(np.mean(xs)), float(np.mean(ys)))
    return out


def plot_ieee8500_feeder_topology(
    *,
    dss_dir: Path | str = DEFAULT_DSS_DIR,
    out_dir: Path | str | None = None,
    out_basename: str = "ieee8500_feeder_topology",
    style: Literal["paper", "draft"] = "paper",
    show_triplex: bool = False,
    show_loads: bool = True,
    show_all_load_transformers: bool = False,
    show_bus_dots: bool = False,
    label_autonomous_controllers: bool = False,
    annotate_key_devices: bool = False,
    show_device_icons: bool = True,
    paper_device_icons_only: bool = False,
    show_substation_icon: bool = True,
    show_capacitor_icons: bool = True,
    show_regulator_icons: bool = True,
    regulator_name_filter: list[str] | None = None,
    bus_color_values: dict[str, float] | None = None,
    bus_color_cmap: str = "magma",
    bus_color_point_size: float = 6.0,
    bus_color_alpha: float = 0.92,
    bus_color_vmin: float | None = None,
    bus_color_vmax: float | None = None,
    bus_color_colorbar: bool = True,
    bus_color_label: str = "Attention mass",
    bus_color_cbar_tick_labels: list[str] | list[float] | None = None,
    bus_color_log: bool = False,
    bus_color_power_gamma: float = 1.0,
    bus_color_upper_frac: float | None = None,
    bus_color_fade_cmap_frac: float = 0.22,
    coord_snap_ft: float | None = None,
    legend_mode: Literal["full", "paper_reference"] = "full",
    taper_line_width: bool = True,
    line_width_mode: LineWidthMode | None = None,
    ampacity_thickness_min: float | None = None,
    ampacity_thickness_max: float | None = None,
    ampacity_gamma: float | None = None,
    ampacity_scale: AmpacityScale | None = None,
    ampacity_qmax_percentile: float | None = None,
    ampacity_smooth_blend: float | None = None,
    ampacity_downstream_weight: float | None = None,
    use_opendss_power: bool = True,
    solve_hour: float | None = None,
    solve_sec: float = 0.0,
    label_font_size: int = 9,
    figsize: tuple[float, float] | None = None,
    png_dpi: int = PAPER_PNG_DPI_DEFAULT,
    allow_png_dpi_fallback: bool = False,
    highlight_bus_groups: list[dict[str, Any]] | None = None,
    bus_aliases: dict[str, str] | None = None,
    device_icon_paths: dict[str, str | Path] | None = None,
    device_icon_zooms: dict[str, float] | None = None,
    icon_size_scale: float = 1.4,
    svg_icon_raster_scale: float = SVG_ICON_RASTER_SCALE,
    svg_savefig_dpi: int = SVG_SAVEFIG_DPI_DEFAULT,
    vector_svg_icons: bool | None = None,
    svg_to_pdf: bool | None = None,
    reload_icons: bool = True,
    visio_friendly_svg: bool = False,
    visio_preserve_appearance: bool = True,
    visio_emf_export: bool = False,
    annotate_highlight_labels: bool = True,
    show_title: bool = True,
    title: str | None = None,
    show_legend: bool = True,
    crop_margins: bool = True,
    show: bool = False,
) -> dict[str, Path]:
    """Parse OpenDSS folder and save topology figure.

    Icon caches (raster + parsed SVG fragments) auto-invalidate when an icon
    file's modification time changes. Pass ``reload_icons=True`` (default) to
    clear any stale in-memory entries at the start of each plot call.
    """
    if reload_icons:
        clear_icon_caches()
    dss_dir = Path(dss_dir).resolve()
    out_root = Path(out_dir).resolve() if out_dir is not None else dss_dir
    out_root.mkdir(parents=True, exist_ok=True)

    figsize_user = figsize

    commands = read_all_dss_commands(dss_dir)
    coords_raw = load_buscoords(dss_dir)
    mv_line_edges = collect_mv_line_edges(commands)

    coords = coords_raw
    n_backfilled = 0
    plot_coords = coords_raw
    if style == "paper":
        coords, n_backfilled = backfill_missing_bus_coords(coords_raw, commands)

    model = parse_opendss_model(commands, coords)

    line_edges_by_tier: dict[LineTier, list[tuple[str, str]]] = model["line_edges_by_tier"]
    backbone_edges: list[tuple[str, str]] = []
    service_lateral_edges: list[tuple[str, str]] = []
    # Paper style snaps near-coincident buses (~22 ft) for a cleaner schematic.
    if coord_snap_ft is None:
        _snap_ft = 22.0 if style == "paper" else 0.0
    else:
        _snap_ft = float(coord_snap_ft)

    if style == "paper":
        backbone_edges, service_lateral_edges = split_mv_backbone_and_service_laterals(line_edges_by_tier)
        if _snap_ft > 0.0:
            line_records_preview = collect_line_records(commands, coords)
            snap_buses = {b for _n, b1, b2, _t in line_records_preview for b in (b1, b2)}
            plot_coords = _snap_coords_for_display(coords, snap_buses, snap_ft=_snap_ft)
            print(f"  Coord snap:               {_snap_ft:g} ft (near-coincident buses merged)")
        else:
            plot_coords = coords
            print("  Coord snap:               off (true Buscoords / backfill XY)")
    else:
        plot_coords = coords
    line_records = collect_line_records(commands, plot_coords)
    transformer_edges = model["transformer_edges"]
    transformer_points = model["transformer_points"]
    regulator_points = model["regulator_points"]
    capacitor_points = model["capacitor_points"]
    capcontrol_points = model["capcontrol_points"]
    pv_points = model["pv_points"]
    storage_points = model["storage_points"]
    load_buses: set[str] = model["load_buses"]

    n_lines = sum(len(v) for v in line_edges_by_tier.values())
    print("Parsed IEEE 8500 model:")
    print(f"  DSS folder:             {dss_dir}")
    print(f"  Style:                    {style}")
    print(f"  Bus coordinates:        {len(coords_raw):,}", end="")
    if n_backfilled:
        print(f"  (+{n_backfilled:,} backfilled)")
    else:
        print()
    print(f"  Line edges (total):     {n_lines:,}")
    n_skip = int(model.get("n_disabled_lines_skipped", 0) or 0)
    if n_skip:
        print(f"  Disabled lines omitted: {n_skip:,}  (OpenDSS enabled=false; avoids false loops)")
    if style == "paper":
        print(f"    MV backbone:          {len(backbone_edges):,}")
        print(f"      3-phase primary:    {len(line_edges_by_tier['primary_3ph']):,}")
        print(f"      1-phase trunk:      {len(backbone_edges) - len(line_edges_by_tier['primary_3ph']):,}")
        print(f"    Service laterals:     {len(service_lateral_edges):,}")
    else:
        print(f"    3-phase primary:      {len(line_edges_by_tier['primary_3ph']):,}")
        print(f"    1-phase laterals:     {len(line_edges_by_tier['lateral_1ph']):,}")
    print("    triplex (LV):         not plotted")
    print(f"  RegControls:            {len(regulator_points):,}")
    print(f"  Capacitor banks:        {len(capacitor_points):,}")
    print(f"  PV systems:             {len(pv_points):,}")
    print(f"  Load buses (coords):    {len(load_buses):,}")
    taper_graph = _dedupe_edges(
        line_edges_by_tier["primary_3ph"] + line_edges_by_tier["lateral_1ph"]
    )
    components = _connected_components(taper_graph)
    components.sort(key=len, reverse=True)
    src = resolve_source_bus(plot_coords)
    print(f"  MV graph components:    {len(components)}  (largest {len(components[0]) if components else 0:,} buses)")
    if components and src not in components[0]:
        main_root = resolve_main_feeder_root(taper_graph, plot_coords, src)
        print(f"    substation component: {next((len(c) for c in components if src in c), 0):,} buses")
        print(f"    gap before main body: yes (entry bus {main_root})")
    else:
        print("    substation-to-feeder:   connected")

    line_qty: dict[str, float] = {}
    quantity_max = PAPER_POWER_MAX_DEFAULT
    thickness_max = PAPER_THICKNESS_MAX
    thickness_min = PAPER_THICKNESS_MIN
    ampacity_all_segments = False
    linewidth_mapper = None
    draw_gamma = PAPER_POWER_GAMMA
    # Resolve width mode: explicit line_width_mode wins; else legacy taper_line_width.
    if line_width_mode is None:
        width_mode: LineWidthMode = "power" if taper_line_width else "uniform"
    else:
        width_mode = line_width_mode

    if style == "paper" and width_mode == "ampacity":
        wire_amps = load_wiredata_normamps(dss_dir)
        geom_amps = load_linegeometry_ampacity(dss_dir, wire_amps)
        line_lcs = collect_line_linecodes(commands)
        raw_amps = build_line_ampacity_qty(line_records, line_lcs, geom_amps)
        amps = sorted(v for v in raw_amps.values() if v > 0)
        tmin = float(
            AMPACITY_THICKNESS_MIN if ampacity_thickness_min is None else ampacity_thickness_min
        )
        tmax = float(
            AMPACITY_THICKNESS_MAX if ampacity_thickness_max is None else ampacity_thickness_max
        )
        agamma = float(AMPACITY_GAMMA if ampacity_gamma is None else ampacity_gamma)
        blend = float(
            AMPACITY_SMOOTH_BLEND if ampacity_smooth_blend is None else ampacity_smooth_blend
        )
        dweight = float(
            AMPACITY_DOWNSTREAM_WEIGHT
            if ampacity_downstream_weight is None
            else ampacity_downstream_weight
        )
        # Same continuous OpenDSS daisy map as loading (|qty|/Max → thickness).
        taper_root = resolve_main_feeder_root(taper_graph, plot_coords, src)
        smoothed = smooth_line_qty_radial(
            line_records, raw_amps, taper_root, blend=blend
        )
        line_qty = blend_ampacity_qty_with_downstream(
            line_records,
            smoothed,
            list(backbone_edges) + list(line_edges_by_tier.get("lateral_1ph", [])),
            taper_root,
            downstream_weight=dweight,
        )
        qvals = sorted(v for v in line_qty.values() if v > 0)
        quantity_max = max(qvals) if qvals else float(AMPACITY_DEFAULT_A)
        thickness_max = tmax
        thickness_min = tmin
        ampacity_all_segments = True
        # Use identical mapper path as power mode (no discrete log/gamma ampacity map).
        linewidth_mapper = None

        n_matched = sum(
            1
            for n, *_ in line_records
            if geom_amps.get(line_lcs.get(n.lower(), ""), None) is not None
        )
        print("  Line-width rule:        ampacity → loading-like daisy (smooth + downstream)")
        if amps:
            print(
                f"    ampacity span:        {amps[0]:.0f} .. {amps[-1]:.0f} A  "
                f"(raw WireData/LineGeometry)"
            )
        if qvals:
            print(
                f"    display qty span:     {qvals[0]:.1f} .. {qvals[-1]:.1f}  "
                f"(Max={quantity_max:.1f})"
            )
        print(
            f"    linewidth:            {tmin:g}-{tmax:g} pt  "
            f"| gamma={agamma:g} (same OpenDSS map as loading)  "
            f"| smooth_blend={blend:g}  downstream_w={dweight:g}"
        )
        print(
            f"    geometry matches:     {n_matched:,}/{len(line_records):,}  "
            f"| wire Normamps={len(wire_amps):,}  geometries={len(geom_amps):,}"
        )
        draw_gamma = agamma
    elif style == "paper" and width_mode == "power":
        if use_opendss_power:
            try:
                line_powers_kw = solve_opendss_line_powers_kw(
                    dss_dir,
                    solve_hour=solve_hour,
                    solve_sec=solve_sec,
                )
                backbone_buses_for_qty = {b for e in backbone_edges for b in e}
                line_qty, n_glue = enrich_backbone_lateral_powers(
                    line_records, line_powers_kw, backbone_buses_for_qty
                )
                quantity_max = PAPER_POWER_MAX_DEFAULT
                pvals = sorted(v for v in line_qty.values() if v > 0)
                if pvals:
                    print("  Line-width rule:        OpenDSS daisy (|kW|/Max on trunk lines)")
                    if solve_hour is not None:
                        print(
                            f"    solve time:           hour={float(solve_hour):g} "
                            f"sec={float(solve_sec):g} (daily snapshot)"
                        )
                    else:
                        print("    solve time:           default (full PV irradiance=1.0)")
                    print(
                        f"    solved |P| range:     {pvals[0]:.1f} .. {pvals[-1]:.1f} kW  "
                        f"(Max={quantity_max:.0f})"
                    )
                    print(
                        f"    trunk linewidth:      {PAPER_THICKNESS_MIN}-{PAPER_THICKNESS_MAX} pt  "
                        f"| service laterals {PAPER_LATERAL_FIXED_LW} pt"
                    )
                    print(f"    backbone 1ph glue:    {n_glue:,} segments inherit local 3ph |P|")
            except Exception as exc:
                print(f"  OpenDSS power solve:    unavailable ({exc})")
        if not line_qty:
            edge_qty_preview = compute_plot_edge_quantities(taper_graph, plot_coords, src)
            quantity_max = _quantity_max(edge_qty_preview)
            for name, b1, b2, _tier in line_records:
                line_qty[name.lower()] = edge_qty_preview.get(_norm_edge(b1, b2), 1.0)
            if edge_qty_preview:
                qvals = sorted(edge_qty_preview.values())
                print("  Line-width rule:        subtree fallback (|qty|/Max)")
                print(f"    proxy range:          {qvals[0]:.0f} .. {qvals[-1]:.0f}  (Max={quantity_max:.0f})")
    elif style == "paper" and width_mode == "uniform":
        print("  Line-width rule:        uniform (tier defaults)")

    draw_coords = plot_coords

    all_xy = np.array(list(draw_coords.values()))
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin

    if figsize_user is None:
        if crop_margins:
            figsize = _figsize_for_data_aspect(dx, dy, style=style)
        else:
            figsize = (11.5, 8.5) if style == "paper" else (13.5, 9.0)
    else:
        figsize = figsize_user

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Path glyphs keep bold/size through SVG→PDF→PNG converters
            # (font-name references often drop weight/size).
            "svg.fonttype": "path",
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    use_taper = style == "paper" and bool(line_qty) and width_mode in ("power", "ampacity")

    backbone_buses = {b for e in backbone_edges for b in e}

    all_line_segs: list[list[tuple[float, float]]] = []
    if style == "paper":
        draw_layers: list[tuple[str, list[tuple[str, str, str, LineTier]]]] = [
            ("lateral_1ph", [r for r in line_records if r[3] == "lateral_1ph"]),
            ("primary_3ph", [r for r in line_records if r[3] == "primary_3ph"]),
        ]
    else:
        triplex_records: list[tuple[str, str, str, LineTier]] = []
        if show_triplex:
            for fname, cmd in _unique_line_commands(commands):
                name = parse_object_name(cmd, "line")
                b1 = base_bus(get_param(cmd, "bus1"))
                b2 = base_bus(get_param(cmd, "bus2"))
                if not name or not b1 or not b2:
                    continue
                if classify_line_tier(fname, cmd) != "triplex":
                    continue
                if b1 in draw_coords and b2 in draw_coords:
                    triplex_records.append((name, b1, b2, "triplex"))
        draw_layers = []
        if show_triplex:
            draw_layers.append(("triplex", triplex_records))
        draw_layers.extend(
            [
                ("lateral_1ph", [r for r in line_records if r[3] == "lateral_1ph"]),
                ("primary_3ph", [r for r in line_records if r[3] == "primary_3ph"]),
            ]
        )

    for tier, records in draw_layers:
        if not records:
            continue
        min_len = 8.0 if style == "paper" and tier == "primary_3ph" else 3.0
        if style == "paper":
            if tier == "lateral_1ph" and not ampacity_all_segments:
                segs, seg_lws = _line_segments_mixed_laterals(
                    records,
                    draw_coords,
                    backbone_buses,
                    min_length=min_len,
                    line_qty=line_qty if use_taper else None,
                    quantity_max=quantity_max if use_taper else None,
                    thickness_max=thickness_max,
                    thickness_min=thickness_min,
                    gamma=draw_gamma,
                    linewidth_mapper=linewidth_mapper if use_taper else None,
                )
            else:
                segs, seg_lws = _line_segments(
                    records,
                    draw_coords,
                    min_length=min_len,
                    line_qty=line_qty if use_taper else None,
                    quantity_max=quantity_max if use_taper else None,
                    thickness_max=thickness_max,
                    thickness_min=thickness_min,
                    gamma=draw_gamma,
                    linewidth_mapper=linewidth_mapper if use_taper else None,
                )
        else:
            edges = [(b1, b2) for _n, b1, b2, _t in records]
            segs, seg_lws = _edge_segments(
                edges,
                draw_coords,
                min_length=min_len,
            )
        if not segs:
            continue
        all_line_segs.extend(segs)
        st = TIER_STYLE[tier]
        if style == "paper":
            line_color = PAPER_BACKBONE_COLOR
            line_lws: float | list[float] = seg_lws if use_taper else float(st["lw"])
            line_alpha = 1.0
        else:
            line_color = str(st["color"])
            line_lws = float(st["lw"])
            line_alpha = float(st["alpha"])
        ax.add_collection(
            LineCollection(
                segs,
                linewidths=line_lws,
                colors=line_color,
                alpha=line_alpha,
                zorder=int(st["zorder"]),
                capstyle="round",
                joinstyle="round",
            )
        )

    if show_triplex and style == "paper":
        triplex_edges = line_edges_by_tier.get("triplex", [])
        segs, _ = _edge_segments(triplex_edges, draw_coords)
        if segs:
            all_line_segs.extend(segs)
            st = TIER_STYLE["triplex"]
            ax.add_collection(
                LineCollection(
                    segs,
                    linewidths=float(st["lw"]),
                    colors=str(st["color"]),
                    alpha=float(st["alpha"]),
                    zorder=int(st["zorder"]),
                    capstyle="round",
                    joinstyle="round",
                )
            )

    if transformer_edges and style == "draft":
        xf_segs, _ = _edge_segments(transformer_edges, draw_coords)
        ax.add_collection(
            LineCollection(
                xf_segs,
                linewidths=0.25,
                colors="#9a9a9a",
                alpha=0.28,
                zorder=0,
                capstyle="round",
                joinstyle="round",
            )
        )

    if show_loads and load_buses:
        load_xy = np.array([draw_coords[b] for b in load_buses if b in draw_coords])
        if len(load_xy):
            ax.scatter(
                load_xy[:, 0],
                load_xy[:, 1],
                s=7 if style == "paper" else 4,
                facecolors="#000000",
                edgecolors="none",
                linewidths=0,
                marker="o",
                zorder=4,
                alpha=1.0,
            )

    if show_bus_dots:
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=1.2,
            c="#222222",
            alpha=0.25,
            linewidths=0,
            zorder=2,
        )

    bus_color_mappable = None
    if bus_color_values:
        # Color physical buses by a scalar (e.g. aggregated regulator-token attention).
        xs_c: list[float] = []
        ys_c: list[float] = []
        cs_c: list[float] = []
        for bus_name, val in bus_color_values.items():
            key = resolve_bus_in_coords(str(bus_name), draw_coords, bus_aliases=bus_aliases)
            if key is None:
                continue
            x, y = draw_coords[key]
            xs_c.append(float(x))
            ys_c.append(float(y))
            cs_c.append(float(val))
        if xs_c:
            c_arr = np.asarray(cs_c, dtype=float)
            if bus_color_log:
                c_plot = np.log10(np.clip(c_arr, 1e-16, None))
            else:
                c_plot = c_arr
            vmin = float(np.min(c_plot) if bus_color_vmin is None else bus_color_vmin)
            vmax = float(np.max(c_plot) if bus_color_vmax is None else bus_color_vmax)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.min(c_plot)), float(np.max(c_plot) + 1e-12)
            # Color mapping: legend ticks stay in data units [vmin, vmax].
            # - upper_frac: top fraction of the data range gets most of the colormap (sharp)
            # - bottom (1-upper_frac) is compressed into fade_cmap_frac (aggressive fade)
            # - else optional PowerNorm via bus_color_power_gamma
            scatter_kw: dict[str, Any] = {
                "cmap": bus_color_cmap if not isinstance(bus_color_cmap, str) else str(bus_color_cmap),
                "alpha": float(bus_color_alpha),
                "linewidths": 0,
                "zorder": 5,
            }
            _norm_note = ""
            if bus_color_upper_frac is not None:
                from matplotlib.colors import FuncNorm

                _uf = float(np.clip(bus_color_upper_frac, 1e-3, 1.0))
                _ff = float(np.clip(bus_color_fade_cmap_frac, 1e-3, 0.95))
                _lo = 1.0 - _uf  # bottom data fraction that fades hard
                _vmin_f, _vmax_f = float(vmin), float(vmax)
                _span = max(_vmax_f - _vmin_f, 1e-12)
                _gamma = float(bus_color_power_gamma) if float(bus_color_power_gamma) > 0 else 1.0

                def _fwd(x: np.ndarray) -> np.ndarray:
                    t = (np.asarray(x, dtype=float) - _vmin_f) / _span
                    t = np.clip(t, 0.0, 1.0)
                    u = np.where(
                        t < _lo,
                        (t / max(_lo, 1e-12)) * _ff,
                        _ff + ((t - _lo) / max(_uf, 1e-12)) * (1.0 - _ff),
                    )
                    # gamma>1: mid/low stay paler, highs pop (more contrast)
                    if abs(_gamma - 1.0) > 1e-12:
                        u = np.clip(u, 0.0, 1.0) ** _gamma
                    return u

                def _inv(y: np.ndarray) -> np.ndarray:
                    u = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
                    if abs(_gamma - 1.0) > 1e-12:
                        u = np.clip(u, 0.0, 1.0) ** (1.0 / _gamma)
                    t = np.where(
                        u < _ff,
                        (u / max(_ff, 1e-12)) * _lo,
                        _lo + ((u - _ff) / max(1.0 - _ff, 1e-12)) * _uf,
                    )
                    return _vmin_f + t * _span

                scatter_kw["norm"] = FuncNorm((_fwd, _inv), vmin=_vmin_f, vmax=_vmax_f)
                _norm_note = f" upper_frac={_uf:g} fade_cmap={_ff:g}"
                if abs(_gamma - 1.0) > 1e-12:
                    _norm_note += f" power_gamma={_gamma:g}"
            else:
                _gamma = float(bus_color_power_gamma)
                if _gamma > 0.0 and abs(_gamma - 1.0) > 1e-9:
                    from matplotlib.colors import PowerNorm

                    scatter_kw["norm"] = PowerNorm(gamma=_gamma, vmin=vmin, vmax=vmax)
                    _norm_note = f" power_gamma={_gamma:g}"
                else:
                    scatter_kw["vmin"] = vmin
                    scatter_kw["vmax"] = vmax
            sc = ax.scatter(
                xs_c,
                ys_c,
                c=c_plot,
                s=float(bus_color_point_size),
                **scatter_kw,
            )
            bus_color_mappable = sc
            # Colorbar is attached after layout (crop_margins) so it is not crushed.
            print(
                f"  Bus color overlay:      {len(xs_c):,} buses  "
                f"[{vmin:.3g}, {vmax:.3g}] cmap={bus_color_cmap}{_norm_note}"
            )

    if show_all_load_transformers and transformer_points:
        xf_xy = np.array(list(transformer_points.values()))
        ax.scatter(
            xf_xy[:, 0],
            xf_xy[:, 1],
            s=5,
            c="#8f8f8f",
            alpha=0.22,
            linewidths=0,
            marker=".",
            zorder=3,
        )

    # Optional RegControl name filter (OpenDSS names, e.g. "vreg2_a", "feeder_rega").
    # Matching is case-insensitive; also accepts bank labels ("Reg 2", "Substation LTC").
    if regulator_name_filter:
        want = {str(x).strip().lower() for x in regulator_name_filter if str(x).strip()}
        filtered: dict[str, tuple[float, float]] = {}
        for name, xy in regulator_points.items():
            nm = str(name).strip().lower()
            lab = _reg_group_label(name).strip().lower()
            if nm in want or lab in want or any(w in nm or w in lab for w in want):
                filtered[name] = xy
        regulator_points = filtered
        print(f"  Regulator filter:       kept {len(regulator_points)} RegControl(s) matching {sorted(want)}")

    reg_plot = _dedupe_regulator_points(regulator_points) if show_regulator_icons else {}
    cap_plot = (
        _dedupe_device_points_by_location(capacitor_points) if show_capacitor_icons else {}
    )
    # Paper uses a uniform base zoom; ``device_icon_zooms`` overrides absolute base
    # per device kind (applied after), then ``icon_size_scale`` multiplies placements.
    icon_zooms = {
        key: float(PAPER_UNIFORM_ICON_ZOOM if style == "paper" else zoom)
        for key, zoom in DEFAULT_DEVICE_ICON_ZOOM.items()
    }
    if device_icon_zooms:
        for key, zoom in device_icon_zooms.items():
            icon_zooms[str(key)] = float(zoom)
    icon_paths = dict(device_icon_paths or {})
    icon_placements: list[dict[str, Any]] = []

    def _device_icon(device_key: str) -> str | Path | None:
        return _resolve_device_icon_path(device_key, icon_paths)

    def _add_device_icons(
        points: dict[str, tuple[float, float]],
        *,
        device_key: str,
        kind: str,
        zorder: int,
    ) -> None:
        icon_path = _device_icon(device_key)
        if icon_path is None:
            return
        for name, (x, y) in points.items():
            icon_placements.append(
                _icon_placement(
                    x=float(x),
                    y=float(y),
                    icon_path=icon_path,
                    zoom=icon_zooms[device_key],
                    zorder=zorder,
                    kind=kind,
                    bus=str(name),
                )
            )

    if show_device_icons:
        if cap_plot:
            _add_device_icons(cap_plot, device_key="capacitor", kind="capacitor", zorder=8)
        if reg_plot:
            _add_device_icons(reg_plot, device_key="regulator", kind="regulator", zorder=9)
        if pv_points and not paper_device_icons_only:
            _add_device_icons(pv_points, device_key="pv", kind="pv", zorder=10)
        if storage_points and not paper_device_icons_only:
            _add_device_icons(storage_points, device_key="storage", kind="storage", zorder=10)
        if src in draw_coords:
            sx, sy = draw_coords[src]
            sub_icon = _device_icon("substation") if show_substation_icon else None
            if sub_icon is not None:
                icon_placements.append(
                    _icon_placement(
                        x=sx,
                        y=sy,
                        icon_path=sub_icon,
                        zoom=icon_zooms["substation"],
                        zorder=12,
                        kind="substation",
                        bus=src,
                        stack_direction="up_right",
                        stack_offset_ft=max(140.0, 0.009 * max(dx, dy)),
                    )
                )
                icon_placements[-1]["keep_on_map"] = True
            # FEEDER_REGA/B/C (Substation LTC) sit on regxfmr_HVMV_Sub_LSB ↔ _HVMV_Sub_LSB,
            # ~40 ft from the source bus — easy to drop as "off-network" or hide under the
            # enlarged transformer. Pin them under the substation icon.
            for p in icon_placements:
                if p.get("kind") != "regulator":
                    continue
                if str(p.get("bus", "")).strip().lower() != "substation ltc":
                    continue
                p["x"], p["y"] = float(sx), float(sy)
                p["stack_direction"] = "down_right"
                p["stack_offset_ft"] = max(220.0, 0.014 * max(dx, dy))
                p["keep_on_map"] = True
                print(
                    "  Substation LTC icon:     pinned under transformer "
                    f"(stack down_right, offset={p['stack_offset_ft']:.0f} ft)"
                )
    else:
        sx, sy = draw_coords[src]

    flat_segs = _flatten_segments(all_line_segs)
    min_line_clear = 0.018 * max(dx, dy)
    min_label_sep = 0.035 * max(dx, dy)

    if label_autonomous_controllers:
        placed_ctrl: list[tuple[float, float]] = []
        for name, (x, y) in regulator_points.items():
            lx, ly = _choose_label_xy(
                x,
                y,
                flat_segs,
                placed_ctrl,
                dx,
                dy,
                min_line_clear=min_line_clear,
                min_label_sep=min_label_sep,
            )
            placed_ctrl.append((lx, ly))
            ax.text(
                lx,
                ly,
                name,
                fontsize=label_font_size - 2,
                color="#9c4f00",
                ha="center",
                va="center",
                zorder=20,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
            )

    if annotate_key_devices:
        placed_ann: list[tuple[float, float]] = []
        ann: list[tuple[tuple[float, float], str]] = []
        for name in sorted(pv_points):
            if name.lower() in ("pv1", "pv2"):
                ann.append((pv_points[name], name.upper().replace("PV", "PV ")))
        for _lab, xy in sorted(reg_plot.items())[:2]:
            ann.append((xy, _lab))
        for (x, y), label in ann:
            _annotate_clear(
                ax,
                anchor=(x, y),
                label=label,
                flat_segs=flat_segs,
                placed=placed_ann,
                dx=dx,
                dy=dy,
                label_font_size=label_font_size,
                min_line_clear=min_line_clear,
                min_label_sep=min_label_sep,
            )

    highlight_legend: list[Line2D] = []
    highlight_avoid: list[tuple[float, float]] = []
    highlight_icon_placements: list[dict[str, Any]] = []
    if highlight_bus_groups:
        highlight_legend, highlight_avoid, _, highlight_icon_placements = _plot_highlight_bus_groups(
            ax,
            groups=highlight_bus_groups,
            coords=draw_coords,
            bus_aliases=bus_aliases,
            flat_segs=flat_segs,
            dx=dx,
            dy=dy,
            label_font_size=label_font_size,
            min_line_clear=min_line_clear,
            min_label_sep=min_label_sep,
            annotate_labels=annotate_highlight_labels,
            raster_scale=svg_icon_raster_scale,
            legend_mode=legend_mode,
            paper_icon_zoom=PAPER_UNIFORM_ICON_ZOOM,
        )
        icon_placements.extend(highlight_icon_placements)

    if icon_size_scale != 1.0:
        for placement in icon_placements:
            placement["zoom"] = float(placement["zoom"]) * float(icon_size_scale)

    icon_placements, suppressed_monitored = _suppress_monitored_icon_placements(
        icon_placements,
        map_span_ft=max(dx, dy),
    )
    if suppressed_monitored:
        preview = ", ".join(suppressed_monitored[:8])
        suffix = " ..." if len(suppressed_monitored) > 8 else ""
        print(
            f"  Suppressed {len(suppressed_monitored)} monitored-bus icon(s) "
            f"at co-located / DER buses: {preview}{suffix}"
        )

    icon_placements, suppressed_bottom = _drop_bottom_stack_icons_when_coclustered(
        icon_placements,
    )
    if suppressed_bottom:
        preview = ", ".join(suppressed_bottom[:8])
        suffix = " ..." if len(suppressed_bottom) > 8 else ""
        print(
            f"  Suppressed {len(suppressed_bottom)} bottom-stack icon(s) "
            f"at co-located buses: {preview}{suffix}"
        )

    if icon_placements:
        use_vector_svg = _resolve_vector_svg_icons_flag(
            vector_svg_icons,
            *icon_paths.values(),
            *(p.get("icon_path") for p in icon_placements),
        )
        vector_icon_placements: list[dict[str, Any]] = []
        placed_avoid, _ = _place_icon_placements_stacked(
            ax,
            icon_placements,
            dx=dx,
            dy=dy,
            flat_segs=flat_segs,
            raster_scale=svg_icon_raster_scale,
            vector_svg=use_vector_svg,
            vector_placements_out=vector_icon_placements,
        )
        highlight_avoid.extend(placed_avoid)
    else:
        use_vector_svg = False
        vector_icon_placements = []

    use_svg_to_pdf = _resolve_svg_to_pdf_flag(
        svg_to_pdf,
        use_vector_svg=use_vector_svg,
    )

    if show_legend:
        legend_items: list[Line2D] = []
        legend_title: str | None = None
        legend_handlelength = 2.0
        if legend_mode == "paper_reference":
            legend_handlelength = 3.0
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=PAPER_BACKBONE_COLOR,
                    lw=2.4,
                    solid_capstyle="round",
                    label="Distribution lines (width $\\propto$ loading)",
                )
            )
        elif style == "paper" and use_taper:
            legend_title = "Line width ∝ solved real power |P| (kW)"
            legend_handlelength = 3.2
            legend_items.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color=PAPER_BACKBONE_COLOR,
                        lw=PAPER_THICKNESS_MAX,
                        solid_capstyle="round",
                        label="higher |P|",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=PAPER_BACKBONE_COLOR,
                        lw=PAPER_LATERAL_FIXED_LW,
                        solid_capstyle="round",
                        label="lower |P|",
                    ),
                ]
            )
        elif style == "paper":
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=PAPER_BACKBONE_COLOR,
                    lw=2.0,
                    label="MV feeder lines",
                )
            )
        else:
            legend_items.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color=TIER_STYLE["primary_3ph"]["color"],
                        lw=2.0,
                        label="3-phase MV primary",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=TIER_STYLE["lateral_1ph"]["color"],
                        lw=1.0,
                        label="Single-phase lateral",
                    ),
                ]
            )
        if show_device_icons:
            def _legend_entry(
                device_key: str,
                *,
                label: str,
                marker: str,
                facecolor: str,
                markersize: float,
            ) -> Line2D | _IconLegendHandle:
                icon_path = _device_icon(device_key)
                if icon_path is not None:
                    return _IconLegendHandle(
                        icon_path,
                        zoom=PAPER_LEGEND_ICON_ZOOM if legend_mode == "paper_reference" else icon_zooms[device_key] * 1.35,
                        label=label,
                        raster_scale=svg_icon_raster_scale,
                    )
                return Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    label=label,
                    markerfacecolor=facecolor,
                    markeredgecolor="#1a1a1a",
                    markersize=markersize,
                )

            if legend_mode == "paper_reference":
                legend_items.extend(
                    [
                        _legend_entry(
                            "substation",
                            label="Substation",
                            marker="*",
                            facecolor="#2171b5",
                            markersize=12,
                        ),
                        _legend_entry(
                            "regulator",
                            label="Voltage regulator",
                            marker="D",
                            facecolor="#d62728",
                            markersize=7,
                        ),
                        _legend_entry(
                            "capacitor",
                            label="Capacitor bank",
                            marker="s",
                            facecolor="#2ca25f",
                            markersize=7,
                        ),
                    ]
                )
                pv_legend_icon = _device_icon("pv")
                if pv_legend_icon is None and highlight_icon_placements:
                    pv_legend_icon = highlight_icon_placements[0]["icon_path"]
                if pv_legend_icon is not None:
                    legend_items.append(
                        _IconLegendHandle(
                            pv_legend_icon,
                            zoom=PAPER_LEGEND_ICON_ZOOM,
                            label="PV connection point",
                            raster_scale=svg_icon_raster_scale,
                        )
                    )
            else:
                legend_items.extend(
                    [
                        _legend_entry(
                            "substation",
                            label="Substation",
                            marker="*",
                            facecolor="#2171b5",
                            markersize=12,
                        ),
                        _legend_entry(
                            "regulator",
                            label="Voltage regulator (LTC)",
                            marker="D",
                            facecolor="#d62728",
                            markersize=7,
                        ),
                        _legend_entry(
                            "capacitor",
                            label="Capacitor bank",
                            marker="s",
                            facecolor="#2ca25f",
                            markersize=7,
                        ),
                    ]
                )
                if pv_points and not paper_device_icons_only:
                    legend_items.append(
                        _legend_entry(
                            "pv",
                            label="PV PCC",
                            marker="*",
                            facecolor="#f1c40f",
                            markersize=12,
                        )
                    )

        if legend_mode != "paper_reference":
            legend_items.extend(highlight_legend)

    if show_title:
        ax.set_title(
            (str(title).strip() if title else "IEEE 8500-Node Unbalanced Distribution Feeder"),
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

    ax.set_aspect("equal", adjustable="box")
    data_pad_frac = TIGHT_DATA_PAD_FRAC if crop_margins else LOOSE_DATA_PAD_FRAC
    pad_x, pad_y = data_pad_frac * dx, data_pad_frac * dy
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # Leave full figure width; colorbar (if any) is carved from ax via
    # make_axes_locatable. bbox_inches='tight' then crops to content
    # (including the colorbar label) without large uniform white borders.
    _want_cbar = bool(bus_color_colorbar and bus_color_mappable is not None)
    if crop_margins and not show_title:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if _want_cbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.6%", pad=0.10)
        cbar = fig.colorbar(bus_color_mappable, cax=cax)
        # Match paper-style vertical scale: label beside ticks, readable ticks.
        _cbar_label = str(bus_color_label).strip().replace("\n", " ")
        if bus_color_log and not _cbar_label.lower().startswith("log"):
            _cbar_label = f"log10({_cbar_label})"
        # IEEE colorbar: title 15 pt bold; tick numbers 10 pt bold.
        _label_fs = 15
        # Space between tick numbers and the rotated label (not export edge pad).
        _labelpad = 28
        cbar.set_label(
            _cbar_label,
            fontsize=_label_fs,
            fontweight="bold",
            fontfamily="serif",
            rotation=270,
            labelpad=_labelpad,
        )
        _ylab = cbar.ax.yaxis.label
        _ylab.set_clip_on(False)
        _ylab.set_fontsize(_label_fs)
        _ylab.set_fontweight("bold")
        _ylab.set_fontfamily("serif")
        cbar.ax.tick_params(labelsize=10, length=3, width=0.8)
        for _t in cbar.ax.get_yticklabels():
            _t.set_fontweight("bold")
            _t.set_fontsize(10)
            _t.set_clip_on(False)
        if bus_color_cbar_tick_labels is not None:
            _labs = list(bus_color_cbar_tick_labels)
            if len(_labs) >= 2:
                _norm = getattr(bus_color_mappable, "norm", None)
                _ymin = float(getattr(_norm, "vmin", 0.0) if _norm is not None else 0.0)
                _ymax = float(getattr(_norm, "vmax", 1.0) if _norm is not None else 1.0)
                if not np.isfinite(_ymin) or not np.isfinite(_ymax) or _ymax <= _ymin:
                    _ymin, _ymax = 0.0, 1.0
                _tick_pos = np.linspace(_ymin, _ymax, len(_labs))
                cbar.set_ticks(_tick_pos)
                cbar.set_ticklabels(
                    [f"{x:.3g}" if isinstance(x, (int, float, np.floating)) else str(x) for x in _labs]
                )
        for _t in cbar.ax.get_yticklabels():
            _t.set_clip_on(False)
            _t.set_fontsize(10)
            _t.set_fontweight("bold")
        cbar.outline.set_linewidth(0.8)
        print(f"  Color scale:              '{_cbar_label}' (right)")

    if show_legend:
        legend_w, legend_h = _estimate_legend_box(dx, dy, len(legend_items))
        if legend_title:
            legend_h += 0.018 * dy
        avoid_xy: list[tuple[float, float]] = [(sx, sy)]
        avoid_xy.extend(cap_plot.values())
        avoid_xy.extend(reg_plot.values())
        avoid_xy.extend(pv_points.values())
        avoid_xy.extend(highlight_avoid)
        if show_loads and load_buses:
            avoid_xy.extend(draw_coords[b] for b in load_buses if b in draw_coords)
        if legend_mode == "paper_reference":
            _add_paper_reference_legend(
                ax,
                icon_paths=icon_paths,
                highlight_icon_placements=highlight_icon_placements,
                fontsize=9,
            )
        elif legend_items:
            legend_kwargs: dict[str, Any] = {
                "handles": legend_items,
                "title": legend_title,
                "frameon": True,
                "framealpha": 0.97,
                "facecolor": "white",
                "edgecolor": "#666666",
                "fontsize": 9 if legend_mode == "paper_reference" else 8,
                "title_fontsize": 8,
                "handlelength": legend_handlelength,
                "borderpad": 0.6,
                "handler_map": {
                    _IconLegendHandle: _HandlerIconLegend(),
                    _CircledStarLegendHandle: _HandlerCircledStarLegend(),
                },
            }
            legend_ul = _choose_legend_anchor(
                xmin=xmin - pad_x,
                xmax=xmax + pad_x,
                ymin=ymin - pad_y,
                ymax=ymax + pad_y,
                box_w=legend_w,
                box_h=legend_h,
                flat_segs=flat_segs,
                avoid_points=avoid_xy,
                min_line_clear=0.018 * max(dx, dy),
                point_margin=0.012 * max(dx, dy),
            )
            ax.legend(
                loc="upper left",
                bbox_to_anchor=legend_ul,
                bbox_transform=ax.transData,
                **legend_kwargs,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    pdf_path = out_root / f"{out_basename}.pdf"
    svg_path = out_root / f"{out_basename}.svg"
    png_path = out_root / f"{out_basename}.png"
    _cbar_label_save = locals().get("_cbar_label", str(bus_color_label))
    _bbox_inches, _savefig_pad = _ieee_tight_bbox_inches(
        fig,
        crop_margins=bool(crop_margins),
        want_cbar=bool(_want_cbar),
        cbar_label=str(_cbar_label_save),
    )
    # Icon embed path expects the pre-expansion tight pad (left/top), not right extra.
    _embed_pad = (
        TIGHT_SAVEFIG_PAD_INCHES if crop_margins else float(plt.rcParams["savefig.pad_inches"])
    )
    save_kwargs: dict[str, Any] = {
        "bbox_inches": _bbox_inches,
        "facecolor": "white",
        "pad_inches": _savefig_pad,
    }
    if use_vector_svg and vector_icon_placements:
        fig.savefig(svg_path, dpi=int(svg_savefig_dpi), **save_kwargs)
        n_embedded = _embed_vector_icons_in_svg(
            svg_path,
            vector_icon_placements,
            fig=fig,
            ax=ax,
            pad_inches=_embed_pad,
        )
        print(f"  Vector SVG icons:       {n_embedded} nested groups embedded")
        for placement in vector_icon_placements:
            _place_bus_icon(
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
    # Skip SVG viewBox crop when a colorbar is present — crop uses axes_1 only
    # and would clip the right-side scale (separate matplotlib axes).
    if crop_margins and not _want_cbar and _crop_svg_viewbox_to_content(svg_path):
        print("  SVG viewBox:            cropped to axes content")
    elif crop_margins and _want_cbar:
        print("  SVG viewBox:            kept full (colorbar on right)")
    if visio_friendly_svg:
        visio_stats = postprocess_svg_for_visio(
            svg_path,
            preserve_appearance=visio_preserve_appearance,
        )
        mode = "appearance-preserving" if visio_preserve_appearance else "aggressive"
        print(
            f"  Visio-friendly SVG ({mode}): "
            f"{visio_stats['path_count_before']:,} -> {visio_stats['path_count_after']:,} paths "
            f"({visio_stats['path_reduction_pct']:.1f}% fewer), "
            f"{visio_stats['bytes_before'] / 1024:.1f} -> {visio_stats['bytes_after'] / 1024:.1f} KiB "
            f"({visio_stats['size_reduction_pct']:.1f}% smaller)"
        )
        if not visio_preserve_appearance and visio_stats.get("linecollections_skipped_variable_width"):
            print(
                "  Visio merge skipped:   "
                f"{visio_stats['linecollections_skipped_variable_width']} tapered LineCollection group(s)"
            )
        if visio_emf_export:
            emf_path = out_root / f"{out_basename}.emf"
            if _try_export_emf_from_svg(svg_path, emf_path):
                paths["emf"] = emf_path
                print(f"  EMF (Inkscape):       {emf_path}")
            else:
                print(
                    "  EMF export skipped:   Inkscape not found on PATH "
                    "(install Inkscape and re-run with visio_emf_export=True)"
                )

    if use_svg_to_pdf:
        try:
            backend = _export_pdf_from_svg(svg_path, pdf_path)
            n_icons = _count_vector_device_icons_in_svg(svg_path)
            icon_note = f", {n_icons} vector icon group(s)" if n_icons else ""
            print(f"  PDF from SVG ({backend}{icon_note}): {pdf_path}")
        except RuntimeError as exc:
            if use_vector_svg and vector_icon_placements:
                fig.savefig(pdf_path, **save_kwargs)
                print(f"  PDF fallback (matplotlib): {pdf_path}")
                print(f"  Warning: {exc}")
            else:
                raise

    # Prefer vector-faithful PNG: rasterize finalized PDF/SVG (icons match SVG/PDF).
    if use_vector_svg or use_svg_to_pdf:
        png_backend = overwrite_png_from_vector(
            png_path,
            pdf_path=pdf_path,
            svg_path=svg_path,
            png_dpi=int(png_dpi),
            allow_dpi_fallback=bool(allow_png_dpi_fallback),
        )
        if png_backend:
            print(f"  PNG from vector ({png_backend}): {png_path}")
            if "FALLBACK" in png_backend:
                print(
                    "  WARNING: PNG used a lower zoom than png_dpi "
                    f"(allow_png_dpi_fallback=True). Requested dpi={int(png_dpi)}."
                )
        else:
            print(
                "  PNG from vector FAILED at exact "
                f"png_dpi={int(png_dpi)} (zoom={int(png_dpi)/72:g}); "
                "no lower-DPI retry. "
                f"Kept matplotlib PNG @ {int(png_dpi)} dpi (icons may differ)."
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
    p = argparse.ArgumentParser(description="Plot IEEE 8500 OpenDSS feeder topology.")
    p.add_argument("--dss-dir", type=Path, default=DEFAULT_DSS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-basename", default="ieee8500_feeder_topology")
    p.add_argument("--style", choices=("paper", "draft"), default="paper")
    p.add_argument("--show-triplex", action="store_true")
    p.add_argument("--no-loads", action="store_true")
    p.add_argument("--transformer-dots", action="store_true")
    p.add_argument("--show-bus-dots", action="store_true")
    p.add_argument("--all-controller-labels", action="store_true")
    p.add_argument("--annotate", action="store_true", help="Arrow labels for PV / a few regs (off by default)")
    p.add_argument(
        "--lines-only",
        action="store_true",
        help="Draw MV lines only (no substation/reg/cap/PV markers or device legend entries)",
    )
    p.add_argument(
        "--no-taper",
        action="store_true",
        help="Disable OpenDSS-style MV line-width taper (paper style only)",
    )
    p.add_argument(
        "--line-width-mode",
        choices=("power", "ampacity", "uniform"),
        default=None,
        help="Line width: power (|P| daisy), ampacity (WireData.Normamps), or uniform. "
        "When set, overrides --no-taper.",
    )
    p.add_argument("--ampacity-lw-min", type=float, default=None, help="Ampacity mode min linewidth (pt)")
    p.add_argument("--ampacity-lw-max", type=float, default=None, help="Ampacity mode max linewidth (pt)")
    p.add_argument(
        "--ampacity-gamma",
        type=float,
        default=None,
        help="Ampacity mode gamma (1.0 = same linear map as loading)",
    )
    p.add_argument(
        "--ampacity-scale",
        choices=("linear", "log"),
        default=None,
        help="Deprecated for daisy map; kept for CLI compatibility",
    )
    p.add_argument(
        "--ampacity-qmax-percentile",
        type=float,
        default=None,
        help="Deprecated; daisy Max is max(display qty)",
    )
    p.add_argument(
        "--ampacity-smooth-blend",
        type=float,
        default=None,
        help="Radial EMA blend (1=raw ampacity, 0=fully smooth from parent)",
    )
    p.add_argument(
        "--ampacity-downstream-weight",
        type=float,
        default=None,
        help="Blend weight for continuous subtree size (loading-like taper)",
    )
    p.add_argument(
        "--no-opendss",
        action="store_true",
        help="Skip OpenDSS solve; use subtree proxy for line width (paper style only)",
    )
    p.add_argument(
        "--solve-hour",
        type=float,
        default=None,
        help="Daily-mode snapshot hour (0-24). Applies PV Daily=IrradDay001; e.g. 3 for zero PV.",
    )
    p.add_argument(
        "--solve-sec",
        type=float,
        default=0.0,
        help="Seconds within the solve hour (default 0).",
    )
    p.add_argument("--png-dpi", type=int, default=PAPER_PNG_DPI_DEFAULT)
    p.add_argument(
        "--icon-size-scale",
        type=float,
        default=1.4,
        help="Uniform multiplier for on-map device/highlight icon size (default 1.4 = 40%% larger)",
    )
    p.add_argument("--no-legend", action="store_true", help="Omit legend box entirely")
    p.add_argument(
        "--no-crop-margins",
        action="store_true",
        help="Keep legacy figure margins (looser savefig pad and data padding)",
    )
    p.add_argument(
        "--no-vector-svg-icons",
        action="store_true",
        help="Rasterize SVG icons inside the exported SVG (legacy behavior)",
    )
    p.add_argument(
        "--no-svg-to-pdf",
        action="store_true",
        help="Save PDF via matplotlib instead of converting the finalized SVG",
    )
    p.add_argument(
        "--visio-friendly-svg",
        action="store_true",
        help="Post-process SVG for Microsoft Visio (safe cleanup by default)",
    )
    p.add_argument(
        "--visio-aggressive",
        action="store_true",
        help="Aggressive Visio SVG flattening (may change appearance; use with --visio-friendly-svg)",
    )
    p.add_argument(
        "--visio-emf-export",
        action="store_true",
        help="Also export EMF via Inkscape CLI (requires --visio-friendly-svg)",
    )
    p.add_argument(
        "--coord-snap-ft",
        type=float,
        default=None,
        help="Merge near-coincident paper buses within this distance (ft). "
        "0 disables. Default: 22 for paper style.",
    )
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    plot_ieee8500_feeder_topology(
        dss_dir=args.dss_dir,
        out_dir=args.out_dir,
        out_basename=args.out_basename,
        style=args.style,
        show_triplex=args.show_triplex,
        show_loads=not args.no_loads,
        show_all_load_transformers=args.transformer_dots,
        show_bus_dots=args.show_bus_dots,
        label_autonomous_controllers=args.all_controller_labels,
        annotate_key_devices=args.annotate,
        show_device_icons=not args.lines_only,
        taper_line_width=not args.no_taper,
        line_width_mode=args.line_width_mode,
        ampacity_thickness_min=args.ampacity_lw_min,
        ampacity_thickness_max=args.ampacity_lw_max,
        ampacity_gamma=args.ampacity_gamma,
        ampacity_scale=args.ampacity_scale,
        ampacity_qmax_percentile=args.ampacity_qmax_percentile,
        ampacity_smooth_blend=args.ampacity_smooth_blend,
        ampacity_downstream_weight=args.ampacity_downstream_weight,
        use_opendss_power=not args.no_opendss,
        solve_hour=args.solve_hour,
        solve_sec=args.solve_sec,
        png_dpi=args.png_dpi,
        icon_size_scale=args.icon_size_scale,
        show_legend=not args.no_legend,
        crop_margins=not args.no_crop_margins,
        vector_svg_icons=False if args.no_vector_svg_icons else None,
        svg_to_pdf=False if args.no_svg_to_pdf else None,
        visio_friendly_svg=args.visio_friendly_svg,
        visio_preserve_appearance=not args.visio_aggressive,
        visio_emf_export=args.visio_emf_export,
        coord_snap_ft=args.coord_snap_ft,
        show=args.show,
    )


if __name__ == "__main__":
    main()
