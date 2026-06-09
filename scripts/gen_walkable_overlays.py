#!/usr/bin/env python3
"""
Generate walkable-area overlays for all 12 floors (B21 F1-F6, B22 F1-F6).

Algorithm:
  walkable = union(bruto).difference(dest_rooms_union).difference(barrier_buffer)

  dest_rooms = neto polys matched to destination rooms (offices, labs, lecture halls, etc.)
  Corridor/foyer/lobby rooms are NOT subtracted — they are part of walkable space.
  Unmatched neto polys (no room in graph.json within range) → treated as destination.

Rendering:
  Blue  = walkable area
  Pink  = destination room interiors
  Cyan* = staircase / elevator waypoint
  Green = door thresholds
"""
import sys, math, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import LineString, Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT   = Path(__file__).resolve().parent.parent
DATA21 = ROOT / "data" / "21"
DATA22 = ROOT / "data" / "22"
DOCS   = ROOT / "docs"

with open(ROOT / "docs" / "b22_transform.json") as f:
    _t = json.load(f)
B22_OX, B22_OY = _t["offset_x"], _t["offset_y"]

def b22(x, y): return x + B22_OX, y + B22_OY

with open(ROOT / "backend" / "graph.json") as f:
    _g = json.load(f)
ROOMS_BY_FLOOR = defaultdict(list)
for r in _g["nodes"]:
    ROOMS_BY_FLOOR[(r["building"], r["floor"])].append(r)

DXF_FILES = {
    ("21",1): DATA21/"02011.dxf", ("21",2): DATA21/"02012.dxf",
    ("21",3): DATA21/"02013.dxf", ("21",4): DATA21/"02014.dxf",
    ("21",5): DATA21/"02015.dxf", ("21",6): DATA21/"02016.dxf",
    ("22",1): DATA22/"02021.dxf",
    ("22",2): DATA22/"02022.dxf", ("22",3): DATA22/"02023.dxf",
    ("22",4): DATA22/"02024.dxf", ("22",5): DATA22/"02025.dxf",
    ("22",6): DATA22/"02026.dxf",
}

BARRIER_LAYERS = {"Beton", "Window", "Mabat"}

# These room types are walkable circulation space — their neto polys stay blue.
CORRIDOR_TYPES = {
    "פויאה", "פויה",
    "מבוא",
    "מעבר",
    "פרוזדור",
    "מרפסת",
    "חדר מדרגות", "מדרגות",
    "פיר מעלית", "פיר",
}
STAIR_TYPES = {"חדר מדרגות", "מדרגות", "פיר מעלית", "פיר"}

# Max distance from neto poly centroid to a graph.json room centroid for matching
MATCH_DIST = 300


def read_pts(e, bld):
    pts = [(p[0], p[1]) for p in e.get_points()]
    if bld == "22":
        pts = [b22(x, y) for x, y in pts]
    return pts


def get_bruto(msp, bld):
    prefix = "0201" if bld == "21" else "0202"
    layer = f"{prefix}Shetah-Bruto"
    accept = {layer, layer.replace("0202", "020 2")}
    polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in accept or e.dxftype() != "LWPOLYLINE": continue
        pts = read_pts(e, bld)
        if len(pts) < 3: continue
        try:
            p = make_valid(Polygon(pts))
            if not p.is_empty and p.geom_type == "Polygon":
                polys.append(p)
        except Exception: pass
    return polys


def get_neto(msp, bld):
    prefix = "0201" if bld == "21" else "0202"
    layer = f"{prefix}Shetah-Neto"
    accept = {layer, layer.replace("0202", "020 2")}
    polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in accept or e.dxftype() != "LWPOLYLINE": continue
        pts = read_pts(e, bld)
        if len(pts) < 3: continue
        try:
            p = make_valid(Polygon(pts))
            if not p.is_empty and p.geom_type == "Polygon":
                polys.append(p)
        except Exception: pass
    return polys


def get_barriers(msp, bld):
    lines = []
    for e in msp:
        layer = e.dxf.get("layer", "")
        if layer not in BARRIER_LAYERS: continue
        t = e.dxftype()
        if t == "LINE":
            x1, y1 = e.dxf.start.x, e.dxf.start.y
            x2, y2 = e.dxf.end.x, e.dxf.end.y
            if bld == "22": x1, y1 = b22(x1, y1); x2, y2 = b22(x2, y2)
            lines.append((layer, LineString([(x1,y1),(x2,y2)])))
        elif t == "LWPOLYLINE":
            verts = list(e.get_points())
            if layer == "Window" and len(verts) == 10: continue
            verts = [b22(p[0],p[1]) for p in verts] if bld == "22" else [(p[0],p[1]) for p in verts]
            if e.closed and verts: verts = verts + [verts[0]]
            for i in range(len(verts)-1):
                lines.append((layer, LineString([verts[i], verts[i+1]])))
    return lines


def get_doors(msp, bld):
    pts = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door" or e.dxftype() != "LWPOLYLINE": continue
        verts = list(e.get_points())
        if len(verts) < 2: continue
        x = (verts[0][0] + verts[1][0]) / 2
        y = (verts[0][1] + verts[1][1]) / 2
        if bld == "22": x, y = b22(x, y)
        pts.append((x, y))
    return pts


def classify_neto(neto_polys, floor_rooms):
    """
    Split neto polys into dest (non-walkable) and corridor (walkable).
    Matching: find nearest room in graph.json by poly centroid distance.
    - Corridor type → corridor list (stay walkable, not rendered pink)
    - Destination type or no match → dest list (subtract + render pink)
    """
    dest, corridor = [], []
    for poly in neto_polys:
        cx, cy = poly.centroid.x, poly.centroid.y
        best_d, best_room = float("inf"), None
        for r in floor_rooms:
            d = math.hypot(cx - r["x"], cy - r["y"])
            if d < best_d:
                best_d, best_room = d, r
        if best_room and best_d < MATCH_DIST and best_room.get("type","") in CORRIDOR_TYPES:
            corridor.append(poly)
        else:
            dest.append(poly)
    return dest, corridor


def compute_walkable(msp, bld, floor):
    bruto_polys = get_bruto(msp, bld)
    if not bruto_polys:
        return None, [], [], [], []

    outer = make_valid(unary_union(bruto_polys))
    neto_polys = get_neto(msp, bld)
    floor_rooms = ROOMS_BY_FLOOR.get((bld, floor), [])

    dest_polys, corridor_polys = classify_neto(neto_polys, floor_rooms)

    walkable = outer
    if dest_polys:
        try:
            walkable = make_valid(outer.difference(unary_union(dest_polys)))
        except Exception as ex:
            print(f"    WARN dest subtraction: {ex}", file=sys.stderr)

    barriers = get_barriers(msp, bld)
    if barriers:
        try:
            buf = unary_union([ls.buffer(8) for _, ls in barriers])
            walkable = make_valid(walkable.difference(buf))
        except Exception as ex:
            print(f"    WARN barrier subtraction: {ex}", file=sys.stderr)

    print(f"    dest={len(dest_polys)} corridor={len(corridor_polys)} barriers={len(barriers)}", file=sys.stderr)
    return walkable, bruto_polys, dest_polys, corridor_polys, barriers


def patches(geom, **kw):
    result = []
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon": continue
        result.append(MplPolygon(np.array(p.exterior.coords), **kw))
    return result


def render_floor(ax, bld, floor, msp):
    walkable, bruto_polys, dest_polys, corridor_polys, barriers = compute_walkable(msp, bld, floor)

    ax.set_aspect("equal")
    ax.set_title(f"B{bld} F{floor}", fontsize=9, pad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Walkable — blue
    if walkable and not walkable.is_empty:
        for p in patches(walkable, closed=True, facecolor="#b3d9ff", edgecolor="none", alpha=0.85):
            ax.add_patch(p)

    # Destination rooms — pink
    for poly in dest_polys:
        ax.add_patch(MplPolygon(np.array(poly.exterior.coords), closed=True,
                                facecolor="#ffb3b3", edgecolor="#cc6666", linewidth=0.4, alpha=0.7))

    # Bruto outline
    for bp in bruto_polys:
        ax.plot(*bp.exterior.xy, color="#222222", lw=1.0, alpha=0.9)

    # Barrier lines
    colours = {"Beton": "#333333", "Window": "#3366ff", "Mabat": "#ff8800"}
    for layer, ls in barriers:
        ax.plot(*ls.xy, color=colours.get(layer, "gray"), lw=0.35, alpha=0.6)

    # Door thresholds
    door_pts = get_doors(msp, bld)
    if door_pts:
        dx, dy = zip(*door_pts)
        ax.scatter(dx, dy, s=10, c="#00aa00", zorder=6, linewidths=0)

    # Stair / elevator waypoints
    floor_rooms = ROOMS_BY_FLOOR.get((bld, floor), [])
    for r in floor_rooms:
        if r.get("type", "") in STAIR_TYPES:
            ax.scatter([r["door_x"]], [r["door_y"]], s=80, c="cyan",
                       marker="*", zorder=7, edgecolors="#007777", linewidths=0.5)

    # Axis limits
    if bruto_polys:
        minx, miny, maxx, maxy = unary_union(bruto_polys).bounds
        pad = max(maxx-minx, maxy-miny) * 0.04
        ax.set_xlim(minx-pad, maxx+pad)
        ax.set_ylim(miny-pad, maxy+pad)

    warea = int(walkable.area) if walkable and not walkable.is_empty else 0
    ax.text(0.02, 0.02, f"doors={len(door_pts)}\nwalkable={warea:,}",
            transform=ax.transAxes, fontsize=6, va="bottom", color="#333333")


# ── main ──────────────────────────────────────────────────────────────────────

FLOORS = [("21",1),("21",2),("21",3),("21",4),("21",5),("21",6),
          ("22",1),("22",2),("22",3),("22",4),("22",5),("22",6)]

NCOLS = 4
NROWS = math.ceil(len(FLOORS) / NCOLS)
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS*6, NROWS*6))
axes = axes.flatten()
for ax in axes[len(FLOORS):]: ax.set_visible(False)

for idx, (bld, floor) in enumerate(FLOORS):
    dxf_path = DXF_FILES.get((bld, floor))
    ax = axes[idx]
    if dxf_path is None or not dxf_path.exists():
        ax.set_title(f"B{bld} F{floor} — NO FILE", fontsize=9); continue

    print(f"Processing B{bld} F{floor} ...", file=sys.stderr)
    try:
        doc = ezdxf.readfile(str(dxf_path))
        render_floor(ax, bld, floor, doc.modelspace())

        fig_ind, ax_ind = plt.subplots(figsize=(12, 12))
        doc2 = ezdxf.readfile(str(dxf_path))
        render_floor(ax_ind, bld, floor, doc2.modelspace())
        out_ind = DOCS / f"walkable_b{bld}_f{floor}.png"
        fig_ind.tight_layout()
        fig_ind.savefig(str(out_ind), dpi=150, bbox_inches="tight")
        plt.close(fig_ind)
        print(f"  -> {out_ind}", file=sys.stderr)
    except Exception as e:
        ax.set_title(f"B{bld} F{floor} — ERROR", fontsize=9)
        ax.text(0.5, 0.5, str(e)[:80], transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="red")
        print(f"  ERROR B{bld} F{floor}: {e}", file=sys.stderr)

legend_handles = [
    mpatches.Patch(facecolor="#b3d9ff", label="Walkable (corridor / foyer / lobby)"),
    mpatches.Patch(facecolor="#ffb3b3", edgecolor="#cc6666", label="Destination room (non-walkable)"),
    mpatches.Patch(facecolor="#333333", label="Beton"),
    mpatches.Patch(facecolor="#3366ff", label="Window"),
    mpatches.Patch(facecolor="#ff8800", label="Mabat"),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#00aa00', markersize=6, label="Door"),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='cyan', markersize=8, label="Stair/Elev"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Campus Nav — Walkable area (blue) vs destination rooms (pink) — all 12 floors",
             fontsize=13, y=1.01)
out = DOCS / "walkable_all_floors.png"
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(str(out), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined: {out}", file=sys.stderr)
print("Done.")
