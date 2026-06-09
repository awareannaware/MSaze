#!/usr/bin/env python3
"""
skeleton_one_floor.py  —  grid-based nav nodes for B22 F5

Algorithm:
1. Erode walkable polygon by WALL_INSET (0.3 m) → navigable interior
2. Fill navigable with a regular grid every GRID_STEP (1 m)
3. Single-pass merge: pairs closer than GRID_STEP → midpoint (narrow corridors)
4. Connect each node to grid neighbours within CONNECT_R if segment stays inside navigable
5. Connect door nodes to their nearest grid node
"""

import json, math, sys
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_walkable import compute_walkable
from build_skeleton_graph import _extract_door_nodes, _dist, CORRIDOR_TYPES

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

BLD, FLOOR = "22", 5

# ── tunables ──────────────────────────────────────────────────────────────────
WALL_INSET  = 30    # 0.3 m  — navigable = walkable eroded by this
GRID_STEP   = 100   # 1 m    — grid spacing inside navigable
CONNECT_R   = 155   # ~√2 × GRID_STEP + margin
DOOR_JOIN_R = 250   # max door → grid node distance
# ─────────────────────────────────────────────────────────────────────────────

with open(ROOT / "backend" / "rooms.json") as f:
    all_rooms = json.load(f)

floor_rooms = [r for r in all_rooms if r["building"] == BLD and r["floor"] == FLOOR]
dest_rooms  = [r for r in floor_rooms if r.get("type", "") not in CORRIDOR_TYPES]

walkable, _, _, _ = compute_walkable(BLD, FLOOR, floor_rooms)
assert walkable is not None

# ── 1. Erode walkable ─────────────────────────────────────────────────────────
navigable = walkable.buffer(-WALL_INSET)
if navigable.is_empty:
    raise RuntimeError("Eroded polygon is empty — WALL_INSET too large")

# ── 2. Fill navigable with a regular grid ─────────────────────────────────────
minx, miny, maxx, maxy = navigable.bounds
all_raw = []
for x in np.arange(minx, maxx + GRID_STEP, GRID_STEP):
    for y in np.arange(miny, maxy + GRID_STEP, GRID_STEP):
        if navigable.contains(Point(x, y)):
            all_raw.append((x, y))

# ── 3. Single-pass merge: pairs closer than GRID_STEP → midpoint ──────────────
pairs = sorted(
    (_dist(all_raw[i], all_raw[j]), i, j)
    for i in range(len(all_raw))
    for j in range(i + 1, len(all_raw))
    if _dist(all_raw[i], all_raw[j]) < GRID_STEP
)
used = [False] * len(all_raw)
merged = []
for _, i, j in pairs:
    if used[i] or used[j]:
        continue
    merged.append(((all_raw[i][0] + all_raw[j][0]) / 2,
                   (all_raw[i][1] + all_raw[j][1]) / 2))
    used[i] = used[j] = True

grid_nodes = merged + [all_raw[i] for i in range(len(all_raw)) if not used[i]]
print(f"Raw grid nodes: {len(all_raw)}  after narrow-merge: {len(grid_nodes)}")

# ── 4. Connect neighbours ─────────────────────────────────────────────────────
def los(a, b):
    return navigable.covers(LineString([a, b]))

grid_edges = []
for i in range(len(grid_nodes)):
    for j in range(i + 1, len(grid_nodes)):
        d = _dist(grid_nodes[i], grid_nodes[j])
        if d <= CONNECT_R and los(grid_nodes[i], grid_nodes[j]):
            grid_edges.append((i, j))

# ensure every grid node has at least one edge
connected = set(k for e in grid_edges for k in e)
for i, g in enumerate(grid_nodes):
    if i not in connected:
        nearest = min((j for j in range(len(grid_nodes)) if j != i),
                      key=lambda j: _dist(g, grid_nodes[j]))
        grid_edges.append((i, nearest))

print(f"Grid edges: {len(grid_edges)}")

# ── 5. Door nodes ─────────────────────────────────────────────────────────────
door_nodes = _extract_door_nodes(BLD, FLOOR, dest_rooms)
door_nodes = [d for d in door_nodes
              if walkable.distance(Point(d["x"], d["y"])) <= WALL_INSET]

door_edges = []
for di, d in enumerate(door_nodes):
    pd = (d["x"], d["y"])
    by_dist = sorted(range(len(grid_nodes)), key=lambda gi: _dist(pd, grid_nodes[gi]))
    for gi in by_dist:
        seg = LineString([pd, grid_nodes[gi]])
        if walkable.covers(seg):
            door_edges.append((di, gi))
            break
    else:
        door_edges.append((di, by_dist[0]))

print(f"Door nodes: {len(door_nodes)}  connected: {len(door_edges)}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 18))
ax.set_aspect("equal")
ax.set_title(f"B{BLD} F{FLOOR} — grid nodes (1 m spacing, 0.3 m wall inset)", fontsize=12)

geoms = list(walkable.geoms) if walkable.geom_type == "MultiPolygon" else [walkable]
for gm in geoms:
    if gm.geom_type == "Polygon":
        ax.fill(*gm.exterior.xy, alpha=0.12, color="steelblue")

nav_geoms = list(navigable.geoms) if navigable.geom_type == "MultiPolygon" else [navigable]
for gm in nav_geoms:
    if gm.geom_type == "Polygon":
        ax.plot(*gm.exterior.xy, color="steelblue", lw=0.6, alpha=0.4, linestyle="--")

for i, j in grid_edges:
    a, b = grid_nodes[i], grid_nodes[j]
    ax.plot([a[0], b[0]], [a[1], b[1]], color="navy", lw=0.5, alpha=0.4, zorder=2)

for di, gi in door_edges:
    a = (door_nodes[di]["x"], door_nodes[di]["y"])
    b = grid_nodes[gi]
    ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=0.8, alpha=0.6, zorder=3)

ax.scatter([g[0] for g in grid_nodes], [g[1] for g in grid_nodes],
           s=8, c="red", zorder=5)

for d in door_nodes:
    ax.scatter(d["x"], d["y"], s=14, c="green", zorder=6)
    if d.get("room_id"):
        ax.annotate(d["room_id"], (d["x"], d["y"]),
                    fontsize=4, color="darkgreen", ha="center", va="bottom", zorder=7)

ax.legend(handles=[
    mpatches.Patch(color="steelblue", alpha=0.2, label="Walkable"),
    mpatches.Patch(color="steelblue", alpha=0.4, label="Navigable (0.3 m inset)"),
    mpatches.Patch(color="red",   label="Grid node (1 m)"),
    mpatches.Patch(color="green", label="Door node"),
    mpatches.Patch(color="navy",  label="Grid edge"),
    mpatches.Patch(color="green", alpha=0.5, label="Door→grid edge"),
], loc="upper right", fontsize=9)

out = DOCS / f"skeleton_test_b{BLD}_f{FLOOR}.png"
fig.savefig(str(out), dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
