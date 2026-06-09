#!/usr/bin/env python3
"""
Place a node at the midpoint of every door opening on B21 F3.
Also mark stairwell/elevator rooms from rooms.json.
Output: docs/door_nodes_b21_f3.png
"""
import json, math
from pathlib import Path
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent.parent
DATA21  = ROOT / "data" / "21"
DOCS    = ROOT / "docs"

BLD, FLOOR = "21", 3
DXF_PATH = DATA21 / "02013.dxf"

# Load rooms for stair/elevator overlay
with open(ROOT / "backend" / "rooms.json") as f:
    all_rooms = json.load(f)

floor_rooms = [r for r in all_rooms if r["building"] == BLD and r["floor"] == FLOOR]

STAIR_TYPES = {"חדר מדרגות", "מדרגות"}
ELEV_TYPES  = {"פיר מעלית", "פיר"}
VERT_TYPES  = STAIR_TYPES | ELEV_TYPES

vert_rooms = [r for r in floor_rooms if r.get("type","") in VERT_TYPES]

# Read DXF
doc = ezdxf.readfile(str(DXF_PATH))
msp = doc.modelspace()

# Collect Beton wall segments for background
beton_segs = []
for e in msp:
    if e.dxf.get("layer","") != "Beton": continue
    t = e.dxftype()
    if t == "LINE":
        beton_segs.append(((e.dxf.start.x, e.dxf.start.y),(e.dxf.end.x, e.dxf.end.y)))
    elif t == "LWPOLYLINE":
        pts = [(p[0],p[1]) for p in e.get_points()]
        for i in range(len(pts)-1):
            beton_segs.append((pts[i], pts[i+1]))

# Door nodes: centroid of each Door LWPOLYLINE, then merge pairs < 25 units apart
MERGE_MIN  = 75  # door width lower bound
MERGE_MAX  = 95  # door width upper bound (~85 units = standard door)
raw = []
for e in msp:
    if e.dxf.get("layer","") != "Door": continue
    if e.dxftype() != "LWPOLYLINE": continue
    verts = list(e.get_points())
    if len(verts) < 2: continue
    x = sum(p[0] for p in verts) / len(verts)
    y = sum(p[1] for p in verts) / len(verts)
    raw.append((x, y))

# Greedy merge: if two raw centroids are within MERGE_DIST, replace with their midpoint
used = [False] * len(raw)
door_nodes = []
for i, (x1, y1) in enumerate(raw):
    if used[i]: continue
    merged = False
    for j, (x2, y2) in enumerate(raw):
        if j <= i or used[j]: continue
        d = math.hypot(x1-x2, y1-y2)
        if MERGE_MIN <= d <= MERGE_MAX:
            door_nodes.append(((x1+x2)/2, (y1+y2)/2))
            used[i] = used[j] = True
            merged = True
            break
    if not merged:
        door_nodes.append((x1, y1))
        used[i] = True

print(f"Door nodes: {len(door_nodes)}")
print(f"Stair/elevator rooms: {len(vert_rooms)}")
for r in vert_rooms:
    print(f"  {r['id']} ({r['type']}) centroid=({r['centroid_x']:.1f}, {r['centroid_y']:.1f})")

# Plot
fig, ax = plt.subplots(figsize=(18, 18))
ax.set_aspect("equal")
ax.set_title(f"B21 F3 — Door nodes (green) + Stair/Elev rooms (orange)\n{len(door_nodes)} doors, {len(vert_rooms)} vert rooms")
ax.set_facecolor("#f8f8f8")

for (x1,y1),(x2,y2) in beton_segs:
    ax.plot([x1,x2],[y1,y2], color="#555", lw=0.5, alpha=0.6)

if door_nodes:
    dx, dy = zip(*door_nodes)
    ax.scatter(dx, dy, s=10, c="green", zorder=5, label=f"Door node ({len(door_nodes)})")

for r in vert_rooms:
    ax.scatter(r["centroid_x"], r["centroid_y"], s=120, c="orange",
               marker="*", zorder=6)
    ax.annotate(r["id"], (r["centroid_x"], r["centroid_y"]),
                fontsize=6, color="darkorange", ha="center", va="bottom")

ax.legend(loc="upper right", fontsize=9)
out = DOCS / "door_nodes_b21_f3.png"
fig.savefig(str(out), dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
