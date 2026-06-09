#!/usr/bin/env python3
"""
Generate door node PNGs + backend/door_nodes.json for all floors.
Output: docs/door_nodes_b{bld}_f{floor}.png
        backend/door_nodes.json  — [{x, y, building, floor, room_id}]
B22 DXF files are in local coordinates; apply b22_transform.json offset to get global frame.
"""
import json, math
from pathlib import Path
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent.parent
DOCS   = ROOT / "docs"

with open(ROOT / "backend" / "rooms.json") as f:
    all_rooms = json.load(f)

with open(ROOT / "docs" / "b22_transform.json") as f:
    b22_tf = json.load(f)
B22_OX = b22_tf["offset_x"]
B22_OY = b22_tf["offset_y"]

STAIR_TYPES = {"חדר מדרגות", "מדרגות"}
ELEV_TYPES  = {"פיר מעלית", "פיר"}
VERT_TYPES  = STAIR_TYPES | ELEV_TYPES
CORRIDOR_ROOM_TYPES = {"פויאה", "פויה", "מבוא", "מעבר", "פרוזדור", "מרפסת"} | VERT_TYPES

MERGE_MIN = 65
MERGE_MAX = 110

FLOORS = [
    ("21", 1, ROOT / "data" / "21" / "02011.dxf"),
    ("21", 2, ROOT / "data" / "21" / "02012.dxf"),
    ("21", 3, ROOT / "data" / "21" / "02013.dxf"),
    ("21", 4, ROOT / "data" / "21" / "02014.dxf"),
    ("21", 5, ROOT / "data" / "21" / "02015.dxf"),
    ("21", 6, ROOT / "data" / "21" / "02016.dxf"),
    ("22", 2, ROOT / "data" / "22" / "02022.dxf"),
    ("22", 3, ROOT / "data" / "22" / "02023.dxf"),
    ("22", 4, ROOT / "data" / "22" / "02024.dxf"),
    ("22", 5, ROOT / "data" / "22" / "02025.dxf"),
    ("22", 6, ROOT / "data" / "22" / "02026.dxf"),
]

results = []
all_door_nodes_json = []

for bld, floor, dxf_path in FLOORS:
    label = f"B{bld} F{floor}"
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        print(f"ERROR loading {label}: {e}")
        results.append((bld, floor, None, str(e)))
        continue

    msp = doc.modelspace()
    is_b22 = (bld == "22")

    def to_global(x, y):
        if is_b22:
            return x + B22_OX, y + B22_OY
        return x, y

    # Collect Beton wall segments
    beton_segs = []
    for e in msp:
        if e.dxf.get("layer", "") != "Beton":
            continue
        t = e.dxftype()
        if t == "LINE":
            x1, y1 = to_global(e.dxf.start.x, e.dxf.start.y)
            x2, y2 = to_global(e.dxf.end.x, e.dxf.end.y)
            beton_segs.append(((x1, y1), (x2, y2)))
        elif t == "LWPOLYLINE":
            pts = [to_global(p[0], p[1]) for p in e.get_points()]
            for i in range(len(pts) - 1):
                beton_segs.append((pts[i], pts[i + 1]))

    # Collect raw door centroids + long-axis direction
    raw = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door":
            continue
        if e.dxftype() != "LWPOLYLINE":
            continue
        verts = list(e.get_points())
        if len(verts) < 2:
            continue
        cx = sum(p[0] for p in verts) / len(verts)
        cy = sum(p[1] for p in verts) / len(verts)
        gx, gy = to_global(cx, cy)
        # Long axis = direction of the longest edge of the rectangle
        best_len, ax, ay = 0, 1, 0
        for k in range(len(verts)):
            dx = verts[(k+1) % len(verts)][0] - verts[k][0]
            dy = verts[(k+1) % len(verts)][1] - verts[k][1]
            L = math.hypot(dx, dy)
            if L > best_len:
                best_len, ax, ay = L, dx / L, dy / L
        raw.append((gx, gy, ax, ay))

    # Greedy merge: two entities are the same door if the vector connecting them
    # is perpendicular to their long axis (they sit side-by-side across the opening).
    # No fixed distance limit — works for any door width.
    AXIS_ALIGN_MAX = 0.3   # dot product threshold: <0.3 = mostly perpendicular = same door
    DIST_MAX = 300         # sanity cap — ignore pairs farther than this
    used = [False] * len(raw)
    door_nodes = []
    for i, (x1, y1, ax1, ay1) in enumerate(raw):
        if used[i]:
            continue
        merged = False
        best = (float("inf"), -1)
        for j, (x2, y2, ax2, ay2) in enumerate(raw):
            if j <= i or used[j]:
                continue
            d = math.hypot(x1 - x2, y1 - y2)
            if d > DIST_MAX:
                continue
            vx, vy = (x2 - x1) / d, (y2 - y1) / d
            align = abs(ax1 * vx + ay1 * vy)
            if align < AXIS_ALIGN_MAX and d < best[0]:
                best = (d, j)
        if best[1] >= 0:
            j = best[1]
            x2, y2 = raw[j][0], raw[j][1]
            door_nodes.append(((x1 + x2) / 2, (y1 + y2) / 2))
            used[i] = used[j] = True
            merged = True
        if not merged:
            door_nodes.append((x1, y1))
            used[i] = True

    # Stair/elevator rooms for overlay
    floor_rooms = [r for r in all_rooms if r["building"] == bld and r["floor"] == floor]
    vert_rooms = [r for r in floor_rooms if r.get("type", "") in VERT_TYPES]
    dest_rooms = [r for r in floor_rooms if r.get("type", "") not in CORRIDOR_ROOM_TYPES]

    # Match each door node to nearest destination room by centroid distance
    ROOM_MATCH_DIST = 400
    door_labels = []
    for nx, ny in door_nodes:
        best_d, best_r = float("inf"), None
        for r in dest_rooms:
            d = math.hypot(nx - r["centroid_x"], ny - r["centroid_y"])
            if d < best_d:
                best_d, best_r = d, r
        door_labels.append(best_r["id"] if (best_r and best_d < ROOM_MATCH_DIST) else "")

    # Save to JSON list
    for (nx, ny), rid in zip(door_nodes, door_labels):
        all_door_nodes_json.append({
            "x": round(nx, 1), "y": round(ny, 1),
            "building": bld, "floor": floor, "room_id": rid
        })

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_aspect("equal")
    ax.set_title(
        f"B{bld} F{floor} — Door nodes (green) + Stair/Elev rooms (orange)\n"
        f"{len(door_nodes)} doors, {len(vert_rooms)} vert rooms"
    )
    ax.set_facecolor("#f8f8f8")

    for (sx1, sy1), (sx2, sy2) in beton_segs:
        ax.plot([sx1, sx2], [sy1, sy2], color="#555", lw=0.5, alpha=0.6)

    for (nx, ny), label in zip(door_nodes, door_labels):
        ax.scatter(nx, ny, s=10, c="green", zorder=5)
        ax.annotate(label, (nx, ny), fontsize=5, color="darkgreen",
                    ha="center", va="bottom", zorder=7)

    for r in vert_rooms:
        ax.scatter(r["centroid_x"], r["centroid_y"], s=120, c="orange", marker="*", zorder=6)
        ax.annotate(
            r["id"], (r["centroid_x"], r["centroid_y"]),
            fontsize=6, color="darkorange", ha="center", va="bottom"
        )

    ax.legend(loc="upper right", fontsize=9)
    out = DOCS / f"door_nodes_b{bld}_f{floor}.png"
    fig.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"{label}: {len(door_nodes)} door nodes, {len(vert_rooms)} vert rooms — saved {out.name}")
    results.append((bld, floor, len(door_nodes), None))

out_json = ROOT / "backend" / "door_nodes.json"
with open(out_json, "w") as f:
    json.dump(all_door_nodes_json, f)
print(f"\nSaved {len(all_door_nodes_json)} door nodes → {out_json.name}")
labeled = sum(1 for d in all_door_nodes_json if d["room_id"])
print(f"  {labeled} with room_id, {len(all_door_nodes_json)-labeled} unlabeled")

print("\n=== Summary ===")
for bld, floor, count, err in results:
    if err:
        print(f"  B{bld} F{floor}: ERROR — {err}")
    else:
        print(f"  B{bld} F{floor}: {count} door nodes")
