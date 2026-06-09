#!/usr/bin/env python3
"""
build_skeleton_graph.py
-----------------------
Builds nav_graph2.json for all 11 floors (B21 F1-F6, B22 F2-F6).

Per-floor algorithm:
1. Erode walkable polygon by WALL_INSET (0.3 m) → navigable
2. Fill navigable with a regular grid every GRID_STEP (1 m)
3. Single-pass merge: grid pairs closer than GRID_STEP → midpoint (narrow corridors)
4. Connect neighbours within CONNECT_R if segment stays inside navigable (.covers)
5. Every node gets at least 1 edge
6. Door nodes: connect each to nearest grid node via walkable LOS

Cross-floor:
7. Stair edges: group stair rooms by (building, last-2-digits), link consecutive floors
8. Cross-building edge: B21 F2 ↔ B22 F2 minimum-gap pair

Output:
  backend/nav_graph2.json
  docs/nav_graph_b{bld}_f{floor}.png  (verification overlays)
"""

import json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_walkable import compute_walkable, B22_OX, B22_OY

import ezdxf

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BACK = ROOT / "backend"

# ── tunables ──────────────────────────────────────────────────────────────────
WALL_INSET   = 30    # 0.3 m  — navigable = walkable eroded by this
GRID_STEP    = 100   # 1 m    — grid node spacing
CONNECT_R    = 155   # ~√2 × GRID_STEP + margin
DOOR_JOIN_R  = 300   # max door → grid node search distance
STAIR_COST   = 400   # cost per floor for stairs
ELEV_COST    = 100   # cost per floor for elevator (much faster)
CROSS_COST   = 300   # B21 ↔ B22 lobby link cost
# ─────────────────────────────────────────────────────────────────────────────

FLOOR_LIST = [
    ("21", 1), ("21", 2), ("21", 3), ("21", 4), ("21", 5), ("21", 6),
    ("22", 1), ("22", 2), ("22", 3), ("22", 4), ("22", 5), ("22", 6),
]
DXF_FILES = {
    ("21", 1): ROOT / "data" / "21" / "02011.dxf",
    ("21", 2): ROOT / "data" / "21" / "02012.dxf",
    ("21", 3): ROOT / "data" / "21" / "02013.dxf",
    ("21", 4): ROOT / "data" / "21" / "02014.dxf",
    ("21", 5): ROOT / "data" / "21" / "02015.dxf",
    ("21", 6): ROOT / "data" / "21" / "02016.dxf",
    ("22", 1): ROOT / "data" / "22" / "02021.dxf",
    ("22", 2): ROOT / "data" / "22" / "02022.dxf",
    ("22", 3): ROOT / "data" / "22" / "02023.dxf",
    ("22", 4): ROOT / "data" / "22" / "02024.dxf",
    ("22", 5): ROOT / "data" / "22" / "02025.dxf",
    ("22", 6): ROOT / "data" / "22" / "02026.dxf",
}
STAIR_TYPES    = {"חדר מדרגות", "מדרגות"}
ELEV_TYPES     = {"פיר מעלית", "פיר"}
ELEV_SHAFT_IDS: set = set()  # populated at runtime from rooms.json
VERT_TYPES     = STAIR_TYPES | ELEV_TYPES
CORRIDOR_TYPES = {"פויאה", "פויה", "מבוא", "מעבר", "פרוזדור", "מרפסת"} | VERT_TYPES
AXIS_ALIGN_MAX = 0.3
DIST_MAX       = 300
ROOM_MATCH_DIST = 400


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── Load pre-computed door nodes (from door_nodes_all.py) ─────────────────────
with open(BACK / "door_nodes.json") as _f:
    _raw_door_nodes = json.load(_f)
_DOOR_NODES_BY_FLOOR: dict = {}
for _d in _raw_door_nodes:
    key = (_d["building"], _d["floor"])
    _DOOR_NODES_BY_FLOOR.setdefault(key, []).append(_d)


# ── Door node extraction (same as door_nodes_all.py) ─────────────────────────
def _extract_door_nodes(bld, floor, dest_rooms):
    dxf_path = DXF_FILES.get((bld, floor))
    if not dxf_path or not dxf_path.exists():
        return []
    doc  = ezdxf.readfile(str(dxf_path))
    msp  = doc.modelspace()
    is22 = (bld == "22")

    def to_global(x, y):
        return (x + B22_OX, y + B22_OY) if is22 else (x, y)

    raw = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door" or e.dxftype() != "LWPOLYLINE":
            continue
        verts = list(e.get_points())
        if len(verts) < 2:
            continue
        cx = sum(p[0] for p in verts) / len(verts)
        cy = sum(p[1] for p in verts) / len(verts)
        gx, gy = to_global(cx, cy)
        best_len, ax, ay = 0, 1.0, 0.0
        for k in range(len(verts)):
            dx = verts[(k + 1) % len(verts)][0] - verts[k][0]
            dy = verts[(k + 1) % len(verts)][1] - verts[k][1]
            L = math.hypot(dx, dy)
            if L > best_len:
                best_len, ax, ay = L, dx / L, dy / L
        raw.append((gx, gy, ax, ay))

    used = [False] * len(raw)
    nodes = []
    for i, (x1, y1, ax1, ay1) in enumerate(raw):
        if used[i]:
            continue
        best = (float("inf"), -1)
        for j, (x2, y2, ax2, ay2) in enumerate(raw):
            if j <= i or used[j]:
                continue
            d = _dist((x1, y1), (x2, y2))
            if d > DIST_MAX:
                continue
            vx, vy = (x2 - x1) / d, (y2 - y1) / d
            if abs(ax1 * vx + ay1 * vy) < AXIS_ALIGN_MAX and d < best[0]:
                best = (d, j)
        if best[1] >= 0:
            j = best[1]
            mx, my = (x1 + raw[j][0]) / 2, (y1 + raw[j][1]) / 2
            used[i] = used[j] = True
        else:
            mx, my = x1, y1
            used[i] = True
        room_id = ""
        best_d = float("inf")
        for r in dest_rooms:
            d = _dist((mx, my), (r["centroid_x"], r["centroid_y"]))
            if d < best_d:
                best_d, room_id = d, r["id"]
        if best_d > ROOM_MATCH_DIST:
            room_id = ""
        nodes.append({"x": mx, "y": my, "room_id": room_id})
    return nodes


# ── Per-floor graph builder ───────────────────────────────────────────────────
def build_floor_graph(bld, floor, floor_rooms):
    walkable, _, _, _ = compute_walkable(bld, floor, floor_rooms)
    if walkable is None or walkable.is_empty:
        print(f"  B{bld} F{floor}: no walkable — skip")
        return [], []

    navigable = walkable.buffer(-WALL_INSET)
    if navigable is None or navigable.is_empty:
        print(f"  B{bld} F{floor}: navigable empty after erosion — skip")
        return [], []

    dest_rooms = [r for r in floor_rooms if r.get("type", "") not in CORRIDOR_TYPES]
    vert_rooms = [r for r in floor_rooms if r.get("type", "") in VERT_TYPES]

    # ── Grid fill — interior ──────────────────────────────────────────────────
    minx, miny, maxx, maxy = navigable.bounds
    all_raw = []
    for x in np.arange(minx, maxx + GRID_STEP, GRID_STEP):
        for y in np.arange(miny, maxy + GRID_STEP, GRID_STEP):
            if navigable.contains(Point(x, y)):
                all_raw.append((x, y))

    # ── Boundary nodes — sample navigable exterior at 1 m intervals ───────────
    # Ensures thin corridors and narrow connections always get nodes even when
    # the interior grid misses them.
    def sample_ring(ring):
        L = ring.length
        return [(ring.interpolate(d).x, ring.interpolate(d).y)
                for d in np.arange(0, L, GRID_STEP)]

    nav_polys = list(navigable.geoms) if navigable.geom_type == "MultiPolygon" else [navigable]
    for poly in nav_polys:
        if poly.geom_type == "Polygon":
            all_raw.extend(sample_ring(poly.exterior))
            for interior in poly.interiors:
                all_raw.extend(sample_ring(interior))

    # ── Narrow-corridor merge (single pass) ───────────────────────────────────
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
    grid_pts = merged + [all_raw[i] for i in range(len(all_raw)) if not used[i]]

    # ── Build node list ───────────────────────────────────────────────────────
    nodes = []
    for x, y in grid_pts:
        nodes.append({
            "id":       f"G-{bld}-F{floor}-{len(nodes)}",
            "type":     "GRID",
            "building": bld,
            "floor":    floor,
            "x":        round(x, 1),
            "y":        round(y, 1),
            "room_id":  "",
        })

    # Stair/elevator rooms
    stair_start = len(nodes)
    for r in vert_rooms:
        nodes.append({
            "id":          f"S-{bld}-F{floor}-{r['id']}",
            "type":        "STAIR_LANDING",
            "building":    bld,
            "floor":       floor,
            "x":           round(r["centroid_x"], 1),
            "y":           round(r["centroid_y"], 1),
            "room_id":     r["id"],
            "stair_shaft": r["id"][-2:],
        })

    # ── Grid edges ────────────────────────────────────────────────────────────
    edges = []

    def add_edge(i, j, etype, cost=None):
        xi, yi = nodes[i]["x"], nodes[i]["y"]
        xj, yj = nodes[j]["x"], nodes[j]["y"]
        edges.append({
            "from": nodes[i]["id"],
            "to":   nodes[j]["id"],
            "type": etype,
            "cost": round(cost if cost is not None else _dist((xi, yi), (xj, yj)), 1),
        })

    n_grid = len(grid_pts)
    for i in range(n_grid):
        for j in range(i + 1, n_grid):
            a = (nodes[i]["x"], nodes[i]["y"])
            b = (nodes[j]["x"], nodes[j]["y"])
            if _dist(a, b) <= CONNECT_R and navigable.covers(LineString([a, b])):
                add_edge(i, j, "CORRIDOR")

    # ensure every grid node has at least one edge
    connected = set(k for e in edges for k in
                    [next(idx for idx, n in enumerate(nodes) if n["id"] == e["from"]),
                     next(idx for idx, n in enumerate(nodes) if n["id"] == e["to"])])
    for i in range(n_grid):
        if i not in connected:
            nearest = min((j for j in range(n_grid) if j != i),
                          key=lambda j: _dist((nodes[i]["x"], nodes[i]["y"]),
                                              (nodes[j]["x"], nodes[j]["y"])))
            add_edge(i, nearest, "CORRIDOR")

    # ── Door nodes ────────────────────────────────────────────────────────────
    door_raw = _DOOR_NODES_BY_FLOOR.get((bld, floor), [])
    door_raw = [d for d in door_raw
                if walkable.distance(Point(d["x"], d["y"])) <= WALL_INSET]

    door_start = len(nodes)
    for d in door_raw:
        nodes.append({
            "id":       f"D-{bld}-F{floor}-{len(nodes) - door_start}",
            "type":     "DOOR",
            "building": bld,
            "floor":    floor,
            "x":        round(d["x"], 1),
            "y":        round(d["y"], 1),
            "room_id":  d["room_id"],
        })

    # Use walkable expanded slightly so doors at the boundary can connect through
    walkable_loose = walkable.buffer(5)
    for di in range(door_start, len(nodes)):
        pd = (nodes[di]["x"], nodes[di]["y"])
        by_dist = sorted(range(n_grid), key=lambda gi: _dist(pd, (nodes[gi]["x"], nodes[gi]["y"])))
        # Connect to up to 2 nearest reachable grid nodes — ensures both sides of a doorway connect
        connected_count = 0
        for gi in by_dist:
            if _dist(pd, (nodes[gi]["x"], nodes[gi]["y"])) > DOOR_JOIN_R:
                break
            seg = LineString([pd, (nodes[gi]["x"], nodes[gi]["y"])])
            if walkable_loose.covers(seg):
                add_edge(di, gi, "DOOR_LINK")
                connected_count += 1
                if connected_count >= 2:
                    break
        if connected_count == 0:
            add_edge(di, by_dist[0], "DOOR_LINK")

    # Stair/elevator nodes → nearest grid nodes, NO LOS check.
    # Beton walls enclose these rooms so LOS always fails, but they MUST connect
    # because they are the only floor-change mechanism.
    # Build main grid component so we can guarantee stair connects into it.
    _gadj = defaultdict(set)
    for _e in edges:
        if _e["type"] == "CORRIDOR":
            _gadj[_e["from"]].add(_e["to"])
            _gadj[_e["to"]].add(_e["from"])
    _grid_ids = {nodes[gi]["id"] for gi in range(n_grid)}
    _unvis = set(_grid_ids); _comps = []
    while _unvis:
        _s = next(iter(_unvis)); _c = set(); _q = [_s]
        while _q:
            _u = _q.pop(); _c.add(_u); _unvis.discard(_u)
            for _v in _gadj[_u]:
                if _v in _unvis: _q.append(_v)
        _comps.append(_c)
    _main_ids = max(_comps, key=len)
    _main_gi = [gi for gi in range(n_grid) if nodes[gi]["id"] in _main_ids]

    for si in range(stair_start, stair_start + len(vert_rooms)):
        ps = (nodes[si]["x"], nodes[si]["y"])
        by_dist = sorted(range(n_grid), key=lambda gi: _dist(ps, (nodes[gi]["x"], nodes[gi]["y"])))
        connected = []
        for gi in by_dist[:2]:
            add_edge(si, gi, "DOOR_LINK"); connected.append(gi)
        # If nearest 2 nodes are all in isolated pockets, also link to nearest main-comp node
        if not any(nodes[gi]["id"] in _main_ids for gi in connected):
            best = min(_main_gi, key=lambda gi: _dist(ps, (nodes[gi]["x"], nodes[gi]["y"])))
            add_edge(si, best, "DOOR_LINK")

    print(f"  B{bld} F{floor}: {n_grid} grid + {len(vert_rooms)} stairs + "
          f"{len(door_raw)} doors → {len(edges)} edges")
    return nodes, edges


# ── Overlay PNG ───────────────────────────────────────────────────────────────
def render_overlay(bld, floor, floor_rooms, nodes, edges):
    walkable, _, _, _ = compute_walkable(bld, floor, floor_rooms)
    if not walkable:
        return
    navigable = walkable.buffer(-WALL_INSET)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect("equal")
    ax.set_title(f"B{bld} F{floor} — nav graph (1 m grid, 0.3 m inset)", fontsize=11)

    for gm in (list(walkable.geoms) if walkable.geom_type == "MultiPolygon" else [walkable]):
        if gm.geom_type == "Polygon":
            ax.fill(*gm.exterior.xy, alpha=0.12, color="steelblue")
    for gm in (list(navigable.geoms) if navigable.geom_type == "MultiPolygon" else [navigable]):
        if gm.geom_type == "Polygon":
            ax.plot(*gm.exterior.xy, color="steelblue", lw=0.5, alpha=0.35, ls="--")

    id2node = {n["id"]: n for n in nodes}
    colors  = {"CORRIDOR": "navy", "DOOR_LINK": "green", "STAIR": "orange"}
    for e in edges:
        a, b = id2node[e["from"]], id2node[e["to"]]
        c = colors.get(e["type"], "gray")
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=c, lw=0.5, alpha=0.45, zorder=2)

    for n in nodes:
        c = {"GRID": "red", "DOOR": "green", "STAIR_LANDING": "cyan"}.get(n["type"], "gray")
        s = {"GRID": 6, "DOOR": 14, "STAIR_LANDING": 80}.get(n["type"], 6)
        ax.scatter(n["x"], n["y"], s=s, c=c, zorder=5)
        if n.get("room_id") and n["type"] == "DOOR":
            ax.annotate(n["room_id"], (n["x"], n["y"]),
                        fontsize=3.5, color="darkgreen", ha="center", va="bottom", zorder=6)

    ax.legend(handles=[
        mpatches.Patch(color="steelblue", alpha=0.2, label="Walkable"),
        mpatches.Patch(color="red",    label="Grid node"),
        mpatches.Patch(color="green",  label="Door node"),
        mpatches.Patch(color="cyan",   label="Stair landing"),
        mpatches.Patch(color="navy",   label="Corridor edge"),
        mpatches.Patch(color="green",  alpha=0.5, label="Door/stair link"),
    ], loc="upper right", fontsize=8)

    out = DOCS / f"nav_graph_b{bld}_f{floor}.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    overlay → {out.name}")


# ── Stair + cross-building edges ──────────────────────────────────────────────
def add_vertical_edges(all_nodes, all_edges):
    by_shaft = defaultdict(list)
    for n in all_nodes:
        if n["type"] == "STAIR_LANDING":
            by_shaft[(n["building"], n["stair_shaft"])].append(n)
    for (bld, shaft), group in by_shaft.items():
        is_elev = shaft in ELEV_SHAFT_IDS
        cost = ELEV_COST if is_elev else STAIR_COST
        etype = "ELEVATOR" if is_elev else "STAIR"
        for a, b in zip(sorted(group, key=lambda n: n["floor"]),
                        sorted(group, key=lambda n: n["floor"])[1:]):
            all_edges.append({"from": a["id"], "to": b["id"],
                               "type": etype, "cost": cost})


def add_cross_building_edge(all_nodes, all_edges):
    b21 = [n for n in all_nodes if n["building"] == "21" and n["floor"] == 2]
    b22 = [n for n in all_nodes if n["building"] == "22" and n["floor"] == 2]
    if not b21 or not b22:
        return

    # Build main component of B21 so we only pick a connected node
    adj_tmp = defaultdict(set)
    for e in all_edges:
        adj_tmp[e["from"]].add(e["to"])
        adj_tmp[e["to"]].add(e["from"])
    b21_ids = {n["id"] for n in b21}
    unvis = set(b21_ids); comps = []
    while unvis:
        s = next(iter(unvis)); comp = set(); q = [s]
        while q:
            u = q.pop()
            if u in comp: continue
            comp.add(u); unvis.discard(u)
            for v in adj_tmp[u]:
                if v in unvis: q.append(v)
        comps.append(comp)
    b21_main = max(comps, key=len)
    b21_connected = [n for n in b21 if n["id"] in b21_main]

    best = min(((a, b) for a in b21_connected for b in b22),
               key=lambda ab: _dist((ab[0]["x"], ab[0]["y"]), (ab[1]["x"], ab[1]["y"])))
    a, b = best
    gap = _dist((a["x"], a["y"]), (b["x"], b["y"]))
    all_edges.append({"from": a["id"], "to": b["id"],
                      "type": "CROSS_BLD", "cost": CROSS_COST})
    print(f"  Cross-building: {a['id']} ↔ {b['id']}  gap={gap:.0f}u")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(BACK / "rooms.json") as f:
        all_rooms = json.load(f)

    # Record which shaft IDs are elevators (for lower cost edges)
    for _r in all_rooms:
        if _r.get("type", "") in ELEV_TYPES:
            ELEV_SHAFT_IDS.add(_r["id"][-2:])

    all_nodes, all_edges = [], []

    for bld, floor in FLOOR_LIST:
        floor_rooms = [r for r in all_rooms if r["building"] == bld and r["floor"] == floor]
        nodes, edges = build_floor_graph(bld, floor, floor_rooms)
        if not nodes:
            continue
        render_overlay(bld, floor, floor_rooms, nodes, edges)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    add_vertical_edges(all_nodes, all_edges)
    add_cross_building_edge(all_nodes, all_edges)

    out_path = BACK / "nav_graph2.json"
    with open(out_path, "w") as f:
        json.dump({"nodes": all_nodes, "edges": all_edges}, f,
                  ensure_ascii=False, indent=2)

    print(f"\nDone: {len(all_nodes)} nodes, {len(all_edges)} edges → {out_path}")
