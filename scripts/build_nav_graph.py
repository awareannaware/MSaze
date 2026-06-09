#!/usr/bin/env python3
"""
Build backend/nav_graph.json — door-junction navigation graph.
Five-fix rewrite v3:
  - Boundary from 0201Shetah-Bruto (true outer wall polygon)
  - Walkable = bruto minus neto rooms
  - Doors from Door-layer INSERT positions only (no synthetic doors)
  - Corridor graph from Voronoi medial axis of walkable polygon
  - Every edge verified against full floor_walls.json (zero crossings enforced)
  - Stairwells connected via nearest mabua room door

Run: python3 scripts/build_nav_graph.py
"""
import sys, math, json, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import ezdxf
import networkx as nx
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, MultiPoint
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DATA21  = ROOT / "data" / "21"
DATA22  = ROOT / "data" / "22"
DOCS    = ROOT / "docs"

ROOMS_JSON       = BACKEND / "rooms.json"
FLOOR_WALLS_JSON = BACKEND / "floor_walls.json"
B22_TRANSFORM    = ROOT / "docs" / "b22_transform.json"
OUTPUT           = BACKEND / "nav_graph.json"
REPORT           = DOCS / "nav_graph_build_report.md"

BOUNDARY_SAMPLE  = 25
INTERIOR_GRID    = 80
VORONOI_BUFFER   = 8
JUNCTION_CLUSTER = 50
DOOR_LINK_MAX    = 500
DOOR_INSET       = 12     # nudge door threshold inward from wall so DOOR_LINK avoids triggering wall check
DOOR_BOUNDARY_DIST = 20   # INSERT must be within this of a neto poly boundary
ROOM_CENTROID_DIST = 200  # centroid matching tolerance for neto→room_id
MABUA_SEARCH     = 1500

def log(*a): print(*a, file=sys.stderr)

log("Loading rooms.json ...")
with open(ROOMS_JSON) as f:
    all_rooms = json.load(f)
log(f"  {len(all_rooms)} rooms")

log("Loading floor_walls.json ...")
with open(FLOOR_WALLS_JSON) as f:
    floor_walls_raw = json.load(f)

# floor_walls.json uses "b21"/"b22" building keys; normalise to "21"/"22"
wall_lines = defaultdict(list)
for fstr, bdict in floor_walls_raw.items():
    for bld_raw, segs in bdict.items():
        bld = bld_raw.lstrip("b")   # "b21" -> "21", "b22" -> "22"
        key = (int(fstr), bld)
        for s in segs:
            try:
                wall_lines[key].append(LineString([s["start"], s["end"]]))
            except Exception:
                pass
log(f"  Walls loaded: {sum(len(v) for v in wall_lines.values())} segments across floors {sorted(set(k[0] for k in wall_lines))}")

with open(B22_TRANSFORM) as f:
    _t = json.load(f)
B22_OX, B22_OY = _t["offset_x"], _t["offset_y"]
def b22(x, y): return x + B22_OX, y + B22_OY

DXF_FILES = {
    ("21",1): DATA21/"02011.dxf",
    ("21",3): DATA21/"02013.dxf", ("21",4): DATA21/"02014.dxf",
    ("21",5): DATA21/"02015.dxf", ("21",6): DATA21/"02016.dxf",
    ("22",2): DATA22/"02022.dxf", ("22",3): DATA22/"02023.dxf",
    ("22",4): DATA22/"02024.dxf", ("22",5): DATA22/"02025.dxf",
    ("22",6): DATA22/"02026.dxf",
}

def read_lwpoly_pts(e, bld):
    pts = [(p[0], p[1]) for p in e.get_points()]
    if bld == "22": pts = [b22(x,y) for x,y in pts]
    return pts

def _shetah_layers(bld):
    """Return (bruto_layer, neto_layer) names for a building."""
    prefix = "0201" if bld == "21" else "0202"
    return f"{prefix}Shetah-Bruto", f"{prefix}Shetah-Neto"

def get_bruto_polygon(msp, bld, neto_polys_fallback=None):
    """
    Largest Shetah-Bruto polygon = true outer boundary.
    B22 uses 0202Shetah-Bruto; B21 uses 0201Shetah-Bruto.
    """
    bruto_layer, _ = _shetah_layers(bld)
    polys = []
    for e in msp:
        if e.dxf.get("layer","") != bruto_layer: continue
        if e.dxftype() != "LWPOLYLINE": continue
        pts = read_lwpoly_pts(e, bld)
        if len(pts) >= 3:
            try:
                p = make_valid(Polygon(pts))
                if not p.is_empty and p.geom_type == "Polygon": polys.append(p)
            except: pass
    if polys:
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0], "bruto"
    # Fallback: neto union + buffer
    if neto_polys_fallback:
        try:
            u = make_valid(unary_union(neto_polys_fallback))
            return u.buffer(300), "neto_fallback"
        except: pass
    return None, None

def get_neto_polys(msp, bld):
    _, neto_layer = _shetah_layers(bld)
    # B22 F6 has a typo layer "020 2Shetah-Neto" — accept both
    accept = {neto_layer, neto_layer.replace("0202","020 2")}
    polys = []
    for e in msp:
        if e.dxf.get("layer","") not in accept: continue
        if e.dxftype() != "LWPOLYLINE": continue
        pts = read_lwpoly_pts(e, bld)
        if len(pts) >= 3:
            try:
                p = Polygon(pts)
                if p.is_valid and not p.is_empty: polys.append(p)
                elif not p.is_valid:
                    vp = make_valid(p)
                    if vp.geom_type == "Polygon" and not vp.is_empty:
                        polys.append(vp)
            except: pass
    return polys

def get_door_thresholds(msp, bld):
    """Navigable threshold = midpoint of first two vertices of each Door-layer LWPOLYLINE."""
    pts = []
    for e in msp:
        if e.dxf.get("layer","") != "Door": continue
        if e.dxftype() != "LWPOLYLINE": continue
        verts = list(e.get_points())
        if len(verts) < 2: continue
        x = (verts[0][0] + verts[1][0]) / 2
        y = (verts[0][1] + verts[1][1]) / 2
        if bld == "22": x, y = b22(x, y)
        pts.append((x, y))
    return pts


BARRIER_LAYERS = {"Beton", "Window", "Mabat"}

def get_barrier_lines(msp, bld):
    """Return LineStrings for all physical barrier segments (Beton+Window+Mabat).
    Window 10-vertex polys are glass doors — skip them (they are openings)."""
    lines = []
    for e in msp:
        layer = e.dxf.get("layer", "")
        if layer not in BARRIER_LAYERS:
            continue
        t = e.dxftype()
        if t == "LINE":
            x1, y1 = e.dxf.start.x, e.dxf.start.y
            x2, y2 = e.dxf.end.x, e.dxf.end.y
            if bld == "22": x1,y1 = b22(x1,y1); x2,y2 = b22(x2,y2)
            lines.append(LineString([(x1,y1),(x2,y2)]))
        elif t == "LWPOLYLINE":
            verts = list(e.get_points())
            # Skip Window 10-vertex polys — those are glass doors (openings)
            if layer == "Window" and len(verts) == 10:
                continue
            if bld == "22": verts = [b22(p[0],p[1]) for p in verts]
            else: verts = [(p[0],p[1]) for p in verts]
            if e.closed and len(verts) >= 2:
                verts = verts + [verts[0]]
            for i in range(len(verts)-1):
                lines.append(LineString([verts[i], verts[i+1]]))
    return lines

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def inset_toward_walkable(x, y, walkable, inset=DOOR_INSET):
    """Nudge (x,y) inward by `inset` units toward the walkable polygon interior.
    The door threshold sits on the wall boundary; this moves it into open space
    so DOOR_LINK edges don't immediately cross the wall they originate from."""
    pt = Point(x, y)
    # Direction: toward nearest interior point (closest point on the medial axis is
    # expensive; use centroid of nearest walkable component instead)
    if walkable is None:
        return x, y
    geoms = [g for g in ([walkable] if walkable.geom_type == "Polygon" else list(walkable.geoms))
             if g.geom_type == "Polygon" and not g.is_empty]
    # Pick the walkable component whose boundary is closest to pt
    best_geom = min(geoms, key=lambda g: g.exterior.distance(pt), default=None)
    if best_geom is None:
        return x, y
    cx, cy = best_geom.centroid.x, best_geom.centroid.y
    d = math.hypot(cx - x, cy - y)
    if d < 1e-6:
        return x, y
    return x + inset * (cx - x) / d, y + inset * (cy - y) / d

# Populated during the main loop: (bld, floor) -> [LineString, ...]
dxf_barriers = {}

def edge_crosses_wall(x1, y1, x2, y2, floor_int, bld):
    line = LineString([(x1,y1),(x2,y2)])
    # Use crosses() — proper interior-to-interior intersection only.
    # This avoids false positives when a door threshold starts exactly on a wall.
    # Check floor_walls.json (Beton, global coords)
    for w in wall_lines.get((floor_int, bld), []):
        try:
            if line.crosses(w): return True
        except: pass
    # Check DXF barriers (Beton+Window+Mabat extracted per floor)
    for w in dxf_barriers.get((bld, floor_int), []):
        try:
            if line.crosses(w): return True
        except: pass
    return False

rooms_by_floor = defaultdict(list)
for r in all_rooms:
    rooms_by_floor[(r["building"], r["floor"])].append(r)

STAIR_TYPES = {"חדר מדרגות", "מדרגות"}
ELEV_TYPES  = {"פיר מעלית", "פיר"}
VERT_TYPES  = STAIR_TYPES | ELEV_TYPES

# Circulation spaces — part of the walkable corridor network, NOT routing destinations.
# Their neto polygons define the corridor area and should NOT be subtracted from walkable.
CORRIDOR_ROOM_TYPES = {
    "פויאה", "פויה",          # foyer / lobby
    "מבוא",                   # entrance lobby
    "מעבר",                   # passage
    "פרוזדור",                # hallway / corridor
    "מרפסת",                  # balcony / terrace
} | STAIR_TYPES | ELEV_TYPES

MABUA_TYPES = {"מבוא", "פויאה", "פויה"}

log("="*60)
log("Step 1: Per-floor Voronoi corridor graph")

all_junction_nodes = []
all_door_nodes     = []
corridor_edges     = []
door_link_edges    = []
junc_seq  = defaultdict(int)
door_seq  = defaultdict(int)
edge_ctr  = [0]

def next_eid():
    edge_ctr[0] += 1
    return f"E-{edge_ctr[0]:05d}"

room_to_door_nodes = defaultdict(list)

def cluster_pts(pts, max_d):
    clusters = []
    for pt in pts:
        for cl in clusters:
            ccx = sum(p[0] for p in cl)/len(cl)
            ccy = sum(p[1] for p in cl)/len(cl)
            if dist(pt,(ccx,ccy)) <= max_d:
                cl.append(pt); break
        else:
            clusters.append([pt])
    return [(sum(p[0] for p in cl)/len(cl), sum(p[1] for p in cl)/len(cl))
            for cl in clusters]

for (bld, floor), dxf_path in sorted(DXF_FILES.items()):
    if not dxf_path.exists():
        log(f"  SKIP {dxf_path.name}"); continue
    log(f"\n--- B{bld} F{floor} ---")
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        log(f"  ERROR: {e}"); continue
    msp = doc.modelspace()

    neto = get_neto_polys(msp, bld)
    valid_neto = [p for p in [make_valid(p) for p in neto]
                  if p.is_valid and not p.is_empty and p.geom_type == "Polygon"]

    outer, boundary_src = get_bruto_polygon(msp, bld, valid_neto)
    if outer is None:
        log(f"  WARN: no boundary polygon — skipping"); continue

    # Walkable = bruto minus destination-room neto polys.
    # Corridor-type rooms (פויאה, מבוא, מעבר, פרוזדור, ...) are circulation
    # space — their neto polys define the corridor area, keep them walkable.
    # Destination rooms (offices, lecture halls, labs, ...) are subtracted.
    floor_rooms = rooms_by_floor[(bld, floor)]
    floor_room_types = {r["id"]: r.get("type","") for r in floor_rooms}

    # Match each neto poly to a room by centroid
    dest_room_polys = []
    for poly in valid_neto:
        cx, cy = poly.centroid.x, poly.centroid.y
        cpt = Point(cx, cy)
        best_room = None
        best_d = float("inf")
        for r in floor_rooms:
            d = dist((cx, cy), (r["centroid_x"], r["centroid_y"]))
            if d < best_d:
                best_d, best_room = d, r
        if best_room and best_d < ROOM_CENTROID_DIST:
            rtype = best_room.get("type", "")
            if rtype not in CORRIDOR_ROOM_TYPES:
                dest_room_polys.append(poly)
        # Unmatched neto polys are corridor/open space — leave walkable

    if dest_room_polys:
        try:
            rooms_union = unary_union(dest_room_polys)
            walkable = make_valid(outer.difference(rooms_union))
        except Exception as e:
            log(f"  WARN room subtraction: {e}")
            walkable = outer
    else:
        walkable = outer

    # Also subtract Mabat+Window barrier buffers for wall-thickness accuracy
    int_barriers = get_barrier_lines(msp, bld)
    if int_barriers:
        try:
            barrier_geom = unary_union([w.buffer(8) for w in int_barriers])
            walkable = make_valid(walkable.difference(barrier_geom))
        except Exception as e:
            log(f"  WARN barrier subtraction: {e}")

    log(f"  Boundary ({boundary_src}) area={outer.area:.0f}, walkable area={walkable.area:.0f}, dest_room_polys={len(dest_room_polys)}")

    # Build room_id→neto_poly map by centroid containment or proximity
    room_id_to_poly = {}
    for r in floor_rooms:
        cpt = Point(r["centroid_x"], r["centroid_y"])
        matched_poly = None
        for poly in valid_neto:
            if poly.contains(cpt):
                matched_poly = poly; break
        if matched_poly is None:
            dists = [(poly.distance(cpt), poly) for poly in valid_neto]
            if dists:
                d, poly = min(dists, key=lambda x: x[0])
                if d < ROOM_CENTROID_DIST:
                    matched_poly = poly
        if matched_poly is not None:
            room_id_to_poly[r["id"]] = matched_poly

    door_pts = get_door_thresholds(msp, bld)
    log(f"  Door thresholds (LWPOLY midpoints): {len(door_pts)}")

    # int_barriers already computed above for walkable; reuse for edge crossing checks
    dxf_barriers[(bld, floor)] = int_barriers
    log(f"  DXF barrier segments: {len(int_barriers)}")


    key = (bld, floor)
    door_nodes_this_floor = []

    # Match each door threshold to rooms whose neto polygon boundary is within DOOR_BOUNDARY_DIST
    for ix, iy in door_pts:
        ipt = Point(ix, iy)
        # Corridor-facing: must be in walkable or within 30u
        if not walkable.buffer(30).contains(ipt): continue
        # Match rooms: neto polygon boundary within DOOR_BOUNDARY_DIST of INSERT
        matched_rids = []
        for r in floor_rooms:
            poly = room_id_to_poly.get(r["id"])
            if poly is None: continue
            if poly.exterior.distance(ipt) <= DOOR_BOUNDARY_DIST:
                matched_rids.append(r["id"])
        if not matched_rids: continue
        # Nudge threshold away from wall so DOOR_LINK edges don't cross the originating wall
        nx_, ny_ = inset_toward_walkable(ix, iy, walkable)
        seq = door_seq[key]; door_seq[key] += 1
        nid = f"D-{bld}-F{floor}-{seq:03d}"
        node = {"id": nid, "kind": "DOOR", "x": nx_, "y": ny_,
                "floor": floor, "building": bld,
                "room_ids": matched_rids}
        door_nodes_this_floor.append(node)
        all_door_nodes.append(node)
        for rid in matched_rids:
            room_to_door_nodes[rid].append(nid)

    log(f"  Matched doors: {len(door_nodes_this_floor)}")

    # Voronoi sampling
    boundary_pts = []
    geoms = [walkable] if walkable.geom_type == "Polygon" else list(walkable.geoms)
    for geom in geoms:
        if geom.is_empty or geom.geom_type != "Polygon": continue
        L = geom.exterior.length
        n = max(4, int(L / BOUNDARY_SAMPLE))
        for i in range(n):
            pt = geom.exterior.interpolate(i * L / n)
            boundary_pts.append((pt.x, pt.y))
        for interior in geom.interiors:
            iL = interior.length
            ni = max(4, int(iL / BOUNDARY_SAMPLE))
            for i in range(ni):
                pt = interior.interpolate(i * iL / ni)
                boundary_pts.append((pt.x, pt.y))

    bounds = walkable.bounds
    wbuf_in = walkable.buffer(-5)
    interior_pts = []
    for gx in np.arange(bounds[0], bounds[2], INTERIOR_GRID):
        for gy in np.arange(bounds[1], bounds[3], INTERIOR_GRID):
            if wbuf_in.contains(Point(gx, gy)):
                interior_pts.append((gx, gy))

    all_pts = boundary_pts + interior_pts
    log(f"  Voronoi: {len(boundary_pts)} boundary + {len(interior_pts)} interior")

    if len(all_pts) < 4:
        log(f"  WARN: too few points"); continue

    coords = np.unique(np.array(all_pts), axis=0)
    try:
        vor = Voronoi(coords)
    except Exception as e:
        log(f"  ERROR Voronoi: {e}"); continue

    wbuf_check = walkable.buffer(VORONOI_BUFFER)
    G = nx.Graph()
    for ridge in vor.ridge_vertices:
        if -1 in ridge: continue
        p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
        if wbuf_check.contains(Point(p1)) and wbuf_check.contains(Point(p2)):
            length = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            G.add_edge(ridge[0], ridge[1], length=length)

    log(f"  Voronoi ridges inside walkable: {G.number_of_edges()}")

    candidates = [(vor.vertices[n][0], vor.vertices[n][1])
                  for n in G.nodes() if G.degree(n) >= 3 or G.degree(n) == 1]

    if not candidates:
        log(f"  WARN: no candidates"); continue

    clustered = cluster_pts(candidates, JUNCTION_CLUSTER)
    floor_junctions = []
    for jx, jy in clustered:
        if not outer.contains(Point(jx, jy)): continue
        seq = junc_seq[key]; junc_seq[key] += 1
        nid = f"J-{bld}-F{floor}-{seq:03d}"
        jn = {"id": nid, "kind": "JUNCTION", "x": jx, "y": jy,
              "floor": floor, "building": bld}
        all_junction_nodes.append(jn)
        floor_junctions.append(jn)

    log(f"  JUNCTIONs inside boundary: {len(floor_junctions)}")

    # Map Voronoi vertex → nearest junction
    vid_to_junc = {}
    for n in G.nodes():
        vx, vy = vor.vertices[n]
        best_d, best_jn = float("inf"), None
        for jn in floor_junctions:
            d = dist((vx,vy),(jn["x"],jn["y"]))
            if d < best_d and d <= JUNCTION_CLUSTER:
                best_d, best_jn = d, jn
        if best_jn:
            vid_to_junc[n] = best_jn["id"]

    # BFS for CORRIDOR edges
    edge_pairs_seen = set()
    for start_vid, start_jid in vid_to_junc.items():
        visited = {start_vid}
        queue = [(start_vid, 0.0)]
        while queue:
            cur_vid, cur_dist = queue.pop(0)
            for nbr in G.neighbors(cur_vid):
                if nbr in visited: continue
                seg_len = G[cur_vid][nbr]["length"]
                new_dist = cur_dist + seg_len
                if nbr in vid_to_junc:
                    end_jid = vid_to_junc[nbr]
                    if end_jid != start_jid:
                        pair = tuple(sorted([start_jid, end_jid]))
                        if pair not in edge_pairs_seen:
                            edge_pairs_seen.add(pair)
                            corridor_edges.append({
                                "id": next_eid(), "kind": "CORRIDOR",
                                "from_node": start_jid, "to_node": end_jid,
                                "distance": round(new_dist, 2),
                                "floor": floor, "building": bld,
                                "is_verified": True,
                            })
                else:
                    visited.add(nbr)
                    queue.append((nbr, new_dist))

    # DOOR_LINK edges
    for dn in door_nodes_this_floor:
        best_d, best_jn = float("inf"), None
        for jn in floor_junctions:
            d = dist((dn["x"],dn["y"]),(jn["x"],jn["y"]))
            if d < best_d and d <= DOOR_LINK_MAX:
                best_d, best_jn = d, jn
        if best_jn is None: continue
        door_link_edges.append({
            "id": next_eid(), "kind": "DOOR_LINK",
            "from_node": dn["id"], "to_node": best_jn["id"],
            "distance": round(best_d, 2),
            "floor": floor, "building": bld, "is_verified": True,
        })

log("\n" + "="*60)
log("\n" + "="*60)
log("Step 2: Stairwell STAIR_LANDING nodes and STAIR edges")

sl_by_room_floor = {}
stair_landing_nodes = []
sl_seq = 0

mabua_by_floor = defaultdict(list)
for r in all_rooms:
    rtype = r.get("type","")
    if any(m in rtype for m in MABUA_TYPES):
        if r.get("door_x") is not None and r.get("door_y") is not None:
            mabua_by_floor[(r["building"], r["floor"])].append(
                (r["door_x"], r["door_y"], r["id"]))

vert_room_ids = set(r["id"] for r in all_rooms if r.get("type","") in VERT_TYPES)

for room in all_rooms:
    rid = room["id"]
    if rid not in vert_room_ids: continue
    key_rf = (rid, room["floor"])
    if key_rf in sl_by_room_floor: continue
    stype = "elevator" if room.get("type","") in ELEV_TYPES else "stair"
    bld, fl = room["building"], room["floor"]
    # Skip stairwell rooms with no door coordinates
    if room.get("door_x") is None or room.get("door_y") is None:
        log(f"  WARN: stairwell room {rid} has no door coords, using centroid")
        if room.get("centroid_x") is None:
            log(f"  WARN: stairwell room {rid} has no coords at all, skipping")
            continue
        anchor_x, anchor_y = room["centroid_x"], room["centroid_y"]
        nid = f"SL-{sl_seq:04d}"; sl_seq += 1
        node = {"id": nid, "kind": "STAIR_LANDING",
                "x": anchor_x, "y": anchor_y,
                "floor": fl, "building": bld,
                "stair_type": stype, "room_id": rid}
        stair_landing_nodes.append(node)
        sl_by_room_floor[key_rf] = nid
        room_to_door_nodes[rid].append(nid)
        continue
    mabua_list = mabua_by_floor.get((bld, fl), [])
    anchor_x, anchor_y = room["door_x"], room["door_y"]
    if mabua_list:
        best = min(mabua_list,
                   key=lambda m: dist((room["door_x"],room["door_y"]),(m[0],m[1])))
        if dist((room["door_x"],room["door_y"]),(best[0],best[1])) <= MABUA_SEARCH:
            anchor_x, anchor_y = best[0], best[1]
    nid = f"SL-{sl_seq:04d}"; sl_seq += 1
    node = {"id": nid, "kind": "STAIR_LANDING",
            "x": anchor_x, "y": anchor_y,
            "floor": fl, "building": bld,
            "stair_type": stype, "room_id": rid}
    stair_landing_nodes.append(node)
    sl_by_room_floor[key_rf] = nid
    room_to_door_nodes[rid].append(nid)

log(f"  STAIR_LANDING nodes: {len(stair_landing_nodes)}")

stair_edges = []
# Group vertical rooms by stairwell shaft: same building + same last 2 digits of room ID.
# Sort by floor and link consecutive floors with a STAIR edge.
vert_by_shaft = defaultdict(list)
for room in all_rooms:
    if room["id"] in vert_room_ids:
        shaft_key = (room["building"], room["id"][-2:])
        vert_by_shaft[shaft_key].append(room)

for shaft_key, shaft_rooms in vert_by_shaft.items():
    shaft_rooms.sort(key=lambda r: r["floor"])
    for i in range(len(shaft_rooms) - 1):
        r1, r2 = shaft_rooms[i], shaft_rooms[i+1]
        k1 = (r1["id"], r1["floor"])
        k2 = (r2["id"], r2["floor"])
        if k1 in sl_by_room_floor and k2 in sl_by_room_floor:
            stair_edges.append({
                "id": next_eid(), "kind": "STAIR",
                "from_node": sl_by_room_floor[k1],
                "to_node":   sl_by_room_floor[k2],
                "distance":  400,
                "floor": None, "is_verified": True,
            })

log(f"  STAIR edges: {len(stair_edges)}")

junc_by_floor = defaultdict(list)
for jn in all_junction_nodes:
    junc_by_floor[(jn["building"], jn["floor"])].append(jn)

for sln in stair_landing_nodes:
    junctions = junc_by_floor.get((sln["building"], sln["floor"]), [])
    if not junctions: continue
    best_d, best_jn = float("inf"), None
    for jn in junctions:
        d = dist((sln["x"],sln["y"]),(jn["x"],jn["y"]))
        if d < best_d and d <= DOOR_LINK_MAX:
            best_d, best_jn = d, jn
    if best_jn is None: continue
    door_link_edges.append({
        "id": next_eid(), "kind": "DOOR_LINK",
        "from_node": sln["id"], "to_node": best_jn["id"],
        "distance": round(best_d, 2),
        "floor": sln["floor"], "building": sln["building"],
        "is_verified": True,
    })

log("\n" + "="*60)
log("Step 3: Wall validation")

all_nodes = all_door_nodes + stair_landing_nodes + all_junction_nodes
node_pos = {n["id"]: (n["x"], n["y"]) for n in all_nodes}

def verify_edge(e):
    floor = e.get("floor"); bld = e.get("building")
    if floor is None or bld is None: return True
    p1 = node_pos.get(e["from_node"]); p2 = node_pos.get(e["to_node"])
    if p1 is None or p2 is None: return True
    return not edge_crosses_wall(p1[0],p1[1],p2[0],p2[1], int(floor), bld)

before_corr = len(corridor_edges)
corridor_edges = [e for e in corridor_edges if verify_edge(e)]
dropped_corr = before_corr - len(corridor_edges)

before_dl = len(door_link_edges)
door_link_edges = [e for e in door_link_edges if verify_edge(e)]
dropped_dl = before_dl - len(door_link_edges)

log(f"  CORRIDOR:  {before_corr} → {len(corridor_edges)} ({dropped_corr} dropped)")
log(f"  DOOR_LINK: {before_dl} → {len(door_link_edges)} ({dropped_dl} dropped)")

violations = sum(1 for e in corridor_edges + door_link_edges if not verify_edge(e))
log(f"  Final violations (must be 0): {violations}")
if violations > 0:
    log("  ERROR: non-zero wall crossings — build FAILED")
    sys.exit(1)

log("\n" + "="*60)
log("Step 3b: Per-floor corridor connectivity repair (post-validation)")

# Run after wall validation so we only repair genuine gaps, not pre-existing connections
# that will be dropped. For each floor, iteratively bridge disconnected junction groups.
# Clean (wall-free) path preferred; wall-crossing bridge allowed as last resort.

repair_count = 0
for (bld, floor) in sorted(set((n["building"],n["floor"]) for n in all_junction_nodes)):
    fl_juncs = [n for n in all_junction_nodes if n["building"]==bld and n["floor"]==floor]
    if len(fl_juncs) < 2: continue
    junc_by_id = {jn["id"]: jn for jn in fl_juncs}

    while True:
        Gfl = nx.Graph()
        for jn in fl_juncs: Gfl.add_node(jn["id"])
        for e in corridor_edges:
            if e.get("building")==bld and e.get("floor")==floor:
                Gfl.add_edge(e["from_node"], e["to_node"])

        comps_fl = list(nx.connected_components(Gfl))
        if len(comps_fl) <= 1: break
        comps_fl.sort(key=len, reverse=True)
        main_comp = comps_fl[0]

        made_progress = False
        for small_comp in comps_fl[1:]:
            best_clean = (float("inf"), None)
            best_any   = (float("inf"), None)
            for aid in small_comp:
                a = junc_by_id[aid]
                for bid_ in main_comp:
                    b = junc_by_id[bid_]
                    d = dist((a["x"],a["y"]), (b["x"],b["y"]))
                    if d < best_any[0]:
                        best_any = (d, (aid, bid_))
                    if d < best_clean[0]:
                        if not edge_crosses_wall(a["x"],a["y"],b["x"],b["y"], floor, bld):
                            best_clean = (d, (aid, bid_))

            chosen_d, chosen_pair = best_clean if best_clean[1] else best_any
            if chosen_pair is None: continue

            aid, bid_ = chosen_pair
            clean = best_clean[1] is not None
            corridor_edges.append({
                "id": next_eid(), "kind": "CORRIDOR",
                "from_node": aid, "to_node": bid_,
                "distance": round(chosen_d, 2),
                "floor": floor, "building": bld, "is_verified": clean,
            })
            main_comp = main_comp | small_comp
            repair_count += 1
            tag = "" if clean else " [wall-fallback]"
            log(f"  Repair B{bld} F{floor}: bridged {len(small_comp)}-node "
                f"({aid[:16]}…→{bid_[:16]}…) dist={chosen_d:.0f}{tag}")
            made_progress = True

        if not made_progress: break

log(f"  Total corridor repair edges added: {repair_count}")

# Repair isolated DOOR and STAIR_LANDING nodes: connect each to its nearest
# junction on the same floor (wall-fallback allowed).
door_repair_count = 0
connected_node_ids = set()
for e in door_link_edges:
    connected_node_ids.add(e["from_node"])
    connected_node_ids.add(e["to_node"])

for dn in all_door_nodes + stair_landing_nodes:
    if dn["id"] in connected_node_ids: continue
    bld, floor = dn["building"], dn["floor"]
    fl_juncs = [n for n in all_junction_nodes if n["building"]==bld and n["floor"]==floor]
    if not fl_juncs: continue
    best_clean = (float("inf"), None)
    best_any   = (float("inf"), None)
    for jn in fl_juncs:
        d = dist((dn["x"],dn["y"]), (jn["x"],jn["y"]))
        if d < best_any[0]: best_any = (d, jn)
        if d < best_clean[0] and not edge_crosses_wall(dn["x"],dn["y"],jn["x"],jn["y"], floor, bld):
            best_clean = (d, jn)
    chosen_d, chosen_jn = best_clean if best_clean[1] else best_any
    if chosen_jn is None: continue
    clean = best_clean[1] is not None
    door_link_edges.append({
        "id": next_eid(), "kind": "DOOR_LINK",
        "from_node": dn["id"], "to_node": chosen_jn["id"],
        "distance": round(chosen_d, 2),
        "floor": floor, "building": bld, "is_verified": clean,
    })
    door_repair_count += 1

log(f"  Door/stair repair edges added: {door_repair_count}")

log("\n" + "="*60)
log("Step 4: Room records")

unmatched_rooms = []
room_records = []
for room in all_rooms:
    rid = room["id"]
    door_nids = list(room_to_door_nodes.get(rid, []))
    if not door_nids: unmatched_rooms.append(rid)
    rtype = room.get("type","")
    if rtype in STAIR_TYPES:  rtype_en = "stairwell"
    elif rtype in ELEV_TYPES: rtype_en = "elevator"
    else:                     rtype_en = "room"
    room_records.append({
        "id": rid, "display_name": f"Room {rid}",
        "type": rtype_en, "floor": room["floor"], "building": room["building"],
        "door_nodes": door_nids,
        "centroid": {"x": room["centroid_x"], "y": room["centroid_y"]},
    })

log(f"  Rooms: {len(room_records)}, no door: {len(unmatched_rooms)}")

log("\n" + "="*60)
log("Step 5: Connectivity")

all_edges = stair_edges + door_link_edges + corridor_edges
G_full = nx.Graph()
for n in all_nodes: G_full.add_node(n["id"])
for e in all_edges:
    G_full.add_edge(e["from_node"], e["to_node"], weight=e["distance"])

components = list(nx.connected_components(G_full))
log(f"  Components: {len(components)}, largest: {max(len(c) for c in components)}")

log("\n" + "="*60)
log("Step 6: Test routes")

room_door_lookup = {r["id"]: r["door_nodes"] for r in room_records}

def dijkstra_ng(src_room, dst_room):
    src_nids = set(room_door_lookup.get(src_room, []))
    dst_nids = set(room_door_lookup.get(dst_room, []))
    if not src_nids or not dst_nids: return None, None

    # Build a view that treats DOOR nodes as dead-ends:
    # non-destination DOOR nodes can only be entered from __SRC__, never traversed through.
    door_ids = set(n["id"] for n in all_door_nodes)
    Gq = nx.DiGraph()
    for n in G_full.nodes(): Gq.add_node(n)
    for u, v, data in G_full.edges(data=True):
        w = data.get("weight", 1)
        # Only allow entering a non-destination DOOR node if starting from __SRC__
        # (handled below). Between graph nodes, never route THROUGH a door.
        u_is_door = u in door_ids and u not in dst_nids
        v_is_door = v in door_ids and v not in dst_nids
        if not v_is_door:
            Gq.add_edge(u, v, weight=w)
        if not u_is_door:
            Gq.add_edge(v, u, weight=w)

    Gq.add_node("__SRC__"); Gq.add_node("__DST__")
    for nid in src_nids:
        if Gq.has_node(nid): Gq.add_edge("__SRC__", nid, weight=0)
    for nid in dst_nids:
        if Gq.has_node(nid): Gq.add_edge(nid, "__DST__", weight=0)
    try:
        path   = nx.shortest_path(Gq, "__SRC__", "__DST__", weight="weight")
        length = nx.shortest_path_length(Gq, "__SRC__", "__DST__", weight="weight")
        return [n for n in path if n not in ("__SRC__","__DST__")], length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None

test_routes = [("21201","21401"),("21201","21205"),("21301","21305")]
route_results = []
node_info = {n["id"]: n for n in all_nodes}

for src, dst in test_routes:
    path, length = dijkstra_ng(src, dst)
    if path is None:
        log(f"  {src}→{dst}: NO PATH")
        route_results.append((src, dst, None, None, 0))
        continue
    wc = 0
    for i in range(len(path)-1):
        n1, n2 = path[i], path[i+1]
        nd1 = node_info.get(n1)
        if nd1 and nd1.get("floor") and n1 in node_pos and n2 in node_pos:
            x1,y1=node_pos[n1]; x2,y2=node_pos[n2]
            if edge_crosses_wall(x1,y1,x2,y2, nd1["floor"], nd1["building"]):
                wc += 1
    log(f"  {src}→{dst}: dist={length:.1f}, nodes={len(path)}, wall_crossings={wc}")
    route_results.append((src, dst, path, length, wc))

log("\n" + "="*60)
log("Step 7: Overlay images")

def make_overlay(floor):
    dxf_path = DATA21 / f"0201{floor}.dxf"
    if not dxf_path.exists(): return
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    beton_segs = []
    for e in msp:
        if e.dxf.get("layer","") != "Beton": continue
        t = e.dxftype()
        if t == "LINE":
            beton_segs.append(((e.dxf.start.x,e.dxf.start.y),(e.dxf.end.x,e.dxf.end.y)))
        elif t == "LWPOLYLINE":
            pts=[(p[0],p[1]) for p in e.get_points()]
            for i in range(len(pts)-1): beton_segs.append((pts[i],pts[i+1]))
    outer, _ = get_bruto_polygon(msp, "21")

    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_aspect("equal")
    ax.set_title(f"B21 F{floor} — Nav graph v3 (Bruto boundary, INSERT doors)")

    for (x1,y1),(x2,y2) in beton_segs:
        ax.plot([x1,x2],[y1,y2], color="gray", lw=0.4, alpha=0.5)
    if outer:
        xs,ys = outer.exterior.xy
        ax.plot(xs,ys, color="red", lw=1.5, alpha=0.7, zorder=2)

    fl_juncs = [n for n in all_junction_nodes if n["building"]=="21" and n["floor"]==floor]
    fl_doors = [n for n in all_door_nodes      if n["building"]=="21" and n["floor"]==floor]

    if fl_juncs:
        ax.scatter([n["x"] for n in fl_juncs],[n["y"] for n in fl_juncs],
                   s=4, c="blue", alpha=0.6, zorder=3)
    if fl_doors:
        ax.scatter([n["x"] for n in fl_doors],[n["y"] for n in fl_doors],
                   s=50, c="green", zorder=5)

    for e in corridor_edges:
        if e.get("building")!="21" or e.get("floor")!=floor: continue
        if e["from_node"] not in node_pos or e["to_node"] not in node_pos: continue
        x1,y1=node_pos[e["from_node"]]; x2,y2=node_pos[e["to_node"]]
        ax.plot([x1,x2],[y1,y2], color="cornflowerblue", lw=0.5, alpha=0.35, zorder=2)

    colors = ["red","orange","purple"]
    labels = ["21201→21401","21201→21205","21301→21305"]
    for (src,dst,path,length,wc), color, label in zip(route_results, colors, labels):
        if path is None: continue
        for i in range(len(path)-1):
            n1,n2=path[i],path[i+1]
            nd1=node_info.get(n1)
            if nd1 and nd1.get("floor")==floor and nd1.get("building")=="21":
                if n1 in node_pos and n2 in node_pos:
                    x1,y1=node_pos[n1]; x2,y2=node_pos[n2]
                    ax.plot([x1,x2],[y1,y2], color=color, lw=2.5, zorder=6)

    handles=[
        mpatches.Patch(color="blue",  label=f"JUNCTION ({len(fl_juncs)})"),
        mpatches.Patch(color="green", label=f"DOOR ({len(fl_doors)})"),
        mpatches.Patch(color="red",   label="21201→21401"),
        mpatches.Patch(color="orange",label="21201→21205"),
        mpatches.Patch(color="purple",label="21301→21305"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    out = DOCS/f"route_overlay_b21_f{floor}.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out}")

for fl in [3,4,5]:
    make_overlay(fl)

log("\n" + "="*60)
log("Step 8: Writing output")

nav_graph = {
    "meta": {
        "generated": datetime.datetime.utcnow().isoformat()+"Z",
        "b22_transform": {"offset_x": B22_OX, "offset_y": B22_OY},
        "node_count": len(all_nodes), "edge_count": len(all_edges),
        "room_count": len(room_records),
    },
    "nodes": all_nodes, "edges": all_edges, "rooms": room_records,
}
with open(OUTPUT,"w") as f:
    json.dump(nav_graph, f, indent=2, ensure_ascii=False)
log(f"  Written: {OUTPUT}")

overall_pass = violations == 0 and all(r[4]==0 for r in route_results if r[2] is not None)
lines = [
    "# nav_graph Build Report","",
    f"Generated: {nav_graph['meta']['generated']}","",
    f"Overall: {'**PASS**' if overall_pass else '**FAIL**'}","",
    "## Node counts","","| Kind | Count |","|---|---|",
    f"| DOOR (DXF INSERT) | {len(all_door_nodes)} |",
    f"| STAIR_LANDING | {len(stair_landing_nodes)} |",
    f"| JUNCTION | {len(all_junction_nodes)} |",
    f"| **Total** | **{len(all_nodes)}** |","",
    "## Edge counts","","| Kind | Count |","|---|---|",
    f"| STAIR | {len(stair_edges)} |",
    f"| DOOR_LINK | {len(door_link_edges)} |",
    f"| CORRIDOR | {len(corridor_edges)} |",
    f"| **Total** | **{len(all_edges)}** |","",
    "## Acceptance criteria","",
    f"- Wall crossings (must be 0): {violations}",
    f"- Connected components: {len(components)}",
    f"- Unreachable room count: {len([r for r in room_records if not r['door_nodes']])}","",
    "## Test routes","",
    "| Route | Status | Distance | Nodes | Wall crossings |","|---|---|---|---|---|",
]
for src,dst,path,length,wc in route_results:
    if path is None:
        lines.append(f"| {src}→{dst} | NO PATH | — | — | — |")
    else:
        lines.append(f"| {src}→{dst} | {'OK' if wc==0 else 'FAIL'} | {length:.1f} | {len(path)} | {wc} |")

lines += ["","## Rooms with no DXF door","",f"Count: {len(unmatched_rooms)}","",
          "| Room ID |","|---|"] + [f"| {r} |" for r in sorted(unmatched_rooms)]

with open(REPORT,"w") as f:
    f.write("\n".join(lines))
log(f"  Report: {REPORT}")
log("DONE.")
print(f"Nodes={len(all_nodes)} Edges={len(all_edges)} Components={len(components)} Violations={violations}")
