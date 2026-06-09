"""
extract_building.py
-------------------
Reusable DXF extraction script for any building floor plan.
Produces a rooms list and walkable polygon for each floor.

Usage:
    python3 extract_building.py

Configure the BUILDING_CONFIG dict below for a new building.

Verified on B21 (floors 1-6) and B22 (floors 2-6) of the Ruppin campus.
See docs/dxf_extraction_algorithm.md for the full algorithm description.
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Configuration — edit this for a new building
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

BUILDING_CONFIG = {
    # Add entries for each new building here.
    # Keys: building prefix (str), floors (list), dxf_dir (Path),
    #       neto_layer, bruto_layer, offset_x, offset_y
    "21": {
        "floors": [1, 2, 3, 4, 5, 6],
        "dxf_files": {
            1: ROOT / "data/21/02011.dxf",
            2: ROOT / "data/21/02012.dxf",
            3: ROOT / "data/21/02013.dxf",
            4: ROOT / "data/21/02014.dxf",
            5: ROOT / "data/21/02015.dxf",
            6: ROOT / "data/21/02016.dxf",
        },
        "neto_layers":  {"0201Shetah-Neto"},
        "bruto_layers": {"0201Shetah-Bruto"},
        "offset_x": 0.0,
        "offset_y": 0.0,
    },
    "22": {
        "floors": [2, 3, 4, 5, 6],
        "dxf_files": {
            2: ROOT / "data/22/02022.dxf",
            3: ROOT / "data/22/02023.dxf",
            4: ROOT / "data/22/02024.dxf",
            5: ROOT / "data/22/02025.dxf",
            6: ROOT / "data/22/02026.dxf",
        },
        # F6 has a typo layer name — include both spellings
        "neto_layers":  {"0202Shetah-Neto", "020 2Shetah-Neto"},
        "bruto_layers": {"0202Shetah-Bruto", "020 2Shetah-Bruto"},
        "offset_x": -2479.0,   # from docs/b22_transform.json
        "offset_y": -104.01,
    },
    # Template for a new building:
    # "23": {
    #     "floors": [1, 2, 3],
    #     "dxf_files": { 1: ROOT/"data/23/02031.dxf", ... },
    #     "neto_layers":  {"0203Shetah-Neto"},
    #     "bruto_layers": {"0203Shetah-Bruto"},
    #     "offset_x": 0.0,
    #     "offset_y": 0.0,
    # },
}

# Corridor / circulation room types — polys containing these are WALKABLE
CORRIDOR_ROOM_TYPES = {
    "פויאה", "פויה", "מבוא", "מעבר", "פרוזדור",
    "מרפסת", "חדר מדרגות", "מדרגות", "פיר מעלית", "פיר",
}

# Tuning constants
NEAR_POLY_MAX       = 600     # max dist (units) label→poly for pass-2 snap
ROOM_CENTROID_DIST  = 500     # max dist for pass-2 room-centric fallback
WALL_BUFFER         = 20      # buffer around Beton/Window walls
MABAT_ARC_MIN_R     = 50      # min arc radius to treat as solid obstacle
CENTER_SNAP         = 30      # arc center snap distance (staircase grouping)
STAIR_MERGE_DIST    = 600     # max dist to merge staircase arc groups
DOOR_MAX_DIST       = 300     # max dist centroid→door for assignment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _decode(raw: str) -> str:
    """Decode Windows-1255 text that arrived via latin-1 transit."""
    try:
        return raw.encode("latin-1", "replace").decode("windows-1255", "replace").strip()
    except Exception:
        return raw.strip()


def _transform(x, y, cfg):
    return x + cfg["offset_x"], y + cfg["offset_y"]


def _load_neto_polys(msp, cfg):
    """Load Shetah-Neto LWPOLYLINE polygons → list of Shapely Polygons."""
    layers = cfg["neto_layers"]
    polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in layers or e.dxftype() != "LWPOLYLINE":
            continue
        pts = [_transform(p[0], p[1], cfg) for p in e.get_points()]
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon":
                for sub in poly.geoms:
                    if sub.area > 0:
                        polys.append(sub)
            elif poly.area > 100:
                polys.append(poly)
        except Exception:
            pass
    return polys


def _load_bruto(msp, cfg):
    """Load building boundary (bruto) → unified Shapely geometry."""
    layers = cfg["bruto_layers"]
    polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in layers or e.dxftype() != "LWPOLYLINE":
            continue
        pts = [_transform(p[0], p[1], cfg) for p in e.get_points()]
        if len(pts) >= 3:
            try:
                p = make_valid(Polygon(pts))
                if not p.is_empty:
                    polys.append(p)
            except Exception:
                pass
    return unary_union(polys) if polys else None


def _load_insert_blocks(msp, cfg, bld):
    """
    Load INSERT blocks from TEXT layer.
    Returns list of dicts: {x, y, room_id, type, suffix, attribs}
    """
    blocks = []
    for e in msp:
        if e.dxf.get("layer", "") != "TEXT" or e.dxftype() != "INSERT":
            continue
        x, y = _transform(e.dxf.insert.x, e.dxf.insert.y, cfg)
        try:
            attribs = [_decode(a.dxf.get("text", "")) for a in e.attribs]
        except Exception:
            attribs = []

        # Find 5-digit room id matching this building prefix
        room_id = next(
            (a.strip().rstrip(".") for a in attribs
             if len(a.strip().rstrip(".")) == 5
             and a.strip().rstrip(".").isdigit()
             and a.strip().rstrip(".").startswith(bld)),
            None
        )
        if room_id is None:
            continue

        # Find suffix (single letter after room_id in attribs)
        suffix = ""
        found_id = False
        for a in attribs:
            if found_id:
                suffix = a.strip()
                break
            if a.strip().rstrip(".") == room_id:
                found_id = True

        # Hebrew type: first non-ASCII non-digit attrib len > 2
        room_type = ""
        for a in attribs:
            a2 = a.strip()
            if len(a2) > 2 and not a2.replace(".", "").isdigit() and not a2.isascii():
                room_type = a2
                break

        blocks.append({
            "x": x, "y": y,
            "room_id": room_id,
            "type": room_type,
            "suffix": suffix,
            "attribs": attribs,
        })
    return blocks


def _load_doors(msp, cfg):
    """
    Load door threshold positions from Door layer LWPOLYLINEs.
    Threshold = midpoint of first two vertices.
    """
    doors = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door" or e.dxftype() != "LWPOLYLINE":
            continue
        pts = list(e.get_points())
        if len(pts) < 2:
            continue
        mx = (pts[0][0] + pts[1][0]) / 2
        my = (pts[0][1] + pts[1][1]) / 2
        doors.append(_transform(mx, my, cfg))
    return doors


def _arc_to_segments(cx, cy, r, a0_deg, a1_deg, seg_len=20):
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    n = max(3, int(r * abs(a1 - a0) / seg_len))
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
            cy + r * math.sin(a0 + (a1 - a0) * i / n))
           for i in range(n + 1)]
    return [LineString([pts[i], pts[i + 1]]) for i in range(len(pts) - 1)]


def _load_walls_and_waypoints(msp, cfg):
    """
    Returns (wall_buffer_polygon, waypoints_list).
    Walls: Beton + Window lines, buffered WALL_BUFFER units.
    Waypoints: one per staircase unit (Mabat ARC groups).
    """
    wall_lines = []
    mabat_solid = []
    arc_by_center = defaultdict(list)

    for e in msp:
        layer = e.dxf.get("layer", "")
        t = e.dxftype()

        # Beton + Window structural walls
        if layer in ("Beton", "Window"):
            if t == "LINE":
                sx, sy = _transform(e.dxf.start.x, e.dxf.start.y, cfg)
                ex_, ey = _transform(e.dxf.end.x, e.dxf.end.y, cfg)
                wall_lines.append(LineString([(sx, sy), (ex_, ey)]))
            elif t == "LWPOLYLINE":
                pts = [_transform(p[0], p[1], cfg) for p in e.get_points()]
                if e.closed:
                    pts = pts + [pts[0]]
                for i in range(len(pts) - 1):
                    wall_lines.append(LineString([pts[i], pts[i + 1]]))

        # Mabat: arcs = solid staircase walls + waypoint source
        if layer == "Mabat":
            if t == "ARC":
                cx, cy = _transform(e.dxf.center.x, e.dxf.center.y, cfg)
                r = e.dxf.radius
                if r >= MABAT_ARC_MIN_R:
                    mabat_solid.extend(
                        _arc_to_segments(cx, cy, r,
                                         e.dxf.start_angle, e.dxf.end_angle)
                    )
                    key = (round(cx / CENTER_SNAP) * CENTER_SNAP,
                           round(cy / CENTER_SNAP) * CENTER_SNAP)
                    arc_by_center[key].append((cx, cy))
            elif t == "CIRCLE":
                cx, cy = _transform(e.dxf.center.x, e.dxf.center.y, cfg)
                r = e.dxf.radius
                if r >= MABAT_ARC_MIN_R:
                    mabat_solid.extend(
                        _arc_to_segments(cx, cy, r, 0, 360)
                    )

    # Wall buffer
    all_lines = wall_lines + mabat_solid
    wall_buf = None
    if all_lines:
        try:
            wall_buf = make_valid(
                unary_union([ln.buffer(WALL_BUFFER) for ln in all_lines])
            )
        except Exception:
            pass

    # Staircase waypoints: merge arc groups within STAIR_MERGE_DIST
    groups = []
    for key, centers in arc_by_center.items():
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        merged = False
        for g in groups:
            if _dist((cx_avg, cy_avg), g["center"]) < STAIR_MERGE_DIST:
                g["points"].extend(centers)
                n = len(g["points"])
                g["center"] = (sum(p[0] for p in g["points"]) / n,
                               sum(p[1] for p in g["points"]) / n)
                merged = True
                break
        if not merged:
            groups.append({"center": (cx_avg, cy_avg), "points": centers})

    waypoints = [g["center"] for g in groups]

    return wall_buf, waypoints


# ---------------------------------------------------------------------------
# Core extraction: one floor
# ---------------------------------------------------------------------------

def extract_floor(bld: str, floor: int, graph_nodes: dict = None):
    """
    Extract all data for one floor.

    Returns dict with keys:
        rooms        — list of room dicts
        walkable     — Shapely polygon (walkable area)
        dest_polys   — list of subtracted room polygons
        waypoints    — list of (x, y) staircase waypoints
        doors        — list of (x, y) door thresholds
    """
    cfg = BUILDING_CONFIG[bld]
    dxf_path = cfg["dxf_files"].get(floor)
    if dxf_path is None or not Path(dxf_path).exists():
        print(f"  B{bld} F{floor}: DXF not found, skipping")
        return None

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    neto_polys    = _load_neto_polys(msp, cfg)
    bruto         = _load_bruto(msp, cfg)
    insert_blocks = _load_insert_blocks(msp, cfg, bld)
    doors         = _load_doors(msp, cfg)
    wall_buf, waypoints = _load_walls_and_waypoints(msp, cfg)

    # ------------------------------------------------------------------
    # Room extraction: match INSERT blocks to neto polygons
    # ------------------------------------------------------------------
    room_data = {}   # room_id -> {type, polys}
    claimed   = set()

    # Pass 1: INSERT inside a neto poly
    for block in insert_blocks:
        rid, rtype = block["room_id"], block["type"]
        pt = Point(block["x"], block["y"])
        for i, poly in enumerate(neto_polys):
            if i in claimed:
                continue
            if poly.contains(pt):
                if rid not in room_data:
                    room_data[rid] = {"type": rtype, "polys": []}
                room_data[rid]["polys"].append(poly)
                claimed.add(i)
                break

    # Pass 2: INSERT outside all polys → snap to nearest unclaimed poly
    for block in insert_blocks:
        rid, rtype = block["room_id"], block["type"]
        if rid in room_data and room_data[rid]["polys"]:
            continue
        pt = Point(block["x"], block["y"])
        # Skip if inside an unclaimed poly (would have been caught in pass 1)
        if any(neto_polys[i].contains(pt)
               for i in range(len(neto_polys)) if i not in claimed):
            continue
        best_i, best_d = None, float("inf")
        for i, poly in enumerate(neto_polys):
            if i in claimed:
                continue
            d = poly.distance(pt)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= NEAR_POLY_MAX:
            if rid not in room_data:
                room_data[rid] = {"type": rtype, "polys": []}
            room_data[rid]["polys"].append(neto_polys[best_i])
            claimed.add(best_i)

    # Build room records: union sub-polys per room_id
    rooms = []
    for rid, data in sorted(room_data.items()):
        if not data["polys"]:
            continue
        union  = unary_union(data["polys"])
        cx, cy = union.centroid.x, union.centroid.y
        area   = union.area

        # Type: use DXF if available, else fall back to graph.json
        rtype = data["type"]
        if not rtype and graph_nodes and rid in graph_nodes:
            rtype = graph_nodes[rid].get("type", "")

        # Nearest door
        door = None
        if doors:
            nearest_d = min(doors, key=lambda d: _dist((cx, cy), d))
            if _dist((cx, cy), nearest_d) <= DOOR_MAX_DIST:
                door = nearest_d

        rooms.append({
            "id":         rid,
            "building":   bld,
            "floor":      floor,
            "type":       rtype,
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "door_x":     round(door[0], 1) if door else None,
            "door_y":     round(door[1], 1) if door else None,
            "poly_area":  round(area, 1),
        })

    # ------------------------------------------------------------------
    # Walkable area: bruto − dest_rooms − walls
    # ------------------------------------------------------------------
    walkable, dest_polys = bruto, []

    if bruto is not None and neto_polys:
        # Classify each neto poly by nearest room centroid
        # Largest poly always = main corridor (protected)
        corr_homes = set()
        corr_homes.add(max(range(len(neto_polys)),
                           key=lambda i: neto_polys[i].area))

        floor_rooms = rooms  # use newly extracted rooms
        candidate_rooms = [r for r in floor_rooms if r["type"] != "מחסן"]

        for i, poly in enumerate(neto_polys):
            if i in corr_homes:
                continue
            if not candidate_rooms:
                dest_polys.append(poly)
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            nearest = min(candidate_rooms,
                          key=lambda r: _dist((cx, cy),
                                              (r["centroid_x"], r["centroid_y"])))
            if nearest["type"] in CORRIDOR_ROOM_TYPES:
                corr_homes.add(i)
            else:
                dest_polys.append(poly)

        if dest_polys:
            rooms_union = unary_union(dest_polys)
            walkable = make_valid(bruto.difference(rooms_union))

        if wall_buf is not None and walkable is not None:
            walkable = make_valid(walkable.difference(wall_buf))

    pct = (walkable.area / bruto.area * 100) if (bruto and not bruto.is_empty) else 0
    print(f"  B{bld} F{floor}: {len(rooms)} rooms, {len(dest_polys)} dest polys, "
          f"{len(waypoints)} waypoints, walkable={pct:.1f}%")

    return {
        "rooms":      rooms,
        "walkable":   walkable,
        "dest_polys": dest_polys,
        "waypoints":  waypoints,
        "doors":      doors,
    }


# ---------------------------------------------------------------------------
# Rendering: save a PNG overlay for visual verification
# ---------------------------------------------------------------------------

def render_floor(bld, floor, result, out_path):
    """Save a verification PNG for one floor."""
    if result is None:
        return
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f"B{bld} F{floor} — Walkable | dest rooms | doors | waypoints")

    walkable = result["walkable"]
    if walkable is not None:
        geoms = list(walkable.geoms) if walkable.geom_type == "MultiPolygon" \
                else [walkable]
        for gm in geoms:
            if gm.geom_type == "Polygon":
                x, y = gm.exterior.xy
                ax.fill(x, y, alpha=0.25, color="steelblue")

    for poly in result["dest_polys"]:
        if hasattr(poly, "exterior"):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, color="salmon")

    for dx, dy in result["doors"]:
        ax.scatter(dx, dy, c="green", s=20, zorder=6)

    for wx, wy in result["waypoints"]:
        ax.scatter(wx, wy, c="cyan", s=300, marker="*", zorder=8)

    for room in result["rooms"]:
        ax.scatter(room["centroid_x"], room["centroid_y"],
                   c="red", s=40, zorder=7)
        ax.annotate(room["id"],
                    (room["centroid_x"], room["centroid_y"]),
                    fontsize=5, ha="center", va="top", color="red")

    legend = [
        mpatches.Patch(color="steelblue", alpha=0.4, label="Walkable"),
        mpatches.Patch(color="salmon",    alpha=0.4, label="Dest rooms"),
        plt.Line2D([0], [0], marker="o",   color="w",
                   markerfacecolor="green", markersize=6, label="Doors"),
        plt.Line2D([0], [0], marker="*",   color="cyan",
                   markersize=10, label="Stair waypoints"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load graph.json as type fallback (needed for B21 Hebrew encoding)
    graph_nodes = {}
    graph_path = ROOT / "backend" / "graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            g = json.load(f)
        for n in g["nodes"]:
            if n["id"] not in graph_nodes or \
               (not graph_nodes[n["id"]].get("type") and n.get("type")):
                graph_nodes[n["id"]] = n

    all_rooms = []
    docs_dir  = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    for bld, cfg in sorted(BUILDING_CONFIG.items()):
        for floor in cfg["floors"]:
            print(f"B{bld} F{floor}:")
            result = extract_floor(bld, floor, graph_nodes)
            if result is None:
                continue
            all_rooms.extend(result["rooms"])
            render_floor(bld, floor, result,
                         docs_dir / f"extract_b{bld}_f{floor}.png")

    # Add corridor rooms not found in DXF (no neto poly)
    existing = {r["id"] for r in all_rooms}
    added = 0
    for n in g["nodes"]:
        if n["id"] not in existing and n.get("type", "") in CORRIDOR_ROOM_TYPES:
            all_rooms.append({
                "id": n["id"], "building": n["building"], "floor": n["floor"],
                "type": n["type"],
                "centroid_x": round(n.get("x", 0), 1),
                "centroid_y": round(n.get("y", 0), 1),
                "door_x": n.get("door_x"), "door_y": n.get("door_y"),
                "poly_area": None,
            })
            existing.add(n["id"])
            added += 1

    out = ROOT / "backend" / "rooms.json"
    with open(out, "w") as f:
        json.dump(all_rooms, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_rooms)} rooms written to {out}")
    print(f"Corridor rooms added from graph.json: {added}")
    print("Overlay PNGs saved to docs/extract_b*_f*.png")


if __name__ == "__main__":
    main()
