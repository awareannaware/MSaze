"""
build_rooms_json.py
-------------------
Build backend/rooms.json purely from DXF source files.
No dependency on graph.json.

Output format (list of room objects):
  {
    "id":         "22602",
    "building":   "22",
    "floor":      6,
    "type":       "חדר חוקרים",
    "centroid_x": -563.1,
    "centroid_y": 2273.4,
    "door_x":     -560.0,    # from Door layer; null if none found
    "door_y":     2100.0,
    "poly_area":  195000.0   # total area (union of sub-polys)
  }

Multi-poly rooms (e.g. 22602A..M) are merged into one entry using
the union centroid of all their polygons.

Coordinate frames:
  B21 DXF → global directly
  B22 DXF → global via offset from docs/b22_transform.json
"""

import json
import math
from pathlib import Path

import ezdxf
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

ROOT   = Path(__file__).resolve().parent.parent
DATA21 = ROOT / "data" / "21"
DATA22 = ROOT / "data" / "22"

with open(ROOT / "docs" / "b22_transform.json") as f:
    _t = json.load(f)
B22_OX, B22_OY = _t["offset_x"], _t["offset_y"]


def _b22(x, y):
    return x + B22_OX, y + B22_OY


DXF_FILES = {
    ("21", 1): DATA21 / "02011.dxf",
    ("21", 2): DATA21 / "02012.dxf",
    ("21", 3): DATA21 / "02013.dxf",
    ("21", 4): DATA21 / "02014.dxf",
    ("21", 5): DATA21 / "02015.dxf",
    ("21", 6): DATA21 / "02016.dxf",
    ("22", 2): DATA22 / "02022.dxf",
    ("22", 3): DATA22 / "02023.dxf",
    ("22", 4): DATA22 / "02024.dxf",
    ("22", 5): DATA22 / "02025.dxf",
    ("22", 6): DATA22 / "02026.dxf",
}

NETO_LAYERS = {
    "21": {"0201Shetah-Neto"},
    "22": {"0202Shetah-Neto", "020 2Shetah-Neto"},  # F6 has space typo
}


def _decode(raw: str) -> str:
    try:
        return raw.encode("latin-1", "replace").decode("windows-1255", "replace").strip()
    except Exception:
        return raw.strip()


def _load_neto_polys(msp, bld):
    """Return list of Shapely Polygons from Shetah-Neto layer."""
    layers = NETO_LAYERS[bld]
    polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in layers:
            continue
        if e.dxftype() != "LWPOLYLINE":
            continue
        pts = [(p[0], p[1]) for p in e.get_points()]
        if bld == "22":
            pts = [_b22(x, y) for x, y in pts]
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon":
                for sub in poly.geoms:
                    if not sub.is_empty and sub.area > 0:
                        polys.append(sub)
            elif not poly.is_empty and poly.area > 100:
                polys.append(poly)
        except Exception:
            pass
    return polys


def _collect_insert_blocks(msp, bld):
    """
    Collect INSERT blocks from TEXT layer.
    Returns list of dicts with all attrib values and insert position.
    """
    blocks = []
    for e in msp:
        if e.dxf.get("layer", "") != "TEXT" or e.dxftype() != "INSERT":
            continue
        ix, iy = e.dxf.insert.x, e.dxf.insert.y
        if bld == "22":
            ix, iy = _b22(ix, iy)
        try:
            attribs = [_decode(a.dxf.get("text", "")) for a in e.attribs]
        except Exception:
            attribs = []
        blocks.append({"x": ix, "y": iy, "attribs": attribs})
    return blocks


def _collect_doors(msp, bld):
    """
    Return list of (x, y) door threshold positions from Door layer LWPOLYLINEs.
    Threshold = midpoint of first two vertices.
    """
    doors = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door" or e.dxftype() != "LWPOLYLINE":
            continue
        pts = list(e.get_points())
        if len(pts) < 2:
            continue
        x = (pts[0][0] + pts[1][0]) / 2
        y = (pts[0][1] + pts[1][1]) / 2
        if bld == "22":
            x, y = _b22(x, y)
        doors.append((x, y))
    return doors


def _nearest_door(cx, cy, doors, max_dist=300):
    """Return (door_x, door_y) of the nearest door within max_dist, else None."""
    best, best_d = None, float("inf")
    for dx, dy in doors:
        d = math.dist((cx, cy), (dx, dy))
        if d < best_d:
            best_d, best = d, (dx, dy)
    if best and best_d <= max_dist:
        return best
    return None


NEAR_POLY_MAX = 600  # units — max distance from INSERT to nearest poly in pass 2


def _extract_room_id_and_type(attribs, bld):
    """
    From an INSERT block's attrib list return (room_id, room_type) or (None, None).
    Filters to only this building's prefix.
    The attrib format is typically: [room_code, building_code, floor, room_id, suffix, type, ...]
    """
    room_id = None
    for a in attribs:
        c = a.strip().rstrip(".")
        if len(c) == 5 and c.isdigit() and c.startswith(bld):
            room_id = c
            break
    if room_id is None:
        return None, None

    room_type = ""
    for a in attribs:
        a2 = a.strip()
        if len(a2) > 2 and not a2.replace(".", "").isdigit() and not a2.isascii():
            room_type = a2
            break
    return room_id, room_type


def process_floor(bld, floor):
    """
    Process one floor.  Returns list of room dicts.

    Pass 1: INSERT block is inside a neto poly → use that poly.
    Pass 2: INSERT block is outside all polys → snap to nearest unclaimed poly
            within NEAR_POLY_MAX units.
    Multi-poly rooms (sub-rooms A/B/C…) are unioned into one centroid.
    """
    dxf_path = DXF_FILES.get((bld, floor))
    if dxf_path is None or not dxf_path.exists():
        print(f"  B{bld} F{floor}: DXF not found, skipping")
        return []

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    neto_polys    = _load_neto_polys(msp, bld)
    insert_blocks = _collect_insert_blocks(msp, bld)
    doors         = _collect_doors(msp, bld)

    # room_id -> {type, polys:[Polygon], label_pos:(x,y)}
    room_data   = {}
    claimed     = set()   # indices of polys already assigned

    # ------------------------------------------------------------------
    # Pass 1: INSERT inside a poly
    # Each poly can belong to at most one room (sub-rooms A/B/C share same
    # room_id so they all accumulate under the same entry).
    # ------------------------------------------------------------------
    for block in insert_blocks:
        room_id, room_type = _extract_room_id_and_type(block["attribs"], bld)
        if room_id is None:
            continue
        pt = Point(block["x"], block["y"])
        for i, poly in enumerate(neto_polys):
            if not poly.contains(pt):
                continue
            # Allow the same room_id to claim multiple polys (sub-rooms)
            # but don't let two different room_ids claim the same poly.
            if i in claimed:
                break
            if room_id not in room_data:
                room_data[room_id] = {"type": room_type, "polys": [], "label_pos": (block["x"], block["y"])}
            room_data[room_id]["polys"].append(poly)
            claimed.add(i)
            break

    # ------------------------------------------------------------------
    # Pass 2: INSERT outside all polys → nearest unclaimed poly
    # ------------------------------------------------------------------
    for block in insert_blocks:
        room_id, room_type = _extract_room_id_and_type(block["attribs"], bld)
        if room_id is None:
            continue
        # Already matched in pass 1?  Skip.
        if room_id in room_data and room_data[room_id]["polys"]:
            continue
        pt = Point(block["x"], block["y"])
        # Skip only if inside an unclaimed poly (it was handled in pass 1)
        if any(poly.contains(pt) for i, poly in enumerate(neto_polys) if i not in claimed):
            continue

        # Find nearest unclaimed poly
        best_i, best_d = None, float("inf")
        for i, poly in enumerate(neto_polys):
            if i in claimed:
                continue
            d = poly.distance(pt)
            if d < best_d:
                best_d, best_i = d, i

        if best_i is not None and best_d <= NEAR_POLY_MAX:
            if room_id not in room_data:
                room_data[room_id] = {"type": room_type, "polys": [], "label_pos": (block["x"], block["y"])}
            room_data[room_id]["polys"].append(neto_polys[best_i])
            claimed.add(best_i)

    # ------------------------------------------------------------------
    # Build output records — union all sub-polys per room
    # ------------------------------------------------------------------
    rooms = []
    for room_id, data in sorted(room_data.items()):
        polys = data["polys"]
        if not polys:
            continue
        union = unary_union(polys)
        cx, cy = union.centroid.x, union.centroid.y
        area   = union.area

        door = _nearest_door(cx, cy, doors)

        rooms.append({
            "id":         room_id,
            "building":   bld,
            "floor":      floor,
            "type":       data["type"],
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "door_x":     round(door[0], 1) if door else None,
            "door_y":     round(door[1], 1) if door else None,
            "poly_area":  round(area, 1),
        })

    print(f"  B{bld} F{floor}: {len(rooms)} rooms from {len(neto_polys)} polys, {len(doors)} doors")
    return rooms


CORRIDOR_ROOM_TYPES = {
    "פויאה", "פויה", "מבוא", "מעבר", "פרוזדור", "מרפסת",
    "חדר מדרגות", "מדרגות", "פיר מעלית", "פיר",
}


def main():
    all_rooms = []

    for (bld, floor) in sorted(DXF_FILES.keys()):
        rooms = process_floor(bld, floor)
        all_rooms.extend(rooms)

    # Merge in data from graph.json:
    # 1. Types: DXF type extraction fails for some B21 rooms (encoding issue).
    #    Use graph.json type when DXF returned empty string.
    # 2. Corridor/stairwell rooms: have no neto poly so they're absent from DXF
    #    extraction. Add them from graph.json so rooms.json is the single source.
    graph_path = ROOT / "backend" / "graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            g = json.load(f)

        # Build lookup: id -> first occurrence WITH a non-empty type in graph.json
        graph_nodes = {}
        for n in g["nodes"]:
            if n["id"] not in graph_nodes:
                graph_nodes[n["id"]] = n
            elif not graph_nodes[n["id"]].get("type") and n.get("type"):
                graph_nodes[n["id"]] = n  # upgrade to entry with type

        # 1. Fix empty types from DXF extraction
        fixed = 0
        for r in all_rooms:
            if not r["type"] and r["id"] in graph_nodes:
                r["type"] = graph_nodes[r["id"]].get("type", "")
                if r["type"]:
                    fixed += 1
        print(f"\nFixed {fixed} empty types from graph.json")

        # 2. Add missing corridor/stairwell rooms
        existing_ids = {r["id"] for r in all_rooms}
        added = 0
        for n in g["nodes"]:
            if n["id"] in existing_ids:
                continue
            if n.get("type", "") not in CORRIDOR_ROOM_TYPES:
                continue
            all_rooms.append({
                "id":         n["id"],
                "building":   n["building"],
                "floor":      n["floor"],
                "type":       n.get("type", ""),
                "centroid_x": round(n.get("x", 0), 1),
                "centroid_y": round(n.get("y", 0), 1),
                "door_x":     n.get("door_x"),
                "door_y":     n.get("door_y"),
                "poly_area":  None,
            })
            existing_ids.add(n["id"])
            added += 1
        print(f"Added {added} corridor/stairwell rooms from graph.json")

    out_path = ROOT / "backend" / "rooms.json"
    with open(out_path, "w") as f:
        json.dump(all_rooms, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_rooms)} rooms total to {out_path}")

    from collections import Counter
    counts = Counter((r["building"], r["floor"]) for r in all_rooms)
    for (bld, floor), n in sorted(counts.items()):
        print(f"  B{bld} F{floor}: {n} rooms")


if __name__ == "__main__":
    main()
