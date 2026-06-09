"""
update_centroids.py  (rewrite — pure DXF, no graph.json proximity matching)
----------------------------------------------------------------------------
Updates graph.json room centroids (x, y) for all floors using neto polygon
centroids read directly from DXF data.

Algorithm per floor:
  Pass 1 — poly-centric:
    For every neto (Shetah-Neto) LWPOLYLINE polygon:
      a. Collect all INSERT-block attribs from the TEXT layer whose INSERT point
         falls INSIDE the polygon.  The attrib that is exactly 5 digits and starts
         with the building prefix is the authoritative room ID.
      b. If no INSERT match, collect TEXT_YASHAN TEXT entities whose insert point
         falls INSIDE the polygon.  Use the first 5-digit label found.
      c. If the identified room is eligible (not מחסן, not already processed for
         this floor) → update x/y to polygon centroid.

  Pass 2 — room-centric fallback:
    For rooms still unmatched: if any INSERT attrib carries a 5-digit room ID that
    was never placed inside a polygon (e.g. the label is just outside), use the
    INSERT position itself as the centroid.

SKIP rules:
  - מחסן (storage closets) — never update
  - Labels ending in A or B (מחסן sub-rooms) — skip the label
  - B21 F2 rooms already verified — skip if they match the verified values

B22 DXF coordinates are in local frame; apply docs/b22_transform.json offset to
convert to the global campus frame before all operations.
"""

import json
import math
from pathlib import Path

import ezdxf
from shapely.geometry import Polygon, Point

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

SKIP_TYPES = {"מחסן"}

# These B21 F2 rooms were manually verified; do not overwrite them even if the
# DXF-derived centroid differs.
B21_F2_VERIFIED = {
    "21201": (1219.2, 2986.5),
    "21203": (2398.4, 1589.3),
    "21205": (4742.9, 990.4),
    "21206": (4832.4, 1810.5),
    "21207": (5090.3, 2926.5),
    "21208": (4356.2, 2704.1),
    "21209": (4198.8, 3521.6),
}


def _decode(raw: str) -> str:
    """Decode Windows-1255 text that arrived via latin-1 transit."""
    try:
        return raw.encode("latin-1", "replace").decode("windows-1255", "replace").strip()
    except Exception:
        return raw.strip()


def _is_5digit_id(label: str, bld: str) -> bool:
    """True if label is a 5-digit room ID for this building."""
    c = label.strip().rstrip(".")
    return len(c) == 5 and c.isdigit() and c.startswith(bld)


def _neto_layer_names(bld: str):
    """Return the set of accepted neto layer names for this building."""
    if bld == "21":
        return {"0201Shetah-Neto"}
    else:
        # B22 F6 uses '020 2Shetah-Neto' (with a space), others use '0202Shetah-Neto'
        return {"0202Shetah-Neto", "020 2Shetah-Neto"}


def _load_neto_polys(msp, bld):
    """
    Load all Shetah-Neto LWPOLYLINE polygons for this building/floor.
    Returns list of Shapely Polygon objects (already in global coords for B22).
    Handles self-intersecting polygons via buffer(0) and splits MultiPolygons.
    """
    layers = _neto_layer_names(bld)
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
            elif not poly.is_empty and poly.area > 0:
                polys.append(poly)
        except Exception:
            pass
    return polys


def _collect_insert_labels(msp, bld):
    """
    Collect (x, y, room_id) from INSERT entities on the TEXT layer.
    Only keeps attribs whose text is exactly 5 digits.
    Returns list of (global_x, global_y, room_id_str) tuples.
    """
    results = []
    for e in msp:
        if e.dxf.get("layer", "") != "TEXT":
            continue
        if e.dxftype() != "INSERT":
            continue
        ix, iy = e.dxf.insert.x, e.dxf.insert.y
        if bld == "22":
            ix, iy = _b22(ix, iy)
        try:
            for attrib in e.attribs:
                raw = attrib.dxf.get("text", "")
                label = _decode(raw)
                if len(label) == 5 and label.isdigit():
                    results.append((ix, iy, label))
                    break  # take first 5-digit attrib per INSERT block
        except Exception:
            pass
    return results


def _collect_text_yashan_labels(msp, bld):
    """
    Collect (x, y, label) from TEXT_YASHAN TEXT entities.
    Skips labels ending in A or B (מחסן sub-room variants).
    Returns list of (global_x, global_y, label_str) tuples.
    """
    results = []
    for e in msp:
        if e.dxf.get("layer", "") != "TEXT_YASHAN":
            continue
        if e.dxftype() != "TEXT":
            continue
        raw = e.dxf.get("text", "")
        label = _decode(raw)
        # Skip labels ending in A or B — these are מחסן sub-rooms
        if label.upper().endswith("A") or label.upper().endswith("B"):
            continue
        x, y = e.dxf.insert.x, e.dxf.insert.y
        if bld == "22":
            x, y = _b22(x, y)
        results.append((x, y, label))
    return results


def _find_id_in_poly(poly, insert_labels, text_labels, bld, eligible_ids, already_matched):
    """
    Try to find a room ID for this polygon by checking which labels fall inside it.
    Priority: INSERT 5-digit > TEXT_YASHAN 5-digit.
    Returns room_id string or None.
    """
    # 1. INSERT attribs: exact 5-digit match
    for ix, iy, room_id in insert_labels:
        if room_id in already_matched:
            continue
        if room_id not in eligible_ids:
            continue
        if poly.contains(Point(ix, iy)):
            return room_id

    # 2. TEXT_YASHAN 5-digit labels
    for lx, ly, label in text_labels:
        c = label.strip().rstrip(".")
        if not _is_5digit_id(c, bld):
            continue
        if c in already_matched:
            continue
        if c not in eligible_ids:
            continue
        if poly.contains(Point(lx, ly)):
            return c

    return None


def update_floor(bld, floor, floor_rooms, nodes_by_id, verbose=True):
    """
    Update x/y centroids in nodes_by_id for rooms on this floor.
    Returns dict: {room_id: (old_x, old_y, new_x, new_y, 'updated'|'skipped'|'verified')}
    """
    dxf_path = DXF_FILES.get((bld, floor))
    if dxf_path is None or not dxf_path.exists():
        if verbose:
            print(f"  B{bld} F{floor}: DXF not found at {dxf_path}")
        return {}

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    polys         = _load_neto_polys(msp, bld)
    insert_labels = _collect_insert_labels(msp, bld)
    text_labels   = _collect_text_yashan_labels(msp, bld)

    # Eligible rooms: not מחסן, must exist in nodes_by_id
    eligible_ids = {
        r["id"] for r in floor_rooms
        if r.get("type", "") not in SKIP_TYPES and r["id"] in nodes_by_id
    }

    # B21 F2 already-verified set (protected from update)
    verified_ids = set()
    if bld == "21" and floor == 2:
        verified_ids = set(B21_F2_VERIFIED.keys())

    results     = {}  # room_id -> result tuple
    matched_ids = set()  # rooms matched in pass 1

    # ------------------------------------------------------------------
    # Pass 1: poly-centric matching
    # ------------------------------------------------------------------
    for poly in polys:
        cx, cy = poly.centroid.x, poly.centroid.y

        room_id = _find_id_in_poly(
            poly, insert_labels, text_labels, bld, eligible_ids, matched_ids
        )

        if room_id is None:
            continue

        node = nodes_by_id[room_id]
        old_x, old_y = node["x"], node["y"]

        if room_id in verified_ids:
            results[room_id] = (old_x, old_y, old_x, old_y, "verified")
            matched_ids.add(room_id)
            continue

        node["x"] = round(cx, 1)
        node["y"] = round(cy, 1)
        results[room_id] = (old_x, old_y, node["x"], node["y"], "updated")
        matched_ids.add(room_id)

    # ------------------------------------------------------------------
    # Pass 2: room-centric fallback via INSERT/TEXT label positions
    # For rooms that were never placed inside a poly, the label may sit
    # just outside the polygon (stairwells, toilets, etc.).
    # Strategy: find the nearest UNMATCHED neto polygon within 600u of
    # the label position and use that polygon's centroid.
    # If no polygon is found within 600u, keep the original graph.json
    # value unchanged (do NOT use the label position directly).
    # Only use exact 5-digit labels here.
    # ------------------------------------------------------------------
    # Build set of poly indices already claimed in pass 1
    claimed_polys = set()
    for room_id2, res in results.items():
        if res[4] == "updated":
            # Find which poly produced this centroid
            for pi, poly in enumerate(polys):
                cx2, cy2 = round(poly.centroid.x, 1), round(poly.centroid.y, 1)
                if cx2 == res[2] and cy2 == res[3]:
                    claimed_polys.add(pi)
                    break

    # Build a combined list of (label_x, label_y, room_id) from both INSERT
    # and TEXT_YASHAN labels for unmatched rooms
    all_label_positions = {}  # room_id -> (lx, ly) — first occurrence wins
    for ix, iy, room_id in insert_labels:
        if room_id not in all_label_positions:
            all_label_positions[room_id] = (ix, iy)
    for lx, ly, label in text_labels:
        c = label.strip().rstrip(".")
        if _is_5digit_id(c, bld) and c not in all_label_positions:
            all_label_positions[c] = (lx, ly)

    MAX_LABEL_POLY_DIST = 600  # units

    for room_id, (lx, ly) in all_label_positions.items():
        if room_id in matched_ids:
            continue
        if room_id not in eligible_ids:
            continue
        if room_id in verified_ids:
            node = nodes_by_id[room_id]
            results[room_id] = (node["x"], node["y"], node["x"], node["y"], "verified")
            matched_ids.add(room_id)
            continue

        # Find nearest unmatched poly within MAX_LABEL_POLY_DIST
        best_pi, best_d = None, float("inf")
        for pi, poly in enumerate(polys):
            if pi in claimed_polys:
                continue
            d = poly.distance(Point(lx, ly))
            if d < best_d:
                best_d, best_pi = d, pi

        node = nodes_by_id[room_id]
        old_x, old_y = node["x"], node["y"]

        if best_pi is not None and best_d <= MAX_LABEL_POLY_DIST:
            cx2, cy2 = polys[best_pi].centroid.x, polys[best_pi].centroid.y
            node["x"] = round(cx2, 1)
            node["y"] = round(cy2, 1)
            results[room_id] = (old_x, old_y, node["x"], node["y"], "near-poly")
            claimed_polys.add(best_pi)
        else:
            # No nearby polygon — keep original graph.json value
            results[room_id] = (old_x, old_y, old_x, old_y, "unmatched")

        matched_ids.add(room_id)

    # rooms that were not matched at all
    for r in floor_rooms:
        rid = r["id"]
        if rid in eligible_ids and rid not in matched_ids:
            node = nodes_by_id[rid]
            results[rid] = (node["x"], node["y"], node["x"], node["y"], "unmatched")

    return results


def main():
    with open(ROOT / "backend" / "graph.json") as f:
        g = json.load(f)

    nodes_by_id = {n["id"]: n for n in g["nodes"]}

    ALL_FLOORS = [
        ("21", 1), ("21", 2), ("21", 3), ("21", 4), ("21", 5), ("21", 6),
        ("22", 2), ("22", 3), ("22", 4), ("22", 5), ("22", 6),
    ]

    grand_updated = 0
    grand_unmatched = []

    for bld, floor in ALL_FLOORS:
        floor_rooms = [r for r in g["nodes"] if r["building"] == bld and r["floor"] == floor]
        results = update_floor(bld, floor, floor_rooms, nodes_by_id)

        updated   = {k: v for k, v in results.items() if v[4] == "updated"}
        near_p    = {k: v for k, v in results.items() if v[4] == "near-poly"}
        verified  = {k: v for k, v in results.items() if v[4] == "verified"}
        unmatched = {k: v for k, v in results.items() if v[4] == "unmatched"}

        grand_updated   += len(updated) + len(near_p)
        grand_unmatched += list(unmatched.keys())

        print(f"\nB{bld} F{floor}:")
        if updated:
            print(f"  Updated via poly centroid ({len(updated)}):")
            for rid, (ox, oy, nx, ny, _) in sorted(updated.items()):
                print(f"    {rid}: ({ox:.1f},{oy:.1f}) → ({nx:.1f},{ny:.1f})")
        if near_p:
            print(f"  Updated via nearest poly centroid ({len(near_p)}):")
            for rid, (ox, oy, nx, ny, _) in sorted(near_p.items()):
                print(f"    {rid}: ({ox:.1f},{oy:.1f}) → ({nx:.1f},{ny:.1f})")
        if verified:
            print(f"  Verified (kept existing) ({len(verified)}): {sorted(verified.keys())}")
        if unmatched:
            print(f"  Unmatched — no nearby poly found ({len(unmatched)}): {sorted(unmatched.keys())}")

    print(f"\n{'='*60}")
    print(f"Total rooms updated: {grand_updated}")
    if grand_unmatched:
        print(f"Total unmatched rooms: {len(grand_unmatched)}: {sorted(grand_unmatched)}")

    # Final sanity check: verify B21 F2 protected values are intact
    print("\nB21 F2 sanity check (protected rooms):")
    all_ok = True
    for rid, (ex, ey) in B21_F2_VERIFIED.items():
        node = nodes_by_id.get(rid)
        if node is None:
            print(f"  {rid}: MISSING from graph.json")
            all_ok = False
            continue
        ax, ay = round(node["x"], 1), round(node["y"], 1)
        status = "OK" if (abs(ax - ex) < 0.5 and abs(ay - ey) < 0.5) else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"  {rid}: expected ({ex},{ey}), actual ({ax},{ay}) → {status}")

    if not all_ok:
        print("ERROR: Some B21 F2 protected values were changed — aborting write!")
        return

    with open(ROOT / "backend" / "graph.json", "w") as f:
        json.dump(g, f, ensure_ascii=False, indent=2)
    print("\ngraph.json written.")


if __name__ == "__main__":
    main()
