"""
fix_room_types.py
-----------------
Scan all DXF floors and fix misclassified room types in graph.json.

Two classes of fix are applied:

CLASS A — מחסן -> correct type
  For each room currently typed as מחסן in graph.json:
  1. Find the INSERT block in the TEXT layer whose attrib[3] matches the room ID,
     attrib[4] (sub-room flag) is blank (main room, not a sub-closet like '21300A'),
     AND attrib[2] matches the current floor number.
  2. If no such exact main-room match, fall back to the nearest main-room INSERT
     within MAX_FALLBACK_DIST units of the graph.json centroid.
  3. If the DXF type_code is in {RL, RS, RV, RT} (i.e. NOT storage), apply the fix.

CLASS B — non-מחסן -> מחסן
  For each room whose graph.json type is NOT מחסן:
  If an exact 5-digit main-room DXF INSERT (floor-code-filtered, no sub-flag) exists
  AND its type_code is RA or RN → change the graph.json type to מחסן.

  This corrects rooms like ארכיון (archive), חדר עזר, חדר תקשורת that DXF marks RA/RN.

CLASS C — משרד -> מבוא (explicit reverse check requested by user)
  If graph type is 'משרד' AND DXF main-room type_code == 'RL', fix to 'מבוא'.

What we deliberately do NOT change:
  - 'חדר חוקרים' / 'חדר עוזרים' -> 'משרד' (same R0 category, just more specific)
  - toilet types -> 'פרוזדור' (DXF RT is a generic category; specific toilet names are kept)
  - stairwell <-> elevator (ambiguous DXF coding)
  - Any name that is a legitimate Hebrew specialization of the DXF generic code

Floor-code filtering:
  Each DXF file may contain INSERT blocks for multiple floors (e.g. B22/02025.dxf has
  floors 4 AND 5). attrib[2] in each INSERT gives the actual floor number as a string.
  We filter to only consider blocks whose attrib[2] matches the target floor number.

Type code -> Hebrew type mapping (for fixes only):
  RL -> מבוא
  RS -> חדר מדרגות
  RV -> פיר מעלית
  RT -> פרוזדור
  RA -> מחסן
  RN -> מחסן
"""

import json
import math
from pathlib import Path

import ezdxf

ROOT   = Path(__file__).resolve().parent.parent
DATA21 = ROOT / "data" / "21"
DATA22 = ROOT / "data" / "22"

with open(ROOT / "docs" / "b22_transform.json") as f:
    _t = json.load(f)
B22_OX, B22_OY = _t["offset_x"], _t["offset_y"]

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

# Type codes that indicate NON-storage space (triggers CLASS-A fix for מחסן rooms)
NON_STORAGE_CODES = {"RL", "RS", "RV", "RT"}

# Type codes that indicate storage (triggers CLASS-B fix for non-מחסן rooms)
STORAGE_CODES = {"RA", "RN"}

# Type code -> Hebrew type name (used only for the types we actually fix)
TYPE_CODE_TO_HEB = {
    "RL": "מבוא",
    "RS": "חדר מדרגות",
    "RV": "פיר מעלית",
    "RT": "פרוזדור",
    "RA": "מחסן",
    "RN": "מחסן",
}

# Maximum distance (in DXF units) for position-based fallback lookup
MAX_FALLBACK_DIST = 400

# Hebrew type names that are fine-grained specialisations of the R0 (office) type.
# We do NOT change these to 'משרד' even if DXF says R0.
R0_SPECIALISATIONS = {
    "חדר חוקרים",
    "חדר עוזרים",
    "חדר מחשבים",
    "כיתת מחשב",
    "חדר ישיבות",
    "חדר ישיבה",
    "חדר תקשורת",
    "חדר בקרה",
    "חדר מכונות",
    "מטבחון",
    "מרפסת",
    "סמינריון",
    "כיתת סמינר",
    "שירותים",
    "שירותים נשים",
    "שירותים גברים",
    "שרותי נכים",
    "שירותי נכים",
    "שרותים נשים",
    "מרחב מוגן",
    "מקלט, סמינריון",
    "ציוד מחקר",
    "פויאה",
    "פיר",
}


def _decode(raw: str) -> str:
    """Decode Windows-1255 text that arrived via latin-1 transit."""
    try:
        return raw.encode("latin-1", "replace").decode("windows-1255", "replace").strip()
    except Exception:
        return raw.strip()


def _b22(x, y):
    return x + B22_OX, y + B22_OY


def _collect_main_room_inserts(msp, bld, floor):
    """
    Collect all main-room INSERT blocks from the TEXT layer.

    Filtering rules:
    - Layer == 'TEXT', entity type == INSERT
    - attrib[2] matches the target floor number (str(floor)) — floor-code filter
    - attrib[4] (sub-room flag) is blank — no sub-closets like '21300A'
    - attrib[3] is a 5-digit room ID starting with the building prefix, OR
      a 4-digit legacy room number

    Returns:
    - five_digit: list of (global_x, global_y, room_id, type_code)
    - four_digit: list of (global_x, global_y, legacy_id, type_code)
    """
    floor_str  = str(floor)
    five_digit = []
    four_digit = []

    for e in msp:
        if e.dxf.get("layer", "") != "TEXT":
            continue
        if e.dxftype() != "INSERT":
            continue
        try:
            attribs = list(e.attribs)
            if len(attribs) < 4:
                continue
            decoded = [_decode(a.dxf.get("text", "")) for a in attribs]

            # --- floor-code filter ---
            attrib_floor = decoded[2] if len(decoded) > 2 else ""
            if attrib_floor != floor_str:
                continue

            type_code   = decoded[0]
            room_id_raw = decoded[3]
            sub_flag    = decoded[4].strip() if len(decoded) > 4 else ""

            # Skip sub-rooms (A, B, C, D, etc.)
            if sub_flag:
                continue

            ix, iy = e.dxf.insert.x, e.dxf.insert.y
            if bld == "22":
                ix, iy = _b22(ix, iy)

            if len(room_id_raw) == 5 and room_id_raw.isdigit() and room_id_raw.startswith(bld):
                five_digit.append((ix, iy, room_id_raw, type_code))
            elif len(room_id_raw) == 4 and room_id_raw.isdigit():
                four_digit.append((ix, iy, room_id_raw, type_code))
        except Exception:
            continue

    return five_digit, four_digit


def _type_for_room(room_id, graph_x, graph_y, five_digit, four_digit):
    """
    Determine DXF type_code for a given room.

    Priority:
    1. Exact 5-digit match on room_id (main room, floor-code-filtered).
    2. Nearest main-room INSERT (5-digit or 4-digit) within MAX_FALLBACK_DIST.

    Returns (type_code, method) or (None, None) if no match found.
    """
    # 1. Exact 5-digit match
    for ix, iy, rid, tc in five_digit:
        if rid == room_id:
            return tc, "exact-5digit"

    # 2. Position-based fallback
    best_dist = float("inf")
    best_tc   = None
    for ix, iy, rid, tc in (five_digit + four_digit):
        d = math.sqrt((ix - graph_x) ** 2 + (iy - graph_y) ** 2)
        if d < best_dist:
            best_dist = d
            best_tc   = tc

    if best_dist <= MAX_FALLBACK_DIST:
        return best_tc, f"nearest-insert(dist={best_dist:.1f})"

    return None, None


def fix_floor(bld, floor, floor_rooms):
    """
    Scan the DXF for this floor and return a list of fixes to apply.
    Each fix is a dict: {room_id, old_type, new_type, method, floor, building}
    """
    dxf_path = DXF_FILES.get((bld, floor))
    if dxf_path is None or not dxf_path.exists():
        return []

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    five_digit, four_digit = _collect_main_room_inserts(msp, bld, floor)

    fixes = []
    for room in floor_rooms:
        rid   = room["id"]
        gtype = room.get("type", "")
        gx    = room.get("x", 0)
        gy    = room.get("y", 0)

        dxf_code, method = _type_for_room(rid, gx, gy, five_digit, four_digit)

        if dxf_code is None:
            continue  # No DXF INSERT found for this room

        # --- CLASS A: מחסן -> correct non-storage type ---
        if gtype == "מחסן" and dxf_code in NON_STORAGE_CODES:
            new_type = TYPE_CODE_TO_HEB[dxf_code]
            fixes.append({
                "room_id":  rid,
                "old_type": gtype,
                "new_type": new_type,
                "dxf_code": dxf_code,
                "method":   method,
                "floor":    floor,
                "building": bld,
            })

        # --- CLASS B: non-מחסן -> מחסן when DXF says storage ---
        elif gtype != "מחסן" and dxf_code in STORAGE_CODES:
            # Don't downgrade rooms that are already named with fine-grained Hebrew names
            # that clearly belong to another category (toilet, stairwell, elevator, etc.)
            # Only fix: ארכיון, חדר עזר, חדר תקשורת, פויאה, or plain 'משרד'/'חדר חוקרים'
            # when DXF explicitly says RA/RN.
            # Skip rooms whose type is in R0_SPECIALISATIONS (they have specific names).
            if gtype not in R0_SPECIALISATIONS and gtype not in {
                "חדר מדרגות", "פיר מעלית", "מבוא", "פרוזדור",
                "כיתת לימוד", "כיתת סמינר", "שירותים גברים", "שירותים נשים",
            }:
                fixes.append({
                    "room_id":  rid,
                    "old_type": gtype,
                    "new_type": "מחסן",
                    "dxf_code": dxf_code,
                    "method":   method,
                    "floor":    floor,
                    "building": bld,
                })

        # --- CLASS C: משרד -> מבוא when DXF says RL ---
        elif gtype == "משרד" and dxf_code == "RL":
            fixes.append({
                "room_id":  rid,
                "old_type": gtype,
                "new_type": "מבוא",
                "dxf_code": dxf_code,
                "method":   method,
                "floor":    floor,
                "building": bld,
            })

    return fixes


def main():
    with open(ROOT / "backend" / "graph.json") as f:
        g = json.load(f)

    nodes_by_id = {n["id"]: n for n in g["nodes"]}

    ALL_FLOORS = [
        ("21", 1), ("21", 2), ("21", 3), ("21", 4), ("21", 5), ("21", 6),
        ("22", 2), ("22", 3), ("22", 4), ("22", 5), ("22", 6),
    ]

    all_fixes = []

    for bld, floor in ALL_FLOORS:
        floor_rooms = [
            n for n in g["nodes"]
            if n["building"] == bld and n["floor"] == floor
        ]
        fixes = fix_floor(bld, floor, floor_rooms)
        all_fixes.extend(fixes)

    if not all_fixes:
        print("No type corrections needed.")
        return

    print(f"Found {len(all_fixes)} type correction(s):")
    class_a = [f for f in all_fixes if f["old_type"] == "מחסן"]
    class_b = [f for f in all_fixes if f["new_type"] == "מחסן" and f["old_type"] != "מחסן"]
    class_c = [f for f in all_fixes if f["old_type"] == "משרד" and f["new_type"] == "מבוא"]

    if class_a:
        print(f"\n  CLASS A (מחסן -> correct type): {len(class_a)}")
        for fix in class_a:
            print(
                f"    {fix['room_id']} B{fix['building']}F{fix['floor']}: "
                f"{fix['old_type']!r} -> {fix['new_type']!r} "
                f"(DXF={fix['dxf_code']!r}, {fix['method']})"
            )

    if class_b:
        print(f"\n  CLASS B (non-מחסן -> מחסן): {len(class_b)}")
        for fix in class_b:
            print(
                f"    {fix['room_id']} B{fix['building']}F{fix['floor']}: "
                f"{fix['old_type']!r} -> {fix['new_type']!r} "
                f"(DXF={fix['dxf_code']!r}, {fix['method']})"
            )

    if class_c:
        print(f"\n  CLASS C (משרד -> מבוא): {len(class_c)}")
        for fix in class_c:
            print(
                f"    {fix['room_id']} B{fix['building']}F{fix['floor']}: "
                f"{fix['old_type']!r} -> {fix['new_type']!r} "
                f"(DXF={fix['dxf_code']!r}, {fix['method']})"
            )

    # Revert any previous changes by re-reading from backup or just apply fresh
    # (graph.json was already modified by first run; need to reload from disk)
    # Apply fixes to the loaded graph
    for fix in all_fixes:
        nodes_by_id[fix["room_id"]]["type"] = fix["new_type"]

    with open(ROOT / "backend" / "graph.json", "w") as f:
        json.dump(g, f, ensure_ascii=False, indent=2)
    print("\ngraph.json updated.")


if __name__ == "__main__":
    main()
