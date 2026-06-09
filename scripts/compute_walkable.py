"""
compute_walkable.py
-------------------
Computes the walkable polygon for a given building/floor from DXF source files.

Walkable = bruto boundary
         - destination-room neto polygons (offices, labs, lecture halls, etc.)
         - buffered Beton wall segments (structural walls)
         - buffered Window wall segments (glass curtain walls)
         - buffered Mabat ARC and CIRCLE entities (curved solid obstacles, treated as Beton)
         - 150u circle fallback for destination rooms with no neto polygon

NOT subtracted:
         - Mabat LINE / LWPOLYLINE (partition walls — shown for reference only)
         - Mabat door-swing arcs (small r<50 arcs — these are notation, not walls)

Forced waypoints (mandatory JUNCTION nodes added to graph):
         - Center of each staircase tread block (from Mabat LINE grid analysis)
         These ensure routing always passes through the staircase landing corridor.

Room type classification
------------------------
CORRIDOR_ROOM_TYPES: spaces that ARE the walkable corridor.
    Their neto polygons are kept as walkable (not subtracted).
    פויאה / פויה  — foyer / lobby
    מבוא          — entrance lobby
    מעבר          — passage
    פרוזדור       — hallway / corridor
    מרפסת         — balcony / terrace
    חדר מדרגות / מדרגות — stairwell (vertical connector)
    פיר מעלית / פיר     — elevator shaft

Everything else (משרד, חדר חוקרים, כיתת לימוד, שירותים, ...) is a destination room
and is subtracted from walkable.
"""

import math
import json
from pathlib import Path
from collections import defaultdict

import ezdxf
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from shapely.ops import unary_union

ROOT    = Path(__file__).resolve().parent.parent
DATA21  = ROOT / "data" / "21"
DATA22  = ROOT / "data" / "22"
DOCS    = ROOT / "docs"

with open(ROOT / "docs" / "b22_transform.json") as f:
    _t = json.load(f)
B22_OX, B22_OY = _t["offset_x"], _t["offset_y"]

def b22(x, y):
    return x + B22_OX, y + B22_OY

DXF_FILES = {
    ("21", 1): DATA21 / "02011.dxf",
    ("21", 2): DATA21 / "02012.dxf",
    ("21", 3): DATA21 / "02013.dxf",
    ("21", 4): DATA21 / "02014.dxf",
    ("21", 5): DATA21 / "02015.dxf",
    ("21", 6): DATA21 / "02016.dxf",
    ("22", 1): DATA22 / "02021.dxf",
    ("22", 2): DATA22 / "02022.dxf",
    ("22", 3): DATA22 / "02023.dxf",
    ("22", 4): DATA22 / "02024.dxf",
    ("22", 5): DATA22 / "02025.dxf",
    ("22", 6): DATA22 / "02026.dxf",
}

CORRIDOR_ROOM_TYPES = {
    "פויאה", "פויה",
    "מבוא",
    "מעבר",
    "פרוזדור",
    "מרפסת",
    "חדר מדרגות", "מדרגות",
    "פיר מעלית", "פיר",
}

# Beton/Window buffer (structural wall half-thickness in DXF units)
WALL_BUFFER = 20
# Mabat ARC/CIRCLE buffer (treat as solid obstacles)
MABAT_SOLID_BUFFER = 20
# Min arc radius to treat as a wall (arcs < this are door-swing notation — ignored)
MABAT_ARC_MIN_R = 50
# Fallback exclusion radius for destination rooms with no neto polygon
FALLBACK_ROOM_BUFFER = 150
# Max distance from neto poly centroid to room centroid for matching
ROOM_CENTROID_DIST = 500


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _arc_to_segments(cx, cy, r, a0_deg, a1_deg, seg_len=20):
    """Approximate a DXF arc as a list of LineStrings."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    n = max(8, int(r * abs(a1 - a0) / seg_len))
    angles = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
    return [LineString([pts[i], pts[i + 1]]) for i in range(len(pts) - 1)]


def compute_walkable(bld: str, floor: int, floor_rooms: list) -> tuple:
    """
    Returns (walkable_polygon, forced_waypoints, dest_polys, fallback_bufs).

    walkable_polygon  — Shapely geometry (Polygon or MultiPolygon)
    forced_waypoints  — list of (x, y) tuples for mandatory JUNCTION nodes
    dest_polys        — neto polygons that were subtracted (destination rooms)
    fallback_bufs     — 150u circle buffers for rooms missing a neto poly
    """
    dxf_path = DXF_FILES.get((bld, floor))
    if dxf_path is None or not dxf_path.exists():
        return None, [], [], []

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    bruto_layer = "0201Shetah-Bruto" if bld == "21" else "0202Shetah-Bruto"
    neto_layer  = "0201Shetah-Neto"  if bld == "21" else "0202Shetah-Neto"
    neto_layer_alt = neto_layer.replace("0202", "020 2")  # B22 F6 typo

    def _read_pts(e):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if bld == "22":
            pts = [b22(x, y) for x, y in pts]
        return pts

    # --- Bruto boundary ---
    bruto_polys = []
    for e in msp:
        if e.dxf.get("layer", "") != bruto_layer or e.dxftype() != "LWPOLYLINE":
            continue
        pts = _read_pts(e)
        if len(pts) >= 3:
            try:
                p = make_valid(Polygon(pts))
                if not p.is_empty:
                    bruto_polys.append(p)
            except Exception:
                pass
    if not bruto_polys:
        return None, [], [], []
    outer = unary_union(bruto_polys)

    # --- Neto polygons ---
    neto_polys = []
    for e in msp:
        if e.dxf.get("layer", "") not in (neto_layer, neto_layer_alt):
            continue
        if e.dxftype() != "LWPOLYLINE":
            continue
        pts = _read_pts(e)
        if len(pts) >= 3:
            try:
                p = Polygon(pts)
                if not p.is_valid:
                    p = p.buffer(0)
                # buffer(0) may return MultiPolygon for self-intersecting rings
                if p.geom_type == "MultiPolygon":
                    for sub in p.geoms:
                        if not sub.is_empty and sub.area > 0:
                            neto_polys.append(sub)
                elif not p.is_empty and p.area > 0:
                    neto_polys.append(p)
            except Exception:
                pass

    # --- Classify neto polys as walkable corridor or destination room ---
    #
    # Rule: for each poly, the NEAREST room centroid determines its class.
    #   - nearest room is corridor type  → walkable (keep)
    #   - nearest room is dest/מחסן type → subtract
    # Safety: always protect the largest poly on the floor (main corridor).
    # מחסן rooms never cause subtraction.

    # Always protect the single largest poly — main corridor on every floor.
    corr_homes = set()
    if neto_polys:
        corr_homes.add(max(range(len(neto_polys)), key=lambda i: neto_polys[i].area))

    dest_idx = set()
    candidate_rooms = [r for r in floor_rooms if r.get("type", "") != "מחסן"]

    for i, poly in enumerate(neto_polys):
        if i in corr_homes:
            continue
        if not candidate_rooms:
            dest_idx.add(i)
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        nearest = min(candidate_rooms,
                      key=lambda r: _dist((cx, cy), (r["centroid_x"], r["centroid_y"])))
        if nearest.get("type", "") in CORRIDOR_ROOM_TYPES:
            corr_homes.add(i)   # corridor → walkable
        else:
            dest_idx.add(i)     # destination → subtract

    dest_polys_pass1 = [neto_polys[i] for i in dest_idx]

    # Pass 2 (room-centric): for each destination room not yet covered,
    # force-subtract the nearest unclaimed poly.
    for r in floor_rooms:
        if r.get("type", "") in CORRIDOR_ROOM_TYPES:
            continue
        if r.get("type", "") == "מחסן":
            continue
        pt = Point(r["centroid_x"], r["centroid_y"])
        if any(neto_polys[i].buffer(200).contains(pt) for i in dest_idx):
            continue
        best_i, best_d = None, float("inf")
        for i, poly in enumerate(neto_polys):
            if i in dest_idx or i in corr_homes:
                continue
            d = poly.distance(pt)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d < ROOM_CENTROID_DIST:
            dest_idx.add(best_i)

    dest_polys = [neto_polys[i] for i in sorted(dest_idx)]

    rooms_union = unary_union(dest_polys) if dest_polys else None
    walkable = make_valid(outer.difference(rooms_union)) if rooms_union else outer

    # --- Structural walls: Beton + Window ---
    # Filter out drafting artifacts: short LINE segments and tiny/open LWPOLYLINEs
    # that appear on the Beton layer but are not real structural walls.
    MIN_LINE_LEN  = 40   # ignore Beton LINE segments shorter than this
    MIN_POLY_AREA = 100  # ignore Beton LWPOLYLINE with enclosed area smaller than this
    wall_lines = []
    for e in msp:
        layer = e.dxf.get("layer", "")
        if layer not in ("Beton", "Window"):
            continue
        t = e.dxftype()
        if t == "LINE":
            px, py = e.dxf.start.x, e.dxf.start.y
            ex, ey = e.dxf.end.x, e.dxf.end.y
            if bld == "22": px, py = b22(px, py); ex, ey = b22(ex, ey)
            if math.hypot(ex - px, ey - py) < MIN_LINE_LEN:
                continue
            wall_lines.append(LineString([(px, py), (ex, ey)]))
        elif t == "LWPOLYLINE":
            pts = _read_pts(e)
            if len(pts) < 2:
                continue
            # skip tiny open polylines — these are hatch/annotation artifacts
            if not e.closed:
                from shapely.geometry import Polygon as _Poly
                try:
                    area = _Poly(pts).area
                except Exception:
                    area = 0
                if area < MIN_POLY_AREA:
                    continue
            seg_pts = pts + [pts[0]] if e.closed else pts
            for i in range(len(seg_pts) - 1):
                wall_lines.append(LineString([seg_pts[i], seg_pts[i + 1]]))

    # --- Mabat solid obstacles: ARC (r >= MABAT_ARC_MIN_R) and CIRCLE ---
    mabat_solid_lines = []
    # Collect all large arcs grouped by center to find concentric staircase pairs
    arc_by_center = defaultdict(list)   # key=(round_cx, round_cy) -> [(r, a0, a1), ...]
    CENTER_SNAP = 30  # arcs within 30u of each other are on the same staircase

    for e in msp:
        if e.dxf.get("layer", "") != "Mabat":
            continue
        t = e.dxftype()
        if t == "ARC":
            cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
            if bld == "22": cx, cy = b22(cx, cy)
            if r >= MABAT_ARC_MIN_R:
                mabat_solid_lines.extend(
                    _arc_to_segments(cx, cy, r, e.dxf.start_angle, e.dxf.end_angle)
                )
                key = (round(cx / CENTER_SNAP) * CENTER_SNAP,
                       round(cy / CENTER_SNAP) * CENTER_SNAP)
                arc_by_center[key].append((cx, cy, r, e.dxf.start_angle, e.dxf.end_angle))
        elif t == "CIRCLE":
            cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
            if bld == "22": cx, cy = b22(cx, cy)
            mabat_solid_lines.extend(
                _arc_to_segments(cx, cy, r, 0, 360)
            )

    # Place one waypoint per staircase unit.
    # Strategy: group arc-centers that are close in Y (same staircase unit with
    # multiple flights). Waypoint = centroid of all arc centers in the group,
    # which lands on the landing between the flights.
    # For isolated single groups, use arc center.
    group_list = sorted(arc_by_center.items())  # sorted by snapped key
    # Merge groups whose snapped keys are within CENTER_SNAP*2 of each other
    merged = []
    for key, arcs in group_list:
        kx, ky = key
        placed = False
        for mg in merged:
            mkx, mky = mg["key"]
            if abs(kx - mkx) <= 600 and abs(ky - mky) <= 600:
                mg["arcs"].extend(arcs)
                placed = True
                break
        if not placed:
            merged.append({"key": key, "arcs": arcs})

    forced_waypoints = []
    for mg in merged:
        all_arcs = mg["arcs"]
        # Centroid of all arc centers → landing between flights
        wx = sum(a[0] for a in all_arcs) / len(all_arcs)
        wy = sum(a[1] for a in all_arcs) / len(all_arcs)
        forced_waypoints.append((wx, wy))

    all_wall_lines = wall_lines + mabat_solid_lines
    if all_wall_lines:
        wall_buf = unary_union([w.buffer(WALL_BUFFER) for w in all_wall_lines])
        walkable = make_valid(walkable.difference(wall_buf))

    # No fallback circle buffers — pass 2 above finds the actual neto poly instead.
    fallback_bufs = []

    # Add elevator rooms as waypoints (using door position from rooms.json)
    ELEV_TYPES = {"פיר מעלית", "פיר"}
    for r in floor_rooms:
        if r.get("type", "") in ELEV_TYPES:
            ex = r.get("door_x") or r["centroid_x"]
            ey = r.get("door_y") or r["centroid_y"]
            if ex is not None and ey is not None:
                forced_waypoints.append((ex, ey))

    # Deduplicate forced waypoints, keep only those inside walkable
    seen = set()
    valid_waypoints = []
    for wx, wy in forced_waypoints:
        key2 = (round(wx, -1), round(wy, -1))
        if key2 not in seen:
            seen.add(key2)
            if walkable.contains(Point(wx, wy)):
                valid_waypoints.append((wx, wy))

    return walkable, valid_waypoints, dest_polys, fallback_bufs


def render_floor(bld, floor, floor_rooms, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc as MplArc

    dxf_path = DXF_FILES.get((bld, floor))
    if dxf_path is None or not dxf_path.exists():
        print(f"  SKIP B{bld} F{floor} — no DXF")
        return

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    walkable, waypoints, dest_polys, fallback_bufs = compute_walkable(bld, floor, floor_rooms)
    if walkable is None:
        print(f"  SKIP B{bld} F{floor} — no bruto polygon")
        return

    print(f"  B{bld} F{floor}: walkable={100*walkable.area/walkable.convex_hull.area:.1f}%  "
          f"dest={len(dest_polys)}  waypoints={len(waypoints)}")

    door_pts = []
    for e in msp:
        if e.dxf.get("layer", "") != "Door" or e.dxftype() != "LWPOLYLINE":
            continue
        verts = list(e.get_points())
        if len(verts) >= 2:
            x = (verts[0][0] + verts[1][0]) / 2
            y = (verts[0][1] + verts[1][1]) / 2
            if bld == "22": x, y = b22(x, y)
            door_pts.append((x, y))

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_aspect("equal")
    ax.set_title(f"B{bld} F{floor} — Walkable | staircase waypoint (cyan star)", fontsize=12)

    geoms = [walkable] if walkable.geom_type == "Polygon" else list(walkable.geoms)
    for gm in geoms:
        if gm.geom_type == "Polygon":
            ax.fill(*gm.exterior.xy, alpha=0.25, color="steelblue")

    for poly in dest_polys:
        if hasattr(poly, "exterior"):
            ax.fill(*poly.exterior.xy, alpha=0.3, color="salmon")

    def _tx(x, y):
        """Apply building coordinate transform for rendering."""
        if bld == "22":
            return b22(x, y)
        return x, y

    for e in msp:
        l, t = e.dxf.get("layer", ""), e.dxftype()
        if l == "Beton":
            if t == "LINE":
                x1,y1 = _tx(e.dxf.start.x, e.dxf.start.y)
                x2,y2 = _tx(e.dxf.end.x,   e.dxf.end.y)
                ax.plot([x1,x2],[y1,y2], color="black", lw=0.8)
            elif t == "LWPOLYLINE":
                pts = [_tx(p[0],p[1]) for p in e.get_points()]
                if e.closed: pts += [pts[0]]
                ax.plot([p[0] for p in pts],[p[1] for p in pts], color="black", lw=0.8)
        elif l == "Mabat":
            if t == "LINE":
                x1,y1 = _tx(e.dxf.start.x, e.dxf.start.y)
                x2,y2 = _tx(e.dxf.end.x,   e.dxf.end.y)
                ax.plot([x1,x2],[y1,y2], color="orange", lw=0.7, alpha=0.9)
            elif t == "LWPOLYLINE":
                pts = [_tx(p[0],p[1]) for p in e.get_points()]
                if e.closed: pts += [pts[0]]
                ax.plot([p[0] for p in pts],[p[1] for p in pts], color="orange", lw=0.7, alpha=0.9)
            elif t == "ARC":
                cx,cy = _tx(e.dxf.center.x, e.dxf.center.y)
                ax.add_patch(MplArc((cx, cy), 2*e.dxf.radius, 2*e.dxf.radius,
                                    angle=0, theta1=e.dxf.start_angle, theta2=e.dxf.end_angle,
                                    color="orange", lw=0.9))
            elif t == "CIRCLE":
                cx,cy = _tx(e.dxf.center.x, e.dxf.center.y)
                ax.add_patch(plt.Circle((cx,cy), e.dxf.radius, fill=False, color="orange", lw=0.9))
        elif l == "Window" and t == "LINE":
            x1,y1 = _tx(e.dxf.start.x, e.dxf.start.y)
            x2,y2 = _tx(e.dxf.end.x,   e.dxf.end.y)
            ax.plot([x1,x2],[y1,y2], color="blue", lw=0.6, alpha=0.6)

    if door_pts:
        ax.scatter([p[0] for p in door_pts], [p[1] for p in door_pts], s=25, c="green", zorder=5)
    if waypoints:
        ax.scatter([p[0] for p in waypoints], [p[1] for p in waypoints],
                   s=300, c="cyan", marker="*", zorder=10, edgecolors="black", lw=0.5)

    ax.legend(handles=[
        mpatches.Patch(color="steelblue", alpha=0.3, label="Walkable"),
        mpatches.Patch(color="salmon",    alpha=0.4, label="Dest rooms"),
        mpatches.Patch(color="black",               label="Beton"),
        mpatches.Patch(color="orange",              label="Mabat"),
        mpatches.Patch(color="blue",                label="Window"),
        mpatches.Patch(color="green",               label="Doors"),
        mpatches.Patch(color="cyan",                label="Stair waypoint"),
    ], loc="upper right")

    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    with open(ROOT / "backend" / "rooms.json") as f:
        all_rooms = json.load(f)

    ALL_FLOORS = [
        ("21", 1), ("21", 2), ("21", 3), ("21", 4), ("21", 5), ("21", 6),
        ("22", 1), ("22", 2), ("22", 3), ("22", 4), ("22", 5), ("22", 6),
    ]

    for bld, floor in ALL_FLOORS:
        floor_rooms = [r for r in all_rooms if r["building"] == bld and r["floor"] == floor]
        out = DOCS / f"walkable_b{bld}_f{floor}.png"
        render_floor(bld, floor, floor_rooms, out)

    print("\nAll floors done.")
