"""
Campus Navigation API  —  English version
Run:  uvicorn main:app --reload
Test: http://localhost:8000/health
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import json, heapq, os, sqlite3, hashlib, math
from collections import defaultdict as _defaultdict

app = FastAPI(title="Campus Navigation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = os.path.dirname(__file__)

_DB = os.path.join(BASE, "analytics.db")
def _init_analytics():
    with sqlite3.connect(_DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS searches(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_room TEXT, to_room TEXT,
            from_floor INTEGER, to_floor INTEGER,
            from_building TEXT, to_building TEXT,
            floor_changes INTEGER,
            ts TEXT DEFAULT (datetime('now','localtime')),
            ip_hash TEXT
        )""")
_init_analytics()

def _log_search(from_room, to_room, floor_changes, ip=""):
    try:
        fr=NODES.get(from_room,{}); tr=NODES.get(to_room,{})
        with sqlite3.connect(_DB) as c:
            c.execute("INSERT INTO searches(from_room,to_room,from_floor,to_floor,from_building,to_building,floor_changes,ip_hash) VALUES(?,?,?,?,?,?,?,?)",
                (from_room,to_room,fr.get("floor"),tr.get("floor"),fr.get("building"),tr.get("building"),floor_changes,hashlib.md5(ip.encode()).hexdigest()[:8]))
    except Exception as e:
        print(f"analytics err: {e}")

# rooms.json is the single source of truth for room data (DXF-derived centroids,
# types, door positions).  graph.json is still loaded for routing edges only.
with open(os.path.join(BASE, "rooms.json"), encoding="utf-8") as f:
    _rooms_data = json.load(f)

NODES: dict = {}
for _r in _rooms_data:
    if _r["id"] not in NODES:
        NODES[_r["id"]] = {
            "id":        _r["id"],
            "building":  _r["building"],
            "floor":     _r["floor"],
            "floor_num": _r["floor"],
            "type":      _r.get("type", ""),
            "x":         _r.get("centroid_x", 0),
            "y":         _r.get("centroid_y", 0),
            "door_x":    _r.get("door_x"),
            "door_y":    _r.get("door_y"),
        }

# Load graph.json for routing edges only (ADJ).
try:
    with open(os.path.join(BASE, "graph.json"), encoding="utf-8") as f:
        GRAPH = json.load(f)
except FileNotFoundError:
    GRAPH = {"nodes": [], "edges": []}

ADJ = {}
for edge in GRAPH["edges"]:
    ADJ.setdefault(edge["from"], []).append((edge["to"],   edge["weight"], edge["type"]))
    ADJ.setdefault(edge["to"],   []).append((edge["from"], edge["weight"], edge["type"]))

TYPE_EN = {
    "כיתת לימוד": "Classroom", "כיתת סמינר": "Seminar Room",
    "משרד": "Office", "חדר חוקרים": "Researchers Room",
    "חדר מדרגות": "Stairwell", "פיר מעלית": "Elevator Shaft",
    "מעלית": "Elevator", "שירותים": "Restrooms", "מעבדה": "Lab",
    "פויאה": "Foyer", "מבוא": "Entrance", "מחסן": "Storage",
    "פרוזדור": "Corridor", "מעבר": "Passage", "שרותי נכים": "Accessible Restrooms",
    "חדר עוזרים": "Assistants Room", "חדר מחשבים": "Computer Room",
}

def type_en(t): 
    for heb, eng in TYPE_EN.items():
        if heb in t: return eng
    return t

# מעברים אסורים בין קומות: כל זוג (קומה_א, קומה_ב) שאין ביניהן גישה ישירה.
# הפורמט: frozenset כדי שהכיוון לא משנה (4↔2 אותו הדבר כמו 2↔4).
FORBIDDEN_FLOOR_JUMPS = {
    frozenset({4, 2}),  # אין מעבר ישיר בין קומה 4 ל-2 — חייבים לעבור דרך 3
    frozenset({4, 1}),  # אין מעבר ישיר בין קומה 4 ל-1
    frozenset({5, 1}),  # אין מעבר ישיר בין קומה 5 ל-1
    frozenset({5, 2}),  # אין מעבר ישיר בין קומה 5 ל-2
    frozenset({6, 1}),
    frozenset({6, 2}),
    frozenset({6, 3}),
}

def _floor_jump_allowed(u_id, v_id):
    """מחזיר True אם המעבר בין שני חדרים מותר מבחינת קומות."""
    u_node = NODES.get(u_id, {}); v_node = NODES.get(v_id, {})
    u_floor = u_node.get("floor_num") or u_node.get("floor")
    v_floor = v_node.get("floor_num") or v_node.get("floor")
    if u_floor is None or v_floor is None: return True          # אין מידע — מאפשרים
    if u_floor == v_floor: return True                           # אותה קומה — תמיד בסדר
    try:
        pair = frozenset({int(u_floor), int(v_floor)})
    except (ValueError, TypeError):
        return True
    return pair not in FORBIDDEN_FLOOR_JUMPS

def dijkstra(start, end):
    dist={start:0}; prev={}; pq=[(0,start)]
    while pq:
        d,u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")): continue
        if u == end: break
        for v,w,et in ADJ.get(u,[]):
            # דלג על מעברים אסורים בין קומות
            if et in ("stairs", "elevator") and not _floor_jump_allowed(u, v):
                continue
            nd = d+w
            if nd < dist.get(v, float("inf")):
                dist[v]=nd; prev[v]=(u,et); heapq.heappush(pq,(nd,v))
    if end not in dist: return None, None
    path=[]; cur=end
    while cur in prev: path.append(cur); cur=prev[cur][0]
    path.append(start); path.reverse()
    etypes=[]
    for i in range(len(path)-1):
        for v,w,et in ADJ.get(path[i],[]):
            if v==path[i+1]: etypes.append(et); break
    return path, etypes

def instructions(path, etypes):
    if not path: return []
    steps=[]
    n=NODES.get(path[0],{}); rt=type_en(n.get("type",""))
    steps.append({"text": f"Start at Room {path[0]}" + (f" ({rt})" if rt else ""), "type":"start","room":path[0]})
    for i,et in enumerate(etypes):
        nxt=path[i+1]; n=NODES.get(nxt,{}); rt=type_en(n.get("type","")); fn=n.get("floor_num",""); b=n.get("building","")
        if et=="horizontal":
            steps.append({"text": f"Go to Room {nxt}" + (f" ({rt})" if rt else "") + (f" — Building {b}" if b else ""), "type":"walk","room":nxt})
        elif et=="stairs":
            steps.append({"text": f"Take the stairs to Floor {fn}", "type":"stairs","room":nxt})
        elif et=="elevator":
            steps.append({"text": f"Take the elevator to Floor {fn}", "type":"elevator","room":nxt})
    n=NODES.get(path[-1],{}); rt=type_en(n.get("type",""))
    steps.append({"text": f"You have arrived! Room {path[-1]}" + (f" ({rt})" if rt else ""), "type":"arrived","room":path[-1]})
    return steps

@app.get("/health")
def health():
    return {"status":"ok","rooms":len(NODES),"edges":len(GRAPH["edges"])}

STAIR_TYPES = ("מדרגות", "מעלית", "פיר")

@app.get("/api/rooms")
def get_rooms(q:str=""):
    seen=set(); results=[]
    for n in GRAPH["nodes"]:
        if n["id"] in seen: continue
        seen.add(n["id"])
        raw_type = n.get("type","")
        if any(s in raw_type for s in STAIR_TYPES): continue
        rt=type_en(raw_type)
        if q and q.lower() not in n["id"].lower() and q.lower() not in rt.lower(): continue
        floor_val = n.get("floor_num") or n.get("floor","")
        results.append({"id":n["id"],"type":rt,"building":n.get("building",""),"floor":floor_val,"floor_num":floor_val})
    return results

@app.get("/api/rooms/{room_id}")
def get_room(room_id:str):
    n=NODES.get(room_id)
    if not n: raise HTTPException(404,f"Room {room_id} not found")
    return {**n,"type_en":type_en(n.get("type",""))}

@app.get("/api/route")
def get_route(from_room:str, to_room:str):
    if from_room not in NODES: raise HTTPException(404,f"Room '{from_room}' not found")
    if to_room   not in NODES: raise HTTPException(404,f"Room '{to_room}' not found")
    if from_room==to_room:
        return {"path":[from_room],"steps":[{"text":"You are already here!","type":"arrived","room":from_room}],"num_steps":0}
    path,etypes=dijkstra(from_room,to_room)
    if path is None: raise HTTPException(404,"No route found between these rooms")
    return {"path":path,"steps":instructions(path,etypes),"num_steps":len(path)-1,
            "from":{"id":from_room,**NODES.get(from_room,{})},"to":{"id":to_room,**NODES.get(to_room,{})}}

@app.get("/api/floors/{floor_num}/rooms")
def get_floor_rooms(floor_num:int):
    rooms=[{**n,"type_en":type_en(n.get("type",""))} for n in GRAPH["nodes"] if n.get("floor_num")==floor_num]
    if not rooms: raise HTTPException(404,f"Floor {floor_num} not found")
    return {"floor":floor_num,"rooms":rooms}


# ── Serve frontend ────────────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(BASE, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Frontend not found — place index.html next to main.py</h1>", 404)


# ── Map data endpoint ─────────────────────────────────────────────────────────
import math as _math

@app.get("/api/map/{floor_num}")
def get_map(floor_num: int):
    """Returns rooms + corridor lines for a given floor."""
    floor_nodes = [n for n in GRAPH["nodes"] if n.get("floor") == floor_num]
    if not floor_nodes:
        raise HTTPException(404, f"Floor {floor_num} not found")

    corridors = []
    seen = set()
    for e in GRAPH["edges"]:
        if e["type"] != "horizontal":
            continue
        a = NODES.get(e["from"])
        b = NODES.get(e["to"])
        if not a or not b:
            continue
        if a.get("floor") != floor_num or b.get("floor") != floor_num:
            continue
        key = tuple(sorted([e["from"], e["to"]]))
        if key in seen:
            continue
        seen.add(key)
        corridors.append({
            "x1": a["door_x"], "y1": a["door_y"],
            "x2": b["door_x"], "y2": b["door_y"],
            "from": e["from"], "to": e["to"]
        })

    all_x = [n["door_x"] for n in floor_nodes] + [n["x"] for n in floor_nodes]
    all_y = [n["door_y"] for n in floor_nodes] + [n["y"] for n in floor_nodes]

    rooms_out = []
    for n in floor_nodes:
        rooms_out.append({**n, "type_en": type_en(n.get("type", ""))})

    return {
        "floor":     floor_num,
        "rooms":     rooms_out,
        "corridors": corridors,
        "bounds": {
            "x_min": min(all_x) - 200,
            "x_max": max(all_x) + 200,
            "y_min": min(all_y) - 200,
            "y_max": max(all_y) + 200,
        }
    }


# ── Floor wall geometry endpoint ──────────────────────────────────────────────
_floor_walls_path = os.path.join(BASE, "floor_walls.json")
print(f"[startup] BASE={BASE}")
print(f"[startup] floor_walls path={_floor_walls_path}, exists={os.path.exists(_floor_walls_path)}")
with open(_floor_walls_path, encoding="utf-8") as _f:
    FLOOR_WALLS = json.load(_f)
print(f"[startup] FLOOR_WALLS keys={list(FLOOR_WALLS.keys())}")

@app.get("/api/floormap/{floor_num}")
def get_floormap(floor_num: int):
    """Returns real wall geometry from DXF for a floor."""
    data = FLOOR_WALLS.get(str(floor_num))
    if data is None:
        raise HTTPException(404, f"Floor {floor_num} not found in floor_walls.json")

    floor_nodes = [
        {**n, "type_en": type_en(n.get("type", ""))}
        for n in GRAPH["nodes"] if n.get("floor") == floor_num
    ]

    walls_b21 = data.get("b21", [])
    walls_b22 = data.get("b22", [])
    all_walls  = walls_b21 + walls_b22

    # Build bounds from walls; fallback to node coordinates if walls are empty
    if all_walls:
        all_x = [l["start"][0] for l in all_walls] + [l["end"][0] for l in all_walls]
        all_y = [l["start"][1] for l in all_walls] + [l["end"][1] for l in all_walls]
    elif floor_nodes:
        all_x = [n["door_x"] for n in floor_nodes if "door_x" in n] + [n.get("x",0) for n in floor_nodes]
        all_y = [n["door_y"] for n in floor_nodes if "door_y" in n] + [n.get("y",0) for n in floor_nodes]
    else:
        all_x = []; all_y = []

    return {
        "floor":     floor_num,
        "rooms":     floor_nodes,
        "walls_b21": walls_b21,
        "walls_b22": walls_b22,
        "bounds": {
            "x_min": min(all_x) - 100 if all_x else -2500,
            "x_max": max(all_x) + 100 if all_x else 6800,
            "y_min": min(all_y) - 100 if all_y else 1000,
            "y_max": max(all_y) + 100 if all_y else 6200,
        }
    }
@app.get("/api/route2")
def get_route2(from_room: str, to_room: str):
    """Corridor-based routing: room → corridor → corridor → room"""
    if from_room not in NODES: raise HTTPException(404, f"Room '{from_room}' not found")
    if to_room   not in NODES: raise HTTPException(404, f"Room '{to_room}' not found")
    if from_room == to_room:
        return {"path":[],"corridor_path":[],"steps":[{"text":"You are already here!","type":"arrived","room":from_room}],"num_steps":0}

    start_cn = _room_to_cnode(from_room)
    end_cn   = _room_to_cnode(to_room)
    if not start_cn: raise HTTPException(404, f"Room '{from_room}' not mapped to corridor")
    if not end_cn:   raise HTTPException(404, f"Room '{to_room}' not mapped to corridor")

    cpath = _dijkstra_corridor(start_cn, end_cn)
    if cpath is None: raise HTTPException(404, "No corridor route found")

    # Enrich corridor path with coordinates
    corridor_coords = [
        {"id": cn, "x": _CNODE_MAP[cn]["x"], "y": _CNODE_MAP[cn]["y"],
         "floor": _CNODE_MAP[cn]["floor"], "rooms": _CNODE_MAP[cn].get("rooms",[])}
        for cn in cpath if cn in _CNODE_MAP
    ]

    # Count floor changes
    floors_seen = []
    for cn in cpath:
        f = _CNODE_MAP.get(cn,{}).get("floor")
        if f and (not floors_seen or floors_seen[-1]!=f):
            floors_seen.append(f)

    return {
        "from_room":      from_room,
        "to_room":        to_room,
        "corridor_path":  corridor_coords,
        "num_corridors":  len(cpath),
        "floor_changes":  len(floors_seen)-1,
        "steps":          _build_corridor_instructions(from_room, to_room, cpath),
    }


# ── Polygon-based corridor routing ───────────────────────────────────────────
from collections import defaultdict as _defdict2

_c2_path = os.path.join(BASE, "corridor2.json")
try:
    with open(_c2_path, encoding="utf-8") as _f:
        C2 = json.load(_f)
except FileNotFoundError:
    C2 = {"floors": {}, "stair_links": [], "cross_building_links": []}

# Build adjacency
_C2ADJ = _defdict2(list)
for _key, _data in C2["floors"].items():
    _bldg, _fl = _key.split("_"); _fl = int(_fl)
    for _e in _data["edges"]:
        _a = (_bldg, _fl, _e["from"])
        _b = (_bldg, _fl, _e["to"])
        _C2ADJ[_a].append((_b, _e["dist"]))
        _C2ADJ[_b].append((_a, _e["dist"]))

for _sl in C2["stair_links"]:
    _a = tuple(_sl["from"]); _b = tuple(_sl["to"])
    _C2ADJ[_a].append((_b, _sl["cost"]))
    _C2ADJ[_b].append((_a, _sl["cost"]))

for _cl in C2.get("cross_building_links", []):
    _a = tuple(_cl["from"]); _b = tuple(_cl["to"])
    _C2ADJ[_a].append((_b, _cl["cost"]))
    _C2ADJ[_b].append((_a, _cl["cost"]))

def _get_room_cpt(room_id):
    n = NODES.get(room_id)
    if not n: return None
    key = f"{n.get('building','21')}_{n.get('floor','')}"
    idx = C2["floors"].get(key, {}).get("room_to_cpt", {}).get(room_id)
    if idx is None: return None
    return (n.get("building","21"), n.get("floor", 0), idx)

def _dijkstra_c2(start, end):
    dist = {start: 0}; prev = {}; pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")): continue
        if u == end: break
        for v, w in _C2ADJ.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if end not in dist: return None
    path = []; cur = end
    while cur in prev: path.append(cur); cur = prev[cur]
    path.append(start); path.reverse()
    return path

@app.get("/api/route3")
def get_route3(from_room: str, to_room: str):
    """Corridor-polygon-based routing through actual corridor space."""
    if from_room not in NODES: raise HTTPException(404, f"Room '{from_room}' not found")
    if to_room   not in NODES: raise HTTPException(404, f"Room '{to_room}' not found")
    if from_room == to_room:
        return {"corridor_path": [], "steps": [{"text":"You are already here!","type":"arrived","room":from_room}], "num_corridors": 0}

    start = _get_room_cpt(from_room)
    end   = _get_room_cpt(to_room)
    if not start: raise HTTPException(404, f"No corridor point for '{from_room}'")
    if not end:   raise HTTPException(404, f"No corridor point for '{to_room}'")

    path = _dijkstra_c2(start, end)
    if path is None: raise HTTPException(404, "No corridor route found")

    # Build coordinate path
    corridor_coords = []
    prev_floor = None
    for bldg, floor, idx in path:
        key = f"{bldg}_{floor}"
        pts = C2["floors"].get(key, {}).get("pts", [])
        if idx < len(pts):
            px, py = pts[idx]
            corridor_coords.append({"x": px, "y": py, "floor": floor, "building": bldg})

    # Count floor changes
    floors_seq = []
    for c in corridor_coords:
        if not floors_seq or floors_seq[-1] != c["floor"]:
            floors_seq.append(c["floor"])
    floor_changes = len(floors_seq) - 1

    # Build steps
    fr_node = NODES.get(from_room, {}); tr_node = NODES.get(to_room, {})
    steps = [{"text": f"Start at Room {from_room}" + (f" ({type_en(fr_node.get('type',''))})" if fr_node.get('type') else ""), "type":"start","room":from_room}]
    prev_f = fr_node.get("floor")
    for c in corridor_coords:
        if c["floor"] != prev_f:
            direction = "up" if c["floor"] > prev_f else "down"
            steps.append({"text": f"Take stairs {direction} to Floor {c['floor']}", "type":"stairs","room":None,"floor":c["floor"]})
            prev_f = c["floor"]
    steps.append({"text": f"You have arrived! Room {to_room}" + (f" ({type_en(tr_node.get('type',''))})" if tr_node.get('type') else ""), "type":"arrived","room":to_room})

    return {
        "from_room": from_room, "to_room": to_room,
        "corridor_path": corridor_coords,
        "num_corridors": len(corridor_coords),
        "floor_changes": floor_changes,
        "steps": steps
    }


# ── Rasterized corridor routing ───────────────────────────────────────────────
_c3_path = os.path.join(BASE, "corridor3.json")
try:
    with open(_c3_path, encoding="utf-8") as _f:
        C3 = json.load(_f)
except FileNotFoundError:
    C3 = {"floors": {}, "stair_links": [], "cross_building_links": []}

# Base adjacency: intra-floor edges + stair links (no cross-building links).
# Used for same-building routes so the path never detours through the other building.
_C3ADJ_SAME = _defaultdict(list)
for _key, _data in C3["floors"].items():
    _bldg, _fl = _key.split("_"); _fl = int(_fl)
    for _e in _data["edges"]:
        _a = (_bldg, _fl, _e["from"]); _b = (_bldg, _fl, _e["to"])
        _C3ADJ_SAME[_a].append((_b, _e["dist"]))
        _C3ADJ_SAME[_b].append((_a, _e["dist"]))
for _sl in C3["stair_links"]:
    _a=tuple(_sl["from"]); _b=tuple(_sl["to"])
    _C3ADJ_SAME[_a].append((_b, _sl["cost"]))
    _C3ADJ_SAME[_b].append((_a, _sl["cost"]))

# Add the missing stair link for building 22 floor 1 → floor 2.
# 22_1 stairwell (22150/22151) maps to cpt_idx 702; 22_2 maps to cpt_idx 597.
_missing_stair_a = ("22", 1, 702); _missing_stair_b = ("22", 2, 597)
_C3ADJ_SAME[_missing_stair_a].append((_missing_stair_b, 600))
_C3ADJ_SAME[_missing_stair_b].append((_missing_stair_a, 600))

# Cross-building adjacency: same as above plus ONE cheapest cross-building link
# per from-node (avoids the full 113 k Cartesian product while keeping connectivity).
_C3ADJ_CROSS = _defaultdict(list, {k: list(v) for k, v in _C3ADJ_SAME.items()})
_min_cross: dict = {}  # exposed for /api/debug/route; from-node → (to-node, cost)
for _cl in C3.get("cross_building_links", []):
    _a = tuple(_cl["from"]); _b = tuple(_cl["to"])
    if _a not in _min_cross or _cl["cost"] < _min_cross[_a][1]:
        _min_cross[_a] = (_b, _cl["cost"])
    if _b not in _min_cross or _cl["cost"] < _min_cross[_b][1]:
        _min_cross[_b] = (_a, _cl["cost"])
for _a, (_b, _cost) in _min_cross.items():
    _C3ADJ_CROSS[_a].append((_b, _cost))
    _C3ADJ_CROSS[_b].append((_a, _cost))

# ── Stairwell direct-access shortcuts ─────────────────────────────────────────
# The rasterised corridor graph sometimes routes to a stairwell via a large loop
# through the stairwell lobby (a big open area with few walls).  We add explicit
# short-cut edges from the nearest main-corridor node to each stairwell transition
# node so Dijkstra prefers the direct path over the lobby loop.
#
# How each shortcut was derived:
#   1. Inspect C3["floors"][key]["pts"][stair_idx] for the stairwell coord.
#   2. Find the nearest corridor node on the same floor that sits on the main
#      corridor grid (within the "direct reach" strip of the stairwell door).
#   3. Measure the straight-line distance → that becomes the shortcut cost.
#
# (bldg, floor, from_idx) ↔ (bldg, floor, to_idx), cost
_STAIR_SHORTCUTS = [
    # Floor 4 bldg21: stairwell #1824 (1660,3800) ↔ main corridor pt #473 (1660,3320)
    # Bypasses the lobby loop that previously went through the open stairwell area.
    (("21", 4, 1824), ("21", 4, 473), 480),
    # Floor 5 bldg21: stairwell #2022 (940,3240) already well-connected; add safety
    (("21", 5, 2022), ("21", 5, 406), 320),
    # Floor 5 bldg21: stairwell #1221 (2140,3800) ↔ nearest pt #898 (2140,3320)
    (("21", 5, 1221), ("21", 5, 898), 480),
    # Floor 6 bldg21: stairwell #1658 (1420,3560) ↔ nearest pt #390 (1420,3080)
    (("21", 6, 1658), ("21", 6, 390), 480),
]
for _sa, _sb, _sc in _STAIR_SHORTCUTS:
    for _adj2 in [_C3ADJ_SAME, _C3ADJ_CROSS]:
        _adj2[_sa].append((_sb, _sc))
        _adj2[_sb].append((_sa, _sc))

# Keep backward-compat alias (used by /api/route3 and tests)
_C3ADJ = _C3ADJ_SAME

def _get_c3_node(room_id, floor_hint=None):
    """Return (building, floor, cpt_idx) for a room.

    floor_hint: if given, used instead of NODES lookup — needed for stairwell
    rooms whose ID appears on multiple floors (NODES dict keeps only last one).
    """
    n = NODES.get(room_id)
    if not n: return None
    bldg = n.get("building", "21")
    # If caller supplies a floor override (e.g. for a stairwell), honour it.
    floor = floor_hint if floor_hint is not None else n.get("floor", "")
    key = f"{bldg}_{floor}"
    idx = C3["floors"].get(key, {}).get("room_to_cpt", {}).get(room_id)
    if idx is None: return None
    return (bldg, int(floor), int(idx))

def _directional_steps(from_room, to_room, corridor_coords):
    """Generate turn-by-turn walking directions from corridor path."""
    if not corridor_coords:
        return []
    fr_n = NODES.get(from_room, {}); tr_n = NODES.get(to_room, {})

    # Group by floor
    segs = []
    for pt in corridor_coords:
        if not segs or segs[-1]["floor"] != pt["floor"]:
            segs.append({"floor": pt["floor"], "pts": []})
        segs[-1]["pts"].append(pt)

    # Room lookup per floor — only meaningful destination rooms (no circulation/utility)
    SKIP_T = ("מדרגות", "מעלית", "פיר", "פרוזדור", "מעבר", "מבוא", "פויאה", "פויה", "מרפסת", "שירותים", "שרותי נכים", "מחסן")
    by_floor = {}
    for n in GRAPH["nodes"]:
        fl = n.get("floor")
        if fl not in by_floor: by_floor[fl] = []
        if not any(t in n.get("type","") for t in SKIP_T) and n.get("type",""):
            by_floor[fl].append(n)

    def heading(dx, dy):  # degrees, 0=east, 90=north
        return math.degrees(math.atan2(dy, dx)) % 360

    def cardinal(deg):
        names = ["east","north-east","north","north-west","west","south-west","south","south-east"]
        return names[round(deg/45) % 8]

    def rooms_near_segment(start_pt, end_pt, floor):
        sx,sy = start_pt["x"],start_pt["y"]; ex,ey = end_pt["x"],end_pt["y"]
        seg2 = (ex-sx)**2+(ey-sy)**2
        if seg2 < 1: return []
        near=[]
        for r in by_floor.get(floor,[]):
            rx,ry = r.get("door_x",r.get("x",0)), r.get("door_y",r.get("y",0))
            t = max(0,min(1,((rx-sx)*(ex-sx)+(ry-sy)*(ey-sy))/seg2))
            d = math.hypot(rx-(sx+t*(ex-sx)), ry-(sy+t*(ey-sy)))
            if d < 250: near.append((d, r["id"]))
        near.sort(); return [r for _,r in near if r not in (from_room,to_room)][:3]

    steps = [{"text": f"Start at Room {from_room}" + (f" ({type_en(fr_n.get('type',''))})" if fr_n.get('type') else ""), "type":"start","room":from_room}]

    for si, seg in enumerate(segs):
        pts = seg["pts"]; floor = seg["floor"]

        if len(pts) >= 2:
            # Initial direction
            dx = pts[1]["x"]-pts[0]["x"]; dy = pts[1]["y"]-pts[0]["y"]
            hd = heading(dx, dy)
            dir_word = "Continue" if si > 0 else "Head"
            near = rooms_near_segment(pts[0], pts[min(6,len(pts)-1)], floor)
            room_txt = f", past rooms {', '.join(near)}" if near else ""
            steps.append({"text": f"{dir_word} {cardinal(hd)}{room_txt}", "type":"walk","room":None,"floor":floor})

            # Detect significant turns (> 35 deg) along simplified waypoints
            sparse = pts[::5] + [pts[-1]]
            for i in range(1, len(sparse)-1):
                p=sparse[i-1]; c=sparse[i]; n=sparse[i+1]
                if p["floor"]!=c["floor"] or c["floor"]!=n["floor"]: continue
                dx1=c["x"]-p["x"]; dy1=c["y"]-p["y"]
                dx2=n["x"]-c["x"]; dy2=n["y"]-c["y"]
                cross=dx1*dy2-dy1*dx2
                dot=dx1*dx2+dy1*dy2
                mag=max(1,(dx1**2+dy1**2)**0.5*(dx2**2+dy2**2)**0.5)
                ang=math.degrees(math.acos(max(-1,min(1,dot/mag))))
                if ang < 35: continue
                turn = "right" if cross < 0 else "left"
                nd=heading(dx2,dy2)
                near_t=rooms_near_segment(c,n,floor)
                rt=f", past rooms {', '.join(near_t)}" if near_t else ""
                steps.append({"text":f"Turn {turn} ({cardinal(nd)}){rt}","type":"walk","room":None,"floor":floor})

        # Floor change to next segment
        if si < len(segs)-1:
            nf=segs[si+1]["floor"]
            ud="up" if nf>floor else "down"
            # Check if this floor change uses elevator
            last_pts = segs[si]["pts"]
            use_elev = any(p.get("is_elevator") for p in last_pts)
            # For elevator: skip intermediate stops, jump to final floor in this elevator run
            if use_elev:
                final_floor = nf
                j = si + 1
                while j < len(segs) - 1 and any(p.get("is_elevator") for p in segs[j]["pts"]):
                    final_floor = segs[j+1]["floor"]
                    j += 1
                if final_floor != nf:
                    # Will be handled when we reach the last elevator seg — skip intermediate
                    pass
                else:
                    steps.append({"text":f"Take elevator {ud} to Floor {final_floor}","type":"elevator","room":None,"floor":final_floor})
            else:
                steps.append({"text":f"Take stairs {ud} to Floor {nf}","type":"stairs","room":None,"floor":nf})

    to_type=type_en(tr_n.get("type",""))
    steps.append({"text": f"You have arrived at Room {to_room}" + (f" ({to_type})" if to_type else ""), "type":"arrived","room":to_room})
    return steps


# ── Stairwell door lookup (for snapping floor-change corridor pts) ─────────────
# Build: (bldg, floor) → list of (door_x, door_y, room_id) for stairwell rooms.
_STAIR_DOOR_BY_FLOOR: dict = {}
for _sn in GRAPH["nodes"]:
    # Include stairwells ("מדרגות"), elevators ("מעלית"), and shafts ("פיר")
    if not any(t in _sn.get("type", "") for t in ("מדרגות", "מעלית", "פיר")):
        continue
    _key = (_sn.get("building", "21"), int(_sn.get("floor", 0)))
    _dx = _sn.get("door_x"); _dy = _sn.get("door_y")
    if _dx is None or _dy is None: continue
    _STAIR_DOOR_BY_FLOOR.setdefault(_key, []).append((_dx, _dy, _sn["id"]))

def _snap_floor_changes_to_stairwell(wire_coords: list) -> list:
    """Annotate floor-change / floor-start corridor points with the nearest
    real stairwell door position.

    We do NOT move the wire point's x/y — the corridor node is already inside
    the stairwell lobby and moving it would create a large visual jump that can
    cross walls.  Instead we add stair_door_x / stair_door_y as metadata so
    the frontend can draw the ↕ marker exactly at the door without displacing
    the corridor wire.
    """
    result = []
    for pt in wire_coords:
        entry = dict(pt)
        if pt.get("is_floor_change") or pt.get("is_floor_start"):
            key = (pt["building"], pt["floor"])
            candidates = _STAIR_DOOR_BY_FLOOR.get(key, [])
            if candidates:
                best = min(candidates,
                           key=lambda c: math.hypot(c[0]-pt["x"], c[1]-pt["y"]))
                dist_to_best = math.hypot(best[0]-pt["x"], best[1]-pt["y"])
                if dist_to_best < 600:
                    # annotate but keep original wire coordinates
                    entry["stair_room"]   = best[2]
                    entry["stair_door_x"] = best[0]
                    entry["stair_door_y"] = best[1]
        result.append(entry)
    return result


def _snap_to_corridor(door_x, door_y, bldg, floor_num, k=5, max_snap=700):
    """Return [(dist, corridor_idx), ...] — K nearest corridor pts to a door."""
    import math as _math
    key = f"{bldg}_{floor_num}"
    pts = C3["floors"].get(key, {}).get("pts", [])
    heap = []
    for idx, (px, py) in enumerate(pts):
        d = _math.sqrt((px - door_x)**2 + (py - door_y)**2)
        if d <= max_snap:
            heapq.heappush(heap, (d, idx))
    result = []
    while heap and len(result) < k:
        result.append(heapq.heappop(heap))
    return result  # [(dist, idx), ...]


def _smooth_corridor(coords, tol=15, max_jump=120):
    """Remove near-collinear intermediate corridor points.

    Parameters chosen conservatively:
    - tol=15  : only remove a midpoint if it deviates < 15 units from the straight
                line — prevents cutting corners around walls.
    - max_jump=120: never create a segment longer than ~1.5 grid cells (grid=80 u)
                so even after simplification the line stays inside the corridor.
    Door connectors, floor-change and floor-start points are always preserved.
    """
    import math as _m
    if len(coords) <= 2: return coords
    result = [coords[0]]
    for i in range(1, len(coords)-1):
        p = result[-1]; c = coords[i]; n = coords[i+1]
        # Never skip floor-change or door marker points
        if (p.get("floor") != c.get("floor") or c.get("floor") != n.get("floor")
                or c.get("is_floor_change") or c.get("is_floor_start") or c.get("is_door")):
            result.append(c); continue
        jump = _m.sqrt((n["x"]-p["x"])**2 + (n["y"]-p["y"])**2)
        if jump > max_jump:
            result.append(c); continue
        dx1=c["x"]-p["x"]; dy1=c["y"]-p["y"]
        dx2=n["x"]-c["x"]; dy2=n["y"]-c["y"]
        cross = abs(dx1*dy2 - dy1*dx2)
        length = max(1, _m.sqrt(dx1**2+dy1**2))
        if cross/length > tol:
            result.append(c)
    result.append(coords[-1])
    return result


ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "campus-admin-2024")

@app.get("/api/admin/stats")
def admin_stats(token: str = ""):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Invalid token")
    with sqlite3.connect(_DB) as c:
        total = c.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        top_routes = c.execute("""SELECT from_room,to_room,COUNT(*) as cnt FROM searches
            GROUP BY from_room,to_room ORDER BY cnt DESC LIMIT 20""").fetchall()
        by_floor = c.execute("""SELECT from_floor,to_floor,COUNT(*) as cnt FROM searches
            WHERE from_floor IS NOT NULL GROUP BY from_floor,to_floor ORDER BY cnt DESC LIMIT 10""").fetchall()
        recent = c.execute("""SELECT from_room,to_room,ts,floor_changes FROM searches
            ORDER BY id DESC LIMIT 50""").fetchall()
    return {
        "total_searches": total,
        "top_routes": [{"from":r[0],"to":r[1],"count":r[2]} for r in top_routes],
        "by_floor": [{"from_floor":r[0],"to_floor":r[1],"count":r[2]} for r in by_floor],
        "recent": [{"from":r[0],"to":r[1],"ts":r[2],"floor_changes":r[3]} for r in recent],
    }


# ── nav_graph2.json routing ───────────────────────────────────────────────────
_ng2_path = os.path.join(BASE, "nav_graph2.json")
with open(_ng2_path, encoding="utf-8") as _f:
    _NG2 = json.load(_f)

_NG2_NODES: dict = {n["id"]: n for n in _NG2["nodes"]}

# room_id → list of door/stair node ids that serve that room
_NG2_ROOMS: dict[str, list[str]] = _defaultdict(list)
for _n in _NG2["nodes"]:
    if _n["type"] in ("DOOR", "STAIR_LANDING") and _n.get("room_id"):
        _NG2_ROOMS[_n["room_id"]].append(_n["id"])

_NG2_ADJ: dict = _defaultdict(list)
_NG2_ELEV_NODES: set = set()  # node IDs that are part of an ELEVATOR edge
for _e in _NG2["edges"]:
    _NG2_ADJ[_e["from"]].append((_e["to"],   _e["cost"]))
    _NG2_ADJ[_e["to"]  ].append((_e["from"], _e["cost"]))
    if _e.get("type") == "ELEVATOR":
        _NG2_ELEV_NODES.add(_e["from"])
        _NG2_ELEV_NODES.add(_e["to"])

# Auto-generate CROSS_BLD edges between CONNECTION nodes across buildings (in-memory only)
_existing_ng2_edges: set = set()
for _e in _NG2["edges"]:
    _existing_ng2_edges.add((_e["from"], _e["to"]))
    _existing_ng2_edges.add((_e["to"], _e["from"]))

_conn_b21: dict = _defaultdict(list)  # floor → list of nodes
_conn_b22: dict = _defaultdict(list)
for _n in _NG2["nodes"]:
    if _n.get("type") == "CONNECTION":
        if _n["building"] == "21":
            _conn_b21[_n["floor"]].append(_n)
        elif _n["building"] == "22":
            _conn_b22[_n["floor"]].append(_n)

_CROSS_BLD_THRESHOLD = 500.0
_cross_bld_added = 0
for _floor in set(_conn_b21) & set(_conn_b22):
    for _na in _conn_b21[_floor]:
        for _nb in _conn_b22[_floor]:
            _dist = _math.hypot(_na["x"] - _nb["x"], _na["y"] - _nb["y"])
            if _dist <= _CROSS_BLD_THRESHOLD:
                if (_na["id"], _nb["id"]) not in _existing_ng2_edges:
                    _NG2_ADJ[_na["id"]].append((_nb["id"], _dist))
                    _NG2_ADJ[_nb["id"]].append((_na["id"], _dist))
                    _existing_ng2_edges.add((_na["id"], _nb["id"]))
                    _existing_ng2_edges.add((_nb["id"], _na["id"]))
                    _cross_bld_added += 1

import logging as _logging
_logging.getLogger(__name__).info(
    "CROSS_BLD auto-edges added: %d (threshold=%g u)", _cross_bld_added, _CROSS_BLD_THRESHOLD
)

# Main connected component (BFS over full graph)
_NG2_MAIN_COMP: set = set()
_visited_mc: set = set()
_start_mc = next(iter(_NG2_NODES))
_queue_mc = [_start_mc]
while _queue_mc:
    _u = _queue_mc.pop()
    if _u in _visited_mc: continue
    _visited_mc.add(_u)
    for _v, _ in _NG2_ADJ.get(_u, []):
        if _v not in _visited_mc: _queue_mc.append(_v)
_NG2_MAIN_COMP = _visited_mc

# Fallback: rooms with no door node → snap to nearest GRID node in main component
_NG2_GRID = [(n["id"], n["x"], n["y"], n["building"], n["floor"])
             for n in _NG2["nodes"] if n["type"] == "GRID" and n["id"] in _NG2_MAIN_COMP]

def _snap_to_grid(room_id: str) -> list[str]:
    """Return nearest main-component GRID node for a room that has no DOOR node."""
    r = NODES.get(room_id)
    if not r: return []
    rx = r.get("door_x") or r.get("x")
    ry = r.get("door_y") or r.get("y")
    bld, flr = r.get("building", ""), int(r.get("floor", 0))
    candidates = sorted(
        [(nid, x, y) for nid, x, y, b, f in _NG2_GRID if b == bld and f == flr],
        key=lambda t: _math.hypot(t[1] - rx, t[2] - ry)
    )
    return [candidates[0][0]] if candidates else []


def _dijkstra_ng2(from_room: str, to_room: str):
    """Dijkstra on nav_graph2 with virtual source/sink for multi-door rooms."""
    src_nodes = _NG2_ROOMS.get(from_room) or _snap_to_grid(from_room)
    dst_nodes = _NG2_ROOMS.get(to_room)   or _snap_to_grid(to_room)
    if not src_nodes or not dst_nodes:
        return None

    V_START = "__start__"
    V_END   = "__end__"

    extra: dict = _defaultdict(list, {V_START: [(nid, 0) for nid in src_nodes]})
    for nid in dst_nodes:
        extra[nid].append((V_END, 0))

    dest_set = set(dst_nodes)

    dist = {V_START: 0}
    prev: dict = {}
    pq = [(0, V_START)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == V_END:
            break
        for v, w in list(_NG2_ADJ.get(u, [])) + extra.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if V_END not in dist:
        return None

    path, cur = [], V_END
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(V_START)
    path.reverse()
    return [n for n in path if n not in (V_START, V_END)]


def _ng2_path_to_corridor(node_ids: list, from_room: str, to_room: str) -> list:
    """Convert nav_graph2 node IDs to a corridor_path list."""
    pts = []
    for nid in node_ids:
        n = _NG2_NODES.get(nid)
        if not n:
            continue
        pt = {"x": n["x"], "y": n["y"], "floor": n["floor"], "building": n["building"]}
        if n["type"] == "DOOR":
            pt["is_door"] = True
            rid = n.get("room_id", "")
            pt["room"] = rid if rid in (from_room, to_room) else rid
        elif n["type"] == "STAIR_LANDING":
            pt["is_stair_landing"] = True
            pt["stair_door_x"] = n["x"]
            pt["stair_door_y"] = n["y"]
            if nid in _NG2_ELEV_NODES:
                pt["is_elevator"] = True
        pts.append(pt)

    for i, pt in enumerate(pts):
        if i == 0:
            continue
        if pts[i - 1]["floor"] != pt["floor"]:
            pts[i - 1]["is_floor_change"] = True
            pt["is_floor_start"] = True

    return pts


@app.get("/api/route5")
def get_route5(from_room: str, to_room: str, request: Request):
    """nav_graph2-based routing."""
    if from_room not in _NG2_ROOMS and not _snap_to_grid(from_room):
        raise HTTPException(404, f"Room '{from_room}' not found in nav_graph2")
    if to_room not in _NG2_ROOMS and not _snap_to_grid(to_room):
        raise HTTPException(404, f"Room '{to_room}' not found in nav_graph2")
    if from_room == to_room:
        return {"corridor_path": [], "steps": [{"text": "You are already here!", "type": "arrived", "room": from_room}], "num_corridors": 0}

    node_path = _dijkstra_ng2(from_room, to_room)
    if node_path is None:
        raise HTTPException(404, "No route found between these rooms")

    corridor_path = _ng2_path_to_corridor(node_path, from_room, to_room)

    floors_seq: list = []
    prev_f = None
    for pt in corridor_path:
        if pt["floor"] != prev_f:
            floors_seq.append(pt["floor"])
            prev_f = pt["floor"]
    floor_changes = len(floors_seq) - 1

    steps = _directional_steps(from_room, to_room, corridor_path)
    _log_search(from_room, to_room, floor_changes, request.client.host if request.client else "")

    return {
        "from_room": from_room,
        "to_room": to_room,
        "corridor_path": corridor_path,
        "num_corridors": len(corridor_path),
        "floor_changes": floor_changes,
        "steps": steps,
    }


@app.get("/api/route4")
def get_route4(from_room: str, to_room: str, request: Request):
    """Rasterized-corridor routing with wire+door-connector architecture."""
    if from_room not in NODES: raise HTTPException(404, f"Room '{from_room}' not found")
    if to_room   not in NODES: raise HTTPException(404, f"Room '{to_room}' not found")
    if from_room == to_room:
        return {"corridor_path":[],"steps":[{"text":"You are already here!","type":"arrived","room":from_room}],"num_corridors":0}

    fr_n = NODES.get(from_room, {}); tr_n = NODES.get(to_room, {})
    from_bldg = fr_n.get("building", "21")
    to_bldg   = tr_n.get("building", "21")
    from_floor = int(fr_n.get("floor", 1))
    to_floor   = int(tr_n.get("floor", 1))

    # Use cross-building adjacency only when the two rooms are in different buildings.
    adj = _C3ADJ_CROSS if from_bldg != to_bldg else _C3ADJ_SAME

    # Door coordinates for from/to rooms
    from_door_x = fr_n.get("door_x") or fr_n.get("x")
    from_door_y = fr_n.get("door_y") or fr_n.get("y")
    to_door_x   = tr_n.get("door_x") or tr_n.get("x")
    to_door_y   = tr_n.get("door_y") or tr_n.get("y")

    if from_door_x is None or from_door_y is None:
        raise HTTPException(404, f"No coordinates for room '{from_room}'")
    if to_door_x is None or to_door_y is None:
        raise HTTPException(404, f"No coordinates for room '{to_room}'")

    # Snap each door to K nearest corridor nodes
    from_snaps = _snap_to_corridor(from_door_x, from_door_y, from_bldg, from_floor)
    to_snaps   = _snap_to_corridor(to_door_x,   to_door_y,   to_bldg,   to_floor)

    # Fallback: if snap fails (e.g. no corridor pts within max_snap), use pre-mapped cpt
    if not from_snaps:
        fs = _get_c3_node(from_room)
        if not fs: raise HTTPException(404, f"No corridor point for '{from_room}'")
        from_snaps = [(0, fs[2])]
        from_bldg_snap, from_floor_snap = fs[0], fs[1]
    else:
        from_bldg_snap, from_floor_snap = from_bldg, from_floor

    if not to_snaps:
        ts = _get_c3_node(to_room)
        if not ts: raise HTTPException(404, f"No corridor point for '{to_room}'")
        to_snaps = [(0, ts[2])]
        to_bldg_snap, to_floor_snap = ts[0], ts[1]
    else:
        to_bldg_snap, to_floor_snap = to_bldg, to_floor

    # Build virtual door nodes and run a single Dijkstra
    DOOR_FROM = ("door", "from")
    DOOR_TO   = ("door", "to")

    # Build augmented adjacency: virtual edges from DOOR_FROM to each from-snap,
    # and from each to-snap to DOOR_TO
    virtual_adj = dict(adj)  # shallow copy; we'll add virtual keys only
    virtual_adj[DOOR_FROM] = [
        ((from_bldg_snap, from_floor_snap, idx), dist_snap)
        for dist_snap, idx in from_snaps
    ]
    for dist_snap, idx in to_snaps:
        node = (to_bldg_snap, to_floor_snap, idx)
        virtual_adj.setdefault(node, list(adj.get(node, [])))
        # avoid mutating the original adjacency lists
        if id(virtual_adj[node]) == id(adj.get(node)):
            virtual_adj[node] = list(adj.get(node, []))
        virtual_adj[node].append((DOOR_TO, dist_snap))

    dist_d = {DOOR_FROM: 0}; prev_d = {}; pq_d = [(0, DOOR_FROM)]
    while pq_d:
        d, u = heapq.heappop(pq_d)
        if d > dist_d.get(u, float("inf")): continue
        if u == DOOR_TO: break
        for v, w in virtual_adj.get(u, []):
            nd = d + w
            if nd < dist_d.get(v, float("inf")):
                dist_d[v] = nd; prev_d[v] = u; heapq.heappush(pq_d, (nd, v))

    if DOOR_TO not in dist_d:
        raise HTTPException(404, "No corridor route found")

    # Reconstruct path, strip virtual door nodes
    raw_path = []; cur = DOOR_TO
    while cur in prev_d: raw_path.append(cur); cur = prev_d[cur]
    raw_path.append(DOOR_FROM); raw_path.reverse()
    # Strip virtual door nodes
    wire_path = [n for n in raw_path if n not in (DOOR_FROM, DOOR_TO)]

    # Build wire_coords from corridor path
    raw_coords = []
    for bldg, floor, idx in wire_path:
        pts = C3["floors"].get(f"{bldg}_{floor}", {}).get("pts", [])
        if int(idx) < len(pts):
            px, py = pts[int(idx)]
            raw_coords.append({"x": px, "y": py, "floor": floor, "building": bldg})

    # Add is_floor_change / is_floor_start flags at floor boundaries
    wire_coords = []
    for i, c in enumerate(raw_coords):
        entry = dict(c)
        if i > 0 and raw_coords[i - 1]["floor"] != c["floor"]:
            wire_coords[-1]["is_floor_change"] = True
            entry["is_floor_start"] = True
        wire_coords.append(entry)

    # Snap floor-change points to nearest real stairwell door
    wire_coords = _snap_floor_changes_to_stairwell(wire_coords)

    # Apply smooth only to wire_coords
    wire_coords_smoothed = _smooth_corridor(wire_coords)

    # Build door connector points
    from_door_pt = {
        "x": from_door_x, "y": from_door_y,
        "floor": from_floor,
        "building": from_bldg,
        "is_door": True, "room": from_room
    }
    to_door_pt = {
        "x": to_door_x, "y": to_door_y,
        "floor": to_floor,
        "building": to_bldg,
        "is_door": True, "room": to_room
    }

    # Handle edge case: door floor differs from first/last wire floor
    if wire_coords_smoothed and from_door_pt["floor"] != wire_coords_smoothed[0]["floor"]:
        from_door_pt["floor"] = wire_coords_smoothed[0]["floor"]
    if wire_coords_smoothed and to_door_pt["floor"] != wire_coords_smoothed[-1]["floor"]:
        to_door_pt["floor"] = wire_coords_smoothed[-1]["floor"]

    corridor_path = [from_door_pt] + wire_coords_smoothed + [to_door_pt]

    floors_seq = []; prev_f = None
    for c in corridor_path:
        if c["floor"] != prev_f: floors_seq.append(c["floor"]); prev_f = c["floor"]
    floor_changes = len(floors_seq) - 1

    steps = _directional_steps(from_room, to_room, corridor_path)
    _log_search(from_room, to_room, floor_changes, request.client.host if request.client else "")

    return {"from_room": from_room, "to_room": to_room,
            "corridor_path": corridor_path,
            "num_corridors": len(corridor_path),
            "floor_changes": floor_changes, "steps": steps,
            "adj_used": "cross_building" if from_bldg != to_bldg else "same_building",
            "snap_debug": {
                "from_snaps": from_snaps[:3],
                "to_snaps": to_snaps[:3],
            }}


@app.get("/api/debug/route")
def debug_route(from_room: str, to_room: str):
    """Debug endpoint: reports graph connectivity stats for a route request."""
    result: dict = {}

    # Room lookup
    fr_n = NODES.get(from_room); tr_n = NODES.get(to_room)
    result["from_room"] = {"id": from_room, "found": fr_n is not None,
                           "building": (fr_n or {}).get("building"), "floor": (fr_n or {}).get("floor")}
    result["to_room"]   = {"id": to_room,   "found": tr_n is not None,
                           "building": (tr_n or {}).get("building"), "floor": (tr_n or {}).get("floor")}

    # Corridor node
    s = _get_c3_node(from_room); e = _get_c3_node(to_room)
    result["from_cpt"] = s; result["to_cpt"] = e

    # Adjacency stats
    from_bldg = (fr_n or {}).get("building","21"); to_bldg = (tr_n or {}).get("building","21")
    adj = _C3ADJ_CROSS if from_bldg != to_bldg else _C3ADJ_SAME
    result["adj_used"] = "cross_building" if from_bldg != to_bldg else "same_building"
    result["adj_from_neighbors"] = len(adj.get(s, [])) if s else 0
    result["adj_to_neighbors"]   = len(adj.get(e, [])) if e else 0

    # Run Dijkstra and report
    if s and e:
        dist = {s: 0}; prev = {}; pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")): continue
            if u == e: break
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if e in dist:
            path = []; cur = e
            while cur in prev: path.append(cur); cur = prev[cur]
            path.append(s); path.reverse()
            buildings_in_path = sorted(set(p[0] for p in path))
            floors_in_path    = [f for i,f in enumerate(p[1] for p in path)
                                   if i==0 or list(p[1] for p in path)[i-1]!=f]
            result["route_found"]      = True
            result["total_cost"]       = dist[e]
            result["num_pts"]          = len(path)
            result["buildings_in_path"] = buildings_in_path
            result["floor_sequence"]   = list(dict.fromkeys(p[1] for p in path))
            result["crosses_buildings"] = len(buildings_in_path) > 1
        else:
            result["route_found"] = False

    # Snap points for from/to rooms
    if fr_n:
        from_door_x = fr_n.get("door_x") or fr_n.get("x")
        from_door_y = fr_n.get("door_y") or fr_n.get("y")
        result["from_snaps"] = _snap_to_corridor(from_door_x, from_door_y, from_bldg, (fr_n or {}).get("floor", 1))
    if tr_n:
        to_door_x = tr_n.get("door_x") or tr_n.get("x")
        to_door_y = tr_n.get("door_y") or tr_n.get("y")
        result["to_snaps"] = _snap_to_corridor(to_door_x, to_door_y, to_bldg, (tr_n or {}).get("floor", 1))

    # Duplicate node IDs
    from collections import Counter
    id_counts = Counter(n["id"] for n in GRAPH["nodes"])
    dups = {k: v for k, v in id_counts.items() if v > 1}
    result["duplicate_ids_in_graph"] = list(dups.keys())

    # Stair link summary
    result["stair_links_count"] = len(C3["stair_links"])
    result["cross_bridge_links_count"] = len(_min_cross)

    return result
