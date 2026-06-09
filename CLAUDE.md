# Campus Navigation — Project Context for Claude

## Quick start
```bash
cd backend && uvicorn main:app --reload   # runs on http://localhost:8000
```

---

## File map (one line each)

| File | Purpose |
|---|---|
| `data/21/02011–02016.dxf` | Building 21 floor plans, floors 1–6; source of truth |
| `data/22/02022–02026.dxf` | Building 22 floor plans, floors 2–6; **local coordinates** (need +offset for global frame) |
| `data/201/–207/*.dwg` | Original binary DWG source files (unreadable by ezdxf) |
| `backend/graph.json` | 283 room nodes + 872 edges; each node has `door_x/door_y` (nav) and `x/y` (centroid, map only) |
| `backend/floor_walls.json` | Wall segments per floor/building in **global** coordinates; rendering only |
| `backend/corridor3.json` | 80-unit raster walkable grid (~600 k nodes); used by the live `/api/route4` endpoint |
| `backend/corridor2.json` | Older polygon-based corridor attempt; not used |
| `backend/corridor_graph.json` | Older hand-built graph; not used |
| `backend/main.py` | FastAPI server; `/api/route4` is the live endpoint (Dijkstra on corridor3 + virtual door snap) |
| `backend/index.html` | Single-page frontend; calls `/api/route4`; draws walls, corridor wire, door connectors, stair markers |
| `backend/analytics.db` | SQLite log of every route search |
| `backend/door_validation.json` | Machine-readable validation output (OK / WARN / MISMATCH per door position) |
| `docs/validation-and-proposal.md` | Full validation report + approved door-junction graph proposal |

---

## Architecture decision — adopt door-junction graph model

**Decision: approved.** Replace the current 600 k-node raster (`corridor3.json`) with a small, explicit door-junction graph stored in a new file (`nav_graph.json`).

### Node types
- `DOOR` — room entrance threshold; references which rooms open here (`room_ids[]`)
- `JUNCTION` — corridor intersection / T-junction / turn
- `STAIR_LANDING` — one node per floor per stairwell/elevator shaft

### Edge types
- `CORRIDOR` — walkable hallway segment between two JUNCTION (or DOOR/JUNCTION) nodes
- `DOOR_LINK` — short link from a DOOR node to the nearest JUNCTION
- `STAIR` — floor-to-floor link between two STAIR_LANDING nodes

### Room record
Room stores `door_nodes: [node_id, …]` (1 or more). Centroid kept for rendering only, never for routing.

### Why
- Current raster misses walls < 80 u thick → paths cross walls visually
- No multi-door room support
- Stairwell IDs collide across floors in the NODES dict
- 600 k nodes vs ~1 000 nodes → routing is orders of magnitude faster
- JUNCTION nodes make verbal directions exact ("turn right at junction J-21-F4-012")

Full schema and trade-off table: `docs/validation-and-proposal.md §3`

---

## Known issues to fix before graph migration

### 1. Two genuine Building 21 door coordinate errors
| Room | Stored door | Nearest DXF door | Gap | Status |
|---|---|---|---|---|
| `21202` | (1599.6, 2385.7) | (2126.1, 2193.5) | ~560 u | ❌ Not fixed |
| `21648` | (1771.9, 4091.5) | (1608.8, 3489.6) | ~624 u | ❌ Not fixed |

Action: open `data/21/02012.dxf` (floor 2) and `data/21/02016.dxf` (floor 6), find the `Door` layer entity nearest to the correct room polygon, update `graph.json`.

### 2. Building 22 coordinate system — NOT an error
Building 22 was deliberately translated ~**−2 530 units in X** (small or zero Y offset) when merging into the shared campus model. `graph.json` and `floor_walls.json` store Building 22 in this global frame. The raw DXF files (`data/22/`) are still in local coordinates. **Do not treat the large DXF-vs-stored distances as errors.** To validate Building 22 against its DXF, apply the local→global transform first.

### 3. Systematic ~63 u offset on all Building 21 floor-6 rooms
All rooms `21601–21648` show a consistent 63-unit gap between stored `door_x/door_y` and the DXF Door entity. Likely an extraction-time offset (different block reference origin in `02016.dxf`). **Not yet corrected; needs investigation.**

---

## Routing implementation notes (current, pre-migration)

- `NODES` dict in `main.py` keeps only the **first** graph.json occurrence of each room ID (important: stairwells repeat across floors; first = lowest floor).
- `_C3ADJ_SAME` (no cross-building links) used for same-building routes; `_C3ADJ_CROSS` (min-cost cross-building link per node) for cross-building.
- Manual stairwell shortcuts added at startup to bypass outdoor lobby loops on floors 4–6:
  - `("21",4,1824) ↔ ("21",4,473)` cost 480
  - `("21",5,2022) ↔ ("21",5,406)` cost 320
  - `("21",5,1221) ↔ ("21",5,898)` cost 480
  - `("21",6,1658) ↔ ("21",6,390)` cost 480
- Missing stair link `("22",1,702) ↔ ("22",2,597)` added at startup.

---

## ⚠ Nav-graph rebuild in progress

See **`docs/rebuild-handoff.md`** for full state.

Next step: execute the five-fix rebuild of `scripts/build_nav_graph.py` — **Fix 1 (skeleton-adjacency edges, not MST) is the prerequisite**. The current `backend/nav_graph.json` has 1,337 wall-safe edges but 1,171 disconnected components; it cannot route anything. `/api/route4` (corridor3.json raster) remains the live fallback. `/api/route5` is wired correctly in `main.py` and `index.html` and will work once `nav_graph.json` is rebuilt.
- `_snap_floor_changes_to_stairwell()` annotates floor-change corridor nodes with `stair_door_x/y` metadata (from graph.json stairwell rooms); the frontend draws the ↕ marker there without displacing the wire.
- Smooth corridor: `tol=15, max_jump=120` — conservative to avoid wall-crossing shortcuts.

---

## Nav-graph rebuild — root cause found (2026-06-07)

Two bugs made every prior "passing" build a lie:

1. **KEY BUG**: wall lookups used keys `"21"`/`"22"` but `floor_walls.json` stores walls under `"b21"`/`"b22"`, so every lookup returned an empty list. Wall-checking validated against nothing. Fix: `bld = bld_raw.lstrip("b")`.

2. **BARRIER-LAYER BUG**: Window layer (glass partitions) and Mabat layer (partition walls) are real physical barriers, not just Beton. Both must be included in the walkable-polygon exclusion AND edge crossing-check.

Verified foundation for all 12 floors (B21 F1–F6, B22 F1–F6):
- Boundary: `0201Shetah-Bruto` / `0202Shetah-Bruto` LWPOLYLINEs (union all per floor — F3 has two)
- Doors: `Door` layer `LWPOLYLINE` only (not INSERT), threshold = midpoint of first two pts
- Barriers: Beton + Window + Mabat
- Basis overlays confirmed by user: `docs/basis_b{bld}_f{floor}.png`

See `docs/rebuild-handoff.md` for full state. Next: fix both bugs, rebuild, verify overlays.
