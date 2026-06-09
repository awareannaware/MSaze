---
name: project-campus-nav
description: Campus navigation graph rebuild — confirmed decisions, verified foundation, current status
metadata:
  type: project
---

## Status (2026-06-08)
Nav-graph rebuild in progress. `/api/route4` (corridor3.json raster) is live fallback. `/api/route5` wired but `nav_graph.json` not yet producing valid routes.

## Confirmed algorithm (from rebuild sessions)
```
walkable = union(bruto_polygons) − rooms_union − barriers_buffer(8u)
```
- **Boundary**: `0201Shetah-Bruto` (B21) / `0202Shetah-Bruto` (B22) LWPOLYLINEs — union ALL per floor (F3 has two)
- **Rooms to subtract**: `0201Shetah-Neto` / `0202Shetah-Neto` — ONLY destination rooms, NOT corridor types
- **Barriers**: Beton + Window + Mabat (all three); skip Window 10-vertex polys (glass door openings)
- **Doors**: `Door` layer LWPOLYLINE only (not INSERT), threshold = midpoint of first two pts

## Two bugs fixed (critical)
1. `floor_walls.json` keys are `"b21"`/`"b22"` — must `lstrip("b")` before lookup
2. Barriers = Beton + Window + Mabat (not Beton only)

## B22 coordinate transform
`global_x = dxf_x − 2479.0`, `global_y = dxf_y − 104.01` (stored in `docs/b22_transform.json`)

## Verified basis overlays
All 12 floors confirmed in `docs/basis_b{bld}_f{floor}.png`

## Door counts per floor (confirmed)
B21: F1=14, F2=114, F3=93, F4=90, F5=84, F6=48
B22: F1=7, F2=54, F3=123, F4=100, F5=104, F6=38

## Corridor room types (never subtract from walkable)
פויאה, פויה, מבוא, מעבר, פרוזדור, מרפסת, חדר מדרגות, מדרגות, פיר מעלית, פיר

## Restrooms excluded from routing
שירותים גברים, שירותים נשים, שרותי נכים, שירותים, שרותים נשים, שירותי נכים

**Why:** These are not routing destinations. Routing through them is wrong.
**How to apply:** Filter these room IDs out before building door nodes and room records.
