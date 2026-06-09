#!/usr/bin/env python3
"""
Investigation: outer boundary (0201Shetah-Bruto) + door detection for F3/F4.
"""
import sys, math, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DATA21 = ROOT / "data" / "21"
DOCS   = ROOT / "docs"

def log(*a): print(*a)

def get_segs(msp, layer):
    segs = []
    for e in msp:
        if e.dxf.get("layer","") != layer: continue
        t = e.dxftype()
        if t == "LINE":
            segs.append(((e.dxf.start.x, e.dxf.start.y),(e.dxf.end.x, e.dxf.end.y)))
        elif t == "LWPOLYLINE":
            pts = [(p[0],p[1]) for p in e.get_points()]
            for i in range(len(pts)-1): segs.append((pts[i],pts[i+1]))
            if e.is_closed and len(pts)>=2: segs.append((pts[-1],pts[0]))
    return segs

def get_lwpolys(msp, layer):
    polys = []
    for e in msp:
        if e.dxf.get("layer","") != layer: continue
        if e.dxftype() != "LWPOLYLINE": continue
        pts = [(p[0],p[1]) for p in e.get_points()]
        if len(pts)>=3:
            try: polys.append(Polygon(pts))
            except: pass
    return polys

# ---------------------------------------------------------------------------
# PART 1 — Outer boundary: 0201Shetah-Bruto for F3 and F4
# ---------------------------------------------------------------------------
for floor, fname in [(3,"02013.dxf"),(4,"02014.dxf")]:
    doc = ezdxf.readfile(str(DATA21/fname))
    msp = doc.modelspace()

    # Collect 0201Shetah-Bruto polygons
    bruto_polys = get_lwpolys(msp, "0201Shetah-Bruto")
    neto_polys  = get_lwpolys(msp, "0201Shetah-Neto")
    beton_segs  = get_segs(msp, "Beton")

    log(f"\nF{floor}: 0201Shetah-Bruto: {len(bruto_polys)} polygon(s)")
    for i,p in enumerate(bruto_polys):
        log(f"  [{i}] area={p.area:.0f}, bounds={tuple(round(v) for v in p.bounds)}")
    log(f"F{floor}: 0201Shetah-Neto: {len(neto_polys)} room polygons")

    # Largest bruto polygon = building outer boundary
    bruto_polys.sort(key=lambda p: p.area, reverse=True)
    outer = bruto_polys[0] if bruto_polys else None

    if outer:
        from shapely.validation import make_valid
        valid_neto = [make_valid(p) for p in neto_polys]
        rooms_union = unary_union(valid_neto) if valid_neto else None
        if rooms_union:
            corridor = outer.difference(rooms_union)
            log(f"  Corridor area (bruto minus rooms): {corridor.area:.0f}")

    # --- Overlay image ---
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_aspect("equal")
    ax.set_title(f"B21 F{floor} — Outer boundary (0201Shetah-Bruto) investigation")

    for (x1,y1),(x2,y2) in beton_segs:
        ax.plot([x1,x2],[y1,y2], color="gray", lw=0.4, alpha=0.5)

    for poly in neto_polys:
        if poly.is_valid and not poly.is_empty:
            xs,ys = poly.exterior.xy
            ax.fill(xs,ys,color="lightblue",alpha=0.35,zorder=1)

    if outer:
        xs,ys = outer.exterior.xy
        ax.plot(xs,ys,color="red",lw=2.5,zorder=5,label="0201Shetah-Bruto (outer boundary)")
        # Show corridor (walkable) in green
        if rooms_union:
            try:
                corr = outer.difference(rooms_union)
                if hasattr(corr,"geoms"):
                    for g in corr.geoms:
                        if g.is_valid and not g.is_empty:
                            xs2,ys2 = g.exterior.xy
                            ax.fill(xs2,ys2,color="lightgreen",alpha=0.35,zorder=2)
                else:
                    xs2,ys2 = corr.exterior.xy
                    ax.fill(xs2,ys2,color="lightgreen",alpha=0.35,zorder=2)
            except: pass

    ax.legend(loc="upper right", fontsize=9)
    out = DOCS/f"invest_boundary_b21_f{floor}.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# PART 2 — Door investigation on F4
# ---------------------------------------------------------------------------
log("\n=== F4 door investigation ===")

with open(ROOT/"backend"/"graph.json") as f:
    gdata = json.load(f)

target_rooms = {n["id"]:n for n in gdata["nodes"] if n["id"] in ("21428","21429")}
cluster_cx = sum(n["door_x"] for n in target_rooms.values())/len(target_rooms)
cluster_cy = sum(n["door_y"] for n in target_rooms.values())/len(target_rooms)
log(f"  21428/21429 cluster center: ({cluster_cx:.1f},{cluster_cy:.1f})")

doc4  = ezdxf.readfile(str(DATA21/"02014.dxf"))
msp4  = doc4.modelspace()

# Count and inspect Door-layer entities
d_lwpoly, d_line, d_insert, d_arc = [], [], [], []
for e in msp4:
    if e.dxf.get("layer","") != "Door": continue
    t = e.dxftype()
    if t=="LWPOLYLINE": d_lwpoly.append(e)
    elif t=="LINE":     d_line.append(e)
    elif t=="INSERT":   d_insert.append(e)
    elif t=="ARC":      d_arc.append(e)

log(f"  Door layer: {len(d_lwpoly)} LWPOLYLINE, {len(d_line)} LINE, "
    f"{len(d_insert)} INSERT, {len(d_arc)} ARC")

# Inspect INSERT blocks — are they door symbols?
if d_insert:
    block_names = defaultdict(int)
    for e in d_insert:
        block_names[e.dxf.name] += 1
    log(f"  INSERT block names: {dict(block_names)}")

# Inspect nearby entities (within 400 units of cluster)
RADIUS = 400
log(f"\n  Entities within {RADIUS}u of cluster (ALL Door-layer):")
log(f"  {'Type':12s} {'dist':7s}  details")
nearby_doors = []
for elist, color_hint in [(d_lwpoly,"LWP"),(d_line,"LIN"),(d_insert,"INS"),(d_arc,"ARC")]:
    for e in elist:
        t = e.dxftype()
        if t=="LWPOLYLINE":
            pts = [(p[0],p[1]) for p in e.get_points()]
            if not pts: continue
            cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
        elif t=="LINE":
            cx=(e.dxf.start.x+e.dxf.end.x)/2; cy=(e.dxf.start.y+e.dxf.end.y)/2
        elif t=="INSERT":
            cx=e.dxf.insert.x; cy=e.dxf.insert.y
        elif t=="ARC":
            cx=e.dxf.center.x; cy=e.dxf.center.y
        d=math.hypot(cx-cluster_cx,cy-cluster_cy)
        if d<=RADIUS:
            nearby_doors.append((d,t,cx,cy,e))

nearby_doors.sort(key=lambda x: x[0])
for d,t,cx,cy,e in nearby_doors:
    if t=="ARC":
        extra=f"r={e.dxf.radius:.1f}"
    elif t=="INSERT":
        extra=f"block={e.dxf.name}"
    elif t=="LWPOLYLINE":
        pts=[(p[0],p[1]) for p in e.get_points()]
        extra=f"{len(pts)} pts, closed={e.is_closed}"
    elif t=="LINE":
        extra=f"({e.dxf.start.x:.0f},{e.dxf.start.y:.0f})->({e.dxf.end.x:.0f},{e.dxf.end.y:.0f})"
    else:
        extra=""
    log(f"  {t:12s} {d:7.1f}  ({cx:.0f},{cy:.0f})  {extra}")

# ---------------------------------------------------------------------------
# PART 3 — Door overlay F4 (all door entities, colored by type)
# ---------------------------------------------------------------------------
beton4 = get_segs(msp4,"Beton")
neto4  = get_lwpolys(msp4,"0201Shetah-Neto")
bruto4 = get_lwpolys(msp4,"0201Shetah-Bruto")
bruto4.sort(key=lambda p:p.area,reverse=True)

fig, ax = plt.subplots(figsize=(16,16))
ax.set_aspect("equal")
ax.set_title("B21 F4 — Door detection investigation")

for (x1,y1),(x2,y2) in beton4:
    ax.plot([x1,x2],[y1,y2], color="gray", lw=0.5, alpha=0.6)
for poly in neto4:
    if poly.is_valid and not poly.is_empty:
        xs,ys=poly.exterior.xy
        ax.fill(xs,ys,color="lightyellow",alpha=0.4,zorder=1)
if bruto4:
    xs,ys=bruto4[0].exterior.xy
    ax.plot(xs,ys,color="red",lw=2,zorder=2,label="Bruto boundary")

# Draw all Door-layer entities
for e in d_lwpoly:
    pts=[(p[0],p[1]) for p in e.get_points()]
    if not pts: continue
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color="blue",lw=1,alpha=0.7,zorder=3)
for e in d_line:
    ax.plot([e.dxf.start.x,e.dxf.end.x],[e.dxf.start.y,e.dxf.end.y],
            color="orange",lw=1.2,alpha=0.8,zorder=3)
for e in d_arc:
    cx2,cy2,r=e.dxf.center.x,e.dxf.center.y,e.dxf.radius
    a1,a2=math.radians(e.dxf.start_angle),math.radians(e.dxf.end_angle)
    if a2<a1: a2+=2*math.pi
    theta=np.linspace(a1,a2,30)
    ax.plot(cx2+r*np.cos(theta),cy2+r*np.sin(theta),color="cyan",lw=1.5,alpha=0.9,zorder=4)
for e in d_insert:
    ax.plot(e.dxf.insert.x,e.dxf.insert.y,"r*",ms=8,zorder=5)

# Highlight cluster box
RBOX=400
rect=plt.Rectangle((cluster_cx-RBOX,cluster_cy-RBOX),2*RBOX,2*RBOX,
                    fill=False,edgecolor="red",lw=2.5,zorder=7)
ax.add_patch(rect)
ax.text(cluster_cx,cluster_cy+RBOX+30,"21428/21429",color="red",fontsize=9,ha="center",zorder=8)
for rid,n in target_rooms.items():
    ax.plot(n["door_x"],n["door_y"],"kx",ms=12,mew=2.5,zorder=9)
    ax.text(n["door_x"]+30,n["door_y"],rid,fontsize=8,color="black",zorder=9)

handles=[
    mpatches.Patch(color="blue",  label=f"LWPOLYLINE ({len(d_lwpoly)})"),
    mpatches.Patch(color="orange",label=f"LINE ({len(d_line)})"),
    mpatches.Patch(color="cyan",  label=f"ARC ({len(d_arc)})"),
    mpatches.Patch(color="red",   label=f"INSERT ({len(d_insert)})"),
    mpatches.Patch(color="black", label="graph.json door pos (×)"),
]
ax.legend(handles=handles,loc="upper right",fontsize=9)

out=DOCS/"invest_doors_b21_f4.png"
fig.savefig(str(out),dpi=120,bbox_inches="tight")
plt.close(fig)
log(f"\nSaved: {out}")

# Per-floor door counts
log("\n=== Door-layer entity counts per floor ===")
for floor, fname in [(3,"02013.dxf"),(4,"02014.dxf"),(5,"02015.dxf"),(6,"02016.dxf")]:
    doc2=ezdxf.readfile(str(DATA21/fname))
    msp2=doc2.modelspace()
    counts=defaultdict(int)
    for e in msp2:
        if e.dxf.get("layer","")=="Door":
            counts[e.dxftype()]+=1
    log(f"  F{floor}: {dict(counts)}  => total={sum(counts.values())}")
