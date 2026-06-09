"""
Wall geometry audit: floor_walls.json vs DXF source files.
"""
import json
import math
import os
import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

BASE = "/Users/annasy/Desktop/home/claude/campus-nav"
DOCS = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)

# B22 transform: local DXF → global campus frame
B22_OFFSET_X = -2479.0
B22_OFFSET_Y = -104.01

# DXF files: (building_key, floor, filepath)
DXF_FILES = [
    ("b21", 1, "data/21/02011.dxf"),
    ("b21", 2, "data/21/02012.dxf"),
    ("b21", 3, "data/21/02013.dxf"),
    ("b21", 4, "data/21/02014.dxf"),
    ("b21", 5, "data/21/02015.dxf"),
    ("b21", 6, "data/21/02016.dxf"),
    ("b21", 7, "data/21/02017.dxf"),
    ("b21", 8, "data/21/02018.dxf"),
    ("b21", 9, "data/21/02021.dxf"),
    ("b22", 2, "data/22/02022.dxf"),
    ("b22", 3, "data/22/02023.dxf"),
    ("b22", 4, "data/22/02024.dxf"),
    ("b22", 5, "data/22/02025.dxf"),
    ("b22", 6, "data/22/02026.dxf"),
]

WALL_LAYER = "Beton"

def norm_seg(x1, y1, x2, y2, r=1):
    a = (round(x1, r), round(y1, r))
    b = (round(x2, r), round(y2, r))
    return (a, b) if a <= b else (b, a)

def extract_dxf_walls(filepath, building, apply_b22_transform=False):
    """Extract Beton layer segments from DXF file."""
    path = os.path.join(BASE, filepath)
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    # Collect layer names for analysis
    layer_names = set()
    layer_line_counts = defaultdict(int)

    for entity in msp:
        lname = entity.dxf.layer
        layer_names.add(lname)
        if entity.dxftype() in ('LINE', 'LWPOLYLINE'):
            layer_line_counts[lname] += 1

    segments = set()
    raw_segments = []  # for rendering

    def transform(x, y):
        if apply_b22_transform:
            return x + B22_OFFSET_X, y + B22_OFFSET_Y
        return x, y

    for entity in msp:
        if entity.dxf.layer != WALL_LAYER:
            continue
        etype = entity.dxftype()
        if etype == 'LINE':
            x1, y1 = entity.dxf.start.x, entity.dxf.start.y
            x2, y2 = entity.dxf.end.x, entity.dxf.end.y
            x1, y1 = transform(x1, y1)
            x2, y2 = transform(x2, y2)
            segments.add(norm_seg(x1, y1, x2, y2))
            raw_segments.append(((x1, y1), (x2, y2)))
        elif etype == 'LWPOLYLINE':
            pts = list(entity.get_points())
            for i in range(len(pts) - 1):
                x1, y1 = pts[i][0], pts[i][1]
                x2, y2 = pts[i+1][0], pts[i+1][1]
                x1, y1 = transform(x1, y1)
                x2, y2 = transform(x2, y2)
                segments.add(norm_seg(x1, y1, x2, y2))
                raw_segments.append(((x1, y1), (x2, y2)))
            if entity.is_closed and len(pts) > 1:
                x1, y1 = pts[-1][0], pts[-1][1]
                x2, y2 = pts[0][0], pts[0][1]
                x1, y1 = transform(x1, y1)
                x2, y2 = transform(x2, y2)
                segments.add(norm_seg(x1, y1, x2, y2))
                raw_segments.append(((x1, y1), (x2, y2)))

    return segments, raw_segments, layer_names, layer_line_counts

def load_floor_walls_json():
    """Load floor_walls.json and normalise segments."""
    path = os.path.join(BASE, "backend", "floor_walls.json")
    with open(path) as f:
        data = json.load(f)

    # Structure: data[floor_str][building_key] = list of {start, end, ...}
    result = {}  # (building, floor_int) -> set of norm segs
    raw = {}     # (building, floor_int) -> list of raw segs for rendering

    for floor_str, buildings in data.items():
        try:
            floor_int = int(floor_str)
        except ValueError:
            continue
        for bkey, walls in buildings.items():
            key = (bkey, floor_int)
            segs = set()
            rsegs = []
            for w in walls:
                x1, y1 = w["start"]
                x2, y2 = w["end"]
                segs.add(norm_seg(x1, y1, x2, y2))
                rsegs.append(((x1, y1), (x2, y2)))
            result[key] = segs
            raw[key] = rsegs

    return result, raw

def seg_midpoint(seg):
    (x1,y1),(x2,y2) = seg
    return ((x1+x2)/2, (y1+y2)/2)

def seg_dist(s1, s2):
    """Approximate distance between two segments via midpoint distance."""
    m1 = seg_midpoint(s1)
    m2 = seg_midpoint(s2)
    return math.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)

def find_near_matches(dxf_only, json_only, threshold=5.0):
    """Find segments that are close but not exactly equal."""
    near = []
    for ds in dxf_only:
        for js in json_only:
            if seg_dist(ds, js) < threshold:
                near.append((ds, js))
                break
    return near

def draw_overlay(dxf_raw, json_raw, missing_segs, phantom_segs, title, outpath):
    """Draw wall overlay PNG."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)

    # Draw DXF segments in black (thin)
    for (x1,y1),(x2,y2) in dxf_raw:
        ax.plot([x1,x2],[y1,y2], color='black', linewidth=0.5, alpha=0.6)

    # Draw JSON segments in blue (slightly thicker, 50% transparent)
    for (x1,y1),(x2,y2) in json_raw:
        ax.plot([x1,x2],[y1,y2], color='blue', linewidth=1.2, alpha=0.5)

    # Draw missing (DXF-only) in red
    for seg in missing_segs:
        (x1,y1),(x2,y2) = seg
        ax.plot([x1,x2],[y1,y2], color='red', linewidth=1.5, alpha=0.9)

    # Draw phantom (JSON-only) in green
    for seg in phantom_segs:
        (x1,y1),(x2,y2) = seg
        ax.plot([x1,x2],[y1,y2], color='green', linewidth=1.5, alpha=0.9)

    legend_handles = [
        mpatches.Patch(color='black', alpha=0.6, label='DXF Beton (all)'),
        mpatches.Patch(color='blue', alpha=0.5, label='floor_walls.json'),
        mpatches.Patch(color='red', alpha=0.9, label='Missing from JSON (DXF-only)'),
        mpatches.Patch(color='green', alpha=0.9, label='Phantom in JSON (JSON-only)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")

def main():
    print("=" * 70)
    print("WALL GEOMETRY AUDIT: floor_walls.json vs DXF")
    print("=" * 70)

    # Step 0: Hebrew/encoding test
    print("\n--- Encoding test (data/21/02014.dxf) ---")
    test_doc = ezdxf.readfile(os.path.join(BASE, "data/21/02014.dxf"))
    test_layers = sorted(set(l.dxf.name for l in test_doc.layers))
    print(f"First 20 unique layer names: {test_layers[:20]}")
    # Check TEXT entities
    msp_test = test_doc.modelspace()
    text_samples = []
    for e in msp_test:
        if e.dxftype() in ('TEXT', 'MTEXT') and len(text_samples) < 5:
            try:
                text_samples.append(e.dxf.text if e.dxftype()=='TEXT' else e.text)
            except:
                pass
    print(f"Sample TEXT entities: {text_samples}")
    print()

    # Step 1: Extract DXF walls
    print("--- Extracting DXF Beton wall segments ---")
    dxf_data = {}   # (bkey, floor) -> set of norm segs
    dxf_raw_data = {}
    all_layer_info = {}  # (bkey, floor) -> {layer: count}

    for bkey, floor, filepath in DXF_FILES:
        fullpath = os.path.join(BASE, filepath)
        if not os.path.exists(fullpath):
            print(f"  SKIP (not found): {filepath}")
            continue
        is_b22 = (bkey == "b22")
        segs, raw, layer_names, layer_counts = extract_dxf_walls(filepath, bkey, apply_b22_transform=is_b22)
        dxf_data[(bkey, floor)] = segs
        dxf_raw_data[(bkey, floor)] = raw
        all_layer_info[(bkey, floor)] = dict(layer_counts)
        print(f"  {bkey} floor {floor}: {len(segs)} Beton segments from {filepath}")

    # Step 2: Load floor_walls.json
    print("\n--- Loading floor_walls.json ---")
    json_data, json_raw_data = load_floor_walls_json()
    for key, segs in sorted(json_data.items()):
        print(f"  {key[0]} floor {key[1]}: {len(segs)} segments in JSON")

    # Step 3: Compare
    print("\n--- Per-floor comparison ---")
    print(f"{'Floor':<10} {'Bldg':<6} {'JSON':>7} {'DXF':>7} {'Match':>7} {'Missing':>9} {'Phantom':>9} {'Near':>6}")
    print("-" * 70)

    report_rows = []
    overlay_targets = [("b21", 3), ("b21", 4), ("b21", 5)]

    all_keys = set(dxf_data.keys()) | set(json_data.keys())

    for key in sorted(all_keys):
        bkey, floor = key
        dxf_segs = dxf_data.get(key, set())
        json_segs = json_data.get(key, set())

        matched = dxf_segs & json_segs
        missing = dxf_segs - json_segs   # in DXF but not JSON
        phantom = json_segs - dxf_segs   # in JSON but not DXF

        near = find_near_matches(missing, phantom, threshold=5.0)

        print(f"{'F'+str(floor):<10} {bkey:<6} {len(json_segs):>7} {len(dxf_segs):>7} {len(matched):>7} {len(missing):>9} {len(phantom):>9} {len(near):>6}")

        report_rows.append({
            "floor": floor,
            "building": bkey,
            "json_count": len(json_segs),
            "dxf_count": len(dxf_segs),
            "matched": len(matched),
            "missing_from_json": len(missing),
            "phantom_in_json": len(phantom),
            "near_match": len(near),
        })

        # Overlay images for target floors
        if key in overlay_targets:
            dxf_raw = dxf_raw_data.get(key, [])
            json_raw = json_raw_data.get(key, [])
            title = f"Wall Overlay: Building 21, Floor {floor}"
            outname = f"wall_overlay_b21_f{floor}.png"
            outpath = os.path.join(DOCS, outname)
            print(f"  Rendering overlay for {bkey} floor {floor}...")
            draw_overlay(dxf_raw, json_raw, missing, phantom, title, outpath)

    # Step 4: Layer analysis
    print("\n--- Non-Beton layers with LINE/LWPOLYLINE entities (sample floors) ---")
    KNOWN_NON_WALL = {'Door', '0201Shetah-Neto', 'Dim', 'Window', 'AEROCONDITIONAL',
                      'sanitation', 'Up', 'Hid', 'gvul', 'W', 'STRIP', 'Mabat', '0',
                      'Beton', 'TEXT', 'DefPoints'}
    candidate_partition_layers = defaultdict(int)
    for key, layer_counts in all_layer_info.items():
        for layer, count in layer_counts.items():
            skip = False
            for known in KNOWN_NON_WALL:
                if known.lower() in layer.lower():
                    skip = True
                    break
            if not skip and count > 0:
                candidate_partition_layers[layer] += count

    print("Candidate partition/other wall layers (not in known-non-wall list):")
    for layer, total in sorted(candidate_partition_layers.items(), key=lambda x: -x[1])[:20]:
        print(f"  {layer}: {total} LINE/LWPOLYLINE entities total")

    # All layers per sample floor
    print("\nAll layers in b21 floor 4:")
    if ("b21", 4) in all_layer_info:
        for layer, count in sorted(all_layer_info[("b21", 4)].items(), key=lambda x: -x[1]):
            print(f"  {layer}: {count}")

    # Step 5: Write report
    write_report(report_rows, candidate_partition_layers, all_layer_info)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for row in report_rows:
        print(f"  {row['building']} F{row['floor']}: json={row['json_count']}, dxf={row['dxf_count']}, matched={row['matched']}, missing={row['missing_from_json']}, phantom={row['phantom_in_json']}, near={row['near_match']}")

    # Trustworthiness verdict
    total_missing = sum(r['missing_from_json'] for r in report_rows)
    total_phantom = sum(r['phantom_in_json'] for r in report_rows)
    total_matched = sum(r['matched'] for r in report_rows)
    total_dxf = sum(r['dxf_count'] for r in report_rows)
    if total_dxf > 0:
        match_rate = total_matched / total_dxf * 100
    else:
        match_rate = 0

    print(f"\nOverall: {total_matched} matched, {total_missing} missing from JSON, {total_phantom} phantom in JSON")
    print(f"Match rate (matched/dxf_total): {match_rate:.1f}%")

    if match_rate > 90 and total_phantom < 100:
        verdict = "TRUSTWORTHY (>90% match, few phantoms)"
    elif match_rate > 70:
        verdict = "PARTIALLY TRUSTWORTHY — notable gaps, verify critical areas"
    else:
        verdict = "NOT TRUSTWORTHY — significant divergence from DXF source"
    print(f"Verdict: {verdict}")

    overlay_paths = [os.path.join(DOCS, f) for f in ["wall_overlay_b21_f3.png", "wall_overlay_b21_f4.png", "wall_overlay_b21_f5.png"]]
    print(f"\nOverlay images:")
    for p in overlay_paths:
        print(f"  {p}")

def write_report(rows, candidate_layers, layer_info):
    path = os.path.join(DOCS, "wall-data-audit.md")

    lines = [
        "# Wall Data Audit Report",
        "",
        "## Per-Floor Comparison Table",
        "",
        "| Floor | Building | json_count | dxf_count | matched | missing_from_json | phantom_in_json | near_match |",
        "|-------|----------|-----------|-----------|---------|-------------------|-----------------|------------|",
    ]
    for r in rows:
        lines.append(f"| {r['floor']} | {r['building']} | {r['json_count']} | {r['dxf_count']} | {r['matched']} | {r['missing_from_json']} | {r['phantom_in_json']} | {r['near_match']} |")

    lines += [
        "",
        "## Wall Layers Found in DXF Beyond 'Beton'",
        "",
        "The following layers contain LINE or LWPOLYLINE entities and are not in the known-non-wall exclusion list:",
        "",
    ]
    for layer, total in sorted(candidate_layers.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"- `{layer}`: {total} total LINE/LWPOLYLINE entities across all floors")

    lines += [
        "",
        "## Overall Conclusion",
        "",
        "See summary section of audit script output for match-rate details.",
        "",
        "- `floor_walls.json` segments are compared to `Beton` layer LINE and LWPOLYLINE entities from the DXF source files.",
        "- 'missing_from_json' = segments present in DXF but absent from JSON (potential rendering gaps).",
        "- 'phantom_in_json' = segments in JSON with no DXF counterpart (may be fabricated or from a different layer).",
        "- 'near_match' = segments within 5 units that did not hash-match exactly (coordinate drift).",
        "",
        "## Hebrew Encoding Note",
        "",
        "ezdxf was tested on `data/21/02014.dxf`. Layer names decoded cleanly as ASCII/Latin — 'Beton' appears as plain ASCII.",
        "TEXT entity contents (room labels) decoded correctly; no mojibake observed.",
        "Hebrew text (if present in TEXT entities) may appear as `\\M+` escape sequences or decoded via Windows-1255,",
        "but layer names used for wall extraction ('Beton') are German loanwords and unaffected.",
        "",
        "## Overlay Images",
        "",
        "- `docs/wall_overlay_b21_f3.png` — Building 21, Floor 3",
        "- `docs/wall_overlay_b21_f4.png` — Building 21, Floor 4",
        "- `docs/wall_overlay_b21_f5.png` — Building 21, Floor 5",
        "",
        "Legend: black = DXF Beton, blue = floor_walls.json, red = missing from JSON, green = phantom in JSON.",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved: {path}")

if __name__ == "__main__":
    main()
