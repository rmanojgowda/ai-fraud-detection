"""
parse_osm.py  —  India OSM → GMap Dataset
==========================================
Reads india-260327.osm.pbf and outputs:
  1. D:\GoogleMap\include\MapLoader_OSM.h     (C++ — all city sizes)
  2. D:\GoogleMap\data\india_osm_data.js      (JavaScript — frontend)
  3. D:\GoogleMap\data\osm_stats.txt          (summary report)

HOW TO RUN:
  Step 1 — Install osmium (one time only):
    pip install osmium

  Step 2 — Run from D:\GoogleMap:
    python parse_osm.py

  Step 3 — Results appear in include/ and data/ folders.

WHAT IT EXTRACTS:
  Nodes  : cities, towns, villages with population data
  Ways   : roads tagged as motorway, trunk, primary, secondary
  Output : up to 5000 cities + all connecting roads between them
"""

import os, sys, json, math, time

# ── Check osmium is installed ─────────────────────────────────
try:
    import osmium
except ImportError:
    print("\n❌ osmium not installed.")
    print("   Run:  pip install osmium")
    print("   Then re-run this script.\n")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(SCRIPT_DIR, "data", "india-260327.osm.pbf")
OUTPUT_H     = os.path.join(SCRIPT_DIR, "include", "MapLoader_OSM.h")
OUTPUT_JS    = os.path.join(SCRIPT_DIR, "data", "india_osm_data.js")
OUTPUT_STATS = os.path.join(SCRIPT_DIR, "data", "osm_stats.txt")

# City size limits — we extract largest cities first to keep graph manageable
MAX_CITIES   = 5000    # cap on output cities
MIN_POP      = 0       # minimum population (0 = include all tagged places)

# Road type priority (higher = more important highway)
ROAD_PRIORITY = {
    'motorway':      10,
    'trunk':          9,
    'primary':        8,
    'secondary':      7,
    'tertiary':       6,
    'motorway_link':  5,
    'trunk_link':     4,
    'primary_link':   3,
    'secondary_link': 2,
}

# Place types to extract as cities
PLACE_TYPES = {'city', 'town', 'village', 'suburb', 'hamlet', 'municipality'}

# India bounding box (filter out stray nodes)
LAT_MIN, LAT_MAX =  6.0,  37.5
LON_MIN, LON_MAX = 68.0,  97.5

print("=" * 60)
print("  GMap OSM Parser")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    print(f"\n❌ File not found: {INPUT_FILE}")
    print(f"   Make sure india-260327.osm.pbf is in D:\\GoogleMap\\data\\")
    sys.exit(1)

print(f"\n📂 Input  : {INPUT_FILE}")
print(f"📝 Output : {OUTPUT_H}")
print(f"           {OUTPUT_JS}")
print()

# ═════════════════════════════════════════════════════════════
#  PASS 1: Extract city nodes
# ═════════════════════════════════════════════════════════════
print("🔍 Pass 1/2 — Extracting city nodes...")
t0 = time.time()

class CityHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.cities = []          # list of dicts
        self.node_id_to_idx = {}  # osm_node_id → our 0-based index

    def node(self, n):
        tags = n.tags
        place = tags.get('place', '')
        if place not in PLACE_TYPES:
            return

        lat, lon = float(n.location.lat), float(n.location.lon)
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            return

        name = tags.get('name:en', '') or tags.get('name', '')
        if not name:
            return

        # Population (used for priority sorting)
        try:
            pop = int(tags.get('population', 0))
        except:
            pop = 0

        # Place type weight for sorting
        type_weight = {'city': 5, 'town': 4, 'municipality': 3,
                       'suburb': 2, 'village': 1, 'hamlet': 0}.get(place, 0)

        self.cities.append({
            'osm_id': n.id,
            'name':   name,
            'lat':    round(lat, 5),
            'lon':    round(lon, 5),
            'pop':    pop,
            'type':   place,
            'weight': type_weight * 1_000_000 + pop,
        })

city_handler = CityHandler()
city_handler.apply_file(INPUT_FILE, locations=True)

# Sort by importance (city > town > village, then by population)
all_cities = sorted(city_handler.cities, key=lambda c: -c['weight'])

# Deduplicate by name (keep highest-weight entry)
seen_names = {}
deduped = []
for c in all_cities:
    key = c['name'].lower().strip()
    if key not in seen_names:
        seen_names[key] = True
        deduped.append(c)

# Cap at MAX_CITIES
cities = deduped[:MAX_CITIES]

# Assign 0-based IDs and build lookup
for i, c in enumerate(cities):
    c['id'] = i

osm_id_to_idx = {c['osm_id']: c['id'] for c in cities}

print(f"   Found {len(all_cities):,} place nodes → kept {len(cities):,} after dedup + cap")
print(f"   Time: {time.time()-t0:.1f}s")

# ═════════════════════════════════════════════════════════════
#  PASS 2: Extract roads between our cities
# ═════════════════════════════════════════════════════════════
print("\n🔍 Pass 2/2 — Extracting roads...")
t1 = time.time()

# Build spatial grid for fast nearest-city lookup
# Cell size ~0.5° ≈ 55km
CELL = 0.5
city_grid = {}  # (grid_row, grid_col) → list of city indices

for c in cities:
    key = (int(c['lat'] / CELL), int(c['lon'] / CELL))
    city_grid.setdefault(key, []).append(c['id'])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(max(0, a)))

def nearest_city(lat, lon, max_dist_km=80):
    """Find the closest city to a lat/lon within max_dist_km."""
    gr, gc = int(lat / CELL), int(lon / CELL)
    best_d, best_id = 9999, -1
    # Search 3×3 grid cells around the point
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            for cid in city_grid.get((gr+dr, gc+dc), []):
                c = cities[cid]
                d = haversine(lat, lon, c['lat'], c['lon'])
                if d < best_d:
                    best_d, best_id = d, cid
    return (best_id, best_d) if best_d <= max_dist_km else (-1, 9999)

class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.edges = {}   # (min_id, max_id) → distance_km

    def way(self, w):
        hw = w.tags.get('highway', '')
        if hw not in ROAD_PRIORITY:
            return

        # Walk the nodes of this way, collect lat/lon
        try:
            coords = [(float(n.location.lat), float(n.location.lon))
                      for n in w.nodes
                      if n.location.valid()]
        except:
            return

        if len(coords) < 2:
            return

        # For each node in the way, find nearest city
        # Connect consecutive city-nearest-pairs
        prev_cid = -1
        accum_d  = 0.0
        prev_lat, prev_lon = coords[0]

        for lat, lon in coords:
            accum_d += haversine(prev_lat, prev_lon, lat, lon)
            prev_lat, prev_lon = lat, lon

            cid, _ = nearest_city(lat, lon, max_dist_km=15)
            if cid == -1 or cid == prev_cid:
                continue

            if prev_cid != -1:
                key = (min(prev_cid, cid), max(prev_cid, cid))

                # FIX: road distance must be at least the straight-line
                # distance between the two cities. GPS snapping radius is
                # 15km so accum_d (measured along road GPS points) can be
                # far shorter than the actual city-to-city distance.
                # Use max(accum_d, haversine_between_cities) * 1.35
                city_a = cities[prev_cid]
                city_b = cities[cid]
                min_d = haversine(city_a['lat'], city_a['lon'],
                                  city_b['lat'], city_b['lon'])
                d = max(round(max(accum_d, min_d) * 1.35), 1)

                # Keep shorter distance if edge already exists
                if key not in self.edges or self.edges[key] > d:
                    self.edges[key] = d
                accum_d = 0.0

            prev_cid = cid

road_handler = RoadHandler()
road_handler.apply_file(INPUT_FILE, locations=True)

road_edges = [(u, v, d) for (u, v), d in road_handler.edges.items() if u != v]
print(f"   Found {len(road_edges):,} road edges")
print(f"   Time: {time.time()-t1:.1f}s")

# ═════════════════════════════════════════════════════════════
#  CONNECTIVITY CHECK + FALLBACK PROXIMITY EDGES
# ═════════════════════════════════════════════════════════════
print("\n🔗 Checking connectivity...")

# Build adjacency
adj = {i: [] for i in range(len(cities))}
for u, v, d in road_edges:
    adj[u].append(v); adj[v].append(u)

# BFS from node 0
visited = set(); q = [0]
while q:
    u = q.pop()
    if u in visited: continue
    visited.add(u)
    q.extend(adj[u])

pct = len(visited) / len(cities) * 100
print(f"   Connected: {len(visited):,}/{len(cities):,} ({pct:.1f}%)")

# Connect isolated cities to nearest reachable city
# MAX_PROXIMITY_KM cap is critical — without it, isolated Northeast cities
# connect to the nearest reachable city anywhere in India, creating phantom
# shortcuts like Delhi→Mumbai = 106km instead of ~1400km.
# 150km allows connecting cities within the same region (state/district)
# but prevents cross-country phantom edges.
MAX_PROXIMITY_KM = 150

isolated = [i for i in range(len(cities)) if i not in visited]
extra_edges = 0
skipped_isolated = 0

for iso in isolated:
    best_d, best_j = 9999, -1
    for j in visited:
        d = haversine(cities[iso]['lat'], cities[iso]['lon'],
                      cities[j]['lat'],  cities[j]['lon'])
        if d < best_d:
            best_d, best_j = d, j

    if best_j >= 0 and best_d <= MAX_PROXIMITY_KM:
        # Only connect if within 150km — keeps edges realistic
        road_d = max(int(best_d * 1.35), 5)
        u, v = min(iso, best_j), max(iso, best_j)
        road_edges.append((u, v, road_d))
        adj[u].append(v); adj[v].append(u)
        visited.add(iso)
        extra_edges += 1
    else:
        # City is truly isolated (island, border area, no nearby roads)
        # Accept it as unreachable rather than create a phantom shortcut
        skipped_isolated += 1

if extra_edges:
    print(f"   Added {extra_edges} proximity edges (max {MAX_PROXIMITY_KM}km cap)")
if skipped_isolated:
    print(f"   Skipped {skipped_isolated} truly isolated cities (no road within {MAX_PROXIMITY_KM}km)")

print(f"   Final: {len(road_edges):,} total edges")

# ═════════════════════════════════════════════════════════════
#  OUTPUT 1: C++ MapLoader_OSM.h
# ═════════════════════════════════════════════════════════════
print(f"\n💾 Writing {OUTPUT_H} ...")

lines = []
lines.append("#pragma once")
lines.append("// ============================================================")
lines.append("//  MapLoader_OSM.h  —  Real India OSM Road Network")
lines.append(f"//  {len(cities):,} cities  |  {len(road_edges):,} roads")
lines.append("//  Source: OpenStreetMap (india-260327.osm.pbf)")
lines.append("//  Parser: parse_osm.py")
lines.append("// ============================================================")
lines.append("")
lines.append('#include "../include/Graph.h"')
lines.append("#include <iostream>")
lines.append("")
lines.append("inline void loadIndiaMapOSM(Graph& g) {")
lines.append("")
lines.append("    // ── Cities ─────────────────────────────────────────────")
for c in cities:
    name = c['name'].replace('"', '\\"')
    lines.append(f'    g.addCity("{name}", {c["lat"]}, {c["lon"]});')
lines.append("")
lines.append("    // ── Roads ──────────────────────────────────────────────")
for u, v, d in road_edges:
    lines.append(f"    g.addRoad({u}, {v}, {d});")
lines.append("")
lines.append(f'    std::cout << "\\n  ✅  OSM India map: '
             f'{len(cities):,} cities, {len(road_edges):,} roads.\\n";')
lines.append("}")

with open(OUTPUT_H, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"   ✅ Written ({len(lines)} lines)")

# ═════════════════════════════════════════════════════════════
#  OUTPUT 2: JavaScript data file
# ═════════════════════════════════════════════════════════════
print(f"\n💾 Writing {OUTPUT_JS} ...")

js_cities = [{"id": c["id"], "name": c["name"],
              "lat": c["lat"], "lon": c["lon"],
              "type": c["type"], "pop": c["pop"]}
             for c in cities]
js_roads  = [[u, v, d] for u, v, d in road_edges]

js_out = f"""// ============================================================
//  india_osm_data.js  —  Real India OSM Road Network
//  {len(cities):,} cities  |  {len(road_edges):,} roads
//  Source: OpenStreetMap  (CC BY-SA)
// ============================================================

const CITIES_OSM = {json.dumps(js_cities, ensure_ascii=False, separators=(',', ':'))};

const ROADS_OSM = {json.dumps(js_roads, separators=(',', ':'))};
"""

with open(OUTPUT_JS, "w", encoding="utf-8") as f:
    f.write(js_out)
print(f"   ✅ Written")

# ═════════════════════════════════════════════════════════════
#  OUTPUT 3: Stats report
# ═════════════════════════════════════════════════════════════
type_counts = {}
for c in cities:
    type_counts[c['type']] = type_counts.get(c['type'], 0) + 1

with_pop = sum(1 for c in cities if c['pop'] > 0)

stats = f"""
GMap OSM Dataset Report
========================
Source file : india-260327.osm.pbf
Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}

CITIES
------
Total kept   : {len(cities):,}
With pop data: {with_pop:,}

By type:
{chr(10).join(f'  {k:20s}: {v:,}' for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))}

ROADS
-----
Total edges  : {len(road_edges):,}
Extra (prox) : {extra_edges}
Connectivity : {len(visited):,}/{len(cities):,} ({pct:.1f}%)

SCALE COMPARISON
----------------
Original (hand-coded) : 31 cities,   56 roads
Generated (proximity) : 268 cities, 465 roads
OSM (real data)       : {len(cities):,} cities, {len(road_edges):,} roads

NEXT STEPS
----------
1. Copy MapLoader_OSM.h to D:\\GoogleMap\\include\\
2. Copy india_osm_data.js to D:\\GoogleMap\\data\\
3. Update main.cpp menu option 3 → loadIndiaMapOSM(g)
4. Run benchmark: watch Dijkstra vs A* gap widen at this scale
"""

with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    f.write(stats)

print(stats)
total_time = time.time() - t0
print(f"✅ Done in {total_time:.1f}s")
print(f"\n   C++ header : {OUTPUT_H}")
print(f"   JS data    : {OUTPUT_JS}")
print(f"   Stats      : {OUTPUT_STATS}")
print(f"\n   Now rebuild: cd D:\\GoogleMap\\build && cmake --build .")
