# Phase 2 — Real Polish Rail Network: Implementation Plan

Status: draft for review
Depends on: Phase 1 (`PLAN.md`), commits 1–15 merged
Related: `docs/algorithm_spec.md`, `docs/data.md`

## 1. Goal

Phase 1 optimizes an idealized Euclidean network between approximate city centres. Its only link to reality is a visual overlay. Phase 2 moves the problem onto real data and answers three questions:

1. **How good is the real network under our own objective?** Score the real PKP PLK network with the same `C`, `D`, `FT`, `J` metrics. Compare FGO and the real network numerically, not only visually.
2. **What would FGO build if construction followed real constraints?** This is the corridor-aware Steiner network (formulation A). Building along existing corridors is cheap. Terrain, water and protected areas are expensive.
3. **Which upgrades or closures of the existing network give the best trade-off?** This is existing-network subgraph selection (formulation B), trading construction cost against travel time and resilience. The result is a Pareto front, not a single answer.

Throughout, the FGO optimizer is reused **unchanged**. `FGOOptimizer` accepts any `objective(vector) -> object with .score`, so every Phase 2 formulation is a new *problem/decoder*, not a new optimizer.

## 2. What we learned from Phase 1 data (Nov 2026)

`scripts/fetch_osm_rail.py` (main and branch `railway=rail`, inside Poland, no service tracks) returns:

| Fact | Value | Consequence |
| --- | --- | --- |
| Ways | ~27,500 | Too many to optimize over directly; contract to a topological graph. |
| Vertices | ~470,000 unique OSM nodes; ~27,000 shared by ≥2 ways | Node IDs are in the Overpass cache (`elements[].nodes`), so topology can be rebuilt exactly. The GeoJSON export currently drops them. |
| `ref` tag | 100% of ways | The PKP PLK line number (e.g. `1`, `3`, `271`). Use it to group track into lines and corridors. |
| `maxspeed` | 98% | Enough to compute realistic travel times. |
| `electrified` | 100% | Enough for an electrification upgrade decision. |
| `railway:track_ref` | 78% | **Each track of a double-track line is a separate way.** Parallel ways must be merged into one edge with `tracks = k`. |
| `tracks` | 4% | Do not rely on it; infer the track count from parallel ways. |
| Border clipping | Clipped to PL | Some connections exist only via neighbouring countries. Handle this explicitly; see §8. |

FGO vs. real rail, Phase 1 runs (seed 42, 20k evals): the share of FGO link length within 5 km of real track is 64% (n=15) and 59% (n=30). Random 60 km segments score 47%. This is the baseline that formulation A should improve on.

## 3. Prerequisites (finish Phase 1 leftovers first)

These are small and unblock Phase 2 work:

- **P1. Map readability.** ✅ Done (2026-11-15). Terminal labels avoid each other and the nodes. A bundled Poland outline (Natural Earth) is drawn when most terminals fall inside it. Use `--no-labels` / `--no-outline` to turn them off.
- **P2. Real units.** ✅ Done (2026-11-15). This uncovered a Phase 1 distortion: x and y were normalized independently (649 vs 592 km per unit for `poland_30`), so distances depended on direction by up to ~10%. Normalization is now isotropic; `Projection.km_per_unit` and the `C_km` / `C_mst_km` metric fields are new. Old runs still reload exactly. Still to do for Phase 2: make `epsg2180` the default, because the equirectangular approximation is off by up to ~6% across Poland.
- **P3. Faster metrics.** ✅ Done (2026-11-15). SciPy `csgraph.dijkstra` for `D`, plus a single-DFS bridge count for `FT`; both are checked against the old NetworkX code. Decode speed: `poland_15` 315 → 518 evals/s, `poland_30` 103 → 218 evals/s. Two-thirds of the time is now `build_network` (pure-Python Kruskal plus Delaunay). Speeding it up with `scipy.sparse.csgraph.minimum_spanning_tree` would change tie-breaking, so it needs its own regression check.
- **P4. Problem interface.** Introduce `problem/base.py` with a `Problem` protocol: `dimension`, `evaluate(vector) -> Evaluation(score, metrics: dict, payload)`, `to_dict()`, `kind: str`. Make `NetworkProblem` conform as `kind="euclidean"`. Make `Metrics` serializable as a plain dict, so viz and export can handle per-problem metric sets.
- **P5. Schema v2.** Record `problem.kind` and the problem parameters in `manifest.json`. `load_run` must still read v1 runs.

## 4. Workstreams

```text
 2A data ingestion ──► 2B real-network baseline ──► 2C formulation A (corridor Steiner)
        │                       │                          │
        └──────────────────────►└──► 2D formulation B ─────┴──► 2E viz/Pareto ──► 2F experiments & docs
```

2A and 2B are required for everything else. 2C and 2D are independent of each other, so two people can work in parallel. If time is short, ship **2A + 2B + 2C** (the clearest story) and treat 2D as the stretch goal.

---

### 2A. Data ingestion: rail graph, stations, demand

**New package:** `src/grzybki/rail/`

| Module | Responsibility |
| --- | --- |
| `rail/osm.py` | Parse the Overpass JSON cache into `nodes: {id: (lon, lat)}` and `ways: [{id, node_ids, tags}]`. No network access. |
| `rail/topology.py` | Build and simplify the topological graph (steps below). |
| `rail/graph.py` | `RailGraph` dataclass: node coordinates (EPSG:2180, metres), edges with polyline geometry, `length_km`, `maxspeed`, `tracks`, `electrified`, `line_refs`. `save()`/`load()` to `.npz` + `.json` sidecar. `to_csr()` for SciPy. |
| `rail/stations.py` | Load stations, snap them to graph nodes, and select subsets. |
| `rail/demand.py` | Gravity-model origin–destination demand between stations. |

**Graph build steps** (`rail/topology.py`):

1. Create a node for every OSM node that is a way endpoint or is shared by ≥2 ways. These are junctions and endpoints.
2. Split each way at those nodes into segments. Each segment keeps its polyline and tags.
3. **Merge parallel tracks.** Segments whose endpoints are within ~50 m of each other at both ends, have the same `ref`, and have similar length (±10%) collapse into one edge with `tracks = count`. Snap near-coincident junction nodes first: cluster nodes within 30 m using a `cKDTree`.
4. **Contract degree-2 chains** into single edges. Length = sum. `maxspeed` = length-weighted harmonic mean, so travel time is preserved. `electrified` = all segments electrified. Keep the polyline for drawing.
5. Keep the largest connected component, and log what was dropped (km, number of components).
6. Defaults for missing tags. Use these only when the tag is absent, and count how often each default was used:
   - `maxspeed`: 80 km/h on main lines, 60 km/h on branch lines.
   - `tracks`: 1.

Expected result: roughly 5–15k nodes and edges. This is cheap for SciPy shortest paths.

**Stations:**

- Add `scripts/fetch_osm_stations.py`. Overpass query for `railway=station` and `railway=halt` nodes (with `name`, `uic_ref`, `public_transport`, `train`), clipped to PL. Cache it under `data/reference/`.
- Optionally add `scripts/fetch_gtfs.py`, which downloads PolishTrainsGTFS feeds (PKP Intercity at minimum). Each station's daily departure count is a proxy for its importance. Check the feed licence before redistributing anything derived from it.
- `rail/stations.py`:
  - Snap each station to the nearest graph node, within 500 m. Otherwise, split the nearest edge and insert a node.
  - Select a subset by (a) matching the Phase 1 city list to each city's main station (e.g. `Warszawa Centralna`, `Kraków Główny`, `Poznań Główny`), or (b) taking the top-k stations by GTFS departures.
- Add `scripts/build_station_instance.py`. It writes `data/instances/poland_stations_15.json` and `poland_stations_30.json` in the existing instance format, plus `metadata.population` and `metadata.station_id`. Because these are real station coordinates, Phase 1 runs can also be redone on them.

**Demand** (`rail/demand.py`):

- `T_ij = P_i · P_j / d_ij^γ`, with `γ = 1` by default (configurable), normalized so that `Σ T_ij = 1`.
- Populations come from GUS (city population). Curate them into the instance file, so the code needs no network access.
- Unweighted demand (`T_ij` constant) must remain available. Phase 1's `D` is unweighted, and the 2B comparison needs both.

**Done when:** `scripts/build_rail_graph.py` turns the existing cache into `data/reference/rail_graph_pl.{npz,json}` in under 2 minutes, and all of the following hold:
- Tests on a tiny hand-made Overpass JSON cover junction splitting, parallel-track merge, degree-2 contraction and missing tags.
- Station instances load offline.
- A sanity check passes: Warszawa Centralna → Kraków Główny network distance is within ±10% of the published ~290 km.

---

### 2B. Real-network baseline and graph comparison

**Goal:** a numerical comparison between FGO and the real network. It is useful on its own, and it is the yardstick for 2C and 2D.

**Reference subnetwork.** The full Polish network is far bigger than what links 15–30 stations, so scoring all of it would be unfair. Define the *reference subnetwork* for a station set `S` as the union of the edges used by the shortest paths between all station pairs in `S`. Its leaves are stations by construction.

**Metrics** (`problem/reference.py`). These use the same definitions as `problem/metrics.py`, so the numbers are directly comparable:

| Metric | Real network | FGO network |
| --- | --- | --- |
| `C` | km of reference subnetwork (×tracks optionally, reported separately) | km of decoded graph |
| `D` | mean of (network path km / straight km) over station pairs | same |
| `FT` | share of edges that are not station-separating bridges | same |
| `J` | normalized by the **same** station-only MST baseline | same |
| `D_time` | mean travel time / (straight km / 160 km/h) | n/a (Euclidean has no speeds) |

Shape similarity, in addition to the score:

- **Buffer overlap:** share of FGO length within `d ∈ {2, 5, 10}` km of the reference subnetwork, and vice versa. Report precision and recall, plus the random-segment control from Phase 1.
- **Pair-level agreement:** Spearman correlation between FGO path length and real path length over station pairs. It answers "are the same pairs hard to connect?"

**CLI:** `grzybki compare RUN_DIR --rail-graph data/reference/rail_graph_pl` writes `comparison.json` and `figures/comparison.png`. The figure shows the reference subnetwork, the FGO network, and the full rail network in light grey.

**Done when:** `compare` runs on the Phase 1 runs and reports `J_real` against `J_FGO`, and tests verify that the reference subnetwork on a toy graph matches a hand-computed result.

---

### 2C. Formulation A: corridor-aware Steiner network

Same encoding as Phase 1: `x = (x1, y1, …, xK, yK, θ)`. Only the **edge cost** changes. A straight link `(u, v)` costs the integral of a cost surface `c(p)` along the segment:

```text
cost(u, v) = ∫ c(p) ds   ≈  Σ_samples c(p_s) · Δs
```

**Cost surface** (`problem/costsurface.py`). A raster over Poland in EPSG:2180. A 1 km grid is about 700 × 650 cells; 2 km is about 350 × 325. The surface is built offline by `scripts/build_cost_surface.py`, stored as `.npz`, and has layers:

| Layer | Factor | Source | Phase |
| --- | --- | --- | --- |
| Base | 1.0 | — | MVP |
| Existing rail corridor | `α` (default 0.3) for cells within `r` km (default 1) of track | OSM rail graph from 2A (rasterized) | **MVP** |
| Urban area | ×2–3 | OSM `landuse=residential`, or CORINE Land Cover | later |
| Water | ×5 (rivers need bridges) | OSM water, or CORINE | later |
| Protected areas | ×3, or forbidden | Natura 2000 (EEA / GDOŚ) | later |
| Slope | `1 + β·slope` | Copernicus DEM GLO-90 (or GUGiK NMT) | later |

**Start with the rail-corridor layer only.** It needs no new data, and it directly tests "does FGO follow corridors when following them is cheaper?" against the 2B overlap metric. Add the other layers one at a time behind CLI flags, each with its own attribution recorded in `manifest.json`.

**Decoder** (`problem/corridor.py`, `kind="corridor"`):

1. Points = stations + K Steiner points. Steiner points map from `[0,1]²` into the raster extent.
2. Pairwise costs for all `(n+K)²/2` pairs: sample each segment every ~1 raster cell, with vectorized bilinear lookup. For n+K ≈ 60 that is ~1,800 pairs × ~300 samples ≈ 0.5M lookups per evaluation, a few milliseconds in NumPy. Cache the station–station pairs once.
3. MST over this **cost** matrix, not Euclidean length.
4. Delaunay candidates as before, with extra edges sorted by cost. `θ` behaves as in Phase 1.
5. Prune degree-1 Steiner points.
6. Metrics:
   - `C_cost / C_cost_MST`: construction cost, normalized by the station-only cost-MST.
   - `D`: geometric detour, unchanged from Phase 1.
   - `FT`: unchanged.

   Same weights and the same `J` form, so `J = 1` is still "the cheapest station-only tree".

**Post-processing, for display only.** For the final best network, route each link along the least-cost path on the raster (`skimage.graph.route_through_array`, or SciPy Dijkstra on the grid graph). The map then shows realistic curved corridors, and we report "routed cost vs straight-line cost". Do not do this inside the optimization loop; it is too slow.

**Experiments:**
- Sweep `α ∈ {1.0, 0.6, 0.3, 0.1}`. At `α = 1` the formulation reduces to Phase 1 and must reproduce Phase 1 `J`; this is a regression check.
- Plot overlap(5 km) and `J` against `α`.
- Expected: overlap rises well above the 47% random control as `α` falls.

**Done when:** `grzybki optimize poland_stations_30 --problem corridor --cost-surface …` runs at ≥100 evals/s, the `α = 1` regression holds, and `compare` shows overlap rising as `α` falls.

---

### 2D. Formulation B: existing-network upgrade / subgraph selection

The optimization now acts **on the real graph**. It decides, per corridor, whether to close it, keep it, or upgrade it, and whether to build a short list of new links.

**Corridor aggregation.** The full graph has too many edges for a decision per edge. Starting from the 2B reference subnetwork for the chosen stations, contract every maximal path between stations and junctions of degree ≥3 into one **corridor**. Expected size: 50–150 corridors for 30 stations.

**Candidate new links:** Delaunay edges between stations that have no direct corridor and whose network detour is above 1.4. Cap them at ~20, as straight links with a new-line cost.

**Encoding:** random keys. `x ∈ [0,1]^(|corridors| + |new links|)`.

- Existing corridor `e`:
  - `x_e < 0.2` → close.
  - `< 0.6` → keep.
  - `< 0.85` → upgrade to 160 km/h.
  - else → upgrade to 200 km/h, with electrification if needed.
- New link `l`: `x_l > 0.5` → build at 200 km/h.
- Thresholds go in config. Every level change is a step, so plateaus are expected, as with `θ` in Phase 1.

**Repair** (deterministic): if the closures disconnect any station, reopen the closed corridors in order of increasing cost until all stations are connected, using Kruskal over closed corridors with their keep cost. Log how often repair happened, and put the count in `details`.

**Objectives** (`problem/upgrade.py`, `kind="upgrade"`). Every term is normalized by the "do nothing" network, so the status quo scores `J = 1`:

| Term | Definition |
| --- | --- |
| `cost` | upgrade capex + new-line capex − closure savings (maintenance over a horizon), using unit costs per km from a params file |
| `time` | demand-weighted mean travel time between station pairs (`T_ij` from 2A), with edge time = length / effective speed |
| `resilience` | demand-weighted share of OD pairs that lose connectivity, or whose travel time rises >50%, under single-corridor failure. Compute exact loss only for the ~10 most loaded corridors per evaluation, and use bridges for the rest |

`J = w1·cost/cost_0 + w2·time/time_0 + w3·(1 − resilience)`. Note that cost_0 = 0 for "do nothing", so normalize cost by a reference budget instead (e.g. the cost of upgrading all corridors to 160 km/h).

**Unit costs** live in `data/params/upgrade_costs_pl.json`: per-km cost of a speed upgrade by level, electrification, a second track, a new line, and annual maintenance. Fill them from published PKP PLK / CPK / EU-funded project figures, **cite every number, and treat them as uncertain**. A sensitivity sweep of ±50% on each parameter is part of 2F.

**Pareto front, without changing FGO:**
- `history.jsonl` already logs every evaluated solution's metrics. Extract the non-dominated set over `(cost, time, 1−resilience)` from the history of each run. This gives a front almost for free.
- Also run a weight sweep, e.g. `w1 ∈ {0.1, …, 0.9}` with `w2 = w3 = (1 − w1)/2`, and merge the archives.
- Report hypervolume against the "do nothing" reference point.

**Done when:** `--problem upgrade` runs on `poland_stations_30`; "do nothing" scores exactly 1.0; repair always yields connected networks (property-based test on random keys); and a Pareto front with ≥10 non-dominated points is produced from a weight sweep.

---

### 2E. Visualization

- `viz/railmap.py`: base map with the Poland outline (Natural Earth admin-0, bundled small GeoJSON, public domain), station labels, and the full rail network in light grey.
- **Comparison map** (2B): reference subnetwork vs FGO network, with an overlap legend.
- **Corridor map** (2C): least-cost-routed links over the cost-surface heat map.
- **Decision map** (2D): corridors coloured by decision (closed / keep / 160 / 200 / new), with line width ∝ demand flow.
- **Pareto plots** (2D): 2-D projections plus a parallel-coordinates plot. Highlight the knee point and the status quo.
- `report.html`: add sections per problem kind, chosen from `manifest.problem.kind`.

**Done when:** `grzybki visualize` produces the right figure set for each problem kind, and figures are readable at report size: labels don't overlap for 30 stations, and colour-blind-safe palettes are used.

---

### 2F. Experiments, baselines and documentation

A metaheuristic result is only meaningful against baselines and across seeds.

- **Seeds:** 10 seeds per configuration. Report median and IQR, not the single best.
- **Baselines** with the same evaluation budget, behind the same `objective` wrapper:
  - Random search.
  - `scipy.optimize.differential_evolution`, with evaluations counted by the wrapper and stopped at the budget.
  - Optionally a simple PSO.
- **Statistics:** Wilcoxon signed-rank test, FGO vs each baseline, per instance.
- **Experiment matrix**, driven by an extended `scripts/benchmark.py`, which writes one JSON per cell:

| Formulation | Instances | Sweeps |
| --- | --- | --- |
| Euclidean (Phase 1, re-run on station coordinates) | stations_15, stations_30 | seeds |
| Corridor (A) | stations_15, stations_30 | `α`, then added cost layers |
| Upgrade (B) | stations_30 | weights (Pareto), unit-cost sensitivity ±50% |

- **Docs:**
  - `docs/phase2_spec.md`: formulations, encodings, metrics and repair, with every deviation and default stated.
  - Update `docs/data.md` with the new sources, licences, retrieval dates and attribution.
  - `docs/results.md`: tables and figures for the report.

**Budget check.** At ≥100 evals/s, 20k evals take about 3.5 min. The matrix is roughly 3 formulations × 2 instances × ~5 settings × 10 seeds × 4 algorithms ≈ 1,200 runs ≈ 70 CPU-hours. Run cells in parallel processes (one run per process; FGO itself stays sequential), or cut to 5 seeds and 2 baselines.

## 5. Commit plan

The rule is the same as Phase 1: one logical change per commit, tests and a CLI smoke run before each commit, and no generated data or `runs/` in Git.

| # | Commit | Workstream | Done when |
| --- | --- | --- | --- |
| 16 | `feat(viz): city labels and Poland outline` | P1 | ✅ network.png shows labels and outline |
| 17 | `refactor(problem): Problem protocol, dict metrics, km units` | P2 ✅, P4 | Phase 1 tests pass unchanged; metrics in km |
| 18 | `perf(metrics): scipy csgraph shortest paths` | P3 | ✅ identical metrics on tests; 2.1× faster on poland_30 |
| 19 | `feat(export): schema v2 with problem kind` | P5 | v1 runs still load |
| 20 | `feat(rail): parse OSM cache and build topological rail graph` | 2A | toy-cache tests; real build < 2 min |
| 21 | `feat(rail): stations, snapping, and station instances` | 2A | `poland_stations_{15,30}.json` load offline |
| 22 | `feat(rail): gravity demand model` | 2A | `Σ T = 1`; unweighted mode |
| 23 | `feat(problem): reference subnetwork and real-network metrics` | 2B | toy graph matches hand result |
| 24 | `feat(cli): grzybki compare` | 2B | `J_real` vs `J_FGO` on Phase 1 runs |
| 25 | `feat(problem): cost surface with rail-corridor layer` | 2C | raster builds; lookup tests |
| 26 | `feat(problem): corridor-aware Steiner decoder` | 2C | `α = 1` reproduces Phase 1 |
| 27 | `feat(viz): least-cost routing and corridor map` | 2C, 2E | routed links render |
| 28 | `feat(problem): corridor aggregation and upgrade encoding with repair` | 2D | property test: always connected |
| 29 | `feat(problem): upgrade objectives and unit-cost params` | 2D | do-nothing `J = 1` |
| 30 | `feat(viz): decision map and Pareto plots` | 2D, 2E | front with ≥10 points |
| 31 | `feat(benchmark): baselines, seeds, and experiment matrix` | 2F | DE/random under same budget |
| 32 | `feat(problem): optional cost layers (land use, water, Natura 2000, DEM)` | 2C | each layer behind a flag, with attribution |
| 33 | `docs(phase2): spec, data sources, results` | 2F | results reproducible from docs |

**Minimum viable Phase 2** = commits 16–27 (comparison plus corridor formulation with the rail layer only). Commits 28–30 (formulation B) and 32 (extra cost layers) are the stretch goals.

## 6. Architecture after Phase 2

```text
src/grzybki/
  problem/
    base.py            # Problem protocol, Evaluation
    decoder.py         # Phase 1 Euclidean (kind="euclidean")
    reference.py       # reference subnetwork + real-network metrics
    costsurface.py     # raster layers, segment integrals
    corridor.py        # formulation A (kind="corridor")
    upgrade.py         # formulation B (kind="upgrade"), corridors, repair
  rail/
    osm.py  topology.py  graph.py  stations.py  demand.py
  fgo/                 # unchanged
  viz/
    railmap.py  pareto.py  (+ existing)
scripts/
  fetch_osm_rail.py  fetch_osm_stations.py  fetch_gtfs.py
  build_rail_graph.py  build_station_instance.py  build_cost_surface.py
  benchmark.py
data/
  instances/poland_stations_{15,30}.json   # curated, committed
  params/upgrade_costs_pl.json             # cited unit costs, committed
  reference/                               # fetched/built, gitignored
```

CLI additions: `optimize --problem {euclidean,corridor,upgrade}` with problem-specific options (`--rail-graph`, `--cost-surface`, `--alpha`, `--params`), and `grzybki compare`.

New dependencies: `pyproj` moves from optional to the Phase 2 default. `scikit-image` is optional, for raster routing; SciPy is the fallback. For DEM and land-cover layers, `rasterio` is optional and only needed by `build_cost_surface.py`, never at optimize time.

## 7. Decisions needed from the team

| Decision | Options | Recommendation |
| --- | --- | --- |
| Scope for the semester | MVP (A only) vs A + B | MVP first; start B only once commit 27 is merged |
| Station selection | Phase 1 city list → main station, vs GTFS top-k | City list: keeps Phase 1 and Phase 2 comparable |
| Instance sizes | 15 / 30 / 50 stations | 15 and 30; 50 only if runtime allows |
| Cost layers beyond rail | none / land use + water / + DEM | Rail-only for results; add layers as sensitivity only if time remains |
| Unit costs for B | literature values vs purely relative costs | Relative costs (new line = 1.0, upgrade = 0.2–0.4, …) plus a sensitivity sweep; avoids defending exact zł figures |
| Border connections | ignore / add virtual cross-border edges | Ignore for a domestic station set; document it |

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| OSM tag gaps or errors (missing `maxspeed`, mistagged `usage`) | Defaults plus counters reported in `manifest.json`; sanity checks on known city-pair distances and times |
| Parallel-track merge fails (tracks diverge at stations) | Merge by `ref` + endpoint proximity; validate total km against published PLK network length (~19,000 km of lines); visual check |
| Border clipping disconnects lines (e.g. near Czech/German border) | Keep the largest component, log what was dropped; station sets avoid border-only stations |
| ODbL share-alike on derived graph | Keep built graphs in gitignored `data/reference/`; commit only scripts and small curated instances, with attribution |
| Formulation B dimension (100–200) slows convergence | Corridor aggregation; cap new links; compare against DE to see whether FGO is the bottleneck |
| Plateaus from discrete decoding (`θ`, upgrade levels) | Already documented for `θ`; consider accepting equal fitness as a config flag (`--accept-equal`) and report its effect as a deviation |
| Runtime of the experiment matrix | P3 speed-up first; parallel runs; reduce seeds if needed |
| Unit-cost figures are contestable | Relative costs plus ±50% sensitivity; cite sources |

## 9. Acceptance criteria for Phase 2 (MVP)

- `scripts/build_rail_graph.py` and `scripts/build_station_instance.py` reproduce the rail graph and station instances from a clean checkout, given network access.
- `grzybki compare` reports `J_real` and `J_FGO` with the same baseline, plus buffer overlap against a random control.
- `grzybki optimize --problem corridor` runs on `poland_stations_30`. `α = 1` reproduces Phase 1, and lower `α` measurably raises corridor overlap.
- All results across 10 seeds, with a DE baseline under the same budget.
- Docs state every formulation choice, default, data source and licence.
