# Data sources and formats

## Poland instances

`scripts/fetch_poland_cities.py` fetches a population ranking of Polish settlements from GeoNames and writes one instance file per size. All files are nested prefixes of one ranking, so `poland_15` is a subset of `poland_30`, which is a subset of `poland_50`, and so on:

| File | Terminals | Purpose |
| --- | --- | --- |
| `data/instances/poland_15.json` | 15 | Fast experiments and smoke runs |
| `data/instances/poland_30.json` | 30 | Main Phase 1 comparison runs |
| `data/instances/poland_{50,100,500,1000,5000}.json` | 50–5,000 | Scaling studies and stress runs |

Regenerate them (writes JSON files into `data/instances/`, caches the ~2 MB dump under `data/reference/`, which is ignored by Git):

```bash
python3 scripts/fetch_poland_cities.py                       # default ladder 15..5000
python3 scripts/fetch_poland_cities.py --sizes 50 100        # subset
python3 scripts/fetch_poland_cities.py --force               # re-download the dump
```

Provenance: GeoNames Poland dump (`https://download.geonames.org/export/dump/PL.zip`), retrieved 2026-11-17. Polish populated places (feature class `P`) with population > 0 are ranked by population; city sections (`PPLX`), historical/abandoned places, and districts recorded as standalone places (Warsaw boroughs, Fordon) are dropped so no city appears twice. Coordinates are rounded to 4 decimals (~11 m). Names follow GeoNames spelling (for example `Warsaw`, not `Warszawa`).

Attribution: GeoNames, CC BY 4.0 — keep the attribution when redistributing derived data.

Each file has this shape:

```json
{
  "name": "poland_15",
  "description": "Polish settlement terminal set (15 largest places by population) for idealized FGO experiments.",
  "source": "GeoNames Poland dump (PL.zip), download.geonames.org, retrieved 2026-11-17; see docs/data.md",
  "license": "CC-BY-4.0 (GeoNames); attribution required.",
  "crs": "EPSG:4326",
  "terminals": [
    {"name": "Warsaw", "lon": 21.0111, "lat": 52.23},
    {"name": "Kraków", "lon": 19.9366, "lat": 50.0614}
  ]
}
```

Coordinates are approximate settlement centres, not station locations. They are suitable for idealized experiments and must be replaced with verified station or node coordinates before operational use.

The files load without network access:

```python
from grzybki.problem import Instance, NetworkProblem

instance = Instance.from_json("data/instances/poland_15.json")
instance.project()
problem = NetworkProblem(instance, K=13)
```

## Synthetic instances

`data/instances/` also ships synthetic terminal sets produced by `scripts/gen_instances.py`. Their terminals carry unit-square `x`/`y` coordinates (`"crs": "unit-square"`), not longitude/latitude: plotting skips the country outline for them, and kilometre metrics such as `C_km` are unavailable because there is no geographic scale.

| Files | Terminals | Layout |
| --- | --- | --- |
| `polygon_{15,30,50,100,500,1000,5000}.json` | 15, 30, 50, 100, 500, 1,000, 5,000 | exact vertices of a regular n-gon (radius 0.45, first vertex at the top) |
| `uniform_{15,30,50,100,500,1000,5000}.json` | 15, 30, 50, 100, 500, 1,000, 5,000 | uniform random points, seed 0 |
| `clustered_{15,30,50,100,500,1000,5000}.json` | 15, 30, 50, 100, 500, 1,000, 5,000 | 3 Gaussian clusters (spread 0.08), seed 0 |
| `ring_{15,30,50,100,500,1000,5000}.json` | 15, 30, 50, 100, 500, 1,000, 5,000 | noisy ring around the centre, seed 0 |
| `grid_{15,30,50,100,500,1000,5000}.json` | 15, 30, 50, 100, 500, 1,000, 5,000 | near-regular grid |

This gives 35 reproducible synthetic instances across five layouts, from small smoke-test inputs to 5,000-terminal stress-test inputs. The larger sizes are useful for scaling tests and may require substantially more runtime and memory when optimized.

The regular-polygon family gives controlled, highly symmetric benchmark sets; the exact polygon instances need no randomness and are fully reproducible, while a `--jitter` option perturbs the vertices.

Generate the full size range (writes JSON files into `data/instances/` by default):

```bash
for kind in uniform clustered ring grid polygon; do
  python3 scripts/gen_instances.py --kind "$kind" \
    --sizes 15 30 50 100 500 1000 5000 --seed 0
done
```

To generate only one layout or a subset of sizes, pass the desired `--kind` and `--sizes`, for example `python3 scripts/gen_instances.py --kind uniform --sizes 50 100 --seed 3`. The default invocation still generates only the default polygon family.

In Python the same layouts are available through `grzybki.problem.synthetic_instance(kind, n, seed=...)` with `kind` in `uniform`, `clustered`, `ring`, `grid`, `polygon` (polygon accepts `radius`, `rotation_deg`, `jitter`).

## Projection

The default projection is a local equirectangular projection fitted to the instance extent, with a 5% margin, normalized to `[0, 1]^2`. This is adequate for the idealized Phase 1 decoder and keeps distances in a comparable unit.

For Poland-wide work, `epsg2180` (PUWG 1992) is available when `pyproj` is installed:

```bash
python3 -m pip install -e '.[geo]'
```

The projection metadata is recorded in `manifest.json`. Raw longitude/latitude values must not be passed directly to the decoder.

## Country outline (bundled)

`src/grzybki/data/outline_PL.geojson` is Poland's border from Natural Earth 1:50m admin-0 countries, v5.1.2, rounded to 4 decimals (6.5 KB, 316 points). `grzybki visualize` draws it under network plots. Natural Earth is public domain; plots carry the note "Made with Natural Earth". To regenerate it, or to add another country:

```bash
python3 scripts/build_country_outline.py --iso PL --scale 50m
```

The raw download is cached under `data/reference/` (gitignored). Only countries listed in `grzybki.viz.basemap.BUNDLED_OUTLINES` are considered when plotting.

## Optional fetchers

Fetchers are optional, require network access, and write to `data/reference/`, which is ignored by Git. They cache raw responses so repeated runs do not re-query remote services.

### OpenStreetMap rail lines

```bash
python3 scripts/fetch_osm_rail.py --out data/reference/osm_rail_poland.geojson
```

By default the script queries the Overpass API for ways tagged `railway=rail` with `usage=main` or `usage=branch`, clipped to Poland's administrative boundary (`--country PL`), and drops service tracks (`service=*`: sidings, yards, spurs). It writes GeoJSON `LineString` features. In November 2026 this returned about 27,500 ways (about 500,000 vertices, 28 MB GeoJSON) and took about 2.5 minutes.

Useful overrides:

- `--bbox min_lon,min_lat,max_lon,max_lat` replaces the country boundary with a bounding box.
- `--railway 'rail|light_rail|narrow_gauge'` widens the `railway=*` filter (a regex).
- `--usage ''` disables the `usage=*` filter; `--include-service` keeps service tracks.

The default cache is `data/reference/osm_rail_poland.json`. Use `--force` to refresh the cache; changing filters without `--force` reuses the old cached response.

Overlay the result on a run with:

```bash
grzybki visualize runs/poland_30_seed42 --reference data/reference/osm_rail_poland.geojson
```

Attribution: © OpenStreetMap contributors, ODbL 1.0.

### BDOT10k WFS

```bash
python3 scripts/fetch_bdot10k.py \
  --type-name ms:BDOT10k_powiaty \
  --out data/reference/bdot10k_counties.geojson
```

The GUGiK WFS endpoint is:

```text
https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k
```

The service exposes the county index (`ms:BDOT10k_powiaty`) as a WFS layer. Railway geometry is distributed in county BDOT10k packages rather than as a single nationwide WFS layer, so the default script fetches the county index and the caller can pass another available `--type-name`. The layer name and endpoint should be re-checked against the current GUGiK service before production use.

Attribution: GUGiK BDOT10k open data.

## Reference data for Phase 2

The following sources are useful for the real-network phase. They are not bundled, and each has its own license and attribution requirements.

| Source | Use | License/terms |
| --- | --- | --- |
| OpenStreetMap / Geofabrik | Rail geometry, stations, tags | ODbL 1.0 |
| OpenRailwayMap | Rendered rail infrastructure reference | ODbL 1.0 for OSM-derived data |
| GUGiK BDOT10k | Topographic rail and infrastructure layers | Open data, attribution required |
| PKP PLK PDP API | Line and station infrastructure data | PKP PLK terms; verify before redistribution |
| PKP PLK SILK WMS | Reference maps and visualization | PKP PLK terms |
| UTK | Rail statistics and network reports | Public reports; check terms |
| GUS | City populations and administrative data | Public statistics; check terms |
| PolishTrainsGTFS | Schedules and station lists | GTFS terms; check upstream license |
| GISCO TEN-T | European TEN-T network geometry | European Commission terms; attribution required |

When combining sources, keep source names, retrieval dates, licenses, and attribution in the derived dataset metadata.

## Comparison helper

`grzybki.problem.comparison` measures how far decoded edges are from reference rail lines. The graph and reference lines must be in the same normalized coordinate space.

```python
import numpy as np
from grzybki.problem import (
    Instance,
    NetworkProblem,
    load_geojson_lines,
    project_lines,
    compare_network_to_reference,
)

instance = Instance.from_json("data/instances/poland_15.json")
instance.project()
problem = NetworkProblem(instance, K=13)

lines = load_geojson_lines("data/reference/osm_rail_poland.geojson")
lines = project_lines(instance.projection, lines)

result = problem.evaluate(np.zeros(problem.dimension))
comparison = compare_network_to_reference(result.graph, lines)
print(comparison["summary"])
```

`compare_network_to_reference` uses each edge midpoint and reports mean, median, 95th percentile, and maximum distance to the nearest reference polyline. This is a diagnostic, not part of the objective.

## Data hygiene

- Do not commit large raw downloads; `data/reference/` is ignored.
- Committed Poland/synthetic instances are produced by reviewed scripts; keep the scripts, not large raw downloads, as the source of truth.
- Record the retrieval date, endpoint, query, and license for every external dataset.
- Treat approximate city centres as placeholders, not as station coordinates.
- Check that coordinates are in the expected CRS before projecting.
