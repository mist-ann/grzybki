# Data sources and formats

## Curated instances

The repository ships two offline terminal sets:

| File | Terminals | Purpose |
| --- | --- | --- |
| `data/instances/poland_15.json` | 15 | Fast experiments and smoke runs |
| `data/instances/poland_30.json` | 30 | Main Phase 1 comparison runs |

Each file has this shape:

```json
{
  "name": "poland_15",
  "description": "Approximate city-centre terminals for an idealized Polish network",
  "source": "Approximate city-centre coordinates; see docs/data.md",
  "license": "CC0-1.0",
  "crs": "EPSG:4326",
  "terminals": [
    {"name": "Warszawa", "lon": 21.0122, "lat": 52.2297},
    {"name": "Krakow", "lon": 19.9445, "lat": 50.0647}
  ]
}
```

Coordinates are approximate city centres, not station locations. They are suitable for idealized experiments and must be replaced with verified station or node coordinates before operational use.

The files load without network access:

```python
from grzybki.problem import Instance, NetworkProblem

instance = Instance.from_json("data/instances/poland_15.json")
instance.project()
problem = NetworkProblem(instance, K=13)
```

## Projection

The default projection is a local equirectangular projection fitted to the instance extent, with a 5% margin, normalized to `[0, 1]^2`. This is adequate for the idealized Phase 1 decoder and keeps distances in a comparable unit.

For Poland-wide work, `epsg2180` (PUWG 1992) is available when `pyproj` is installed:

```bash
python3 -m pip install -e '.[geo]'
```

The projection metadata is recorded in `manifest.json`. Raw longitude/latitude values must not be passed directly to the decoder.

## Optional fetchers

Fetchers are optional, require network access, and write to `data/reference/`, which is ignored by Git. They cache raw responses so repeated runs do not re-query remote services.

### OpenStreetMap rail lines

```bash
python3 scripts/fetch_osm_rail.py \
  --bbox 14.0,49.0,24.2,54.9 \
  --out data/reference/osm_rail_poland.geojson
```

The script queries the Overpass API for ways tagged `railway` with `rail`, `light_rail`, `narrow_gauge`, `subway`, or `tram`, and writes GeoJSON `LineString` features. The default cache is `data/reference/osm_rail_poland.json`. Use `--force` to refresh the cache.

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
- Keep curated instances small and reviewable.
- Record the retrieval date, endpoint, query, and license for every external dataset.
- Treat approximate city centres as placeholders, not as station coordinates.
- Check that coordinates are in the expected CRS before projecting.
