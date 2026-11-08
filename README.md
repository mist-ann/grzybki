# grzybki

Fungal Growth Optimizer (FGO) for idealized railway network design, with a focus on Polish city terminal sets.

The project encodes a network as Steiner points plus a redundancy parameter, decodes each vector into a graph with an MST and extra Delaunay edges, and minimizes a normalized cost/detour/fault-tolerance objective. Phase 1 uses an idealized Euclidean decoder; real PKP PLK geometry and terrain-aware costs are Phase 2 work.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e '.[dev]'   # tests
python3 -m pip install -e '.[geo]'   # optional EPSG:2180 projection via pyproj
```

Python 3.10+ is required. The runtime dependencies are NumPy, SciPy, Matplotlib, NetworkX, Click, and imageio. FFmpeg is optional and only needed for MP4 animations.

## Quick start

```bash
grzybki optimize data/instances/poland_15.json \
  --seed 42 --max-evals 20000 --out runs/poland_15_seed42

grzybki visualize runs/poland_15_seed42 \
  --out runs/poland_15_seed42/figures --animation
```

For a fast smoke run:

```bash
grzybki optimize data/instances/poland_15.json \
  --seed 1 --max-evals 300 --population 10 --snapshot-interval 50 \
  --out runs/smoke

grzybki visualize runs/smoke --out runs/smoke/figures --animation --fps 5
```

Run the test suite with:

```bash
python3 -m pytest
```

## Algorithm summary

A solution is a vector in `[0, 1]^(2K+1)`:

```text
x = (x1, y1, ..., xK, yK, theta)
```

The decoder:

1. Adds the `K` Steiner points to the fixed terminals.
2. Builds the Euclidean minimum spanning tree over all points.
3. Computes the Delaunay triangulation and adds the first `floor(theta * m)` non-MST edges.
4. Recursively removes degree-1 Steiner points.
5. Computes the objective.

The objective is:

```text
J = w1 * C / C_MST + w2 * D / D_MST + w3 * (1 - FT)
```

Defaults are `(w1, w2, w3) = (0.4, 0.4, 0.2)`. The terminal-only MST has `J = 1`; lower values are better. See [docs/algorithm_spec.md](docs/algorithm_spec.md) for formulas, draw order, and documented deviations.

## CLI

### `optimize`

```text
grzybki optimize [INSTANCE] [OPTIONS]
```

`INSTANCE` is a path or a name under `data/instances/` (for example `poland_15`). The `--instance` option is an alias, so `grzybki optimize --instance poland_15` also works.

| Option | Default | Meaning |
| --- | --- | --- |
| `--out PATH` | `runs/<instance>_seed<seed>` | Run directory |
| `--seed INT` | `42` | RNG seed; use `-1` for a random seed |
| `--population INT` | `30` | FGO population size `N` |
| `--max-evals INT` | `20000` | Evaluation budget, including initialization |
| `--K INT` | `n - 2` | Number of Steiner points |
| `--M FLOAT` | `0.6` | Exploration threshold parameter |
| `--Ep FLOAT` | `0.7` | Extra exploration step probability |
| `--R FLOAT` | `0.9` | Movement-toward-best threshold |
| `--w1/--w2/--w3 FLOAT` | `0.4/0.4/0.2` | Objective weights |
| `--projection` | `equirectangular` | `equirectangular` or `epsg2180` |
| `--nutrient-mode` | `fitness` | `fitness` or `inverse` |
| `--snapshot-interval INT` | `100` | Snapshot cadence in evaluations |
| `--checkpoint-interval INT` | `1000` | Checkpoint cadence; `0` disables periodic checkpoints |
| `--save-vectors` | off | Include candidate vectors in `history.jsonl` |
| `--resume PATH` | off | Resume from a checkpoint directory |

### `visualize`

```text
grzybki visualize RUN_DIR [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out PATH` | `RUN_DIR/figures` | Figure directory |
| `--animation` | off | Write a GIF or MP4 animation from snapshots |
| `--animation-format` | `gif` | `gif` or `mp4` |
| `--fps INT` | `10` | Animation frame rate |
| `--dpi INT` | `140` | Figure resolution |
| `--reference PATH` | off | Optional GeoJSON reference lines |
| `--report` | on | Write `report.html` |

## Run artifacts

```text
runs/<run>/
  manifest.json          # instance, problem, config, environment, schema version
  summary.json           # best score, best vector, evaluations, runtime
  best.json              # best vector, metrics, decoded graph
  history.jsonl          # one record per evaluation
  snapshots.jsonl        # periodic best vectors and population fitness summaries
  checkpoints/
    state.npz            # population, fitness, best vector, t, iteration
    state.json           # RNG state
  figures/
    convergence.png
    metrics.png
    network.png
    animation.gif
    report.html
```

`history.jsonl` records include `t`, `iteration`, `phase`, `operator`, `accepted`, `score`, `best_score`, and optional `details.metrics` plus `details.theta`. With `--save-vectors`, each record also includes the candidate `vector`.

## Data

The repository includes two curated, offline terminal sets:

- `data/instances/poland_15.json`
- `data/instances/poland_30.json`

Coordinates are approximate city centres, not station locations. They are intended for idealized experiments. See [docs/data.md](docs/data.md) for sources, licenses, and optional fetchers.

Optional fetchers write to `data/reference/`, which is ignored by Git:

```bash
python3 scripts/fetch_osm_rail.py --out data/reference/osm_rail_poland.geojson
python3 scripts/fetch_bdot10k.py --out data/reference/bdot10k_wfs.geojson
```

OSM data is © OpenStreetMap contributors and available under ODbL 1.0. BDOT10k is open data from GUGiK. Always retain attribution when redistributing derived data.

## Benchmarks

```bash
python3 scripts/benchmark.py \
  --max-evals 20000 --population 30 --seed 42 \
  --out runs/benchmark_20k.json
```

On the development machine, the default 20,000-evaluation budget takes about 65 s for `poland_15` and 212 s for `poland_30`. Use `--max-evals 2000` while iterating, and `--repeat` or `--sweep` for sensitivity studies. See [docs/benchmarks.md](docs/benchmarks.md) for measured runtimes, the `M` sensitivity sweep, and recommended settings.

## Limitations

- The Phase 1 decoder is Euclidean and idealized; it does not use terrain, real rail geometry, schedules, capacity, or construction costs.
- The curated Polish coordinates are approximate and must be verified before operational use.
- The real PKP PLK graph formulation, corridor-aware Steiner networks, and existing-network upgrades are Phase 2 work.
- `theta` has flat regions because it enters through `floor(theta * m)`; strict selection rejects many neutral moves.

## Repository layout

```text
src/grzybki/
  config.py            # objective and FGO configuration
  problem/             # instances, projection, decoder, graph, metrics, comparison
  fgo/                 # operators, RNG, optimizer, checkpoints
  export/              # schema and JSONL writers
  viz/                 # convergence, network, animation, report
  cli.py               # Click entry points
data/instances/        # curated terminal sets
scripts/               # OSM/BDOT10k fetchers and benchmark harness
tests/                 # pytest suite
docs/                  # algorithm, data, and benchmark documentation
```

## License

Code is released under the MIT License. Data files retain their own licenses and attribution requirements; see [docs/data.md](docs/data.md).
