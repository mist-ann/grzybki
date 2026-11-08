# FGO Polish Railway Network — Implementation Plan

Status: draft  
Source specification: `FGO_opis_algorytmu.docx`  
Primary reference: Abdel-Basset, Mohamed, Abouhawwash, *Fungal growth optimizer*, CMAME 437 (2025) 117825.

## 1. Goal

Implement the Fungal Growth Optimizer (FGO) in Python with a CLI that can:

1. optimize a network of Polish cities using the idealized FGO/Steiner decoder;
2. export the full optimization process in a reproducible format;
3. visualize convergence, metrics, the final network, and network evolution;
4. compare the result with the real Polish railway network;
5. later extend the work to a graph-based formulation using actual PKP PLK data.

## 2. Locked decisions

- **Scope:** both, phased.
  - Phase 1: idealized Polish intercity network.
  - Phase 2: extension to real PKP PLK network data.
- **Steiner points:** `--K` configurable, default `K = n - 2`.
  - `n = 15` → `K = 13`, `d = 27`.
  - `n = 30` → `K = 28`, `d = 57`.
- **Export:** JSONL history/snapshots + NPZ checkpoints.
- **Visualization:** PNG + GIF/MP4; no Plotly requirement.
- **Data:** curated offline instances by default; optional OSM/BDOT10k fetchers.
- **Dependencies:** add `pytest` and optional `pyproj`.
- **Runtime target:** 20,000 evaluations for `n = 15` should complete comfortably on a laptop; benchmark before optimizing prematurely.

## 3. Algorithm summary

### 3.1 Encoding

Each hypha is a vector

```text
x = (x1, y1, ..., xK, yK, theta) in [0, 1]^(2K+1)
```

`theta` controls the number of extra Delaunay edges:

```text
extra_edges = floor(theta * m)
```

where `m` is the number of Delaunay edges not already in the MST.

### 3.2 Decoder

For a candidate vector:

1. Read `K` Steiner points.
2. Build `V = terminals + Steiner points`.
3. Compute the Euclidean MST over `V`.
4. Compute Delaunay triangulation of `V`.
5. Sort non-MST Delaunay edges by length and add the first `floor(theta * m)`.
6. Recursively remove Steiner points of degree `<= 1` and their edges.
7. Compute `C`, `D`, `FT`, and `J`.

### 3.3 Objective

```text
J = w1 * C / C_MST + w2 * D / D_MST + w3 * (1 - FT)
w1 + w2 + w3 = 1
```

Defaults: `(w1, w2, w3) = (0.4, 0.4, 0.2)`.

- `C`: total network length.
- `D`: unweighted mean pairwise terminal detour ratio, following docx eq. (5).
- `FT`: fraction of edges that are not bridges.
- `C_MST`, `D_MST`: baselines from the terminal-only MST.

### 3.4 FGO operators (pinned)

Let `tau = t / t_max`, where `t` counts evaluations including the initial `N` evaluations.

- `p_i = (f_i - min f) / (max f - min f + eps)`
- `E_r = M + (1 - M)(1 - tau)`
- Exploration:
  - `F = (f_i / sum(f_k)) * r1 * (1 - tau)^(1 - tau)`
  - `E = exp(F)`
  - `S_new = [R_j < r3] S_i + (1 - [R_j < r3])(S_i + E (S_a - S_b))`
- Nutrient/growth strength:
  - `eta_i = r7` if `t <= t_max / 2`, else `eta_i = f_i`
  - `eta_i = eta_i / sum(eta_k) + 2 * r8`
- Extra exploratory step:
  - `E_c = (r9 - 0.5) * r10 * (S_a - S_b)` with `r9` a vector
- Exploitation state 2, away from `S*`:
  - `D_e2 = r_j * (S_i - S*) * [R_j > r3]`
  - `S_new = S_i + eta_i * D_e2 + E_c * [r11 > r13]`
- Exploitation state 1, toward `S*`:
  - `D_e = r4 * (S_a - S_i) + r5 * (beta * S* - S_i) * [r6 > R]`
  - `beta in {-1, 1}`
  - `S_new = S_i + eta_i * D_e + E_c * [r12 > E_p]`
- State choice: if `r14 < r15`, use state 2; otherwise state 1.
- Branching:
  - `E_L = 1 + exp(f_i / sum(f_k)) * [r6 < r7]`
  - `D_ep1 = S_a - S_b`
  - `D_ep2 = S_c - S*`
  - `S_new = [R_j < r3] S_i + (1 - [R_j < r3])(S_i + r5 * E_L * D_ep1 + (1 - r5) * E_L * D_ep2)`
- Germination:
  - `S_g in {-1, 1}`
  - `mu = S_g * r5 * E`
  - `S_new = [R_j < r3] S_i + (1 - [R_j < r3])((tau * S* + (1 - tau) * S_a + S_b) / 2 + mu * |(S_c + S_a + S_b) / 3 - S_i|)`
- Branching vs germination: `r7 < 0.5` → branching, else germination.
- Selection: strict accept only when `f_new < f_i`; update `S*`.

### 3.5 Ambiguities and defaults

- The docx and an open-source MATLAB reference differ in a few places.
- Follow the docx for:
  - `beta` and `S_g` as `±1`; the MATLAB reference has a missing-parentheses bug producing `0/1`.
  - strict `<` selection; the MATLAB reference accepts equal fitness.
  - `t = N` after initialization; the MATLAB reference starts at `t = 0`.
- Use `eta_i = f_i` in the second half of the budget as written in the docx.
  - Expose a config option for `1 / (f_i + eps)` if tuning shows that minimization needs it.
- Record all deviations in `docs/algorithm_spec.md`.

## 4. Architecture

```text
pyproject.toml
README.md
PLAN.md
docs/
  algorithm_spec.md
  data.md
src/grzybki/
  __init__.py
  cli.py
  config.py
  problem/
    instance.py
    projection.py
    graph.py
    metrics.py
    decoder.py
  fgo/
    rng.py
    operators.py
    optimizer.py
  export/
    schema.py
    writer.py
  viz/
    convergence.py
    network.py
    animation.py
    report.py
data/
  instances/
    poland_15.json
    poland_30.json
  reference/
    poland_rail.geojson        # optional, fetched locally
scripts/
  fetch_osm_rail.py
  fetch_bdot10k.py
tests/
  test_instance.py
  test_graph.py
  test_metrics.py
  test_decoder.py
  test_operators.py
  test_optimizer.py
  test_export.py
  test_cli.py
runs/                         # generated, gitignored
```

## 5. Commit plan

Execution rule: one logical change per commit. Run the relevant tests and a CLI smoke test before each commit. Do not commit generated `runs/` artifacts or large raw datasets.

### Commit 1 — `docs(plan): add implementation plan`

- Add `PLAN.md`.
- Record scope, pinned algorithm details, defaults, architecture, and commit breakdown.
- Done when the plan is reviewable and matches the agreed decisions.

### Commit 2 — `chore(scaffold): add Python package and CLI skeleton`

- Add `pyproject.toml` with runtime and dev dependencies.
- Add `src/grzybki/__init__.py`, `cli.py`, `config.py`.
- Add `grzybki optimize` and `grzybki visualize` command stubs.
- Add `pytest` configuration and a minimal smoke test.
- Done when `python -m grzybki.cli --help` works and `pytest` passes.

### Commit 3 — `feat(instances): add instance loading, projection, and synthetic generators`

- Add `problem/instance.py` and `problem/projection.py`.
- Support JSON/CSV/GeoJSON terminals.
- Add local equirectangular projection; optional `pyproj`/EPSG:2180.
- Add normalized `[0,1]` coordinates with stored inverse transform.
- Add synthetic generators: uniform, clusters, polygon, grid.
- Done when instances round-trip through projection and synthetic instances are reproducible by seed.

### Commit 4 — `feat(decoder): implement graph construction and pruning`

- Add `problem/graph.py`.
- Implement distance matrix, deterministic MST, Delaunay triangulation, extra-edge selection, recursive degree-1 pruning, bridge detection, terminal shortest paths.
- Handle duplicate/collinear points and Delaunay failures.
- Done when the decoder returns a valid graph for edge cases and deterministic inputs.

### Commit 5 — `feat(metrics): add C, D, FT, and J`

- Add `problem/metrics.py` and `problem/decoder.py`.
- Compute terminal-only MST baselines.
- Implement `C`, `D`, `FT`, `J` with default weights.
- Add caching where safe.
- Done when terminal-only MST gives `J = 1` and metrics match hand-computed small examples.

### Commit 6 — `test(problem): add decoder and metrics tests`

- Add tests for known small graphs, pruning, bridges, `J = 1`, `K = 0`, duplicate points, and degenerate Delaunay cases.
- Add fixtures for synthetic instances.
- Done when `pytest tests/test_graph.py tests/test_metrics.py tests/test_decoder.py` passes.

### Commit 7 — `feat(fgo): implement operators`

- Add `fgo/rng.py` and `fgo/operators.py`.
- Implement initialization, exploration, exploitation states, branching, germination, crossover, and bounds clipping.
- Use one seeded `numpy.random.Generator` and a documented draw order.
- Done when each operator is unit-testable in isolation.

### Commit 8 — `feat(fgo): implement optimizer loop, budget, and checkpoints`

- Add `fgo/optimizer.py`.
- Implement main loop, phase selection, strict greedy selection, exact evaluation budget, best update, logging hooks, and checkpoint/resume state.
- Done when the optimizer uses exactly `t_max` evaluations and is reproducible by seed.

### Commit 9 — `test(fgo): add operator and optimizer tests`

- Add tests for bounds, reproducibility, exact budget, strict selection, θ plateau behavior, and convergence on sphere/Rastrigin.
- Add a small end-to-end optimization test.
- Done when `pytest tests/test_operators.py tests/test_optimizer.py` passes.

### Commit 10 — `feat(export): add run artifact schema and writers`

- Add `export/schema.py` and `export/writer.py`.
- Write `manifest.json`, `history.jsonl`, `snapshots.jsonl`, `best.json`, `summary.json`, and `checkpoints/`.
- Include schema version, config, seed, terminals, CRS, transform, and package versions.
- Done when a run can be written, loaded, and resumed.

### Commit 11 — `feat(viz): add convergence, metrics, network, animation, and report`

- Add `viz/convergence.py`, `viz/network.py`, `viz/animation.py`, `viz/report.py`.
- Produce `convergence.png`, `metrics.png`, `network.png`, `network_evolution.gif/.mp4`, and `report.html`.
- Support optional real-rail overlay.
- Done when `grzybki visualize runs/<run_id>` generates all outputs from a small run.

### Commit 12 — `feat(data): add Polish instances and optional data fetchers`

- Add `data/instances/poland_15.json` and `poland_30.json` with names, coordinates, source, and license.
- Add optional `scripts/fetch_osm_rail.py` and `scripts/fetch_bdot10k.py` with caching and attribution.
- Add comparison helpers for edge-to-nearest-rail distance.
- Done when curated instances load without network access and fetchers are documented.

### Commit 13 — `docs(readme): document usage, formats, data sources, and licenses`

- Update `README.md` with install, CLI examples, artifact schema, data sources, and attribution.
- Add `docs/algorithm_spec.md` and `docs/data.md`.
- Done when a new user can reproduce a run from the README.

### Commit 14 — `test(cli): add end-to-end CLI and artifact tests`

- Add `tests/test_cli.py` and `tests/test_export.py`.
- Run `optimize` and `visualize` on a tiny synthetic instance.
- Validate artifact schemas and output files.
- Done when the full test suite passes.

### Commit 15 — `chore(benchmark): add performance and sensitivity harness`

- Add a small benchmark/sensitivity script or CLI mode.
- Measure runtime for `poland_15` and `poland_30`.
- Document recommended defaults and any needed caching.
- Done when runtime is measured and reported in `docs/`.

### Phase 2 commits — real PKP PLK extension

These are separate after Phase 1 results:

1. `feat(data): ingest real rail graph and station data`
2. `feat(problem): add corridor-aware or existing-network decoder`
3. `feat(metrics): add construction cost, travel time, and resilience objectives`
4. `feat(viz): add real-network comparison and Pareto visualizations`
5. `docs(phase2): document formulation, assumptions, and limitations`

Phase 2 formulation order:

1. Corridor-aware Steiner network using real rail/terrain cost surfaces.
2. Existing-network upgrade/subgraph selection with construction cost, travel time, and resilience objectives.

## 6. Acceptance criteria for Phase 1

- `grzybki optimize --instance poland_15 --seed 42 --max-evals 20000` runs from a clean install.
- Runs are deterministic for a fixed seed and use exactly the requested evaluation budget.
- `grzybki visualize` produces convergence, metrics, network, animation, and HTML report.
- Tests cover decoder, metrics, FGO operators, optimizer budget, export/resume, and CLI smoke paths.
- README and docs explain installation, usage, data sources, and licensing.
- Results can be overlaid on the real Polish rail network for qualitative comparison.

## 7. Risks

- The docx equations are partially ambiguous; the spec and tests must pin assumptions before tuning.
- The Euclidean decoder ignores terrain, existing lines, capacity, and schedules; Phase 1 is stylized, not an engineering plan.
- Delaunay degeneracies and θ plateaus need explicit handling.
- 20,000 evaluations require efficient graph and metric code; benchmark early.
- Real rail data has licensing and attribution requirements; record sources and licenses.
