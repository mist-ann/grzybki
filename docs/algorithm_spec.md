# Algorithm specification

This document describes the Phase 1 implementation. It follows `FGO_opis_algorytmu.docx` and the approved `PLAN.md`; documented deviations are called out explicitly.

## 1. Problem representation

An instance contains `n` fixed terminals `p_1, ..., p_n` in the unit square `[0, 1]^2`. A candidate solution has `K` Steiner points and one redundancy parameter:

```text
x = (x_1, y_1, ..., x_K, y_K, theta) in [0, 1]^(2K+1)
```

The default is `K = n - 2`. The optimizer always clips candidates to `[0, 1]^(2K+1)` before evaluation.

Coordinates are normalized after projection. The default projection is a local equirectangular projection with a 5% margin; `epsg2180` is available when `pyproj` is installed. Distances are Euclidean in the normalized plane.

## 2. Decoder

Given `x`:

1. Read the `K` Steiner points from the first `2K` entries.
2. Form `Q = terminals + Steiner points`.
3. Build the Euclidean minimum spanning tree `T` on `Q`.
4. Compute the Delaunay triangulation of `Q` and collect all edges not already in `T`.
5. Sort the remaining edges by length and add the first `floor(theta * m)` of them, where `m` is the number of non-MST Delaunay edges. The added edges are marked `extra`.
6. Repeatedly remove any Steiner point of degree 1. Terminals are never removed.
7. Return the resulting graph and its metrics.

Delaunay failures caused by degenerate point sets fall back to a complete graph over `Q`, from which the same non-MST edge selection is made. Self-loops and duplicate edges are rejected.

`theta` is not clipped by the decoder; callers are expected to keep it in `[0, 1]`. Values outside that range therefore saturate at `0` or `m` extra edges.

## 3. Metrics

Let `E` be the final edge set and `ell_e` the Euclidean length of edge `e`.

### Cost `C`

```text
C = sum_{e in E} ell_e
```

### Detour `D`

For every unordered terminal pair `(i, j)`, let `d_G(i, j)` be the shortest-path length in the final graph and `||p_i - p_j||` the direct Euclidean distance. Then:

```text
D = 2 / (n(n-1)) * sum_{i<j} d_G(i, j) / ||p_i - p_j||
```

This is the unweighted mean pairwise detour ratio, following docx eq. (5). Pairs with coincident terminals are skipped. If a terminal pair is disconnected, `D = inf`. If no valid pair remains, `D = 1`.

### Fault tolerance `FT`

`FT` is the fraction of edges whose individual removal still leaves all terminals in one connected component:

```text
FT = (|E| - B_sep) / |E|
```

where `B_sep` is the number of bridges whose removal separates at least two terminals. Non-terminal bridges count as fault tolerant. If the graph is disconnected or has no edges, `FT = 0`.

### Objective `J`

```text
J = w1 * C / C_MST + w2 * D / D_MST + w3 * (1 - FT)
```

`C_MST` and `D_MST` are computed from the terminal-only Euclidean MST. Defaults are `(w1, w2, w3) = (0.4, 0.4, 0.2)`. The terminal-only MST therefore has `J = 1`, and lower `J` is better.

The baseline is computed once per `NetworkProblem` and reused for every evaluation.

## 4. FGO operators

### 4.1 State and draw order

The population is `S_1, ..., S_N`, initialized uniformly in `[0, 1]^d`. The evaluation budget `t_max` includes the `N` initialization evaluations. After initialization `t = N`; each candidate evaluation increments `t` by one.

For each outer iteration:

```text
tau = t / t_max
```

The optimizer chooses tip growth or branch growth with equal probability:

```text
if rng.random() < rng.random(): tip growth
else: branch growth
```

Within tip growth, each individual is updated in index order:

```text
p_i = (f_i - min(f)) / (max(f) - min(f))
E_r = M + (1 - M)(1 - tau)
if p_i < E_r: exploration
else: exploitation
```

Within branch growth, each individual is updated in index order, choosing branching or germination with equal probability.

The loop stops as soon as `t == t_max`, so the budget is exact even if an iteration is cut short.

### 4.2 Shared quantities

Normalized fitness:

```text
f_i^+ = max(f_i, 0)
share_i = f_i^+ / sum_j f_j^+
```

If the total is zero, shares are uniform. `quality_scores` is the min-max normalization used for `p_i`.

Growth factor for individual `i`:

```text
F_i = share_i * r1 * (1 - tau)^(1 - tau)
G_i = exp(F_i)
```

Nutrient strength is a vector `eta`:

```text
if t <= t_max / 2:
    eta_i = r7_i                         # random vector
else if nutrient_mode == "fitness":
    eta_i = f_i
else if nutrient_mode == "inverse":
    eta_i = 1 / (max(f_i, 0) + eps)
eta = eta / sum(eta) + 2 * r8            # r8 is a scalar
```

If the total is zero, the normalized part is uniform.

Three distinct peers `a, b, c` are drawn uniformly without replacement from the population, excluding `i`.

### 4.3 Crossover

All exploratory operators finish with a component-wise keep-old crossover:

```text
keep_old_j = (R_j < r3)
child_j = old_j if keep_old_j else candidate_j
```

`r3` is a scalar uniform draw; `R_j` is a fresh uniform vector. This matches the approved PLAN and the reference implementation.

### 4.4 Exploration

```text
candidate = S_i + G_i * (S_a - S_b)
```

The candidate is then crossed over with `S_i`. The implementation follows the approved PLAN and reference code, which use `S_a - S_b`. The docx equation (7) prints `S_a - S_c`; that difference is treated as a typo.

### 4.5 Exploitation

```text
Ec = (r9 - 0.5) * r10 * (S_a - S_b)
```

`r9` is a uniform vector, `r10` a scalar. One of two states is chosen with equal probability.

State 2:

```text
r_j = uniform vector
active_j = (r11_j > r12_j)
D_e2 = r_j * (S_i - S_best) * active
candidate = S_i + eta * D_e2 + Ec * (r13 > r14)
```

State 1:

```text
r4 = uniform scalar
r5 = uniform vector
beta = +1 or -1 with equal probability
toward_best = (r6 > R)
D_e = r4 * (S_a - S_i) + r5 * (beta * S_best - S_i) * toward_best
candidate = S_i + eta * De + Ec * (r7 > Ep)
```

`eta` is the scalar `eta_i`. The candidate is clipped to `[0, 1]^d`.

### 4.6 Branching

```text
E_L = 1 + exp(share_i) * (r6 < r7)
D_ep1 = S_a - S_b
D_ep2 = S_c - S_best
candidate = S_i + r5 * E_L * D_ep1 + (1 - r5) * E_L * D_ep2
```

`r5` is a scalar. The candidate is crossed over with `S_i`. This follows the docx and the approved PLAN. The reference MATLAB code uses `S_b - S_c` and `S_a - S_best`; that difference is not used here.

### 4.7 Germination

```text
S_g = +1 or -1 with equal probability
G_i = exp(F_i)
mu = S_g * r5 * G_i
spore = (tau * S_best + (1 - tau) * S_a + S_b) / 2
      + mu * |(S_c + S_a + S_b) / 3 - S_i|
```

The candidate is crossed over with `S_i`. `mu` is a scalar, as specified by the approved PLAN; the MATLAB reference draws a separate `mu` for each dimension.

### 4.8 Selection

Selection is strict:

```text
accept = score(candidate) < fitness_i
```

The population member, its fitness, and the global best are updated only when the candidate is strictly better. Equal scores are rejected.

## 5. Budget, history, and checkpoints

- Initialization evaluates exactly `N` random vectors.
- Every candidate evaluation increments `t` before the history record is written.
- History records contain `t`, `iteration`, `phase` (`init`, `tip`, or `branch`), `operator`, `accepted`, `score`, and `best_score`.
- Metrics and `theta` are stored under `details`; candidate vectors are stored only when `save_vectors` is enabled.
- Snapshots are written every `snapshot_interval` evaluations and forced at the start and end of a run.
- Checkpoints are written every `checkpoint_interval` evaluations and forced at the end. `state.npz` stores population, fitness, best vector, best score, `t`, and iteration; `state.json` stores the RNG state.
- `best_payload` is not serialized in checkpoints. After a resume, the first improvement repopulates it.

## 6. Reproducibility

The RNG is a NumPy `Generator` seeded from `FGOConfig.seed`. A negative seed means a random seed; the resolved seed is recorded in the run manifest. Given the same instance, config, seed, and library versions, the evaluation history is deterministic.

## 7. Known limitations

- `theta` enters through `floor(theta * m)`, so the objective is piecewise constant in `theta` and many candidates are neutral.
- The decoder is Euclidean and does not use real rail geometry, terrain, construction costs, capacity, or schedules.
- The fault-tolerance metric tests single-edge removal only; it does not model node failures or multi-edge failures.
- The Phase 1 graph is undirected and unweighted apart from Euclidean edge length.
