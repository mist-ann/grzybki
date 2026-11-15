# Benchmarks and sensitivity

These measurements are from the Phase 1 idealized decoder. They are indicative, not a performance guarantee.

Last re-measured on 2026-11-15, after two changes:
- isotropic coordinate normalization, which changes `J` slightly (see `docs/algorithm_spec.md` §1);
- SciPy-based metrics (prerequisite P3 in `docs/PHASE2_PLAN.md`).

Numbers from before both changes are kept at the end for comparison.

## Environment

```text
Linux-6.8.0-147-generic-x86_64-with-glibc2.39
Python 3.12.3
NumPy 2.5.4, SciPy 1.18.1, NetworkX 3.7
```

## Runtime

Command:

```bash
python3 scripts/benchmark.py \
  --max-evals 20000 --population 30 --seed 42 \
  --out runs/benchmark_20k.json
```

| Instance | n | K | Dimension | Evaluations | Best J | Wall time | Evaluations/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `poland_15` | 15 | 13 | 27 | 2,000 | 0.912014 | 3.24 s | 617.11 |
| `poland_30` | 30 | 28 | 57 | 2,000 | 0.897078 | 7.46 s | 268.02 |
| `poland_15` | 15 | 13 | 27 | 20,000 | 0.871279 | 36.42 s | 549.22 |
| `poland_30` | 30 | 28 | 57 | 20,000 | 0.832852 | 85.02 s | 235.23 |

The 20,000-evaluation run is the default budget. Runtime is roughly linear in the evaluation count. `poland_30` is about 2.3 times slower per evaluation because it has twice the terminals and a 57-dimensional search space. About two-thirds of decode time is now graph construction (`build_network`), not metrics.

## Sensitivity to `M`

Command:

```bash
python3 scripts/benchmark.py \
  --max-evals 500 --population 30 --seed 42 \
  --sweep M --values 0.4,0.6,0.8 \
  --out runs/sensitivity_M.json
```

| Instance | M | Evaluations | Best J | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `poland_15` | 0.4 | 500 | 0.958430 | 1.09 s |
| `poland_15` | 0.6 | 500 | 0.929738 | 1.04 s |
| `poland_15` | 0.8 | 500 | 0.958801 | 1.07 s |
| `poland_30` | 0.4 | 500 | 0.930866 | 2.23 s |
| `poland_30` | 0.6 | 500 | 0.930866 | 2.09 s |
| `poland_30` | 0.8 | 500 | 0.921102 | 2.29 s |

The differences are small and come from a single seed. Do not treat them as a ranking. Use multiple seeds before changing defaults:

```bash
python3 scripts/benchmark.py \
  --max-evals 2000 --repeat 5 \
  --sweep M --values 0.4,0.6,0.8 \
  --out runs/sensitivity_M_5seeds.json
```

`--repeat 5` runs seeds `42, 43, 44, 45, 46` and writes every run to the JSON file.

## Recommended settings

- Use `--max-evals 2000` while iterating and `--max-evals 20000` for final comparisons.
- Keep `population=30`, `M=0.6`, `Ep=0.7`, `R=0.9` unless a multi-seed sweep shows a consistent improvement.
- Use `--snapshot-interval 100` and `--checkpoint-interval 1000` for normal runs. Set both to `0` in benchmarks to avoid I/O overhead.
- Run `poland_15` first; it is roughly 3 times faster and exposes implementation errors quickly.
- Use `--repeat` rather than a single seed when comparing parameter settings.
- Keep instances offline and pre-projected in benchmark loops; projection is cheap but repeated I/O and projection add noise.
- Cache external downloads under `data/reference/`; the optional fetchers reuse cached responses unless `--force` is given.

## Reproducing the tables

The JSON files used for these tables are under `runs/`, which is ignored by Git. The commands above regenerate them. `scripts/benchmark.py` records the environment, arguments, best score, evaluation count, wall time, and best metrics for every run.

The benchmark uses `snapshot_interval=0` and `checkpoint_interval=0`, so its runtime is slightly lower than a default CLI run with snapshots and checkpoints enabled.

## Previous measurements (before isotropic normalization and SciPy metrics)

Linux 5.15, Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, NetworkX 3.4.2. `J` values are not directly comparable with the tables above, because the coordinate normalization differed.

| Instance | Evaluations | Best J | Wall time | Evaluations/s |
| --- | ---: | ---: | ---: | ---: |
| `poland_15` | 2,000 | 0.909418 | 6.65 s | 300.92 |
| `poland_30` | 2,000 | 0.892138 | 17.94 s | 111.51 |
| `poland_15` | 20,000 | 0.881212 | 64.63 s | 309.44 |
| `poland_30` | 20,000 | 0.858629 | 212.05 s | 94.32 |
