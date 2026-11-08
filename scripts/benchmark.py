#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grzybki.config import FGOConfig, ObjectiveWeights  # noqa: E402
from grzybki.fgo.optimizer import FGOOptimizer  # noqa: E402
from grzybki.problem import Instance, NetworkProblem  # noqa: E402


def run_one(
    instance_path: str | Path,
    *,
    max_evals: int,
    population: int,
    seed: int,
    projection: str,
    nutrient_mode: str,
    M: float,
    Ep: float,
    R: float,
) -> dict[str, Any]:
    instance = Instance.from_json(instance_path)
    instance.project(method=projection)
    problem = NetworkProblem(instance, instance.n - 2, ObjectiveWeights())
    config = FGOConfig(
        population_size=population,
        max_evals=max_evals,
        M=M,
        Ep=Ep,
        R=R,
        seed=seed,
        nutrient_mode=nutrient_mode,
        snapshot_interval=0,
        checkpoint_interval=0,
    )
    optimizer = FGOOptimizer(problem.evaluate, problem.dimension, config)
    started = time.perf_counter()
    result = optimizer.run()
    wall_time = time.perf_counter() - started
    return {
        "instance": instance.name,
        "n": instance.n,
        "K": problem.K,
        "dimension": problem.dimension,
        "max_evals": max_evals,
        "population": population,
        "seed": seed,
        "projection": projection,
        "nutrient_mode": nutrient_mode,
        "M": M,
        "Ep": Ep,
        "R": R,
        "best_score": result.best_score,
        "evaluations": result.evaluations,
        "runtime": result.runtime,
        "wall_time": wall_time,
        "evals_per_second": result.evaluations / wall_time if wall_time else None,
        "metrics": result.best_details,
    }


def parse_values(raw: str, cast: type) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one sweep value is required")
    return [cast(item) for item in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and sensitivity harness for Phase 1 FGO runs."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[
            "data/instances/poland_15.json",
            "data/instances/poland_30.json",
        ],
    )
    parser.add_argument("--max-evals", type=int, default=2_000)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--projection",
        choices=["equirectangular", "epsg2180"],
        default="equirectangular",
    )
    parser.add_argument(
        "--nutrient-mode", choices=["fitness", "inverse"], default="fitness"
    )
    parser.add_argument("--M", type=float, default=0.6)
    parser.add_argument("--Ep", type=float, default=0.7)
    parser.add_argument("--R", type=float, default=0.9)
    parser.add_argument(
        "--sweep",
        choices=["none", "M", "Ep", "R", "population"],
        default="none",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated sweep values, e.g. 0.4,0.6,0.8.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/benchmark.json")
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    runs: list[dict[str, Any]] = []
    for instance_path in args.instances:
        if args.sweep == "none":
            variants = [None]
        else:
            if args.values is None:
                parser.error("--values is required when --sweep is not none")
            cast = int if args.sweep == "population" else float
            variants = parse_values(args.values, cast)

        for variant in variants:
            for repeat in range(args.repeat):
                kwargs = {
                    "max_evals": args.max_evals,
                    "population": args.population,
                    "seed": args.seed + repeat,
                    "projection": args.projection,
                    "nutrient_mode": args.nutrient_mode,
                    "M": args.M,
                    "Ep": args.Ep,
                    "R": args.R,
                }
                if args.sweep != "none" and variant is not None:
                    kwargs[args.sweep] = variant
                run = run_one(instance_path, **kwargs)
                run["repeat"] = repeat
                runs.append(run)
                print(
                    f"{run['instance']:>12}  "
                    f"{args.sweep}={variant if variant is not None else '-'!s:>6}  "
                    f"evals={run['evaluations']:>6}  "
                    f"J={run['best_score']:.6f}  "
                    f"{run['wall_time']:8.2f}s  "
                    f"{run['evals_per_second']:8.2f} eval/s"
                )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "args": {
            "instances": args.instances,
            "max_evals": args.max_evals,
            "population": args.population,
            "seed": args.seed,
            "projection": args.projection,
            "nutrient_mode": args.nutrient_mode,
            "M": args.M,
            "Ep": args.Ep,
            "R": args.R,
            "sweep": args.sweep,
            "values": args.values,
            "repeat": args.repeat,
        },
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
