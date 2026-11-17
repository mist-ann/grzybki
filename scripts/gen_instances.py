#!/usr/bin/env python3
"""Generate synthetic terminal-set instances as JSON files.

By default it writes a family of regular-polygon instances (the terminal
layout is the vertex set of a regular n-gon) into ``data/instances/``.
Other synthetic kinds from ``grzybki.problem.synthetic_instance``
(uniform, clustered, ring, grid) are also available.

Examples:

    python3 scripts/gen_instances.py                    # default polygon family
    python3 scripts/gen_instances.py --sides 7 9 11     # selected polygon sizes
    python3 scripts/gen_instances.py --kind ring --sizes 15 --seed 3
    python3 scripts/gen_instances.py --jitter 0.02 --suffix jitter
"""
from __future__ import annotations

import argparse
from pathlib import Path

from grzybki.problem import Instance, synthetic_instance

# Default family: regular polygons from 8 up to the poland_30-sized instance,
# so results are directly comparable.
DEFAULT_SIDES = (8, 12, 20, 30)

DESCRIPTION = (
    "Synthetic terminal set: vertices of a regular n-gon "
    "inscribed in a circle of radius r inside the unit square."
)


def build_instance(
    kind: str,
    n: int,
    seed: int,
    suffix: str,
    jitter: float,
    radius: float,
    rotation_deg: float,
) -> Instance:
    parameters: dict[str, float] = {}
    name_parts = [kind, str(n)]
    if suffix:
        name_parts.append(suffix)
    if kind == "polygon":
        parameters.update(
            radius=radius, rotation_deg=rotation_deg, jitter=jitter
        )
        if suffix == "" and jitter > 0.0:
            name_parts.append(f"j{int(round(jitter * 1000))}")
    return synthetic_instance(
        kind,
        n,
        seed=seed,
        name="_".join(name_parts),
        **parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        default="polygon",
        choices=["polygon", "uniform", "clustered", "ring", "grid"],
        help="synthetic terminal layout (default: polygon)",
    )
    parser.add_argument(
        "--sides",
        type=int,
        nargs="+",
        default=None,
        help="polygon vertex counts to generate (alias for --sizes)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="terminal counts to generate (default: the polygon family)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Gaussian noise std-dev added to vertices; 0 keeps the exact "
        "regular polygon (polygon kind only, default 0)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.45,
        help="polygon circumradius in the unit square (default 0.45)",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=-90.0,
        help="rotation of the first polygon vertex in degrees (default -90)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="extra name suffix for the generated files",
    )
    parser.add_argument(
        "--out",
        default="data/instances",
        help="output directory (default data/instances)",
    )
    args = parser.parse_args()

    sizes = args.sizes or args.sides
    if sizes is None:
        if args.kind == "polygon":
            sizes = list(DEFAULT_SIDES)
        else:
            sizes = [15, 30]
    sizes = sorted(set(sizes))
    if any(size < 2 for size in sizes):
        parser.error("every size must be at least 2 terminals")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for size in sizes:
        instance = build_instance(
            args.kind, size, args.seed, args.suffix, args.jitter,
            args.radius, args.rotation_deg,
        )
        instance.crs = "unit-square"
        if args.kind == "polygon":
            instance.description = DESCRIPTION
            instance.source = "scripts/gen_instances.py (synthetic)"
        instance.metadata["seed"] = args.seed
        if args.kind == "polygon":
            instance.metadata["sides"] = size
            instance.metadata["radius"] = args.radius
            instance.metadata["rotation_deg"] = args.rotation_deg
            instance.metadata["jitter"] = args.jitter
            instance.metadata["notes"] = (
                "Coordinates are unit-square x/y values, not geographic."
            )
        path = out_dir / f"{instance.name}.json"
        instance.save(path)
        written.append(path)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
