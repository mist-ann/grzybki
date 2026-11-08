from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from .config import FGOConfig, ObjectiveWeights
from .export.schema import SCHEMA_VERSION, manifest_payload, summary_payload, utc_now
from .export.writer import RunWriter, load_run
from .fgo.optimizer import FGOOptimizer, load_checkpoint
from .problem.comparison import load_geojson_lines, project_lines
from .problem.decoder import NetworkProblem
from .problem.instance import Instance
from .viz import (
    plot_convergence,
    plot_decoded,
    plot_metrics,
    write_animation,
    write_report,
)


def run_optimize(
    instance_path: str,
    out_dir: str | None,
    *,
    seed: int = 42,
    population_size: int = 30,
    max_evals: int = 20_000,
    K: int | None = None,
    M: float = 0.6,
    Ep: float = 0.7,
    R: float = 0.9,
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.2,
    projection: str = "equirectangular",
    nutrient_mode: str = "fitness",
    snapshot_interval: int = 100,
    checkpoint_interval: int = 1_000,
    save_vectors: bool = False,
    resume: str | None = None,
) -> None:
    resolved_instance_path = _resolve_instance_path(instance_path)
    instance = Instance.from_json(resolved_instance_path)
    instance.project(method=projection)
    resolved_seed = _resolve_seed(seed)

    steiner_count = instance.n - 2 if K is None else K
    if steiner_count < 0:
        raise click.ClickException("K must be non-negative")
    weights = ObjectiveWeights(w1=w1, w2=w2, w3=w3)
    problem = NetworkProblem(instance, steiner_count, weights)
    config = FGOConfig(
        population_size=population_size,
        max_evals=max_evals,
        M=M,
        Ep=Ep,
        R=R,
        seed=resolved_seed,
        nutrient_mode=nutrient_mode,
        snapshot_interval=snapshot_interval,
        checkpoint_interval=checkpoint_interval,
        save_vectors=save_vectors,
    )

    resume_state = None
    resume_run_dir: Path | None = None
    if resume is not None:
        checkpoint_dir, resume_run_dir = _resolve_resume(resume)
        resume_state = load_checkpoint(checkpoint_dir)

    run_dir = _resolve_run_dir(out_dir, resume_run_dir, instance.name, resolved_seed)
    append = (
        resume_run_dir is not None
        and run_dir.resolve() == resume_run_dir.resolve()
    )

    extra = {
        "instance_path": str(resolved_instance_path.resolve()),
        "instance": instance.to_dict(),
        "resolved_seed": resolved_seed,
    }
    manifest = manifest_payload(
        instance=instance,
        problem=problem,
        config=config,
        extra=extra,
    )

    writer = RunWriter(run_dir, manifest, append=append)
    try:
        optimizer = FGOOptimizer(
            problem.evaluate,
            problem.dimension,
            config,
            checkpoint_dir=writer.checkpoint_dir,
            on_evaluation=writer.append_history,
            on_snapshot=writer.append_snapshot,
        )
        result = optimizer.run(resume=resume_state)
        decoded = problem.evaluate(result.best_vector)
        best_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "best_score": result.best_score,
            "best_vector": result.best_vector,
            "best": decoded.to_dict(),
            "config": result.config.to_dict(),
        }
        writer.write_best(best_payload)
        writer.write_summary(summary_payload(result=result))
    finally:
        writer.close()

    click.echo(f"best J = {result.best_score:.6f}")
    click.echo(f"evaluations = {result.evaluations}")
    click.echo(f"runtime = {result.runtime:.2f} s")
    click.echo(f"run = {run_dir}")


def run_visualize(
    run_dir: str,
    out_dir: str | None,
    *,
    animation: bool = False,
    animation_format: str = "gif",
    fps: int = 10,
    dpi: int = 140,
    reference: str | None = None,
    report: bool = True,
) -> None:
    artifacts = load_run(run_dir)
    manifest = artifacts.manifest
    summary = artifacts.summary
    best = artifacts.best
    if best is None and summary is None:
        raise click.ClickException("run has no best solution or summary")

    instance = _instance_from_manifest(manifest)
    problem_data = manifest.get("problem", {})
    steiner_count = int(problem_data.get("K", instance.n - 2))
    weights = ObjectiveWeights.from_dict(problem_data.get("weights", {}))
    problem = NetworkProblem(instance, steiner_count, weights)

    best_vector = None
    if best is not None:
        best_vector = best.get("best_vector")
    if best_vector is None and summary is not None:
        best_vector = summary.get("best_vector")
    if best_vector is None:
        raise click.ClickException("run does not contain a best vector")
    decoded = problem.evaluate(np.asarray(best_vector, dtype=float))

    output = Path(out_dir) if out_dir is not None else Path(run_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}

    if artifacts.history:
        images["Convergence"] = plot_convergence(
            artifacts.history,
            output / "convergence.png",
            snapshots=artifacts.snapshots,
            title=f"FGO convergence: {instance.name}",
        )
        try:
            images["Metrics"] = plot_metrics(
                artifacts.history,
                output / "metrics.png",
                title=f"Network metrics: {instance.name}",
            )
        except ValueError:
            pass

    reference_payload = None
    if reference is not None:
        lines = load_geojson_lines(reference)
        if instance.projection is not None:
            lines = project_lines(instance.projection, lines)
        reference_payload = _reference_payload(lines)
        if reference_payload is None:
            raise click.ClickException("reference file contains no line features")

    images["Network"] = plot_decoded(
        decoded,
        output / "network.png",
        title=f"Best network: {instance.name}",
        reference=reference_payload,
    )

    if animation:
        if not artifacts.snapshots:
            raise click.ClickException("run contains no snapshots for animation")
        if animation_format not in {"gif", "mp4"}:
            raise click.ClickException("animation format must be gif or mp4")
        images["Animation"] = write_animation(
            problem,
            artifacts.snapshots,
            output / f"animation.{animation_format}",
            fps=fps,
            dpi=dpi,
        )

    if report:
        images["Report"] = write_report(
            output / "report.html",
            manifest=manifest,
            summary=summary,
            best=best,
            images=images,
            title=f"FGO run: {instance.name}",
        )

    for label, path in images.items():
        click.echo(f"{label}: {path}")


def _resolve_instance_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = Path("data") / "instances" / f"{value}.json"
    if candidate.exists():
        return candidate
    raise click.ClickException(
        f"instance not found: {value} (tried {path} and {candidate})"
    )


def _resolve_seed(seed: int) -> int:
    if seed >= 0:
        return seed
    return int(np.random.SeedSequence().entropy)


def _resolve_run_dir(
    out_dir: str | None,
    resume_run_dir: Path | None,
    instance_name: str,
    seed: int,
) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    if resume_run_dir is not None:
        return resume_run_dir
    return Path("runs") / f"{instance_name}_seed{seed}"


def _resolve_resume(resume: str) -> tuple[Path, Path]:
    path = Path(resume)
    if (path / "state.npz").exists():
        return path, path.parent if path.name == "checkpoints" else path
    if (path / "checkpoints" / "state.npz").exists():
        return path / "checkpoints", path
    raise click.ClickException(
        "resume path must contain state.npz or checkpoints/state.npz"
    )


def _instance_from_manifest(manifest: dict[str, Any]) -> Instance:
    extra = manifest.get("extra", {})
    instance_data = extra.get("instance")
    if isinstance(instance_data, dict):
        instance = Instance.from_dict(instance_data)
    else:
        instance_path = extra.get("instance_path")
        if not instance_path:
            raise click.ClickException("manifest does not contain instance data")
        instance = Instance.from_json(instance_path)
    if not all(terminal.has_xy for terminal in instance.terminals):
        projection = manifest.get("instance", {}).get("projection", {})
        instance.project(method=projection.get("method", "equirectangular"))
    return instance


def _reference_payload(lines: list[np.ndarray]) -> dict[str, Any] | None:
    points = []
    edges = []
    offset = 0
    for line in lines:
        line = np.asarray(line, dtype=float)
        if len(line) < 2:
            continue
        points.append(line)
        edges.extend((offset + index, offset + index + 1) for index in range(len(line) - 1))
        offset += len(line)
    if not points:
        return None
    return {
        "points": np.vstack(points),
        "edges": edges,
        "label": "reference rail",
        "color": "#7f7f7f",
        "linewidth": 0.8,
        "alpha": 0.5,
        "size": 2,
    }
