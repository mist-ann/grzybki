from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import FGOConfig
from .operators import (
    branching_candidate,
    clip_vector,
    exploration_candidate,
    exploration_threshold,
    exploitation_candidate,
    germination_candidate,
    initialize_population,
    nutrient_strength,
    quality_scores,
    select_distinct_indices,
)
from .rng import make_rng


@dataclass
class OptimizerState:
    population: np.ndarray
    fitness: np.ndarray
    best_vector: np.ndarray
    best_score: float
    best_payload: Any
    t: int
    iteration: int
    rng_state: dict[str, Any]


@dataclass
class OptimizationResult:
    best_vector: np.ndarray
    best_score: float
    best_details: dict[str, Any] | None
    history: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    evaluations: int
    runtime: float
    config: FGOConfig

    def summary(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_vector": self.best_vector.tolist(),
            "best_details": self.best_details,
            "evaluations": self.evaluations,
            "runtime": self.runtime,
            "config": self.config.to_dict(),
        }


class FGOOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], Any],
        dimension: int,
        config: FGOConfig,
        *,
        checkpoint_dir: str | Path | None = None,
        on_evaluation: Callable[[dict[str, Any]], None] | None = None,
        on_snapshot: Callable[[dict[str, Any]], None] | None = None,
        on_checkpoint: Callable[[OptimizerState], None] | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.objective = objective
        self.dimension = dimension
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.on_evaluation = on_evaluation
        self.on_snapshot = on_snapshot
        self.on_checkpoint = on_checkpoint

        self.rng = make_rng(config.seed)
        self.population = np.empty((config.population_size, dimension), dtype=float)
        self.fitness = np.full(config.population_size, np.inf, dtype=float)
        self.best_vector = np.zeros(dimension, dtype=float)
        self.best_score = float("inf")
        self.best_payload: Any = None
        self.t = 0
        self.iteration = 0
        self.history: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []

    def run(self, resume: OptimizerState | None = None) -> OptimizationResult:
        started = time.perf_counter()
        if resume is None:
            self._initialize()
        else:
            self._restore(resume)

        self._snapshot(force=True)

        while self.t < self.config.max_evals:
            self.iteration += 1
            eta = nutrient_strength(
                self.rng,
                self.fitness,
                self.t,
                self.config.max_evals,
                mode=self.config.nutrient_mode,
            )
            if self.rng.random() < self.rng.random():
                self._tip_growth(eta)
            else:
                self._branch_growth()

        self._snapshot(force=True)
        self._checkpoint(force=True)
        runtime = time.perf_counter() - started
        return OptimizationResult(
            best_vector=self.best_vector.copy(),
            best_score=self.best_score,
            best_details=_details(self.best_payload),
            history=list(self.history),
            snapshots=list(self.snapshots),
            evaluations=self.t,
            runtime=runtime,
            config=self.config,
        )

    def state(self) -> OptimizerState:
        return OptimizerState(
            population=self.population.copy(),
            fitness=self.fitness.copy(),
            best_vector=self.best_vector.copy(),
            best_score=self.best_score,
            best_payload=self.best_payload,
            t=self.t,
            iteration=self.iteration,
            rng_state=dict(self.rng.bit_generator.state),
        )

    def _initialize(self) -> None:
        self.rng = make_rng(self.config.seed)
        self.population = initialize_population(
            self.rng, self.config.population_size, self.dimension
        )
        self.fitness = np.empty(self.config.population_size, dtype=float)
        self.best_score = float("inf")
        self.best_vector = np.zeros(self.dimension, dtype=float)
        self.best_payload = None
        self.t = 0
        self.iteration = 0
        self.history = []
        self.snapshots = []

        for index in range(self.config.population_size):
            score, payload = self._evaluate(self.population[index])
            self.fitness[index] = score
            self.t += 1
            if score < self.best_score:
                self.best_score = score
                self.best_vector = self.population[index].copy()
                self.best_payload = payload
            self._append_history(
                self._record(
                    "init",
                    "initialization",
                    True,
                    score,
                    payload,
                    self.population[index],
                )
            )
            self._maybe_snapshot()
            self._maybe_checkpoint()

    def _restore(self, resume: OptimizerState) -> None:
        self.population = resume.population.copy()
        self.fitness = resume.fitness.copy()
        self.best_vector = resume.best_vector.copy()
        self.best_score = resume.best_score
        self.best_payload = resume.best_payload
        self.t = resume.t
        self.iteration = resume.iteration
        self.rng = make_rng()
        self.rng.bit_generator.state = resume.rng_state
        self.history = []
        self.snapshots = []

    def _tip_growth(self, eta: np.ndarray) -> None:
        for index in range(self.config.population_size):
            if self.t >= self.config.max_evals:
                break
            a, b, c = select_distinct_indices(
                self.rng, self.config.population_size, index, 3
            )
            p = quality_scores(self.fitness)[index]
            Er = exploration_threshold(self.t, self.config.max_evals, self.config.M)
            if p < Er:
                candidate = exploration_candidate(
                    self.rng,
                    self.population[index],
                    self.population[a],
                    self.population[b],
                    self.fitness,
                    index,
                    self.t,
                    self.config.max_evals,
                )
                operator = "exploration"
            else:
                candidate = exploitation_candidate(
                    self.rng,
                    self.population[index],
                    self.population[a],
                    self.population[b],
                    self.best_vector,
                    self.fitness,
                    index,
                    self.t,
                    self.config.max_evals,
                    float(eta[index]),
                    self.config.M,
                    self.config.Ep,
                    self.config.R,
                )
                operator = "exploitation"
            self._consider(index, candidate, "tip", operator)

    def _branch_growth(self) -> None:
        for index in range(self.config.population_size):
            if self.t >= self.config.max_evals:
                break
            a, b, c = select_distinct_indices(
                self.rng, self.config.population_size, index, 3
            )
            if self.rng.random() < 0.5:
                candidate = branching_candidate(
                    self.rng,
                    self.population[index],
                    self.population[a],
                    self.population[b],
                    self.population[c],
                    self.best_vector,
                    self.fitness,
                    index,
                    self.t,
                    self.config.max_evals,
                )
                operator = "branching"
            else:
                candidate = germination_candidate(
                    self.rng,
                    self.population[index],
                    self.population[a],
                    self.population[b],
                    self.population[c],
                    self.best_vector,
                    self.fitness,
                    index,
                    self.t,
                    self.config.max_evals,
                )
                operator = "germination"
            self._consider(index, candidate, "branch", operator)

    def _consider(
        self,
        index: int,
        candidate: np.ndarray,
        phase: str,
        operator: str,
    ) -> None:
        candidate = clip_vector(candidate)
        score, payload = self._evaluate(candidate)
        self.t += 1
        accepted = score < self.fitness[index]
        if accepted:
            self.population[index] = candidate
            self.fitness[index] = score
            if score < self.best_score:
                self.best_score = score
                self.best_vector = candidate.copy()
                self.best_payload = payload
        self._append_history(
            self._record(phase, operator, accepted, score, payload, candidate)
        )
        self._maybe_snapshot()
        self._maybe_checkpoint()

    def _evaluate(self, vector: np.ndarray) -> tuple[float, Any]:
        result = self.objective(vector)
        if isinstance(result, (int, float, np.integer, np.floating)):
            return float(result), None
        if hasattr(result, "score"):
            return float(result.score), result
        if hasattr(result, "metrics") and hasattr(result.metrics, "objective"):
            return float(result.metrics.objective), result
        raise TypeError("objective must return a number or an object with .score")

    def _record(
        self,
        phase: str,
        operator: str,
        accepted: bool,
        score: float,
        payload: Any,
        vector: np.ndarray,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "t": self.t,
            "iteration": self.iteration,
            "phase": phase,
            "operator": operator,
            "accepted": bool(accepted),
            "score": float(score),
            "best_score": float(self.best_score),
        }
        details = _details(payload)
        if details is not None:
            record["details"] = details
        if self.config.save_vectors:
            record["vector"] = np.asarray(vector, dtype=float).tolist()
        return record

    def _append_history(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        if self.on_evaluation is not None:
            self.on_evaluation(record)

    def _maybe_snapshot(self) -> None:
        interval = self.config.snapshot_interval
        if interval > 0 and self.t % interval == 0:
            self._snapshot(force=False)

    def _snapshot(self, force: bool) -> None:
        if not force and self.t == 0:
            return
        if self.snapshots and self.snapshots[-1]["t"] == self.t:
            return
        snapshot = {
            "t": self.t,
            "iteration": self.iteration,
            "best_score": float(self.best_score),
            "best_vector": self.best_vector.tolist(),
            "fitness_min": float(np.min(self.fitness)),
            "fitness_mean": float(np.mean(self.fitness)),
            "fitness_max": float(np.max(self.fitness)),
        }
        details = _details(self.best_payload)
        if details is not None:
            snapshot["best_details"] = details
        self.snapshots.append(snapshot)
        if self.on_snapshot is not None:
            self.on_snapshot(snapshot)

    def _maybe_checkpoint(self) -> None:
        interval = self.config.checkpoint_interval
        if interval > 0 and self.t % interval == 0:
            self._checkpoint(force=False)

    def _checkpoint(self, force: bool) -> None:
        if self.t == 0:
            return
        state = self.state()
        if self.checkpoint_dir is not None:
            save_checkpoint(self.checkpoint_dir, state)
        if self.on_checkpoint is not None:
            self.on_checkpoint(state)


def save_checkpoint(directory: str | Path, state: OptimizerState) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "state.npz",
        population=state.population,
        fitness=state.fitness,
        best_vector=state.best_vector,
        best_score=np.asarray(state.best_score),
        t=np.asarray(state.t),
        iteration=np.asarray(state.iteration),
    )
    with (path / "state.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"rng_state": state.rng_state},
            handle,
            indent=2,
            default=_json_default,
        )
        handle.write("\n")


def load_checkpoint(directory: str | Path) -> OptimizerState:
    path = Path(directory)
    with np.load(path / "state.npz", allow_pickle=False) as data:
        population = np.asarray(data["population"], dtype=float)
        fitness = np.asarray(data["fitness"], dtype=float)
        best_vector = np.asarray(data["best_vector"], dtype=float)
        best_score = float(data["best_score"])
        t = int(data["t"])
        iteration = int(data["iteration"])
    with (path / "state.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return OptimizerState(
        population=population,
        fitness=fitness,
        best_vector=best_vector,
        best_score=best_score,
        best_payload=None,
        t=t,
        iteration=iteration,
        rng_state=metadata["rng_state"],
    )


def _details(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if hasattr(payload, "metrics") and hasattr(payload.metrics, "to_dict"):
        details: dict[str, Any] = {"metrics": payload.metrics.to_dict()}
        if hasattr(payload, "theta"):
            details["theta"] = float(payload.theta)
        return details
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
