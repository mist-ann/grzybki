import json

import numpy as np
import pytest

from grzybki.config import FGOConfig
from grzybki.fgo import FGOOptimizer, load_checkpoint, save_checkpoint
from grzybki.problem import NetworkProblem, synthetic_instance


def sphere(vector: np.ndarray) -> float:
    return float(np.sum(vector**2))


def test_optimizer_respects_budget_and_improves() -> None:
    config = FGOConfig(
        population_size=10,
        max_evals=120,
        seed=3,
        snapshot_interval=20,
        checkpoint_interval=0,
    )
    result = FGOOptimizer(sphere, 4, config).run()
    assert result.evaluations == config.max_evals
    assert len(result.history) == config.max_evals
    assert result.best_score <= min(record["score"] for record in result.history[:10])
    assert np.all((result.best_vector >= 0.0) & (result.best_vector <= 1.0))
    assert result.snapshots
    assert result.snapshots[-1]["t"] == config.max_evals
    json.dumps(result.summary())


def test_optimizer_is_reproducible() -> None:
    config = FGOConfig(
        population_size=8,
        max_evals=80,
        seed=9,
        snapshot_interval=10,
        checkpoint_interval=0,
    )
    first = FGOOptimizer(sphere, 3, config).run()
    second = FGOOptimizer(sphere, 3, config).run()
    assert np.array_equal(first.best_vector, second.best_vector)
    assert first.best_score == second.best_score
    assert [record["score"] for record in first.history] == [
        record["score"] for record in second.history
    ]


def test_strict_selection_rejects_equal_scores() -> None:
    config = FGOConfig(
        population_size=6,
        max_evals=40,
        seed=2,
        snapshot_interval=0,
        checkpoint_interval=0,
    )
    result = FGOOptimizer(lambda vector: 1.0, 2, config).run()
    assert all(record["accepted"] is False for record in result.history[6:])
    assert result.best_score == 1.0


def test_checkpoint_roundtrip(tmp_path) -> None:
    config = FGOConfig(
        population_size=6,
        max_evals=30,
        seed=5,
        snapshot_interval=10,
        checkpoint_interval=10,
    )
    optimizer = FGOOptimizer(sphere, 3, config, checkpoint_dir=tmp_path)
    optimizer.run()
    state = load_checkpoint(tmp_path)
    assert state.t == config.max_evals
    assert np.array_equal(state.population, optimizer.population)
    assert np.array_equal(state.best_vector, optimizer.best_vector)
    assert state.best_score == optimizer.best_score
    assert isinstance(state.rng_state, dict)

    save_checkpoint(tmp_path / "again", state)
    reloaded = load_checkpoint(tmp_path / "again")
    assert np.array_equal(reloaded.fitness, state.fitness)


def test_resume_continues_budget() -> None:
    short = FGOConfig(
        population_size=6,
        max_evals=20,
        seed=1,
        snapshot_interval=0,
        checkpoint_interval=0,
    )
    optimizer = FGOOptimizer(sphere, 2, short)
    optimizer.run()
    state = optimizer.state()

    long = FGOConfig(
        population_size=6,
        max_evals=40,
        seed=1,
        snapshot_interval=10,
        checkpoint_interval=0,
    )
    resumed = FGOOptimizer(sphere, 2, long).run(state)
    assert resumed.evaluations == long.max_evals
    assert len(resumed.history) == 20


def test_invalid_objective_result() -> None:
    config = FGOConfig(
        population_size=4,
        max_evals=4,
        seed=1,
        snapshot_interval=0,
        checkpoint_interval=0,
    )
    with pytest.raises(TypeError):
        FGOOptimizer(lambda vector: "bad", 2, config).run()


def test_network_problem_integration() -> None:
    problem = NetworkProblem(synthetic_instance("uniform", 5, seed=1), K=2)
    config = FGOConfig(
        population_size=8,
        max_evals=32,
        seed=4,
        snapshot_interval=8,
        checkpoint_interval=0,
    )
    result = FGOOptimizer(problem.evaluate, problem.dimension, config).run()
    assert result.best_details is not None
    assert "metrics" in result.best_details
    assert result.best_details["metrics"]["J"] == pytest.approx(result.best_score)
    assert result.snapshots[-1]["best_details"]["metrics"]["J"] == pytest.approx(
        result.best_score
    )
