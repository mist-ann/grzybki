import json
from types import SimpleNamespace

import numpy as np

from grzybki.config import FGOConfig, ObjectiveWeights
from grzybki.export import (
    RunWriter,
    load_run,
    manifest_payload,
    summary_payload,
    to_jsonable,
    write_json,
    read_json,
)
from grzybki.fgo import load_checkpoint
from grzybki.problem import NetworkProblem, synthetic_instance


def test_to_jsonable_handles_numpy_and_nonfinite() -> None:
    payload = {
        "array": np.asarray([1.0, 2.0]),
        "scalar": np.float64(3.5),
        "nan": float("nan"),
        "inf": float("inf"),
        "path": __import__("pathlib").Path("x"),
    }
    result = to_jsonable(payload)
    assert result["array"] == [1.0, 2.0]
    assert result["scalar"] == 3.5
    assert result["nan"] is None
    assert result["inf"] is None
    assert result["path"] == "x"
    json.dumps(result, allow_nan=False)


def test_write_read_json(tmp_path) -> None:
    path = tmp_path / "nested" / "data.json"
    write_json(path, {"a": np.asarray([1, 2])})
    assert read_json(path) == {"a": [1, 2]}


def test_run_writer_and_loader(tmp_path) -> None:
    instance = synthetic_instance("uniform", 5, seed=1)
    problem = NetworkProblem(instance, K=2, weights=ObjectiveWeights())
    config = FGOConfig(population_size=6, max_evals=12, seed=1)
    manifest = manifest_payload(instance=instance, problem=problem, config=config)
    result = SimpleNamespace(
        best_score=0.5,
        best_vector=np.asarray([0.1, 0.2, 0.3]),
        best_details={"metrics": {"J": 0.5}},
        evaluations=12,
        runtime=0.01,
        config=config,
    )
    summary = summary_payload(result=result)

    with RunWriter(tmp_path / "run", manifest) as writer:
        writer.append_history({"t": 1, "score": 1.0})
        writer.append_snapshot({"t": 1, "best_score": 1.0})
        writer.write_best({"vector": [0.1, 0.2], "score": 0.5})
        writer.write_summary(summary)
        optimizer_state = SimpleNamespace(
            population=np.zeros((2, 2)),
            fitness=np.ones(2),
            best_vector=np.zeros(2),
            best_score=1.0,
            best_payload=None,
            t=2,
            iteration=1,
            rng_state={"state": 1},
        )
        writer.save_state(optimizer_state)

    loaded = load_run(tmp_path / "run")
    assert loaded.manifest["schema_version"] == 1
    assert loaded.summary["best_score"] == 0.5
    assert loaded.best["vector"] == [0.1, 0.2]
    assert loaded.history == [{"t": 1, "score": 1.0}]
    assert loaded.snapshots == [{"t": 1, "best_score": 1.0}]
    checkpoint = load_checkpoint(tmp_path / "run" / "checkpoints")
    assert checkpoint.t == 2


def test_run_writer_append(tmp_path) -> None:
    manifest = {"schema_version": 1}
    with RunWriter(tmp_path, manifest) as writer:
        writer.append_history({"t": 1})
    with RunWriter(tmp_path, manifest, append=True) as writer:
        writer.append_history({"t": 2})
    loaded = load_run(tmp_path)
    assert [record["t"] for record in loaded.history] == [1, 2]
