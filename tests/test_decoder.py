import numpy as np
import pytest

from grzybki.problem import Instance, NetworkProblem, decode, synthetic_instance


def test_decode_k0_uses_terminal_mst() -> None:
    terminals = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    decoded = decode(np.asarray([0.0]), terminals, K=0)
    assert decoded.theta == 0.0
    assert decoded.graph.node_count == 3
    assert decoded.graph.steiner_count == 0
    assert decoded.metrics.objective == pytest.approx(1.0)


def test_decode_with_steiner_points() -> None:
    terminals = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vector = np.asarray([0.5, 0.5, 0.5, 0.8, 1.0])
    decoded = decode(vector, terminals, K=2)
    assert decoded.graph.node_count == 6
    assert decoded.graph.steiner_count == 2
    assert decoded.theta == 1.0
    assert np.allclose(decoded.graph.points[:4], terminals)
    assert np.allclose(decoded.graph.points[4:], [[0.5, 0.5], [0.5, 0.8]])


def test_decode_rejects_bad_vector() -> None:
    terminals = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        decode(np.asarray([0.5]), terminals, K=1)
    with pytest.raises(ValueError):
        decode(np.asarray([0.5, 0.5, np.nan]), terminals, K=1)


def test_decode_is_deterministic() -> None:
    terminals = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vector = np.asarray([0.2, 0.3, 0.6, 0.7, 0.4])
    first = decode(vector, terminals, K=2)
    second = decode(vector, terminals, K=2)
    assert first.metrics.to_dict() == second.metrics.to_dict()
    assert [edge.key for edge in first.graph.edges] == [
        edge.key for edge in second.graph.edges
    ]


def test_network_problem_projects_lonlat() -> None:
    instance = Instance.from_lonlat(
        "two",
        [(20.0, 51.0), (21.0, 52.0)],
        names=["A", "B"],
    )
    problem = NetworkProblem(instance, K=0)
    assert problem.dimension == 1
    assert instance.projection is not None
    decoded = problem.evaluate(np.asarray([0.0]))
    assert decoded.metrics.objective == pytest.approx(1.0)


def test_network_problem_dimension_and_baseline() -> None:
    instance = synthetic_instance("uniform", 6, seed=3)
    problem = NetworkProblem(instance, K=3)
    assert problem.dimension == 7
    assert problem.baseline.cost > 0.0
    assert problem.to_dict()["K"] == 3
    with pytest.raises(ValueError):
        NetworkProblem(instance, K=-1)
