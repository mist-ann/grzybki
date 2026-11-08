import numpy as np
import pytest

from grzybki.config import ObjectiveWeights
from grzybki.problem import compute_metrics, terminal_mst_baseline
from grzybki.problem.graph import NetworkGraph, build_network, minimum_spanning_tree_edges


def terminal_mst_graph(points: np.ndarray) -> NetworkGraph:
    edges = minimum_spanning_tree_edges(points)
    return NetworkGraph(
        points=points,
        terminal_count=len(points),
        steiner_count=0,
        edges=edges,
    )


def test_terminal_mst_objective_is_one() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    graph = terminal_mst_graph(points)
    metrics = compute_metrics(graph)
    assert metrics.cost == pytest.approx(metrics.cost_mst)
    assert metrics.detour == pytest.approx(metrics.detour_mst)
    assert metrics.fault_tolerance == 0.0
    assert metrics.objective == pytest.approx(1.0)
    assert metrics.C == metrics.cost
    assert metrics.D == metrics.detour
    assert metrics.FT == metrics.fault_tolerance
    assert metrics.J == metrics.objective


def test_line_detour_is_one() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    metrics = compute_metrics(terminal_mst_graph(points))
    assert metrics.detour == pytest.approx(1.0)
    assert metrics.cost == pytest.approx(2.0)


def test_extra_edges_improve_fault_tolerance() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    graph = build_network(points, terminal_count=4, theta=1.0)
    metrics = compute_metrics(graph, ObjectiveWeights(0.0, 0.0, 1.0))
    assert metrics.fault_tolerance > 0.0
    assert metrics.objective < 1.0
    assert metrics.extra_edge_count == len(graph.extra_edges)


def test_fault_tolerance_ignores_nonterminal_bridge() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0], [2.0, 1.0]])
    graph = NetworkGraph(
        points=points,
        terminal_count=2,
        steiner_count=3,
        edges=[
            _edge(0, 1, 1.0),
            _edge(1, 2, 1.0),
            _edge(2, 0, 1.0),
            _edge(2, 3, 1.0),
            _edge(3, 4, 1.0),
        ],
    )
    metrics = compute_metrics(graph, ObjectiveWeights(0.0, 0.0, 1.0))
    assert metrics.fault_tolerance == pytest.approx(1.0)


def test_baseline_reusable() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    baseline = terminal_mst_baseline(points)
    graph = terminal_mst_graph(points)
    first = compute_metrics(graph, baseline=baseline)
    second = compute_metrics(graph, baseline=baseline)
    assert first.to_dict() == second.to_dict()
    assert baseline.cost == pytest.approx(first.cost_mst)
    assert baseline.detour == pytest.approx(first.detour_mst)


def _edge(u: int, v: int, length: float):
    from grzybki.problem.graph import Edge

    return Edge(u=u, v=v, length=length, kind="mst")
