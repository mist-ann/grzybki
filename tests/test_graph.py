import numpy as np
import pytest

from grzybki.problem.graph import (
    Edge,
    NetworkGraph,
    build_network,
    delaunay_edges,
    minimum_spanning_tree_edges,
    prune_steiner_leaves,
)


def test_mst_square() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    edges = minimum_spanning_tree_edges(points)
    assert len(edges) == 3
    assert sum(edge.length for edge in edges) == pytest.approx(3.0)
    assert all(edge.kind == "mst" for edge in edges)


def test_delaunay_extra_edges_and_floor() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    delaunay = delaunay_edges(points)
    assert len(delaunay) >= 5

    graph = build_network(points, terminal_count=4, theta=1.0)
    assert graph.edge_count >= 5
    assert len(graph.extra_edges) == graph.requested_extra_edges
    assert all(edge.kind in {"mst", "extra"} for edge in graph.edges)

    half = build_network(points, terminal_count=4, theta=0.5)
    assert half.requested_extra_edges == graph.requested_extra_edges // 2


def test_prune_steiner_leaf() -> None:
    points = np.zeros((5, 2))
    edges = [
        Edge(0, 2, 1.0),
        Edge(2, 3, 1.0),
        Edge(3, 1, 1.0),
        Edge(3, 4, 1.0),
    ]
    pruned = prune_steiner_leaves(points, terminal_count=2, steiner_count=3, edges=edges)
    assert len(pruned) == 3
    assert all(4 not in (edge.u, edge.v) for edge in pruned)


def test_prune_is_recursive() -> None:
    points = np.zeros((6, 2))
    edges = [
        Edge(0, 2, 1.0),
        Edge(2, 1, 1.0),
        Edge(2, 3, 1.0),
        Edge(3, 4, 1.0),
        Edge(4, 5, 1.0),
    ]
    pruned = prune_steiner_leaves(points, terminal_count=2, steiner_count=4, edges=edges)
    assert len(pruned) == 2


def test_collinear_delaunay_fallback() -> None:
    points = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edges = delaunay_edges(points)
    assert len(edges) == 6


def test_network_graph_degrees_and_active_steiner() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.1], [0.5, 0.2]])
    graph = NetworkGraph(
        points=points,
        terminal_count=2,
        steiner_count=2,
        edges=[Edge(0, 2, 1.0), Edge(2, 1, 1.0), Edge(2, 3, 1.0)],
    )
    assert graph.degree.tolist() == [1, 1, 3, 1]
    assert graph.active_steiner_indices == [2, 3]
    assert len(graph.adjacency()[2]) == 3
    assert graph.total_length == pytest.approx(3.0)


def test_build_network_validation() -> None:
    with pytest.raises(ValueError):
        build_network(np.zeros((3, 3)), 2, 0.5)
    with pytest.raises(ValueError):
        build_network(np.zeros((3, 2)), 1, 0.5)
    with pytest.raises(ValueError):
        build_network(np.zeros((3, 2)), 4, 0.5)
