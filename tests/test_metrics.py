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


def _networkx_reference(graph: NetworkGraph) -> tuple[float, float]:
    """The pre-v0.2 NetworkX implementation of D and FT, kept as an oracle."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(graph.node_count))
    for edge in graph.edges:
        g.add_edge(edge.u, edge.v, weight=max(edge.length, 1e-12))
    terminals = list(range(graph.terminal_count))
    ratios = []
    detour = 1.0
    for i, source in enumerate(terminals):
        lengths = nx.single_source_dijkstra_path_length(g, source, weight="weight")
        for target in terminals[i + 1 :]:
            direct = float(np.linalg.norm(graph.points[source] - graph.points[target]))
            if direct <= 1e-12:
                continue
            if target not in lengths:
                detour = float("inf")
                break
            ratios.append(lengths[target] / direct)
        if detour == float("inf"):
            break
    if detour != float("inf") and ratios:
        detour = float(np.mean(ratios))

    if graph.edge_count == 0 or not set(terminals) <= nx.node_connected_component(g, 0):
        return detour, 0.0
    separating = 0
    for u, v in list(nx.bridges(g)):
        data = dict(g.get_edge_data(u, v))
        g.remove_edge(u, v)
        hits = len(set(terminals) & nx.node_connected_component(g, u))
        g.add_edge(u, v, **data)
        if 0 < hits < len(terminals):
            separating += 1
    return detour, (graph.edge_count - separating) / graph.edge_count


@pytest.mark.parametrize("seed", range(40))
def test_scipy_metrics_match_networkx_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    k = int(rng.integers(0, 8))
    points = rng.random((n + k, 2))
    graph = build_network(points, terminal_count=n, theta=float(rng.random()))
    if seed % 4 == 0 and graph.edges:
        # Also exercise disconnected graphs and graphs with extra bridges.
        graph.edges = graph.edges[: max(1, len(graph.edges) // 2)]
    metrics = compute_metrics(graph)
    detour, fault_tolerance = _networkx_reference(graph)
    assert metrics.detour == pytest.approx(detour)
    assert metrics.fault_tolerance == pytest.approx(fault_tolerance)


def test_fault_tolerance_cycle_with_terminal_tail() -> None:
    # Triangle 0-1-2 plus a bridge 2-3 to terminal 3: only 2-3 separates terminals.
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 2.0]])
    graph = NetworkGraph(
        points=points,
        terminal_count=4,
        steiner_count=0,
        edges=[_edge(0, 1, 1.0), _edge(1, 2, 1.0), _edge(2, 0, 1.0), _edge(2, 3, 1.0)],
    )
    assert compute_metrics(graph).fault_tolerance == pytest.approx(3 / 4)


def test_km_per_unit_adds_kilometre_fields() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    metrics = compute_metrics(terminal_mst_graph(points), km_per_unit=500.0)
    payload = metrics.to_dict()
    assert payload["C_km"] == pytest.approx(1000.0)
    assert payload["C_mst_km"] == pytest.approx(1000.0)
    assert metrics.cost_km == pytest.approx(1000.0)
    assert "C_km" not in compute_metrics(terminal_mst_graph(points)).to_dict()
