from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from ..config import ObjectiveWeights
from .graph import Edge, NetworkGraph, minimum_spanning_tree_edges

EPSILON = 1e-12


@dataclass(frozen=True)
class MSTBaseline:
    cost: float
    detour: float


@dataclass(frozen=True)
class Metrics:
    cost: float
    detour: float
    fault_tolerance: float
    objective: float
    cost_mst: float
    detour_mst: float
    edge_count: int
    extra_edge_count: int

    @property
    def C(self) -> float:
        return self.cost

    @property
    def D(self) -> float:
        return self.detour

    @property
    def FT(self) -> float:
        return self.fault_tolerance

    @property
    def J(self) -> float:
        return self.objective

    def to_dict(self) -> dict[str, float | int]:
        return {
            "C": self.cost,
            "D": self.detour,
            "FT": self.fault_tolerance,
            "J": self.objective,
            "C_mst": self.cost_mst,
            "D_mst": self.detour_mst,
            "edge_count": self.edge_count,
            "extra_edge_count": self.extra_edge_count,
        }


def terminal_mst_baseline(terminals: np.ndarray) -> MSTBaseline:
    terminals = np.asarray(terminals, dtype=float)
    if terminals.ndim != 2 or terminals.shape[1] != 2:
        raise ValueError("terminals must have shape (n, 2)")
    if terminals.shape[0] < 2:
        raise ValueError("at least two terminals are required")
    edges = minimum_spanning_tree_edges(terminals)
    graph = _make_graph(terminals, edges)
    return MSTBaseline(
        cost=_cost(edges),
        detour=_detour(terminals, graph, list(range(terminals.shape[0]))),
    )


def compute_metrics(
    network: NetworkGraph,
    weights: ObjectiveWeights | None = None,
    baseline: MSTBaseline | None = None,
) -> Metrics:
    weights = weights or ObjectiveWeights()
    if baseline is None:
        baseline = terminal_mst_baseline(network.points[: network.terminal_count])

    graph = _make_graph(network.points, network.edges)
    cost = _cost(network.edges)
    detour = _detour(
        network.points, graph, list(range(network.terminal_count))
    )
    fault_tolerance = _fault_tolerance(
        graph, network.terminal_count, network.edge_count
    )

    cost_ratio = _safe_ratio(cost, baseline.cost)
    detour_ratio = _safe_ratio(detour, baseline.detour)
    objective = (
        weights.w1 * cost_ratio
        + weights.w2 * detour_ratio
        + weights.w3 * (1.0 - fault_tolerance)
    )
    return Metrics(
        cost=cost,
        detour=detour,
        fault_tolerance=fault_tolerance,
        objective=objective,
        cost_mst=baseline.cost,
        detour_mst=baseline.detour,
        edge_count=network.edge_count,
        extra_edge_count=sum(1 for edge in network.edges if edge.kind == "extra"),
    )


def _make_graph(points: np.ndarray, edges: Iterable[Edge]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(points.shape[0])))
    for edge in edges:
        graph.add_edge(
            edge.u,
            edge.v,
            weight=max(float(edge.length), EPSILON),
            length=float(edge.length),
            kind=edge.kind,
        )
    return graph


def _cost(edges: Iterable[Edge]) -> float:
    return float(sum(edge.length for edge in edges))


def _detour(
    points: np.ndarray, graph: nx.Graph, terminals: Sequence[int]
) -> float:
    if len(terminals) < 2:
        return 1.0
    paths = {
        source: dict(
            nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        )
        for source in terminals
    }
    ratios: list[float] = []
    for position, source in enumerate(terminals):
        for target in terminals[position + 1 :]:
            direct = float(np.linalg.norm(points[source] - points[target]))
            if direct <= EPSILON:
                continue
            if target not in paths.get(source, {}):
                return float("inf")
            ratios.append(float(paths[source][target]) / direct)
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


def _fault_tolerance(
    graph: nx.Graph, terminal_count: int, edge_count: int
) -> float:
    if edge_count == 0:
        return 0.0
    terminals = set(range(terminal_count))
    components = list(nx.connected_components(graph))
    terminal_component: set[int] | None = None
    for component in components:
        if 0 in component:
            terminal_component = component
            break
    if terminal_component is None or not terminals.issubset(terminal_component):
        return 0.0

    separating_bridges = 0
    for u, v in list(nx.bridges(graph)):
        edge_data = dict(graph.get_edge_data(u, v))
        graph.remove_edge(u, v)
        component = nx.node_connected_component(graph, u)
        terminal_hits = len(terminals.intersection(component))
        graph.add_edge(u, v, **edge_data)
        if 0 < terminal_hits < terminal_count:
            separating_bridges += 1
    return max(0.0, (edge_count - separating_bridges) / edge_count)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= EPSILON:
        return 1.0
    return numerator / denominator
