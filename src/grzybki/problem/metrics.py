from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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
    km_per_unit: float | None = None

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

    @property
    def cost_km(self) -> float | None:
        return None if self.km_per_unit is None else self.cost * self.km_per_unit

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {
            "C": self.cost,
            "D": self.detour,
            "FT": self.fault_tolerance,
            "J": self.objective,
            "C_mst": self.cost_mst,
            "D_mst": self.detour_mst,
            "edge_count": self.edge_count,
            "extra_edge_count": self.extra_edge_count,
        }
        if self.km_per_unit is not None:
            data["km_per_unit"] = self.km_per_unit
            data["C_km"] = self.cost * self.km_per_unit
            data["C_mst_km"] = self.cost_mst * self.km_per_unit
        return data


def terminal_mst_baseline(terminals: np.ndarray) -> MSTBaseline:
    terminals = np.asarray(terminals, dtype=float)
    if terminals.ndim != 2 or terminals.shape[1] != 2:
        raise ValueError("terminals must have shape (n, 2)")
    if terminals.shape[0] < 2:
        raise ValueError("at least two terminals are required")
    edges = minimum_spanning_tree_edges(terminals)
    matrix, _ = _adjacency(terminals.shape[0], edges)
    return MSTBaseline(
        cost=_cost(edges),
        detour=_detour(terminals, matrix, terminals.shape[0]),
    )


def compute_metrics(
    network: NetworkGraph,
    weights: ObjectiveWeights | None = None,
    baseline: MSTBaseline | None = None,
    km_per_unit: float | None = None,
) -> Metrics:
    weights = weights or ObjectiveWeights()
    if baseline is None:
        baseline = terminal_mst_baseline(network.points[: network.terminal_count])

    points = np.asarray(network.points, dtype=float)
    matrix, neighbours = _adjacency(points.shape[0], network.edges)
    cost = _cost(network.edges)
    detour = _detour(points, matrix, network.terminal_count)
    fault_tolerance = _fault_tolerance(
        neighbours, network.terminal_count, network.edge_count
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
        km_per_unit=km_per_unit,
    )


def _adjacency(
    node_count: int, edges: Iterable[Edge]
) -> tuple[csr_matrix, dict[int, list[int]]]:
    """Symmetric CSR weight matrix plus neighbour lists.

    Duplicate edges keep the last occurrence, matching the previous NetworkX
    ``add_edge`` semantics. Zero lengths are clamped to ``EPSILON`` so that the
    sparse matrix does not drop them.
    """
    weights: dict[tuple[int, int], float] = {}
    for edge in edges:
        if edge.u == edge.v:
            continue
        key = (edge.u, edge.v) if edge.u < edge.v else (edge.v, edge.u)
        weights[key] = max(float(edge.length), EPSILON)
    neighbours: dict[int, list[int]] = {}
    if not weights:
        return csr_matrix((node_count, node_count), dtype=float), neighbours
    keys = np.asarray(list(weights.keys()), dtype=np.int64)
    values = np.fromiter(weights.values(), dtype=float, count=len(weights))
    rows = np.concatenate([keys[:, 0], keys[:, 1]])
    cols = np.concatenate([keys[:, 1], keys[:, 0]])
    matrix = csr_matrix(
        (np.concatenate([values, values]), (rows, cols)),
        shape=(node_count, node_count),
    )
    for u, v in weights:
        neighbours.setdefault(u, []).append(v)
        neighbours.setdefault(v, []).append(u)
    return matrix, neighbours


def _cost(edges: Iterable[Edge]) -> float:
    return float(sum(edge.length for edge in edges))


def _detour(points: np.ndarray, matrix: csr_matrix, terminal_count: int) -> float:
    """Unweighted mean ratio of network distance to straight-line distance."""
    if terminal_count < 2:
        return 1.0
    terminals = np.arange(terminal_count)
    distances = dijkstra(matrix, directed=False, indices=terminals)
    rows, cols = np.triu_indices(terminal_count, k=1)
    direct = np.linalg.norm(points[rows] - points[cols], axis=1)
    keep = direct > EPSILON
    if not np.any(keep):
        return 1.0
    network = distances[rows[keep], cols[keep]]
    if not np.all(np.isfinite(network)):
        return float("inf")
    return float(np.mean(network / direct[keep]))


def _fault_tolerance(
    neighbours: dict[int, list[int]], terminal_count: int, edge_count: int
) -> float:
    """Share of edges that are not terminal-separating bridges.

    A bridge separates terminals when both sides of it contain at least one
    terminal. One iterative DFS (Tarjan low-link) from terminal 0 finds every
    bridge in its component together with the number of terminals in the
    subtree below it.
    """
    if edge_count == 0:
        return 0.0
    if terminal_count < 1:
        return 1.0
    order: dict[int, int] = {0: 0}
    low: dict[int, int] = {0: 0}
    below: dict[int, int] = {0: 1}
    separating = 0
    stack: list[tuple[int, int, int]] = [(0, -1, 0)]
    counter = 1
    while stack:
        node, parent, position = stack[-1]
        adjacent = neighbours.get(node, [])
        if position < len(adjacent):
            stack[-1] = (node, parent, position + 1)
            child = adjacent[position]
            if child == parent:
                continue
            if child in order:
                low[node] = min(low[node], order[child])
            else:
                order[child] = low[child] = counter
                below[child] = 1 if child < terminal_count else 0
                counter += 1
                stack.append((child, node, 0))
            continue
        stack.pop()
        if parent < 0:
            continue
        low[parent] = min(low[parent], low[node])
        below[parent] += below[node]
        if low[node] > order[parent] and 0 < below[node] < terminal_count:
            separating += 1

    if below[0] < terminal_count:
        return 0.0
    return max(0.0, (edge_count - separating) / edge_count)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= EPSILON:
        return 1.0
    return numerator / denominator
