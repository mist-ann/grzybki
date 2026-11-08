from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    length: float
    kind: str = "mst"

    @property
    def key(self) -> tuple[int, int]:
        return canonical_edge(self.u, self.v)


@dataclass
class NetworkGraph:
    points: np.ndarray
    terminal_count: int
    steiner_count: int
    edges: list[Edge]
    requested_extra_edges: int = 0

    @property
    def node_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def terminal_indices(self) -> list[int]:
        return list(range(self.terminal_count))

    @property
    def steiner_indices(self) -> list[int]:
        return list(range(self.terminal_count, self.node_count))

    @property
    def degree(self) -> np.ndarray:
        degrees = np.zeros(self.node_count, dtype=int)
        for edge in self.edges:
            degrees[edge.u] += 1
            degrees[edge.v] += 1
        return degrees

    @property
    def active_steiner_indices(self) -> list[int]:
        degrees = self.degree
        return [
            index
            for index in self.steiner_indices
            if degrees[index] > 0
        ]

    @property
    def total_length(self) -> float:
        return float(sum(edge.length for edge in self.edges))

    @property
    def mst_edges(self) -> list[Edge]:
        return [edge for edge in self.edges if edge.kind == "mst"]

    @property
    def extra_edges(self) -> list[Edge]:
        return [edge for edge in self.edges if edge.kind == "extra"]

    def adjacency(self) -> dict[int, list[Edge]]:
        adjacency: dict[int, list[Edge]] = {
            index: [] for index in range(self.node_count)
        }
        for edge in self.edges:
            adjacency[edge.u].append(edge)
            adjacency[edge.v].append(edge)
        return adjacency

    def to_dict(self) -> dict[str, object]:
        return {
            "terminal_count": self.terminal_count,
            "steiner_count": self.steiner_count,
            "requested_extra_edges": self.requested_extra_edges,
            "points": self.points.tolist(),
            "edges": [
                {
                    "u": edge.u,
                    "v": edge.v,
                    "length": edge.length,
                    "kind": edge.kind,
                }
                for edge in self.edges
            ],
        }


def build_network(points: np.ndarray, terminal_count: int, theta: float) -> NetworkGraph:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if terminal_count < 2 or terminal_count > points.shape[0]:
        raise ValueError("terminal_count must be between 2 and the number of points")

    steiner_count = int(points.shape[0] - terminal_count)
    mst_edges = minimum_spanning_tree_edges(points)
    mst_keys = {edge.key for edge in mst_edges}

    delaunay = delaunay_edges(points)
    candidates = [
        edge for edge in delaunay if canonical_edge(edge[0], edge[1]) not in mst_keys
    ]
    candidates.sort(key=lambda item: (item[2], item[0], item[1]))
    requested = int(math.floor(np.clip(theta, 0.0, 1.0) * len(candidates)))
    extra_edges = [
        Edge(u=int(u), v=int(v), length=float(length), kind="extra")
        for u, v, length in candidates[:requested]
    ]

    edges = prune_steiner_leaves(
        points, terminal_count, steiner_count, [*mst_edges, *extra_edges]
    )
    return NetworkGraph(
        points=points,
        terminal_count=terminal_count,
        steiner_count=steiner_count,
        edges=edges,
        requested_extra_edges=requested,
    )


def minimum_spanning_tree_edges(points: np.ndarray) -> list[Edge]:
    points = np.asarray(points, dtype=float)
    node_count = int(points.shape[0])
    if node_count < 2:
        return []
    distances = cdist(points, points)
    candidates = [
        (float(distances[i, j]), i, j)
        for i, j in combinations(range(node_count), 2)
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = list(range(node_count))
    rank = [0] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    edges: list[Edge] = []
    for length, i, j in candidates:
        if union(i, j):
            edges.append(Edge(u=i, v=j, length=length, kind="mst"))
            if len(edges) == node_count - 1:
                break
    return edges


def delaunay_edges(points: np.ndarray) -> list[tuple[int, int, float]]:
    points = np.asarray(points, dtype=float)
    node_count = int(points.shape[0])
    if node_count < 3:
        return []

    pairs: set[tuple[int, int]] = set()
    try:
        triangulation = Delaunay(points)
    except (QhullError, ValueError):
        triangulation = None

    if triangulation is not None:
        for simplex in triangulation.simplices:
            for left, right in combinations(sorted(int(i) for i in simplex), 2):
                pairs.add((left, right))
    else:
        pairs = set(combinations(range(node_count), 2))

    distances = cdist(points, points)
    result = [
        (int(u), int(v), float(distances[u, v]))
        for u, v in pairs
    ]
    result.sort(key=lambda item: (item[2], item[0], item[1]))
    return result


def prune_steiner_leaves(
    points: np.ndarray,
    terminal_count: int,
    steiner_count: int,
    edges: Iterable[Edge],
) -> list[Edge]:
    current = list(edges)
    if steiner_count == 0:
        return current

    while True:
        degrees = np.zeros(int(points.shape[0]), dtype=int)
        for edge in current:
            degrees[edge.u] += 1
            degrees[edge.v] += 1
        leaves = {
            index
            for index in range(terminal_count, terminal_count + steiner_count)
            if degrees[index] == 1
        }
        if not leaves:
            break
        current = [
            edge
            for edge in current
            if edge.u not in leaves and edge.v not in leaves
        ]
    return current


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)
