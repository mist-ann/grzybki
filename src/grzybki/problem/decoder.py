from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ObjectiveWeights
from .graph import NetworkGraph, build_network
from .instance import Instance
from .metrics import MSTBaseline, Metrics, compute_metrics, terminal_mst_baseline


@dataclass
class DecodedNetwork:
    vector: np.ndarray
    graph: NetworkGraph
    metrics: Metrics
    theta: float

    @property
    def score(self) -> float:
        return self.metrics.objective

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": self.theta,
            "metrics": self.metrics.to_dict(),
            "graph": self.graph.to_dict(),
        }


@dataclass
class NetworkProblem:
    instance: Instance
    K: int
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def __post_init__(self) -> None:
        if self.K < 0:
            raise ValueError("K must be non-negative")
        if not all(terminal.has_xy for terminal in self.instance.terminals):
            if all(terminal.has_lonlat for terminal in self.instance.terminals):
                self.instance.project()
            else:
                raise ValueError("instance must be projected or contain lon/lat")
        self._terminals = self.instance.xy
        self._baseline = terminal_mst_baseline(self._terminals)

    @property
    def dimension(self) -> int:
        return 2 * self.K + 1

    @property
    def terminals(self) -> np.ndarray:
        return self._terminals

    @property
    def baseline(self) -> MSTBaseline:
        return self._baseline

    def evaluate(self, vector: np.ndarray) -> DecodedNetwork:
        return decode(
            vector,
            terminals=self._terminals,
            K=self.K,
            weights=self.weights,
            baseline=self._baseline,
        )

    def evaluate_graph(self, graph: NetworkGraph) -> Metrics:
        return compute_metrics(graph, self.weights, self._baseline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance.name,
            "n": self.instance.n,
            "K": self.K,
            "dimension": self.dimension,
            "weights": self.weights.to_dict(),
            "baseline": {
                "C_mst": self._baseline.cost,
                "D_mst": self._baseline.detour,
            },
        }


def decode(
    vector: np.ndarray,
    terminals: np.ndarray,
    K: int,
    weights: ObjectiveWeights | None = None,
    baseline: MSTBaseline | None = None,
) -> DecodedNetwork:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    expected = 2 * K + 1
    if vector.size != expected:
        raise ValueError(f"expected vector of length {expected}, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    terminals = np.asarray(terminals, dtype=float)
    if terminals.ndim != 2 or terminals.shape[1] != 2:
        raise ValueError("terminals must have shape (n, 2)")

    if K:
        steiner = vector[: 2 * K].reshape(K, 2)
    else:
        steiner = np.empty((0, 2), dtype=float)
    points = np.vstack([terminals, steiner])
    theta = float(vector[-1])
    graph = build_network(points, terminal_count=len(terminals), theta=theta)
    metrics = compute_metrics(graph, weights, baseline)
    return DecodedNetwork(
        vector=vector.copy(),
        graph=graph,
        metrics=metrics,
        theta=theta,
    )
