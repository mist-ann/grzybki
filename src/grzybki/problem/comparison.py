from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .graph import NetworkGraph
from .projection import Projection


def point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-18:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / length_squared)
    projection = float(np.clip(projection, 0.0, 1.0))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    line = np.asarray(polyline, dtype=float)
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("polyline must have shape (m, 2)")
    if len(line) == 0:
        return float("inf")
    if len(line) == 1:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - line[0]))
    return min(
        point_to_segment_distance(point, line[index], line[index + 1])
        for index in range(len(line) - 1)
    )


def point_to_lines_distance(
    point: np.ndarray, lines: Sequence[np.ndarray]
) -> float:
    if not lines:
        return float("inf")
    return min(point_to_polyline_distance(point, line) for line in lines)


def edge_rail_distances(
    graph: NetworkGraph, rail_lines: Sequence[np.ndarray]
) -> list[dict[str, Any]]:
    if not rail_lines:
        raise ValueError("at least one reference line is required")
    points = np.asarray(graph.points, dtype=float)
    distances = []
    for edge in graph.edges:
        midpoint = (points[edge.u] + points[edge.v]) / 2.0
        distance = point_to_lines_distance(midpoint, rail_lines)
        distances.append(
            {
                "u": edge.u,
                "v": edge.v,
                "kind": edge.kind,
                "distance": distance,
            }
        )
    return distances


def compare_network_to_reference(
    graph: NetworkGraph, rail_lines: Sequence[np.ndarray]
) -> dict[str, Any]:
    distances = edge_rail_distances(graph, rail_lines)
    values = np.asarray([item["distance"] for item in distances], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        summary = {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    else:
        summary = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }
    return {
        "edge_count": len(distances),
        "summary": summary,
        "edges": distances,
    }


def load_geojson_lines(path: str | Path) -> list[np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    lines: list[np.ndarray] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if kind == "LineString":
            line = np.asarray(coordinates, dtype=float)
            if line.ndim == 2 and line.shape[1] >= 2 and len(line) >= 2:
                lines.append(line[:, :2])
        elif kind == "MultiLineString":
            for item in coordinates or []:
                line = np.asarray(item, dtype=float)
                if line.ndim == 2 and line.shape[1] >= 2 and len(line) >= 2:
                    lines.append(line[:, :2])
    return lines


def project_lines(
    projection: Projection, lines: Sequence[np.ndarray]
) -> list[np.ndarray]:
    projected = []
    for line in lines:
        line = np.asarray(line, dtype=float)
        if len(line) < 2:
            continue
        projected.append(projection.forward_array(line[:, 0], line[:, 1]))
    return projected
