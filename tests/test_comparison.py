import json
from pathlib import Path

import numpy as np
import pytest

from grzybki.problem import (
    Instance,
    NetworkGraph,
    NetworkProblem,
    compare_network_to_reference,
    edge_rail_distances,
    load_geojson_lines,
    point_to_lines_distance,
    point_to_polyline_distance,
    point_to_segment_distance,
    project_lines,
)
from grzybki.problem.graph import Edge


def test_point_to_segment_distance() -> None:
    assert point_to_segment_distance(
        np.asarray([0.5, 0.5]), np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0])
    ) == pytest.approx(0.5)
    assert point_to_segment_distance(
        np.asarray([2.0, 0.0]), np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0])
    ) == pytest.approx(1.0)


def test_point_to_polyline_and_lines() -> None:
    line = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert point_to_polyline_distance(np.asarray([0.5, 0.25]), line) == pytest.approx(
        0.25
    )
    assert point_to_lines_distance(
        np.asarray([0.5, 0.25]), [line, np.asarray([[0.5, 0.5], [0.5, 1.0]])]
    ) == pytest.approx(0.25)
    assert point_to_lines_distance(np.asarray([0.5, 0.5]), []) == float("inf")


def test_edge_and_network_comparison() -> None:
    graph = NetworkGraph(
        points=np.asarray([[0.0, 0.5], [1.0, 0.5], [0.0, 0.7], [1.0, 0.7]]),
        terminal_count=4,
        steiner_count=0,
        edges=[Edge(0, 1, 1.0), Edge(2, 3, 1.0)],
    )
    rail = [np.asarray([[0.0, 0.5], [1.0, 0.5]])]
    distances = edge_rail_distances(graph, rail)
    assert [item["distance"] for item in distances] == pytest.approx([0.0, 0.2])
    comparison = compare_network_to_reference(graph, rail)
    assert comparison["summary"]["mean"] == pytest.approx(0.1)
    assert comparison["summary"]["max"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        edge_rail_distances(graph, [])


def test_geojson_loading_and_projection(tmp_path) -> None:
    path = tmp_path / "lines.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[20.0, 51.0], [21.0, 52.0]],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[[20.0, 51.0], [20.5, 51.5]]],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    lines = load_geojson_lines(path)
    assert len(lines) == 2
    assert lines[0].shape == (2, 2)

    instance = Instance.from_lonlat(
        "two", [(20.0, 51.0), (21.0, 52.0)], names=["A", "B"]
    )
    instance.project()
    projected = project_lines(instance.projection, lines)
    assert len(projected) == 2
    assert np.all((projected[0] >= 0.0) & (projected[0] <= 1.0))


def test_polish_instances_load_and_project() -> None:
    root = Path(__file__).resolve().parents[1]
    for name, expected in [("poland_15", 15), ("poland_30", 30)]:
        instance = Instance.from_json(root / "data" / "instances" / f"{name}.json")
        assert instance.n == expected
        assert instance.terminals[0].name == "Warszawa"
        instance.project()
        assert np.all((instance.xy >= 0.0) & (instance.xy <= 1.0))
        problem = NetworkProblem(instance, K=expected - 2)
        assert problem.dimension == 2 * (expected - 2) + 1
        assert problem.baseline.cost > 0.0
