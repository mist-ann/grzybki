import json

import numpy as np
import pytest

from grzybki.problem import Instance, Terminal, synthetic_instance
from grzybki.problem.projection import Projection


def test_lonlat_roundtrip_and_projection(tmp_path) -> None:
    instance = Instance.from_lonlat(
        "cities",
        [(21.0122, 52.2297), (19.9445, 50.0647), (16.9252, 52.4064)],
        names=["Warszawa", "Krakow", "Poznan"],
    )
    instance.project()
    xy = instance.xy
    assert xy.shape == (3, 2)
    assert np.all((xy >= 0.0) & (xy <= 1.0))
    assert instance.projection is not None

    lonlat = instance.projection.inverse_array(xy)
    assert np.allclose(lonlat, instance.lonlat, atol=1e-9)

    path = tmp_path / "instance.json"
    instance.save(path)
    loaded = Instance.from_json(path)
    assert loaded.name == instance.name
    assert loaded.names == instance.names
    assert np.allclose(loaded.xy, xy)
    assert loaded.projection is not None


def test_from_xy_validation() -> None:
    instance = Instance.from_xy("tiny", [(0.1, 0.2), (0.8, 0.9)])
    assert instance.n == 2
    assert instance.xy.shape == (2, 2)
    with pytest.raises(ValueError):
        Instance.from_xy("bad", [(0.1, 0.2), (1.2, 0.9)])
    with pytest.raises(ValueError):
        Instance.from_xy("bad", [(0.1, 0.2)])


def test_terminal_validation() -> None:
    with pytest.raises(ValueError):
        Terminal("x", lon=1.0)
    with pytest.raises(ValueError):
        Terminal("x", x=1.0)
    with pytest.raises(ValueError):
        Terminal("x")


def test_project_requires_lonlat() -> None:
    instance = Instance.from_xy("tiny", [(0.1, 0.2), (0.8, 0.9)])
    assert instance.project() is instance


def test_synthetic_instances() -> None:
    for kind in ["uniform", "clustered", "ring", "grid"]:
        instance = synthetic_instance(kind, 15, seed=7)
        assert instance.n == 15
        assert instance.xy.shape == (15, 2)
        assert np.all((instance.xy >= 0.0) & (instance.xy <= 1.0))
        assert instance.metadata["generator"] == kind


def test_synthetic_reproducible() -> None:
    first = synthetic_instance("uniform", 20, seed=11)
    second = synthetic_instance("uniform", 20, seed=11)
    assert np.array_equal(first.xy, second.xy)
    assert first.names == second.names


def test_json_schema_accepts_projection() -> None:
    data = {
        "name": "manual",
        "crs": "EPSG:4326",
        "terminals": [
            {"name": "A", "lon": 20.0, "lat": 51.0},
            {"name": "B", "lon": 21.0, "lat": 52.0},
        ],
    }
    instance = Instance.from_dict(data)
    instance.project()
    payload = instance.to_dict()
    assert payload["projection"]["method"] == "equirectangular"
    json.dumps(payload)


def test_equirectangular_projection_roundtrip() -> None:
    projection = Projection.from_lonlat(
        np.asarray([19.0, 21.0, 23.0]),
        np.asarray([50.0, 52.0, 54.0]),
    )
    points = np.asarray([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    lonlat = projection.inverse_array(points)
    restored = projection.forward_array(lonlat[:, 0], lonlat[:, 1])
    assert np.allclose(restored, points)
