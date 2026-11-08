from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .projection import Projection


@dataclass
class Terminal:
    name: str
    lon: float | None = None
    lat: float | None = None
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("terminal name must not be empty")
        if (self.lon is None) != (self.lat is None):
            raise ValueError(f"terminal {self.name!r} must define both lon and lat")
        if (self.x is None) != (self.y is None):
            raise ValueError(f"terminal {self.name!r} must define both x and y")
        if self.lon is None and self.x is None:
            raise ValueError(f"terminal {self.name!r} has no coordinates")

    @property
    def has_lonlat(self) -> bool:
        return self.lon is not None and self.lat is not None

    @property
    def has_xy(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class Instance:
    name: str
    terminals: list[Terminal]
    description: str = ""
    source: str = ""
    license: str = ""
    crs: str = "EPSG:4326"
    projection: Projection | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("instance name must not be empty")
        if len(self.terminals) < 2:
            raise ValueError("an instance requires at least two terminals")

    @property
    def n(self) -> int:
        return len(self.terminals)

    @property
    def names(self) -> list[str]:
        return [terminal.name for terminal in self.terminals]

    @property
    def xy(self) -> np.ndarray:
        if not all(terminal.has_xy for terminal in self.terminals):
            raise ValueError("instance is not projected; call project() first")
        return np.asarray(
            [[terminal.x, terminal.y] for terminal in self.terminals], dtype=float
        )

    @property
    def lonlat(self) -> np.ndarray:
        if not all(terminal.has_lonlat for terminal in self.terminals):
            raise ValueError("instance has no geographic coordinates")
        return np.asarray(
            [[terminal.lon, terminal.lat] for terminal in self.terminals], dtype=float
        )

    def project(self, method: str = "equirectangular") -> "Instance":
        if all(terminal.has_xy for terminal in self.terminals):
            return self
        if not all(terminal.has_lonlat for terminal in self.terminals):
            raise ValueError("cannot project an instance without lon/lat coordinates")
        lonlat = self.lonlat
        projection = Projection.from_lonlat(lonlat[:, 0], lonlat[:, 1], method=method)
        xy = projection.forward_array(lonlat[:, 0], lonlat[:, 1])
        for terminal, point in zip(self.terminals, xy):
            terminal.x = float(point[0])
            terminal.y = float(point[1])
        self.projection = projection
        return self

    def to_dict(self, include_projection: bool = True) -> dict[str, Any]:
        terminals: list[dict[str, Any]] = []
        for terminal in self.terminals:
            item: dict[str, Any] = {"name": terminal.name}
            if terminal.has_lonlat:
                item["lon"] = terminal.lon
                item["lat"] = terminal.lat
            if terminal.has_xy:
                item["x"] = terminal.x
                item["y"] = terminal.y
            terminals.append(item)
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "license": self.license,
            "crs": self.crs,
            "terminals": terminals,
            "metadata": self.metadata,
        }
        if include_projection and self.projection is not None:
            data["projection"] = self.projection.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        terminals = [
            Terminal(
                name=str(item.get("name", f"T{index + 1:02d}")),
                lon=_optional_float(item.get("lon")),
                lat=_optional_float(item.get("lat")),
                x=_optional_float(item.get("x")),
                y=_optional_float(item.get("y")),
            )
            for index, item in enumerate(data["terminals"])
        ]
        projection_data = data.get("projection")
        projection = (
            Projection.from_dict(projection_data) if projection_data is not None else None
        )
        return cls(
            name=str(data["name"]),
            terminals=terminals,
            description=str(data.get("description", "")),
            source=str(data.get("source", "")),
            license=str(data.get("license", "")),
            crs=str(data.get("crs", "EPSG:4326")),
            projection=projection,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Instance":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @classmethod
    def from_lonlat(
        cls,
        name: str,
        coordinates: Sequence[Sequence[float]],
        names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "Instance":
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2)")
        terminal_names = _terminal_names(len(points), names)
        terminals = [
            Terminal(name=terminal_names[i], lon=float(points[i, 0]), lat=float(points[i, 1]))
            for i in range(len(points))
        ]
        return cls(name=name, terminals=terminals, **kwargs)

    @classmethod
    def from_xy(
        cls,
        name: str,
        coordinates: Sequence[Sequence[float]],
        names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "Instance":
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2)")
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("normalized coordinates must lie in [0, 1]")
        terminal_names = _terminal_names(len(points), names)
        terminals = [
            Terminal(name=terminal_names[i], x=float(points[i, 0]), y=float(points[i, 1]))
            for i in range(len(points))
        ]
        return cls(name=name, terminals=terminals, **kwargs)


def synthetic_instance(
    kind: str,
    n: int,
    seed: int = 0,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> Instance:
    if n < 2:
        raise ValueError("n must be at least 2")
    if kind == "uniform":
        coordinates = _uniform_points(n, seed)
    elif kind == "clustered":
        coordinates = _clustered_points(n, seed, **kwargs)
    elif kind == "ring":
        coordinates = _ring_points(n, seed)
    elif kind == "grid":
        coordinates = _grid_points(n)
    else:
        raise ValueError(f"unknown synthetic instance kind: {kind}")

    instance_name = name or f"{kind}_{n}_{seed}"
    return Instance.from_xy(
        instance_name,
        coordinates,
        metadata={"generator": kind, "seed": seed, "parameters": kwargs},
    )


def _uniform_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _clustered_points(
    n: int, seed: int, clusters: int = 3, spread: float = 0.08
) -> np.ndarray:
    if clusters < 1:
        raise ValueError("clusters must be at least 1")
    if clusters > n:
        raise ValueError("clusters must not exceed n")
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, size=(clusters, 2))
    assignment = np.arange(n) % clusters
    rng.shuffle(assignment)
    points = centers[assignment] + rng.normal(0.0, spread, size=(n, 2))
    return np.clip(points, 0.0, 1.0)


def _ring_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    angles = angles + rng.normal(0.0, 0.015, size=n)
    radius = 0.35 + rng.normal(0.0, 0.025, size=n)
    points = np.column_stack(
        [0.5 + radius * np.cos(angles), 0.5 + radius * np.sin(angles)]
    )
    return np.clip(points, 0.0, 1.0)


def _grid_points(n: int) -> np.ndarray:
    columns = math.ceil(math.sqrt(n))
    rows = math.ceil(n / columns)
    points = []
    for row in range(rows):
        for column in range(columns):
            if len(points) == n:
                break
            points.append(((column + 0.5) / columns, (row + 0.5) / rows))
    return np.asarray(points, dtype=float)


def _terminal_names(n: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [f"T{index + 1:02d}" for index in range(n)]
    if len(names) != n:
        raise ValueError("names length must match number of coordinates")
    return [str(name) for name in names]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
