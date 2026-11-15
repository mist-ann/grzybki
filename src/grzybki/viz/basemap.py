"""Map context for network plots: country outline and terminal labels."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Sequence

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from ..problem.instance import Instance

OUTLINE_ATTRIBUTION = "Made with Natural Earth (public domain)"
BUNDLED_OUTLINES = ("PL",)

# Label candidate offsets in points, tried in order: right, upper right,
# lower right, left, upper left, lower left, above, below.
_LABEL_OFFSETS: tuple[tuple[float, float, str, str], ...] = (
    (5, 0, "left", "center"),
    (4, 4, "left", "bottom"),
    (4, -4, "left", "top"),
    (-5, 0, "right", "center"),
    (-4, 4, "right", "bottom"),
    (-4, -4, "right", "top"),
    (0, 6, "center", "bottom"),
    (0, -6, "center", "top"),
)


@dataclass
class Basemap:
    """Plot context in the same normalized coordinates as the network."""

    labels: list[str] | None = None
    outline: list[np.ndarray] = field(default_factory=list)
    attribution: str | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if not self.outline:
            return None
        stacked = np.vstack(self.outline)
        return (
            float(stacked[:, 0].min()),
            float(stacked[:, 0].max()),
            float(stacked[:, 1].min()),
            float(stacked[:, 1].max()),
        )


def load_outline(iso: str = "PL") -> list[np.ndarray]:
    """Return the bundled outline rings of a country as lon/lat arrays."""
    name = f"outline_{iso.upper()}.geojson"
    try:
        text = resources.files("grzybki").joinpath("data", name).read_text("utf-8")
    except (FileNotFoundError, OSError):
        return []
    payload = json.loads(text)
    rings: list[np.ndarray] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry.get("type") == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            if polygon:
                rings.append(np.asarray(polygon[0], dtype=float)[:, :2])
    return rings


def build_basemap(
    instance: Instance,
    *,
    labels: bool = True,
    outline: bool = True,
    min_inside: float = 0.5,
) -> Basemap:
    """Build plot context for an instance.

    The outline is used only when the instance has geographic coordinates and
    at least ``min_inside`` of its terminals fall inside a bundled country
    outline, so synthetic or foreign instances never get a stray border.
    """
    basemap = Basemap(labels=list(instance.names) if labels else None)
    if not outline or instance.projection is None:
        return basemap
    if not all(terminal.has_lonlat for terminal in instance.terminals):
        return basemap
    lonlat = instance.lonlat
    for iso in BUNDLED_OUTLINES:
        rings = load_outline(iso)
        if not rings:
            continue
        inside = np.zeros(len(lonlat), dtype=bool)
        for ring in rings:
            inside |= MplPath(ring).contains_points(lonlat)
        if inside.mean() >= min_inside:
            basemap.outline = [
                instance.projection.forward_array(ring[:, 0], ring[:, 1])
                for ring in rings
            ]
            basemap.attribution = OUTLINE_ATTRIBUTION
            break
    return basemap


def draw_outline(ax: Axes, basemap: Basemap) -> None:
    for index, ring in enumerate(basemap.outline):
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor="#f4f2ec",
                edgecolor="#8a8a8a",
                linewidth=0.9,
                zorder=0,
                label="border" if index == 0 else None,
            )
        )


def draw_labels(
    ax: Axes,
    points: np.ndarray,
    labels: Sequence[str],
    *,
    fontsize: float = 7.0,
    avoid: np.ndarray | None = None,
) -> list[Any]:
    """Place one label per point, greedily avoiding overlaps.

    Each label tries several offsets around its point and keeps the first
    whose bounding box overlaps no earlier label and no marker; if none is
    free, it keeps the offset with the smallest overlap. Axis limits must be
    final before calling this, because placement works in display space.
    """
    figure = ax.figure
    renderer = figure.canvas.get_renderer()
    marker_boxes = _marker_boxes(ax, points if avoid is None else avoid)
    placed: list[Bbox] = []
    texts = []
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    for point, label in zip(np.asarray(points, dtype=float), labels):
        best = None
        for dx, dy, ha, va in _LABEL_OFFSETS:
            text = ax.annotate(
                label,
                xy=(point[0], point[1]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color="#222222",
                zorder=5,
                path_effects=halo,
            )
            box = text.get_window_extent(renderer).expanded(1.04, 1.1)
            overlap = sum(_overlap_area(box, other) for other in placed)
            overlap += sum(
                _overlap_area(box, other)
                for index, other in enumerate(marker_boxes)
                if not _contains(other, point, ax)
            )
            if best is None or overlap < best[0]:
                if best is not None:
                    best[1].remove()
                best = (overlap, text, box)
            else:
                text.remove()
            if overlap == 0.0:
                break
        assert best is not None
        placed.append(best[2])
        texts.append(best[1])
    return texts


def _marker_boxes(ax: Axes, points: np.ndarray, radius: float = 3.5) -> list[Bbox]:
    display = ax.transData.transform(np.asarray(points, dtype=float))
    scale = ax.figure.dpi / 72.0
    r = radius * scale
    return [Bbox.from_extents(x - r, y - r, x + r, y + r) for x, y in display]


def _contains(box: Bbox, point: np.ndarray, ax: Axes) -> bool:
    x, y = ax.transData.transform(point)
    return bool(box.contains(x, y))


def _overlap_area(first: Bbox, second: Bbox) -> float:
    width = min(first.x1, second.x1) - max(first.x0, second.x0)
    height = min(first.y1, second.y1) - max(first.y0, second.y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return float(width * height)
