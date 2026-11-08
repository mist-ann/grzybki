from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Projection:
    method: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    lon0: float = 0.0
    lat0: float = 0.0
    crs: str = "LOCAL"
    _transformer: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_lonlat(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        method: str = "equirectangular",
        crs: str = "EPSG:2180",
        margin: float = 0.05,
    ) -> "Projection":
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.shape != lat.shape:
            raise ValueError("lon and lat must have the same shape")
        if lon.size == 0:
            raise ValueError("at least one coordinate is required")

        lon0 = float(np.mean(lon))
        lat0 = float(np.mean(lat))

        if method == "equirectangular":
            x, y = _equirectangular_forward(lon, lat, lon0, lat0)
            projection_crs = "LOCAL"
        elif method == "epsg2180":
            transformer = _make_transformer(crs)
            x, y = transformer.transform(lon, lat)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            projection_crs = crs
        else:
            raise ValueError(f"unsupported projection method: {method}")

        x_min, x_max = _expanded_bounds(x, margin)
        y_min, y_max = _expanded_bounds(y, margin)
        transformer = _make_transformer(crs) if method == "epsg2180" else None
        return cls(
            method=method,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            lon0=lon0,
            lat0=lat0,
            crs=projection_crs,
            _transformer=transformer,
        )

    def forward(self, lon: float, lat: float) -> tuple[float, float]:
        if self.method == "equirectangular":
            x, y = _equirectangular_forward(
                np.asarray([lon], dtype=float),
                np.asarray([lat], dtype=float),
                self.lon0,
                self.lat0,
            )
            x_value = float(x[0])
            y_value = float(y[0])
        elif self.method == "epsg2180":
            transformer = self._get_transformer()
            x_value, y_value = transformer.transform(lon, lat)
        else:
            raise ValueError(f"unsupported projection method: {self.method}")
        return self._normalize(x_value, y_value)

    def forward_array(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if self.method == "equirectangular":
            x, y = _equirectangular_forward(lon, lat, self.lon0, self.lat0)
        elif self.method == "epsg2180":
            transformer = self._get_transformer()
            x, y = transformer.transform(lon, lat)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
        else:
            raise ValueError(f"unsupported projection method: {self.method}")
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        return np.column_stack([x_norm, y_norm])

    def inverse(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        x = self.x_min + float(x_norm) * (self.x_max - self.x_min)
        y = self.y_min + float(y_norm) * (self.y_max - self.y_min)
        if self.method == "equirectangular":
            lon, lat = _equirectangular_inverse(
                np.asarray([x]), np.asarray([y]), self.lon0, self.lat0
            )
            return float(lon[0]), float(lat[0])
        if self.method == "epsg2180":
            transformer = self._get_transformer()
            lon, lat = transformer.transform(x, y, direction="INVERSE")
            return float(lon), float(lat)
        raise ValueError(f"unsupported projection method: {self.method}")

    def inverse_array(self, xy_norm: np.ndarray) -> np.ndarray:
        xy_norm = np.asarray(xy_norm, dtype=float)
        x = self.x_min + xy_norm[:, 0] * (self.x_max - self.x_min)
        y = self.y_min + xy_norm[:, 1] * (self.y_max - self.y_min)
        if self.method == "equirectangular":
            lon, lat = _equirectangular_inverse(x, y, self.lon0, self.lat0)
        elif self.method == "epsg2180":
            transformer = self._get_transformer()
            lon, lat = transformer.transform(x, y, direction="INVERSE")
            lon = np.asarray(lon, dtype=float)
            lat = np.asarray(lat, dtype=float)
        else:
            raise ValueError(f"unsupported projection method: {self.method}")
        return np.column_stack([lon, lat])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "lon0": self.lon0,
            "lat0": self.lat0,
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Projection":
        method = str(data["method"])
        crs = str(data.get("crs", "LOCAL"))
        transformer = _make_transformer(crs) if method == "epsg2180" else None
        return cls(
            method=method,
            x_min=float(data["x_min"]),
            x_max=float(data["x_max"]),
            y_min=float(data["y_min"]),
            y_max=float(data["y_max"]),
            lon0=float(data.get("lon0", 0.0)),
            lat0=float(data.get("lat0", 0.0)),
            crs=crs,
            _transformer=transformer,
        )

    def _normalize(self, x: float, y: float) -> tuple[float, float]:
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        return x_norm, y_norm

    def _get_transformer(self) -> Any:
        if self._transformer is None:
            self._transformer = _make_transformer(self.crs)
        return self._transformer


def _expanded_bounds(values: np.ndarray, margin: float) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0:
        span = 1.0
    pad = margin * span
    return low - pad, high + pad


def _equirectangular_forward(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)
    x = EARTH_RADIUS_M * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS_M * (lat_rad - lat0_rad)
    return x, y


def _equirectangular_inverse(
    x: np.ndarray, y: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)
    lon_rad = lon0_rad + x / (EARTH_RADIUS_M * np.cos(lat0_rad))
    lat_rad = lat0_rad + y / EARTH_RADIUS_M
    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def _make_transformer(crs: str) -> Any:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for the epsg2180 projection; install grzybki[geo]"
        ) from exc
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)
