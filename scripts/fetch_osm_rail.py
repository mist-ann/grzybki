#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_COUNTRY = "PL"
DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
DEFAULT_RAILWAY = "rail"
DEFAULT_USAGE = "main|branch"
ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise argparse.ArgumentTypeError("bbox minimums must be below maximums")
    return min_lon, min_lat, max_lon, max_lat


def build_query(
    railway: str,
    timeout: int,
    *,
    country: str | None = DEFAULT_COUNTRY,
    bbox: tuple[float, float, float, float] | None = None,
    usage: str | None = DEFAULT_USAGE,
    include_service: bool = False,
) -> str:
    """Build an Overpass query for railway ways.

    By default the query is clipped to the country's boundary (ISO 3166-1
    code), keeps only ``usage=main|branch`` lines, and drops service tracks
    (sidings, yards, spurs), which would otherwise dominate the geometry.
    """
    filters = f'["railway"~"^({railway})$"]'
    if usage:
        filters += f'["usage"~"^({usage})$"]'
    if not include_service:
        filters += '[!"service"]'
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        scope = f"({min_lat},{min_lon},{max_lat},{max_lon})"
        prefix = ""
    elif country:
        prefix = f'area["ISO3166-1"="{country}"][admin_level=2]->.region;'
        scope = "(area.region)"
    else:
        raise ValueError("either country or bbox is required")
    return (
        f"[out:json][timeout:{timeout}];"
        f"{prefix}"
        f"way{filters}{scope};"
        "out geom;"
    )


def fetch(endpoint: str, query: str, timeout: int) -> dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"User-Agent": "grzybki/0.1 (rail network research)"},
    )
    with urllib.request.urlopen(request, timeout=timeout + 30) as response:
        return json.load(response)


def to_geojson(payload: dict[str, Any]) -> dict[str, Any]:
    features = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        coordinates = [
            [float(point["lon"]), float(point["lat"])]
            for point in geometry
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 2:
            continue
        properties = dict(element.get("tags", {}))
        properties["osm_id"] = element.get("id")
        properties["attribution"] = ATTRIBUTION
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "osm_rail",
        "attribution": ATTRIBUTION,
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSM railway lines as GeoJSON")
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help="ISO 3166-1 code of the clipping boundary (ignored with --bbox)",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=None,
        help="min_lon,min_lat,max_lon,max_lat; overrides --country",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--railway", default=DEFAULT_RAILWAY, help="regex of railway=* values"
    )
    parser.add_argument(
        "--usage",
        default=DEFAULT_USAGE,
        help="regex of usage=* values; empty string disables the filter",
    )
    parser.add_argument(
        "--include-service",
        action="store_true",
        help="keep sidings, yards, spurs and other service=* tracks",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/reference/osm_rail_poland.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/reference/osm_rail_poland.geojson")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists() and not args.force:
        with args.cache.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        query = build_query(
            args.railway,
            args.timeout,
            country=args.country,
            bbox=args.bbox,
            usage=args.usage or None,
            include_service=args.include_service,
        )
        payload = fetch(args.endpoint, query, args.timeout)
        with args.cache.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")

    geojson = to_geojson(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(geojson, handle, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {len(geojson['features'])} railway lines to {args.out}")
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
