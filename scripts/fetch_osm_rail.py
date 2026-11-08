#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BBOX = "14.0,49.0,24.2,54.9"
DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
DEFAULT_RAILWAY = "rail|light_rail|narrow_gauge|subway|tram"
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
    bbox: tuple[float, float, float, float], railway: str, timeout: int
) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    return (
        f"[out:json][timeout:{timeout}];"
        f'way["railway"~"{railway}"]({overpass_bbox});'
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
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--railway", default=DEFAULT_RAILWAY)
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
        query = build_query(args.bbox, args.railway, args.timeout)
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
