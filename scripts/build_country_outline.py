#!/usr/bin/env python3
"""Extract one country's outline from Natural Earth admin-0 boundaries.

The output is the small GeoJSON bundled with the package and drawn under
network plots. Natural Earth is public domain; attribution is appreciated
("Made with Natural Earth").
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "v5.1.2/geojson/ne_{scale}_admin_0_countries.geojson"
)
ATTRIBUTION = "Made with Natural Earth (public domain)"


def fetch(url: str, cache: Path, force: bool) -> dict[str, Any]:
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": "grzybki/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return json.loads(payload)


def extract(payload: dict[str, Any], iso: str) -> dict[str, Any]:
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        if iso in (properties.get("ISO_A2"), properties.get("ISO_A2_EH")):
            geometry = feature["geometry"]
            return {
                "type": "FeatureCollection",
                "name": f"outline_{iso}",
                "attribution": ATTRIBUTION,
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "iso_a2": iso,
                            "name": properties.get("NAME"),
                        },
                        "geometry": _round(geometry, 4),
                    }
                ],
            }
    raise SystemExit(f"country {iso!r} not found")


def _round(geometry: dict[str, Any], digits: int) -> dict[str, Any]:
    def walk(value: Any) -> Any:
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            return [round(float(item), digits) for item in value]
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", default="PL", help="ISO 3166-1 alpha-2 code")
    parser.add_argument("--scale", default="50m", choices=["10m", "50m", "110m"])
    parser.add_argument(
        "--cache", type=Path, default=None, help="raw download cache path"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: src/grzybki/data/outline_<ISO>.geojson",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = args.cache or Path(f"data/reference/ne_{args.scale}_admin_0_countries.geojson")
    out = args.out or Path(f"src/grzybki/data/outline_{args.iso}.geojson")
    payload = fetch(SOURCE_URL.format(scale=args.scale), cache, args.force)
    outline = extract(payload, args.iso)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outline, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
