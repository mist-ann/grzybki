#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
)
DEFAULT_TYPE = "ms:BDOT10k_powiaty"
ATTRIBUTION = "GUGiK BDOT10k open data"


def parse_bbox(value: str) -> str:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat")
    if parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise argparse.ArgumentTypeError("bbox minimums must be below maximums")
    return value


def build_url(
    endpoint: str,
    type_name: str,
    *,
    bbox: str | None,
    srs_name: str,
    count: int | None,
    output_format: str,
) -> str:
    params: dict[str, str] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": type_name,
        "srsName": srs_name,
        "outputFormat": output_format,
    }
    if bbox:
        params["bbox"] = bbox
    if count is not None:
        params["count"] = str(count)
    separator = "&" if "?" in endpoint else "?"
    return endpoint + separator + urllib.parse.urlencode(params)


def fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "grzybki/0.1 (rail network research)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a BDOT10k WFS layer; default is the county index."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--type-name", default=DEFAULT_TYPE)
    parser.add_argument("--bbox", type=parse_bbox, default=None)
    parser.add_argument("--srs-name", default="EPSG:4326")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output-format", default="application/json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/reference/bdot10k_wfs.raw")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/reference/bdot10k_wfs.geojson")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists() and not args.force:
        payload = args.cache.read_bytes()
    else:
        url = build_url(
            args.endpoint,
            args.type_name,
            bbox=args.bbox,
            srs_name=args.srs_name,
            count=args.count,
            output_format=args.output_format,
        )
        payload = fetch(url, args.timeout)
        args.cache.write_bytes(payload)

    text = payload.decode("utf-8", errors="replace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if text.lstrip().startswith(("{", "[")):
        parsed: Any = json.loads(text)
        if isinstance(parsed, dict):
            parsed["attribution"] = ATTRIBUTION
        args.out.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    else:
        args.out.write_text(text, encoding="utf-8")
    print(f"wrote BDOT10k response for {args.type_name} to {args.out}")
    print(ATTRIBUTION)
    print(
        "Note: the default WFS layer is the county index. Railway geometry is "
        "distributed in county BDOT10k packages; see docs/data.md."
    )


if __name__ == "__main__":
    main()
