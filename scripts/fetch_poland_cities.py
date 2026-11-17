#!/usr/bin/env python3
"""Fetch a population-ranked ladder of Polish settlement terminal sets.

Downloads the GeoNames Poland dump (PL.zip), filters populated places with a
recorded population, ranks them by population, and writes one instance file
per requested size into ``data/instances/`` (``poland_15.json``, ...).

The files are nested prefixes of one ranking, so poland_15 is a subset of
poland_30, which is a subset of poland_50, and so on.

City districts recorded as standalone places (Warsaw boroughs, Fordon in
Bydgoszcz) are dropped via a (name, parent-city adm2) blocklist, so villages
with the same name elsewhere are kept.

Data: GeoNames (https://www.geonames.org), license CC BY 4.0.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://download.geonames.org/export/dump/PL.zip"
ATTRIBUTION = "GeoNames, CC BY 4.0"
DEFAULT_SIZES = (15, 30, 50, 100, 500, 1000, 5000)

# Populated-place codes to drop: city sections duplicate their parent city,
# and historical/abandoned places are not terminals.
EXCLUDED_CODES = {"PPLX", "PPLQ", "PPLH", "PPLW", "PPLCH"}

# City districts that GeoNames records as standalone PPL places, keyed by
# their parent city's name (matched through its adm2 code below).
DISTRICTS_BY_CITY = {
    "Warsaw": {
        "Bemowo", "Białołeka", "Białołęka", "Bielany", "Mokotów", "Ochota",
        "Praga Południe", "Praga Północ", "Rembertów", "Śródmieście",
        "Targówek", "Ursus", "Ursynów", "Wawer", "Wesoła", "Wilanów",
        "Włochy", "Wola", "Żoliborz",
    },
    "Bydgoszcz": {"Fordon"},
}


def download(cache: Path, force: bool) -> None:
    if cache.exists() and not force:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "grzybki/0.1 (rail network research)"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    cache.write_bytes(payload)
    print(f"cached {URL} -> {cache} ({len(payload)} bytes)")


def load_ranked(cache: Path) -> list[dict]:
    with zipfile.ZipFile(cache) as archive:
        member = next(
            n for n in archive.namelist() if not n.startswith("readme")
        )
        text = archive.read(member).decode("utf-8")

    rows: list[dict] = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 15:
            continue
        name, lat, lon = fields[1], fields[4], fields[5]
        feature_class, feature_code, country = fields[6], fields[7], fields[8]
        adm2, population = fields[11], fields[14]
        if country != "PL" or feature_class != "P":
            continue
        pop = int(population) if population.isdigit() else 0
        rows.append(
            {
                "name": name,
                "lon": round(float(lon), 4),
                "lat": round(float(lat), 4),
                "population": pop,
                "code": feature_code,
                "adm2": adm2,
            }
        )

    parent_adm2 = {
        city: next(r["adm2"] for r in rows if r["name"] == city)
        for city in DISTRICTS_BY_CITY
    }

    ranked: list[dict] = []
    for row in rows:
        if row["population"] <= 0:
            continue
        if row["code"] in EXCLUDED_CODES:
            continue
        districts = next(
            (names for city, names in DISTRICTS_BY_CITY.items()
             if row["adm2"] == parent_adm2[city]),
            None,
        )
        if districts is not None and row["name"] in districts:
            continue
        ranked.append(row)

    ranked.sort(key=lambda row: row["population"], reverse=True)
    return ranked


def write_instance(rows: list[dict], size: int, out_dir: Path) -> Path:
    path = out_dir / f"poland_{size}.json"
    data = {
        "name": f"poland_{size}",
        "description": (
            f"Polish settlement terminal set ({size} largest places by "
            "population) for idealized FGO experiments."
        ),
        "source": (
            "GeoNames Poland dump (PL.zip), download.geonames.org, retrieved "
            "2026-11-17; see docs/data.md"
        ),
        "license": "CC-BY-4.0 (GeoNames); attribution required.",
        "crs": "EPSG:4326",
        "terminals": [
            {"name": row["name"], "lon": row["lon"], "lat": row["lat"]}
            for row in rows[:size]
        ],
        "metadata": {
            "generator": "geonames_population_ranking",
            "retrieval_date": "2026-11-17",
            "endpoint": URL,
            "selection": (
                "Polish populated places (feature class P, excluding sections "
                "PPLX and historical/abandoned places, plus city districts "
                "recorded as standalone places) with population > 0, ranked "
                "by population"
            ),
            "coordinate_precision": "approximately 0.0001 degrees",
            "names": "Spelling follows GeoNames.",
            "notes": (
                "Coordinates are approximate settlement centres, not station "
                "locations. Nested prefix of the poland_5000 ranking."
            ),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES)
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("data/reference/geonames_PL.zip")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/instances"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download(args.cache, args.force)
    rows = load_ranked(args.cache)
    print(f"ranked Polish populated places: {len(rows)}")

    for size in args.sizes:
        if size > len(rows):
            raise SystemExit(
                f"cannot build poland_{size}: only {len(rows)} ranked places"
            )
        path = write_instance(rows, size, args.out_dir)
        smallest = rows[size - 1]
        print(
            f"wrote {path} ({size} terminals; "
            f"{rows[0]['name']} {rows[0]['population']} -> "
            f"{smallest['name']} {smallest['population']})"
        )
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
