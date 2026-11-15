import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_osm_rail.py"
spec = importlib.util.spec_from_file_location("fetch_osm_rail", SCRIPT)
fetch_osm_rail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_osm_rail)


def test_default_query_is_country_main_line_without_service() -> None:
    query = fetch_osm_rail.build_query("rail", 180)
    assert 'area["ISO3166-1"="PL"][admin_level=2]->.region;' in query
    assert '["railway"~"^(rail)$"]' in query
    assert '["usage"~"^(main|branch)$"]' in query
    assert '[!"service"]' in query
    assert "(area.region)" in query
    assert query.endswith("out geom;")


def test_bbox_overrides_country_and_filters_are_optional() -> None:
    query = fetch_osm_rail.build_query(
        "rail|tram",
        60,
        bbox=(14.0, 49.0, 24.2, 54.9),
        usage=None,
        include_service=True,
    )
    assert "area[" not in query
    assert "(49.0,14.0,54.9,24.2)" in query
    assert "usage" not in query
    assert "service" not in query
    assert '["railway"~"^(rail|tram)$"]' in query


def test_query_requires_scope() -> None:
    with pytest.raises(ValueError):
        fetch_osm_rail.build_query("rail", 60, country=None)
