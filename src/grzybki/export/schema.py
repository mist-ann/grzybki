from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import __version__

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    return str(value)


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_line(handle: Any, payload: Any) -> None:
    handle.write(
        json.dumps(to_jsonable(payload), ensure_ascii=False, allow_nan=False) + "\n"
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def environment_info() -> dict[str, Any]:
    import matplotlib
    import networkx
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "grzybki": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "networkx": networkx.__version__,
    }


def manifest_payload(
    *,
    instance: Any,
    problem: Any,
    config: Any,
    command: str = "optimize",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instance_data = {
        "name": getattr(instance, "name", None),
        "n": getattr(instance, "n", None),
        "description": getattr(instance, "description", ""),
        "source": getattr(instance, "source", ""),
        "license": getattr(instance, "license", ""),
        "crs": getattr(instance, "crs", None),
    }
    projection = getattr(instance, "projection", None)
    if projection is not None:
        instance_data["projection"] = projection.to_dict()
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "command": command,
        "instance": instance_data,
        "problem": problem.to_dict() if hasattr(problem, "to_dict") else problem,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "environment": environment_info(),
    }
    if extra:
        data["extra"] = extra
    return data


def summary_payload(
    *,
    result: Any,
    history_file: str = "history.jsonl",
    snapshots_file: str = "snapshots.jsonl",
    best_file: str = "best.json",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "best_score": result.best_score,
        "best_vector": result.best_vector,
        "best_details": result.best_details,
        "evaluations": result.evaluations,
        "runtime": result.runtime,
        "config": result.config.to_dict() if hasattr(result.config, "to_dict") else result.config,
        "artifacts": {
            "history": history_file,
            "snapshots": snapshots_file,
            "best": best_file,
        },
    }


def merge_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_jsonable(record) for record in records]
