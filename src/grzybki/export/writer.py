from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..fgo.optimizer import OptimizerState, save_checkpoint
from .schema import (
    read_json,
    read_jsonl,
    to_jsonable,
    write_json,
    write_jsonl_line,
)


@dataclass
class RunArtifacts:
    run_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any] | None
    best: dict[str, Any] | None
    history: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]


class RunWriter:
    def __init__(
        self,
        run_dir: str | Path,
        manifest: dict[str, Any],
        *,
        append: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = to_jsonable(manifest)
        self.append = append

        manifest_path = self.run_dir / "manifest.json"
        if not append or not manifest_path.exists():
            write_json(manifest_path, self.manifest)

        mode = "a" if append else "w"
        self.history_handle = (self.run_dir / "history.jsonl").open(
            mode, encoding="utf-8"
        )
        self.snapshot_handle = (self.run_dir / "snapshots.jsonl").open(
            mode, encoding="utf-8"
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def append_history(self, record: dict[str, Any]) -> None:
        write_jsonl_line(self.history_handle, record)

    def append_snapshot(self, snapshot: dict[str, Any]) -> None:
        write_jsonl_line(self.snapshot_handle, snapshot)

    def write_best(self, best: Any) -> None:
        write_json(self.run_dir / "best.json", best)

    def write_summary(self, summary: Any) -> None:
        write_json(self.run_dir / "summary.json", summary)

    def save_state(self, state: OptimizerState) -> None:
        save_checkpoint(self.checkpoint_dir, state)

    def close(self) -> None:
        if not self.history_handle.closed:
            self.history_handle.flush()
            self.history_handle.close()
        if not self.snapshot_handle.closed:
            self.snapshot_handle.flush()
            self.snapshot_handle.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def load_run(run_dir: str | Path) -> RunArtifacts:
    path = Path(run_dir)
    manifest = read_json(path / "manifest.json")
    summary_path = path / "summary.json"
    best_path = path / "best.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    best = read_json(best_path) if best_path.exists() else None
    history_path = path / "history.jsonl"
    snapshots_path = path / "snapshots.jsonl"
    history = read_jsonl(history_path) if history_path.exists() else []
    snapshots = read_jsonl(snapshots_path) if snapshots_path.exists() else []
    return RunArtifacts(
        run_dir=path,
        manifest=manifest,
        summary=summary,
        best=best,
        history=history,
        snapshots=snapshots,
    )
