from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence(
    history: list[dict[str, Any]],
    out_path: str | Path,
    snapshots: list[dict[str, Any]] | None = None,
    title: str | None = None,
) -> Path:
    if not history:
        raise ValueError("history is empty")
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    t = [record["t"] for record in history]
    best = [record["best_score"] for record in history]
    candidate = [record["score"] for record in history]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.scatter(t, candidate, s=8, alpha=0.25, color="#7f7f7f", label="candidate")
    ax.plot(t, best, color="#1f77b4", linewidth=2.0, label="best")
    if snapshots:
        snapshot_t = [snapshot["t"] for snapshot in snapshots]
        snapshot_mean = [snapshot["fitness_mean"] for snapshot in snapshots]
        ax.plot(
            snapshot_t,
            snapshot_mean,
            color="#ff7f0e",
            linewidth=1.2,
            alpha=0.8,
            label="population mean",
        )
    ax.set_xlabel("evaluations")
    ax.set_ylabel("J")
    ax.set_title(title or "FGO convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_metrics(
    history: list[dict[str, Any]],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    records = [
        record
        for record in history
        if isinstance(record.get("details"), dict)
        and isinstance(record["details"].get("metrics"), dict)
    ]
    if not records:
        raise ValueError("history contains no metric details")
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    t = [record["t"] for record in records]
    series = {
        "C": [record["details"]["metrics"]["C"] for record in records],
        "D": [record["details"]["metrics"]["D"] for record in records],
        "FT": [record["details"]["metrics"]["FT"] for record in records],
        "J": [record["details"]["metrics"]["J"] for record in records],
    }
    labels = {
        "C": "cost C",
        "D": "detour D",
        "FT": "fault tolerance FT",
        "J": "objective J",
    }
    colors = {"C": "#1f77b4", "D": "#ff7f0e", "FT": "#2ca02c", "J": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=140, sharex=True)
    for axis, key in zip(axes.ravel(), ["C", "D", "FT", "J"]):
        axis.plot(t, series[key], color=colors[key], linewidth=1.4)
        axis.set_title(labels[key])
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("evaluations")
    fig.suptitle(title or "Network metrics")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
