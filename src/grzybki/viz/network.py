from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..problem.decoder import DecodedNetwork
from ..problem.graph import NetworkGraph


def draw_network(
    ax: Axes,
    graph: NetworkGraph,
    *,
    title: str | None = None,
    reference: dict[str, Any] | None = None,
) -> None:
    points = np.asarray(graph.points, dtype=float)
    if graph.mst_edges:
        segments = [
            [points[edge.u], points[edge.v]] for edge in graph.mst_edges
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors="#1f77b4",
                linewidths=1.6,
                zorder=2,
                label="MST",
            )
        )
    if graph.extra_edges:
        segments = [
            [points[edge.u], points[edge.v]] for edge in graph.extra_edges
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors="#d62728",
                linewidths=1.3,
                linestyles="dashed",
                zorder=2,
                label="extra Delaunay",
            )
        )
    if reference:
        reference_points = np.asarray(reference.get("points", []), dtype=float)
        reference_edges = reference.get("edges", [])
        if len(reference_points) and reference_edges:
            segments = [
                [reference_points[u], reference_points[v]]
                for u, v in reference_edges
            ]
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=reference.get("color", "#7f7f7f"),
                    linewidths=reference.get("linewidth", 1.0),
                    alpha=reference.get("alpha", 0.6),
                    zorder=1,
                    label=reference.get("label", "reference"),
                )
            )
        if len(reference_points):
            ax.scatter(
                reference_points[:, 0],
                reference_points[:, 1],
                s=reference.get("size", 10),
                color=reference.get("color", "#7f7f7f"),
                alpha=reference.get("alpha", 0.6),
                zorder=1,
            )

    terminals = graph.terminal_indices
    steiner = graph.active_steiner_indices
    ax.scatter(
        points[terminals, 0],
        points[terminals, 1],
        s=34,
        color="#111111",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="terminals",
    )
    if steiner:
        ax.scatter(
            points[steiner, 0],
            points[steiner, 1],
            s=22,
            marker="s",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="Steiner",
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def plot_network(
    graph: NetworkGraph,
    out_path: str | Path,
    *,
    title: str | None = None,
    reference: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    draw_network(ax, graph, title=title, reference=reference)
    if metrics:
        label = (
            f"C={metrics['C']:.3f}  D={metrics['D']:.3f}  "
            f"FT={metrics['FT']:.3f}  J={metrics['J']:.3f}"
        )
        ax.text(
            0.01,
            0.01,
            label,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_decoded(
    decoded: DecodedNetwork,
    out_path: str | Path,
    *,
    title: str | None = None,
    reference: dict[str, Any] | None = None,
) -> Path:
    return plot_network(
        decoded.graph,
        out_path,
        title=title,
        reference=reference,
        metrics=decoded.metrics.to_dict(),
    )
