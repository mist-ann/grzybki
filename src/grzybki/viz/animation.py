from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..problem.decoder import DecodedNetwork, NetworkProblem
from ..problem.graph import NetworkGraph
from .network import draw_network


def write_animation(
    problem: NetworkProblem,
    snapshots: list[dict[str, Any]],
    out_path: str | Path,
    *,
    fps: int = 10,
    dpi: int = 100,
) -> Path:
    if not snapshots:
        raise ValueError("snapshots are empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for snapshot in snapshots:
        decoded = problem.evaluate(np.asarray(snapshot["best_vector"], dtype=float))
        frames.append(_render_frame(decoded, int(snapshot["t"]), dpi=dpi))

    suffix = output.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output, frames, duration=1000.0 / fps, loop=0)
    elif suffix in {".mp4", ".m4v"}:
        imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    else:
        raise ValueError("animation output must be .gif, .mp4, or .m4v")
    return output


def render_frame(
    graph: NetworkGraph,
    t: int,
    *,
    title: str | None = None,
    dpi: int = 100,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    draw_network(ax, graph, title=title or f"t = {t}")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _render_frame(decoded: DecodedNetwork, t: int, dpi: int) -> np.ndarray:
    title = f"t = {t}    J = {decoded.metrics.objective:.4f}"
    return render_frame(decoded.graph, t, title=title, dpi=dpi)
