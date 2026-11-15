import numpy as np

from grzybki.problem import NetworkProblem, synthetic_instance
from grzybki.viz import (
    plot_convergence,
    plot_decoded,
    plot_metrics,
    write_animation,
    write_report,
)


def test_visualization_outputs(tmp_path) -> None:
    problem = NetworkProblem(synthetic_instance("uniform", 6, seed=2), K=2)
    vector = np.asarray([0.2, 0.3, 0.6, 0.7, 0.4])
    decoded = problem.evaluate(vector)

    network_path = plot_decoded(decoded, tmp_path / "network.png")
    assert network_path.exists() and network_path.stat().st_size > 0

    history = [
        {
            "t": index + 1,
            "score": 1.0 / (index + 1),
            "best_score": 1.0 / (index + 1),
            "details": {"metrics": decoded.metrics.to_dict(), "theta": decoded.theta},
        }
        for index in range(5)
    ]
    snapshots = [
        {"t": 1, "fitness_mean": 1.0},
        {"t": 3, "fitness_mean": 0.8},
        {"t": 5, "fitness_mean": 0.6},
    ]
    convergence_path = plot_convergence(history, tmp_path / "convergence.png", snapshots)
    assert convergence_path.exists() and convergence_path.stat().st_size > 0

    metrics_path = plot_metrics(history, tmp_path / "metrics.png")
    assert metrics_path.exists() and metrics_path.stat().st_size > 0

    animation_snapshots = [
        {"t": 1, "best_vector": vector.tolist()},
        {"t": 2, "best_vector": (vector * 0.8).tolist()},
        {"t": 3, "best_vector": (vector * 0.6).tolist()},
    ]
    animation_path = write_animation(
        problem, animation_snapshots, tmp_path / "animation.gif", fps=5, dpi=50
    )
    assert animation_path.exists() and animation_path.stat().st_size > 0

    report_path = write_report(
        tmp_path / "report.html",
        manifest={"instance": {"name": "test"}},
        summary={"best_score": decoded.metrics.objective},
        best={"vector": vector.tolist()},
        images={"Network": network_path, "Convergence": convergence_path},
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Network" in report
    assert "Convergence" in report
    assert "test" in report


def test_basemap_for_polish_instance(tmp_path) -> None:
    from grzybki.problem import Instance
    from grzybki.viz import build_basemap, load_outline
    from grzybki.viz.network import plot_decoded as plot

    assert load_outline("PL"), "bundled Poland outline is missing"
    instance = Instance.from_json("data/instances/poland_15.json")
    instance.project()
    basemap = build_basemap(instance)
    assert basemap.labels == instance.names
    assert basemap.outline and basemap.attribution
    x0, x1, y0, y1 = basemap.bounds
    assert x0 < 0.2 and x1 > 0.8 and y0 < 0.2 and y1 > 0.8

    problem = NetworkProblem(instance, K=2)
    decoded = problem.evaluate(np.full(problem.dimension, 0.5))
    path = plot(decoded, tmp_path / "network.png", basemap=basemap, dpi=60)
    assert path.exists() and path.stat().st_size > 0

    no_extras = build_basemap(instance, labels=False, outline=False)
    assert no_extras.labels is None and not no_extras.outline


def test_basemap_skips_outline_for_synthetic_instance() -> None:
    from grzybki.viz import build_basemap

    basemap = build_basemap(synthetic_instance("uniform", 5, seed=1))
    assert basemap.outline == [] and basemap.labels is not None


def test_labels_avoid_each_other() -> None:
    import matplotlib.pyplot as plt

    from grzybki.viz.basemap import draw_labels

    fig, ax = plt.subplots(figsize=(4, 4), dpi=72)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    points = np.asarray([[0.5, 0.5], [0.52, 0.5], [0.5, 0.52]])
    texts = draw_labels(ax, points, ["Alpha", "Beta", "Gamma"])
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in texts]
    assert len(texts) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert not boxes[i].overlaps(boxes[j])
    plt.close(fig)
