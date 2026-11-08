import json
from pathlib import Path

from click.testing import CliRunner

from grzybki.cli import main
from grzybki.export.writer import load_run
from grzybki.problem import Instance


def _make_instance(path: Path) -> Path:
    instance = Instance.from_xy(
        "tiny",
        [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        names=["A", "B", "C", "D"],
        source="test",
        license="CC0-1.0",
    )
    instance.save(path)
    return path


def test_optimize_and_visualize_end_to_end(tmp_path) -> None:
    instance_path = _make_instance(tmp_path / "tiny.json")
    run_dir = tmp_path / "run"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            str(instance_path),
            "--out",
            str(run_dir),
            "--seed",
            "7",
            "--population",
            "8",
            "--max-evals",
            "20",
            "--K",
            "2",
            "--snapshot-interval",
            "5",
            "--checkpoint-interval",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "best J" in result.output
    for name in [
        "manifest.json",
        "summary.json",
        "best.json",
        "history.jsonl",
        "snapshots.jsonl",
        "checkpoints/state.npz",
        "checkpoints/state.json",
    ]:
        assert (run_dir / name).exists(), name

    artifacts = load_run(run_dir)
    assert len(artifacts.history) == 20
    assert artifacts.snapshots
    assert artifacts.best is not None
    assert artifacts.manifest["schema_version"] == 1
    assert artifacts.manifest["extra"]["instance"]["name"] == "tiny"
    assert len(artifacts.best["best"]["graph"]["edges"]) > 0

    figures = tmp_path / "figures"
    result = runner.invoke(
        main,
        [
            "visualize",
            str(run_dir),
            "--out",
            str(figures),
            "--animation",
            "--fps",
            "2",
            "--dpi",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    for name in [
        "convergence.png",
        "metrics.png",
        "network.png",
        "animation.gif",
        "report.html",
    ]:
        assert (figures / name).exists(), name
    report = (figures / "report.html").read_text(encoding="utf-8")
    assert "FGO run: tiny" in report


def test_resume_appends_history(tmp_path) -> None:
    instance_path = _make_instance(tmp_path / "tiny.json")
    run_dir = tmp_path / "run"
    runner = CliRunner()
    common = [
        "optimize",
        str(instance_path),
        "--out",
        str(run_dir),
        "--population",
        "8",
        "--K",
        "2",
        "--snapshot-interval",
        "5",
        "--checkpoint-interval",
        "10",
    ]
    result = runner.invoke(main, common + ["--seed", "3", "--max-evals", "20"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        main,
        common + ["--seed", "3", "--max-evals", "24", "--resume", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    artifacts = load_run(run_dir)
    assert len(artifacts.history) == 24
    assert artifacts.summary is not None
    assert artifacts.summary["evaluations"] == 24


def test_visualize_reference_geojson(tmp_path) -> None:
    instance_path = _make_instance(tmp_path / "tiny.json")
    run_dir = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "optimize",
            str(instance_path),
            "--out",
            str(run_dir),
            "--population",
            "8",
            "--max-evals",
            "12",
            "--K",
            "2",
            "--snapshot-interval",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output

    reference = tmp_path / "reference.geojson"
    reference.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0.0, 0.0], [1.0, 1.0]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    figures = tmp_path / "figures"
    result = runner.invoke(
        main,
        [
            "visualize",
            str(run_dir),
            "--out",
            str(figures),
            "--reference",
            str(reference),
            "--no-report",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (figures / "network.png").exists()
    assert not (figures / "report.html").exists()


def test_instance_option_and_name_resolution(tmp_path) -> None:
    instance_path = _make_instance(tmp_path / "tiny.json")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            "--instance",
            str(instance_path),
            "--out",
            str(tmp_path / "run_path"),
            "--population",
            "8",
            "--max-evals",
            "10",
            "--K",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run_path" / "best.json").exists()

    result = runner.invoke(
        main,
        [
            "optimize",
            "--instance",
            "poland_15",
            "--out",
            str(tmp_path / "run_name"),
            "--population",
            "8",
            "--max-evals",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run_name" / "best.json").exists()
