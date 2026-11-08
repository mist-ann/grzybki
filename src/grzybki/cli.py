import click

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="grzybki")
def main() -> None:
    pass


@main.command()
@click.argument("instance_arg", required=False, type=click.Path(dir_okay=False))
@click.option(
    "--instance",
    "instance_opt",
    type=click.Path(dir_okay=False),
    default=None,
    help="Instance path or a name under data/instances.",
)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--population", "population_size", type=int, default=30, show_default=True)
@click.option("--max-evals", type=int, default=20_000, show_default=True)
@click.option("--K", "K", type=int, default=None)
@click.option("--M", "M", type=float, default=0.6, show_default=True)
@click.option("--Ep", "Ep", type=float, default=0.7, show_default=True)
@click.option("--R", "R", type=float, default=0.9, show_default=True)
@click.option("--w1", type=float, default=0.4, show_default=True)
@click.option("--w2", type=float, default=0.4, show_default=True)
@click.option("--w3", type=float, default=0.2, show_default=True)
@click.option(
    "--projection",
    type=click.Choice(["equirectangular", "epsg2180"]),
    default="equirectangular",
    show_default=True,
)
@click.option(
    "--nutrient-mode",
    type=click.Choice(["fitness", "inverse"]),
    default="fitness",
    show_default=True,
)
@click.option("--snapshot-interval", type=int, default=100, show_default=True)
@click.option("--checkpoint-interval", type=int, default=1_000, show_default=True)
@click.option("--save-vectors", is_flag=True, help="Include candidate vectors in history.")
@click.option(
    "--resume",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Resume from a run directory or checkpoint directory.",
)
def optimize(
    instance_arg: str | None,
    instance_opt: str | None,
    out_dir: str | None,
    seed: int,
    population_size: int,
    max_evals: int,
    K: int | None,
    M: float,
    Ep: float,
    R: float,
    w1: float,
    w2: float,
    w3: float,
    projection: str,
    nutrient_mode: str,
    snapshot_interval: int,
    checkpoint_interval: int,
    save_vectors: bool,
    resume: str | None,
) -> None:
    from .commands import run_optimize

    instance = instance_arg or instance_opt
    if instance is None:
        raise click.UsageError("INSTANCE or --instance is required")

    run_optimize(
        instance,
        out_dir,
        seed=seed,
        population_size=population_size,
        max_evals=max_evals,
        K=K,
        M=M,
        Ep=Ep,
        R=R,
        w1=w1,
        w2=w2,
        w3=w3,
        projection=projection,
        nutrient_mode=nutrient_mode,
        snapshot_interval=snapshot_interval,
        checkpoint_interval=checkpoint_interval,
        save_vectors=save_vectors,
        resume=resume,
    )


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None)
@click.option("--animation", is_flag=True, help="Write a GIF or MP4 animation.")
@click.option(
    "--animation-format",
    type=click.Choice(["gif", "mp4"]),
    default="gif",
    show_default=True,
)
@click.option("--fps", type=int, default=10, show_default=True)
@click.option("--dpi", type=int, default=140, show_default=True)
@click.option(
    "--reference",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional GeoJSON reference rail lines.",
)
@click.option("--report/--no-report", default=True, show_default=True)
def visualize(
    run_dir: str,
    out_dir: str | None,
    animation: bool,
    animation_format: str,
    fps: int,
    dpi: int,
    reference: str | None,
    report: bool,
) -> None:
    from .commands import run_visualize

    run_visualize(
        run_dir,
        out_dir,
        animation=animation,
        animation_format=animation_format,
        fps=fps,
        dpi=dpi,
        reference=reference,
        report=report,
    )
