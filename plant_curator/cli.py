import shutil
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from .ingest import list_photos
from .score import score_image

METRICS = ("sharpness", "exposure", "colorfulness", "combined")


@click.group()
def main() -> None:
    """plant-curator — auto-curate plant photos."""


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--by", "metric", type=click.Choice(METRICS), default="sharpness", show_default=True,
              help="Score to rank by.")
@click.option("--top", type=int, default=0, show_default=True,
              help="Show only the top N (0 = show all).")
@click.option("--copy-to", "copy_to", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Copy the top N picks into this folder.")
def scan(folder: Path, metric: str, top: int, copy_to: Optional[Path]) -> None:
    """Score photos in FOLDER and print a ranked list."""
    photos = list(list_photos(folder))
    if not photos:
        click.echo(f"No JPG files found under {folder}")
        return

    click.echo(f"Found {len(photos)} photos. Scoring…")
    rows = []
    for ph in tqdm(photos, unit="img"):
        s = score_image(ph.path)
        rows.append((ph, s))

    rows.sort(key=lambda r: _value(r[1], metric), reverse=True)
    if top:
        rows = rows[:top]

    click.echo()
    click.echo(f"{'rank':>4}  {'sharp':>7}  {'exp':>5}  {'color':>6}  {'captured':<19}  path")
    click.echo("-" * 90)
    for i, (ph, s) in enumerate(rows, 1):
        ts = ph.captured_at.strftime("%Y-%m-%d %H:%M:%S") if ph.captured_at else "-"
        click.echo(
            f"{i:>4}  {s.sharpness:>7.1f}  {s.exposure:>5.3f}  {s.colorfulness:>6.1f}  "
            f"{ts:<19}  {ph.path.relative_to(folder)}"
        )

    if copy_to and rows:
        copy_to.mkdir(parents=True, exist_ok=True)
        for ph, _ in rows:
            shutil.copy2(ph.path, copy_to / ph.path.name)
        click.echo(f"\nCopied {len(rows)} picks → {copy_to}")


def _value(scores, metric: str) -> float:
    if metric == "combined":
        return scores.combined()
    return getattr(scores, metric)
