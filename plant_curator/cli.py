import shutil
from collections import defaultdict
from datetime import datetime
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

    _print_ranked(rows, folder)

    if copy_to and rows:
        _copy_picks(rows, copy_to, prefix=False)
        click.echo(f"\nCopied {len(rows)} picks → {copy_to}")


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--date", "date_str", default=None,
              help="Day to pick from, YYYY-MM-DD. If omitted, lists available days.")
@click.option("--n", default=20, show_default=True, help="How many to pick.")
@click.option("--by", "metric", type=click.Choice(METRICS), default="combined", show_default=True,
              help="Score to rank by within each time bucket.")
@click.option("--spread/--no-spread", default=True, show_default=True,
              help="Spread picks evenly across the day instead of taking raw top-N.")
@click.option("--out", "out", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Output folder for picks (chronologically prefixed).")
def day(folder: Path, date_str: Optional[str], n: int, metric: str, spread: bool,
        out: Optional[Path]) -> None:
    """Best-of-day picks for storytelling. Output is sorted chronologically."""
    photos = [p for p in list_photos(folder) if p.captured_at]
    if not photos:
        click.echo("No photos with EXIF capture time found.")
        return

    by_date = defaultdict(list)
    for p in photos:
        by_date[p.captured_at.date()].append(p)

    if not date_str:
        click.echo(f"Available days in {folder}:")
        for d, ps in sorted(by_date.items()):
            first = min(ps, key=lambda x: x.captured_at).captured_at
            last = max(ps, key=lambda x: x.captured_at).captured_at
            click.echo(f"  {d.isoformat()}  {len(ps):>4} photos  ({first.strftime('%H:%M')}–{last.strftime('%H:%M')})")
        click.echo("\nRe-run with --date YYYY-MM-DD to pick from one.")
        return

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Bad --date format: {date_str!r}. Use YYYY-MM-DD.")
        return

    day_photos = by_date.get(target)
    if not day_photos:
        click.echo(f"No photos on {target.isoformat()}.")
        click.echo(f"Available: {', '.join(d.isoformat() for d in sorted(by_date))}")
        return

    click.echo(f"Scoring {len(day_photos)} photos from {target.isoformat()}…")
    scored = []
    for p in tqdm(day_photos, unit="img"):
        scored.append((p, score_image(p.path)))

    if spread:
        scored = _spread_picks(scored, n, metric)
    else:
        scored.sort(key=lambda r: _value(r[1], metric), reverse=True)
        scored = scored[:n]
    scored.sort(key=lambda r: r[0].captured_at)

    times = [p.captured_at for p, _ in scored]
    click.echo(f"\nPicked {len(scored)} of {len(day_photos)} "
               f"({times[0].strftime('%H:%M')} → {times[-1].strftime('%H:%M')}):\n")
    _print_ranked(scored, folder, chronological=True)

    if out:
        _copy_picks(scored, out, prefix=True)
        click.echo(f"\nCopied to {out}")


def _value(scores, metric: str) -> float:
    if metric == "combined":
        return scores.combined()
    return getattr(scores, metric)


def _spread_picks(scored, n: int, metric: str):
    """Divide the time range into n equal buckets, pick best from each.
    Empty buckets are filled from leftover by score."""
    if not scored or n >= len(scored):
        return list(scored)

    by_time = sorted(scored, key=lambda r: r[0].captured_at)
    earliest = by_time[0][0].captured_at
    latest = by_time[-1][0].captured_at
    span = (latest - earliest).total_seconds()
    if span <= 0:
        return sorted(scored, key=lambda r: _value(r[1], metric), reverse=True)[:n]

    buckets = [[] for _ in range(n)]
    for row in by_time:
        offset = (row[0].captured_at - earliest).total_seconds()
        idx = min(int(offset / span * n), n - 1)
        buckets[idx].append(row)

    picks = []
    leftover = []
    for bucket in buckets:
        if not bucket:
            continue
        best = max(bucket, key=lambda r: _value(r[1], metric))
        picks.append(best)
        leftover.extend(r for r in bucket if r is not best)

    needed = n - len(picks)
    if needed > 0:
        leftover.sort(key=lambda r: _value(r[1], metric), reverse=True)
        picks.extend(leftover[:needed])

    return picks


def _print_ranked(rows, folder: Path, chronological: bool = False) -> None:
    label = "time" if chronological else "rank"
    click.echo(f"{label:>4}  {'sharp':>7}  {'exp':>5}  {'color':>6}  {'captured':<19}  path")
    click.echo("-" * 90)
    for i, (ph, s) in enumerate(rows, 1):
        ts = ph.captured_at.strftime("%Y-%m-%d %H:%M:%S") if ph.captured_at else "-"
        click.echo(
            f"{i:>4}  {s.sharpness:>7.1f}  {s.exposure:>5.3f}  {s.colorfulness:>6.1f}  "
            f"{ts:<19}  {ph.path.relative_to(folder)}"
        )


def _copy_picks(rows, dest: Path, prefix: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for i, (ph, _) in enumerate(rows, 1):
        name = f"{i:02d}_{ph.path.name}" if prefix else ph.path.name
        shutil.copy2(ph.path, dest / name)
