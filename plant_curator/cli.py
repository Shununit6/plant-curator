import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
from tqdm import tqdm

from . import cache as cache_mod
from . import taste as taste_mod
from .cluster import collapse_bursts, kmeans, select_mmr
from .ingest import Photo, list_photos
from .score import Scores, score_image

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


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--n", default=9, show_default=True, help="How many diverse picks.")
@click.option("--by", "metric", type=click.Choice(METRICS), default="combined", show_default=True,
              help="Score used as MMR quality term.")
@click.option("--lam", default=0.4, show_default=True,
              help="MMR lambda: 1.0 = pure quality, 0.0 = pure diversity.")
@click.option("--drop-pct", default=0.3, show_default=True,
              help="Drop the bottom fraction by sharpness before selection.")
@click.option("--out", "out", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Output folder for picks.")
def portfolio(folder: Path, n: int, metric: str, lam: float, drop_pct: float,
              out: Optional[Path]) -> None:
    """N visually-diverse best picks for a grid post (e.g. Instagram 3x3)."""
    photos = list(list_photos(folder))
    if not photos:
        click.echo(f"No JPG files found under {folder}")
        return
    if len(photos) <= n:
        click.echo(f"Folder has only {len(photos)} photos; nothing to cluster.")
        return

    click.echo(f"Found {len(photos)} photos. Analyzing…")
    rows = _analyze(photos, with_embeddings=True)

    if drop_pct > 0:
        sharps = sorted(r[1].sharpness for r in rows)
        cutoff = sharps[int(drop_pct * len(sharps))]
        before = len(rows)
        rows = [r for r in rows if r[1].sharpness >= cutoff]
        click.echo(f"Dropped bottom {drop_pct:.0%} by sharpness "
                   f"({before - len(rows)} discarded, {len(rows)} remain).")

    # Collapse bursts: photos within 90s and CLIP sim >= 0.75 = one subject
    indexed = [(i, r[0].captured_at, r[2]) for i, r in enumerate(rows)
               if r[0].captured_at is not None]
    no_time = [i for i, r in enumerate(rows) if r[0].captured_at is None]
    groups = collapse_bursts(indexed)
    for i in no_time:
        groups.append([i])
    click.echo(f"Collapsed into {len(groups)} subject groups (from {len(rows)} photos).")

    # Pick the highest-scored representative from each group
    representatives = []
    for grp in groups:
        best = max(grp, key=lambda i: _value(rows[i][1], metric))
        representatives.append(rows[best])

    if len(representatives) <= n:
        click.echo(f"Only {len(representatives)} subjects — returning all.")
        picks = representatives
    else:
        embeddings = np.stack([r[2] for r in representatives])
        scores = np.array([_value(r[1], metric) for r in representatives])
        # Blend in taste-vector aesthetic score, if a model is trained
        taste = taste_mod.compute_taste()
        if taste is not None:
            n_liked = cache_mod.count_liked()
            aesthetic = taste_mod.aesthetic_scores(embeddings, taste)
            # Min-max normalize technical scores into [0,1] before blending
            s_min, s_max = float(scores.min()), float(scores.max())
            tech = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones_like(scores)
            scores = 0.5 * tech + 0.5 * aesthetic
            click.echo(f"Using taste model from {n_liked} liked examples.")
        idxs = select_mmr(embeddings, scores, n=n, lam=lam)
        picks = [representatives[i] for i in idxs]

    picks.sort(key=lambda r: r[0].captured_at or datetime.min)

    click.echo(f"\nPicked {len(picks)} diverse photos (one per cluster):\n")
    click.echo(f"{'#':>2}  {'sharp':>7}  {'exp':>5}  {'color':>6}  {'captured':<19}  path")
    click.echo("-" * 90)
    for i, (ph, s, _) in enumerate(picks, 1):
        ts = ph.captured_at.strftime("%Y-%m-%d %H:%M:%S") if ph.captured_at else "-"
        click.echo(
            f"{i:>2}  {s.sharpness:>7.1f}  {s.exposure:>5.3f}  {s.colorfulness:>6.1f}  "
            f"{ts:<19}  {ph.path.relative_to(folder)}"
        )

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for i, (ph, _, _) in enumerate(picks, 1):
            shutil.copy2(ph.path, out / f"{i:02d}_{ph.path.name}")
        click.echo(f"\nCopied to {out}")


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
def like(folder: Path) -> None:
    """Mark every photo in FOLDER as a liked example (training data for taste)."""
    photos = list(list_photos(folder))
    if not photos:
        click.echo(f"No JPG files found under {folder}")
        return

    click.echo(f"Found {len(photos)} photos. Ensuring scores + embeddings…")
    rows = _analyze(photos, with_embeddings=True)

    for ph, _, _ in rows:
        h = cache_mod.file_hash(ph.path)
        cache_mod.set_liked(h, True)

    total = cache_mod.count_liked()
    click.echo(f"\nMarked {len(rows)} photos as liked.")
    click.echo(f"Total liked examples in your taste model: {total}")
    if total < taste_mod.MIN_EXAMPLES:
        click.echo(f"(Need at least {taste_mod.MIN_EXAMPLES} liked photos before "
                   f"the taste model kicks in. Keep going.)")
    else:
        click.echo("Taste model is active — future portfolio runs will use it.")


@main.command()
def taste() -> None:
    """Show stats on your trained taste model."""
    n = cache_mod.count_liked()
    click.echo(f"Liked examples: {n}")
    if n < taste_mod.MIN_EXAMPLES:
        click.echo(f"Need at least {taste_mod.MIN_EXAMPLES} to activate the taste model.")
    else:
        click.echo("Taste model: active. Used by `portfolio` to bias picks toward your style.")


def _value(scores, metric: str) -> float:
    if metric == "combined":
        return scores.combined()
    return getattr(scores, metric)


def _analyze(photos: List[Photo], with_embeddings: bool):
    """Score (and optionally embed) photos using cache. Returns list of
    (photo, Scores, embedding-or-None)."""
    hashes = [cache_mod.file_hash(p.path) for p in photos]
    cached = cache_mod.get(hashes)

    need_score = [(p, h) for p, h in zip(photos, hashes)
                  if not cached.get(h) or cached[h].sharpness is None]
    if need_score:
        for ph, h in tqdm(need_score, desc="Scoring", unit="img"):
            s = score_image(ph.path)
            cache_mod.put(h, sharpness=s.sharpness, exposure=s.exposure,
                          colorfulness=s.colorfulness, captured_at=ph.captured_at)

    if with_embeddings:
        need_embed = [(p, h) for p, h in zip(photos, hashes)
                      if not cached.get(h) or cached[h].embedding is None]
        if need_embed:
            from .embed import embed_images
            click.echo(f"Embedding {len(need_embed)} photos with CLIP…")
            paths = [p.path for p, _ in need_embed]
            vecs = embed_images(paths)
            for (_, h), v in zip(need_embed, vecs):
                cache_mod.put(h, embedding=v)

    final = cache_mod.get(hashes)
    out: List[Tuple[Photo, Scores, Optional[np.ndarray]]] = []
    for ph, h in zip(photos, hashes):
        row = final[h]
        s = Scores(
            sharpness=row.sharpness or 0.0,
            exposure=row.exposure or 0.0,
            colorfulness=row.colorfulness or 0.0,
        )
        out.append((ph, s, row.embedding))
    return out


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
