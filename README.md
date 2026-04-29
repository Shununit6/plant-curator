# plant-curator

Auto-curate the best photos from a folder of plant shots.

Status: **v0.1 — scan + sharpness scoring only.** Bigger features (visual grouping, web UI, export) coming next.

## Install

```bash
pip install -e .
```

## Use

```bash
plant-curator scan /path/to/photos
```

Lists every JPG in the folder with its sharpness score and EXIF capture time. Originals are never modified.
