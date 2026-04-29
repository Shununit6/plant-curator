from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ExifTags

JPG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
_EXIF_DATETIME_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal")


@dataclass
class Photo:
    path: Path
    captured_at: Optional[datetime]


def list_photos(folder: Path) -> Iterable[Photo]:
    for p in sorted(folder.rglob("*")):
        if p.name.startswith("._") or p.name.startswith("."):
            continue
        if p.suffix in JPG_EXTS and p.is_file():
            yield Photo(path=p, captured_at=_read_capture_time(p))


_EXIF_SUB_IFD = 0x8769  # ExifIFDPointer — where DateTimeOriginal actually lives


def _read_capture_time(path: Path) -> Optional[datetime]:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(_EXIF_SUB_IFD)
            raw = sub.get(_EXIF_DATETIME_TAG) or exif.get(_EXIF_DATETIME_TAG)
        if not raw:
            return None
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError):
        return None
