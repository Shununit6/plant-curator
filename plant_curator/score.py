from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Scores:
    sharpness: float
    exposure: float
    colorfulness: float

    def combined(self) -> float:
        # Min-max normalization happens at ranking time; here we return a raw blend
        # that's roughly comparable across photos of similar size.
        return self.sharpness * self.exposure * (1 + self.colorfulness / 60)


def score_image(path: Path, max_side: int = 1024) -> Scores:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return Scores(0.0, 0.0, 0.0)
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return Scores(
        sharpness=_sharpness(gray),
        exposure=_exposure(gray),
        colorfulness=_colorfulness(bgr),
    )


def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure(gray: np.ndarray) -> float:
    """Fraction of pixels that aren't clipped at the extremes. 1.0 = nothing clipped."""
    total = gray.size
    clipped = int(np.count_nonzero(gray < 5)) + int(np.count_nonzero(gray > 250))
    return 1.0 - clipped / total


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness metric."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma_rgyb = float(np.sqrt(rg.var() + yb.var()))
    mu_rgyb = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return sigma_rgyb + 0.3 * mu_rgyb
