"""Render Mandarin/English captions over photos, Xiaohongshu-style.

Templates compose three layers:
  1. photo treatment   — crop, warm tone, vignette, film grain, optional gradient
  2. paper card        — cream base + real CC0 paper-texture multiply-blended,
                          with deckle (rough) edges and a soft drop shadow
  3. text + ornaments  — Songti/NewYork serif text with soft shadow,
                          plus ink dots, hairlines, red square seals
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ASSET_DIR = Path(__file__).parent / "assets" / "textures"

SONGTI = "/System/Library/Fonts/Songti.ttc"
PINGFANG = "/System/Library/Fonts/PingFang.ttc"
NEWYORK = "/System/Library/Fonts/NewYork.ttf"
DIDOT = "/System/Library/Fonts/Supplemental/Didot.ttc"

# Songti.ttc subfont indices. Index 0 (SC Black) is incomplete — missing some
# Traditional glyphs (細, 節, 裡, 執). Use these instead.
SONGTI_SC_BOLD = 1     # body bold
SONGTI_TC_BOLD = 2     # display bold (full TC + SC coverage)
SONGTI_SC_LIGHT = 3    # elegant light weight for sublines
SONGTI_SC_REGULAR = 6
SONGTI_TC_REGULAR = 7

CREAM = (245, 235, 215)
INK = (38, 28, 18)
SEAL_RED = (170, 45, 45)


# ---------- photo treatment ----------

def crop_cover(img: Image.Image, size=(1080, 1920)) -> Image.Image:
    return ImageOps.fit(img, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def warm_tone(img: Image.Image, strength=0.06) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * (1 + strength), 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * (1 - strength * 0.6), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_grain(img: Image.Image, amount=4, seed=0) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.int16)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount, arr.shape[:2])[..., None]
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def vignette(img: Image.Image, strength=0.32) -> Image.Image:
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1 - d * strength, 0, 1)
    arr = np.array(img.convert("RGB")).astype(np.float32) * mask[..., None]
    return Image.fromarray(arr.astype(np.uint8))


def darken(img: Image.Image, alpha=0.20) -> Image.Image:
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, int(255 * alpha)))
    return Image.alpha_composite(rgba, overlay)


def gradient_bottom(img: Image.Image, height_frac=0.45, alpha=0.55) -> Image.Image:
    rgba = img.convert("RGBA")
    w, h = rgba.size
    grad = np.zeros((h, w), dtype=np.uint8)
    start = int(h * (1 - height_frac))
    for y in range(start, h):
        t = (y - start) / max(1, h - start)
        grad[y, :] = int(255 * alpha * t ** 1.4)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.putalpha(Image.fromarray(grad))
    return Image.alpha_composite(rgba, overlay)


def base_photo(src_path: Path, *, gradient=False, dark=0.0, vig=0.30,
               warm=True, grain=4) -> Image.Image:
    img = crop_cover(Image.open(src_path))
    if warm:
        img = warm_tone(img)
    img = vignette(img, vig)
    if dark > 0:
        img = darken(img, dark)
    if grain:
        img = add_grain(img, grain)
    if gradient:
        img = gradient_bottom(img)
    return img.convert("RGBA")


# ---------- paper / blending ----------

_texture_cache: dict[str, Image.Image] = {}


def _load_texture(name: str) -> Image.Image:
    if name in _texture_cache:
        return _texture_cache[name]
    path = ASSET_DIR / name
    img = Image.open(path).convert("RGBA")
    _texture_cache[name] = img
    return img


def _tile(texture: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Tile a small PNG to cover the given size."""
    w, h = size
    tw, th = texture.size
    out = Image.new("RGBA", (w, h))
    for y in range(0, h, th):
        for x in range(0, w, tw):
            out.paste(texture, (x, y))
    return out


def multiply_blend(base: Image.Image, overlay: Image.Image,
                   strength: float = 1.0) -> Image.Image:
    """Multiply blend overlay onto base. Strength fades the overlay toward white."""
    a = np.array(base.convert("RGBA")).astype(np.float32) / 255.0
    b = np.array(overlay.convert("RGBA")).astype(np.float32) / 255.0
    if strength < 1.0:
        b[..., :3] = b[..., :3] * strength + (1.0 - strength)
    out = a.copy()
    out[..., :3] = a[..., :3] * b[..., :3]
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def paper_card(size: tuple[int, int], *, color=CREAM, texture="paper-fibers.png",
               texture_strength=0.85, deckle=8, seed=11) -> Image.Image:
    """Cream base, multiplied with a real paper texture, with deckle edges."""
    w, h = size
    base = Image.new("RGBA", (w, h), color + (255,))
    tex = _tile(_load_texture(texture), (w, h))
    out = multiply_blend(base, tex, strength=texture_strength)
    # Edge tint: warmer/darker near edges for aged look
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / max(1, w / 2)) ** 2 +
                ((y - h / 2) / max(1, h / 2)) ** 2)
    edge = np.clip(d - 0.55, 0, 1) * 30
    a = np.array(out).astype(np.int16)
    a[..., 0] -= edge.astype(np.int16)
    a[..., 1] -= (edge * 1.2).astype(np.int16)
    a[..., 2] -= (edge * 1.4).astype(np.int16)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if deckle:
        out.putalpha(_deckle_mask(size, irregularity=deckle, seed=seed))
    return out


def paper_circle(diameter: int, *, color=CREAM, texture="paper-fibers.png",
                 texture_strength=0.85) -> Image.Image:
    card = paper_card((diameter, diameter), color=color, texture=texture,
                      texture_strength=texture_strength, deckle=0)
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((4, 4, diameter - 4, diameter - 4), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    card.putalpha(mask)
    return card


def _deckle_mask(size, irregularity=8, seed=3) -> Image.Image:
    w, h = size
    rng = np.random.default_rng(seed)
    mask = np.full((h, w), 255, dtype=np.uint8)
    for y in range(h):
        mask[y, :rng.integers(0, irregularity)] = 0
        mask[y, w - rng.integers(0, irregularity):] = 0
    for x in range(w):
        mask[:rng.integers(0, irregularity), x] = 0
        mask[h - rng.integers(0, irregularity):, x] = 0
    return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(1.5))


def drop_shadow(layer: Image.Image, offset=(0, 6), blur=14, alpha=120) -> Image.Image:
    rgba = layer.convert("RGBA")
    shadow = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    a = rgba.split()[-1]
    black = Image.new("RGBA", rgba.size, (0, 0, 0, alpha))
    shadow.paste(black, mask=a)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    out.alpha_composite(shadow, dest=offset)
    return out


def paste_with_shadow(base: Image.Image, layer: Image.Image, pos, **shadow_kw) -> None:
    pad = 60
    container = Image.new("RGBA",
                          (base.size[0] + pad * 2, base.size[1] + pad * 2),
                          (0, 0, 0, 0))
    sh_pos = (pos[0] + pad, pos[1] + pad)
    sh = drop_shadow(layer, **shadow_kw)
    container.alpha_composite(sh, dest=sh_pos)
    cropped = container.crop((pad, pad, pad + base.size[0], pad + base.size[1]))
    base.alpha_composite(cropped)
    base.alpha_composite(layer, dest=pos)


# ---------- text + decorations ----------

def soft_text(img: Image.Image, xy, text, font, fill, *,
              shadow=(0, 0, 0, 160), offset=(2, 3), blur=5,
              multiline=False, spacing=10, anchor=None) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(layer)
    sx, sy = xy[0] + offset[0], xy[1] + offset[1]
    if multiline:
        sd.multiline_text((sx, sy), text, fill=shadow, font=font, spacing=spacing)
    else:
        sd.text((sx, sy), text, fill=shadow, font=font, anchor=anchor)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(layer)
    d = ImageDraw.Draw(img)
    if multiline:
        d.multiline_text(xy, text, fill=fill, font=font, spacing=spacing)
    else:
        d.text(xy, text, fill=fill, font=font, anchor=anchor)


def kerned_text(img: Image.Image, xy, text, font, fill, kern=3) -> None:
    x, y = xy
    d = ImageDraw.Draw(img)
    for ch in text:
        d.text((x, y), ch, fill=fill, font=font)
        x += int(d.textlength(ch, font=font)) + kern


def red_seal(text: str, size=92) -> Image.Image:
    s = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=4,
                        fill=SEAL_RED + (235,))
    f = ImageFont.truetype(SONGTI, int(size * 0.42), index=2)
    chars = list(text)
    if len(chars) == 4:
        positions = [(size * 0.18, size * 0.10), (size * 0.55, size * 0.10),
                     (size * 0.18, size * 0.50), (size * 0.55, size * 0.50)]
        for ch, p in zip(chars, positions):
            d.text(p, ch, fill=(250, 240, 220, 245), font=f)
    elif len(chars) == 2:
        d.text((size * 0.10, size * 0.08), chars[0], fill=(250, 240, 220, 245), font=f)
        d.text((size * 0.10, size * 0.50), chars[1], fill=(250, 240, 220, 245), font=f)
    else:
        d.text((size / 2, size / 2), text, fill=(250, 240, 220, 245),
               font=f, anchor="mm")
    return s


def ink_dot(size=10, color=CREAM, alpha=220) -> Image.Image:
    s = Image.new("RGBA", (size * 3, size * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    d.ellipse((size, size, size * 2, size * 2), fill=color + (alpha,))
    return s.filter(ImageFilter.GaussianBlur(0.6))


def thin_line(length=120, thickness=2, color=CREAM, vertical=False, alpha=200):
    if vertical:
        s = Image.new("RGBA", (thickness + 2, length), (0, 0, 0, 0))
        ImageDraw.Draw(s).rectangle((1, 0, thickness, length), fill=color + (alpha,))
    else:
        s = Image.new("RGBA", (length, thickness + 2), (0, 0, 0, 0))
        ImageDraw.Draw(s).rectangle((0, 1, length, thickness), fill=color + (alpha,))
    return s


# ---------- templates ----------

@dataclass
class CaptionInputs:
    photo: Path
    out: Path
    headline: str = ""
    subline: str = ""
    seal: str = ""
    accent_color: tuple = CREAM


def _template_banner(i: CaptionInputs) -> None:
    """Vertical Mandarin paper banner on right side, with optional red seal."""
    img = base_photo(i.photo, vig=0.32, dark=0.20, grain=4)
    bw, bh = 240, max(900, 220 + 145 * len(i.headline) + (140 if i.seal else 0))
    bh = min(bh, 1280)
    bx, by = 1080 - bw - 70, max(220, (1920 - bh) // 2)

    paper = paper_card((bw, bh), texture="paper-fibers.png",
                       texture_strength=0.85, deckle=6, seed=11)
    paste_with_shadow(img, paper, (bx, by), offset=(4, 8), blur=18, alpha=140)

    f_v = ImageFont.truetype(SONGTI, 130, index=SONGTI_TC_BOLD)
    cx = bx + bw // 2
    cy = by + 60
    d = ImageDraw.Draw(img)
    for ch in i.headline:
        tw = d.textlength(ch, font=f_v)
        d.text((cx - tw / 2, cy), ch, fill=INK + (250,), font=f_v)
        cy += 145
    if i.seal:
        seal = red_seal(i.seal, size=92)
        img.alpha_composite(seal, dest=(cx - 46, cy + 18))
    if i.subline:
        f_s = ImageFont.truetype(SONGTI, 36, index=SONGTI_SC_LIGHT)
        soft_text(img, (96, 1920 - 240), i.subline, f_s,
                  (250, 240, 220, 220), multiline=True, spacing=14, blur=4)
    img.convert("RGB").save(i.out, "JPEG", quality=92)


def _template_circle(i: CaptionInputs) -> None:
    """Centered round cream paper card with Mandarin or English text."""
    img = base_photo(i.photo, vig=0.34, dark=0.30, grain=4)
    r = 380
    paper = paper_circle(r * 2, texture="paper-fibers.png", texture_strength=0.85)
    cx, cy = 540, 820
    paste_with_shadow(img, paper, (cx - r, cy - r), offset=(4, 10), blur=22, alpha=130)

    lines = [l for l in i.headline.split("\n") if l]
    is_chinese = any("一" <= ch <= "鿿" for ch in i.headline)
    f_h = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             78 if is_chinese else 70,
                             index=SONGTI_TC_BOLD if is_chinese else 0)
    f_s = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             32 if is_chinese else 26,
                             index=SONGTI_SC_LIGHT if is_chinese else 0)
    d = ImageDraw.Draw(img)
    n = len(lines)
    line_h = 100 if is_chinese else 88
    start_y = cy - (n - 1) * line_h // 2
    for k, line in enumerate(lines):
        d.text((cx, start_y + k * line_h), line, fill=INK + (245,),
               font=f_h, anchor="mm")
    if i.subline:
        for j, dx in enumerate((-30, 0, 30)):
            dot = ink_dot(size=6, color=(80, 60, 40), alpha=230)
            img.alpha_composite(dot, dest=(cx + dx - 9,
                                            start_y + (n - 1) * line_h + 80))
        d.text((cx, start_y + (n - 1) * line_h + 140), i.subline,
               fill=(80, 60, 40, 230), font=f_s, anchor="mm")
    img.convert("RGB").save(i.out, "JPEG", quality=92)


def _template_left(i: CaptionInputs) -> None:
    """Mandarin or English headline upper-left on photo, no paper."""
    img = base_photo(i.photo, vig=0.30, dark=0.18, grain=4)
    is_chinese = any("一" <= ch <= "鿿" for ch in i.headline)
    f_h = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             108 if is_chinese else 110,
                             index=SONGTI_TC_BOLD if is_chinese else 0)
    f_s = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             32 if is_chinese else 28,
                             index=SONGTI_SC_LIGHT if is_chinese else 0)
    soft_text(img, (96, 540), i.headline, f_h, (252, 245, 230, 245),
              multiline=True, spacing=22 if is_chinese else 14, blur=6)
    if i.subline:
        soft_text(img, (104, 1090), i.subline, f_s, (250, 240, 220, 215),
                  multiline=True, spacing=10, blur=4)
        line = thin_line(length=80, thickness=2, color=(250, 240, 220))
        img.alpha_composite(line, dest=(102, 1230))
        dot = ink_dot(size=10, color=(250, 240, 220))
        img.alpha_composite(dot, dest=(190, 1212))
    img.convert("RGB").save(i.out, "JPEG", quality=92)


def _template_quote(i: CaptionInputs) -> None:
    """Bottom-third gradient with a quote — minimal, photo-forward."""
    img = base_photo(i.photo, gradient=True, vig=0.35)
    is_chinese = any("一" <= ch <= "鿿" for ch in i.headline)
    f_h = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             92 if is_chinese else 110,
                             index=SONGTI_TC_BOLD if is_chinese else 0)
    f_s = ImageFont.truetype(SONGTI if is_chinese else NEWYORK,
                             32 if is_chinese else 28,
                             index=SONGTI_SC_LIGHT if is_chinese else 0)
    lines = i.headline.split("\n")
    line_h = 110 if is_chinese else 120
    y = 1920 - 280 - line_h * (len(lines) - 1)
    for line in lines:
        soft_text(img, (90, y), line, f_h, (250, 245, 235, 245), blur=6, offset=(2, 4))
        y += line_h
    if i.subline:
        kerned_text(img, (96, 1820), i.subline, f_s, (240, 235, 225, 220), kern=2)
    img.convert("RGB").save(i.out, "JPEG", quality=92)


TEMPLATES: dict[str, Callable[[CaptionInputs], None]] = {
    "banner": _template_banner,
    "circle": _template_circle,
    "left": _template_left,
    "quote": _template_quote,
}


def render(template: str, photo: Path, out: Path, *,
           headline: str = "", subline: str = "", seal: str = "") -> None:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}. "
                         f"Choices: {sorted(TEMPLATES)}")
    TEMPLATES[template](CaptionInputs(
        photo=photo, out=out, headline=headline,
        subline=subline, seal=seal,
    ))
