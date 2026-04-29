"""CLIP image embeddings on Apple Silicon (MPS) with CPU fallback."""
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _ensure_loaded() -> None:
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return
    import torch
    import open_clip

    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _model.eval().to(_device)
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")


def device_name() -> str:
    _ensure_loaded()
    return str(_device)


def embed_images(paths: Iterable[Path], batch_size: int = 16) -> List[np.ndarray]:
    """Return one L2-normalized float32 vector per image, in input order."""
    _ensure_loaded()
    import torch

    paths = list(paths)
    out: List[Optional[np.ndarray]] = [None] * len(paths)

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        tensors = []
        valid_idx = []
        for i, p in enumerate(chunk):
            try:
                with Image.open(p) as img:
                    tensors.append(_preprocess(img.convert("RGB")))
                valid_idx.append(start + i)
            except (OSError, ValueError):
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(_device)
        with torch.no_grad():
            feats = _model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.detach().cpu().numpy().astype(np.float32)
        for k, dest_i in enumerate(valid_idx):
            out[dest_i] = feats_np[k]

    return [v if v is not None else np.zeros(512, dtype=np.float32) for v in out]


def embed_text(prompt: str) -> np.ndarray:
    """Return one L2-normalized text embedding (for text-to-image search)."""
    _ensure_loaded()
    import torch

    tokens = _tokenizer([prompt]).to(_device)
    with torch.no_grad():
        feats = _model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)[0]
