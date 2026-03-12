from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class LabelMaps:
    # key: image stem -> paths
    plane_png: Dict[str, Path]
    mask_png: Dict[str, Path]


def _read_png(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore

        return np.array(Image.open(str(path)))
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        return imageio.imread(str(path))


def load_label_maps(render_dir: Path) -> LabelMaps:
    plane_png: Dict[str, Path] = {}
    mask_png: Dict[str, Path] = {}
    for p in render_dir.glob("*_plane_id.png"):
        stem = p.name.replace("_plane_id.png", "")
        plane_png[stem] = p
    for p in render_dir.glob("*_valid_mask.png"):
        stem = p.name.replace("_valid_mask.png", "")
        mask_png[stem] = p
    return LabelMaps(plane_png=plane_png, mask_png=mask_png)


def query_plane_id(
    maps: LabelMaps,
    image_stem: str,
    u: float,
    v: float,
) -> Optional[int]:
    """
    Returns plane_id or None if unknown/outside/not covered.
    PNG encoding: 0=unknown, plane_id+1 otherwise.
    """
    if image_stem not in maps.plane_png or image_stem not in maps.mask_png:
        return None
    plane = _read_png(maps.plane_png[image_stem])
    mask = _read_png(maps.mask_png[image_stem])

    H, W = plane.shape[0], plane.shape[1]
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= W or vi < 0 or vi >= H:
        return None

    if mask.ndim == 3:
        m = mask[vi, ui, 0]
    else:
        m = mask[vi, ui]
    if int(m) == 0:
        return None

    val = int(plane[vi, ui])  # uint16 typically
    if val <= 0:
        return None
    return val - 1
