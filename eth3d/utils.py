from pathlib import Path
from typing import Tuple

import numpy as np
import pycolmap


def norm_name(name: str) -> str:
    """Normalize image name to a stable key (basename only)."""
    return Path(name).name


def get_cam_from_world(image: pycolmap.Image):
    """
    Get world->camera transform (Rigid3d) from a pycolmap image.
    Some pycolmap builds expose it as a property, others as a callable.
    """
    cw = getattr(image, "cam_from_world", None)
    if cw is None:
        raise AttributeError("pycolmap.Image has no attribute 'cam_from_world'.")
    return cw() if callable(cw) else cw


def get_c2w_R_C(image: pycolmap.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera-to-world rotation and camera center in world coordinates.
    Returns:
      R_wc: (3,3) rotation matrix mapping camera coordinates -> world coordinates
      C_w : (3,) camera center in world coordinates
    """
    # cam_from_world is world->camera (w2c)
    T_cw = get_cam_from_world(image)
    # Invert to get camera->world (c2w)
    T_wc = T_cw.inverse()

    # Rotation and translation are expressed in world coordinates for c2w
    R_wc = np.asarray(T_wc.rotation.matrix(), dtype=np.float64)
    C_w = np.asarray(T_wc.translation, dtype=np.float64)
    return R_wc, C_w
