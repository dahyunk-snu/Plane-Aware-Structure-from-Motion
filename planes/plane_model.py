from dataclasses import dataclass
import numpy as np


@dataclass
class PlaneModel:
    """[PlaneModel] Plane in world coords: n^T X + d = 0 with ||n||=1."""

    n: np.ndarray  # shape (3,), float64
    d: float
