from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class PlaneEq:
    n: np.ndarray  # unit (3,)
    d: float


def point_plane_dist(X: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return np.abs(X @ n + d)


def summarize_dist(
    dist: np.ndarray, tau_list: Tuple[float, float, float]
) -> Dict[str, float]:
    if dist.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "var": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            **{f"inlier@{t:.3f}": float("nan") for t in tau_list},
        }

    dist = dist.astype(np.float64)
    out = {
        "n": int(dist.size),
        "mean": float(np.mean(dist)),
        "var": float(np.var(dist)),  # population variance
        "std": float(np.std(dist)),
        "median": float(np.percentile(dist, 50)),
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
    }
    for t in tau_list:
        out[f"inlier@{t:.3f}"] = float(np.mean(dist < t))
    return out
