from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sim3:
    s: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (self.s * (X @ self.R.T)) + self.t.reshape(1, 3)


def umeyama_sim3(src: np.ndarray, tgt: np.ndarray) -> Sim3:
    """
    Find Sim3 mapping src -> tgt:
      tgt ≈ s R src + t
    """
    assert src.shape == tgt.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    X = src - mu_s
    Y = tgt - mu_t

    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = (X * X).sum() / n
    s = float(np.trace(np.diag(D) @ S) / (var + 1e-12))
    t = mu_t - s * (R @ mu_s)
    return Sim3(s=s, R=R.astype(np.float64), t=t.astype(np.float64))


def rmse(A: np.ndarray, B: np.ndarray) -> float:
    d = A - B
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
