import numpy as np
from typing import List, Tuple, Optional


def rotation_angle_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Rotation error in degrees using trace formula:
      angle = arccos( (trace(R_gt * R_est^T) - 1)/2 )
    """
    M = R_gt @ R_est.T
    cos = (np.trace(M) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def translation_direction_error_deg(
    t_gt: np.ndarray, t_est: np.ndarray, allow_flip: bool = False
) -> float:
    """
    Translation direction error in degrees.
    Since translation scale may be ambiguous, compare only direction:
      angle = arccos( <t_gt/||t_gt||, t_est/||t_est||> )

    If allow_flip=True, also consider (t_est -> -t_est) and take smaller angle.
    """
    ng = float(np.linalg.norm(t_gt))
    ne = float(np.linalg.norm(t_est))
    if ng == 0.0 or ne == 0.0:
        return 180.0

    u = t_gt / ng
    v = t_est / ne

    c1 = float(np.clip(np.dot(u, v), -1.0, 1.0))
    a1 = float(np.degrees(np.arccos(c1)))

    if not allow_flip:
        return a1

    c2 = float(np.clip(np.dot(u, -v), -1.0, 1.0))
    a2 = float(np.degrees(np.arccos(c2)))
    return min(a1, a2)


def compute_recall_curve(errors: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given errors (length N):
      sort errors ascending -> e[0..N-1]
      recall[k] = (k+1)/N
    """
    e = np.asarray(errors, dtype=np.float64)
    e_sort = np.sort(e)
    n = len(e_sort)
    if n == 0:
        return e_sort, np.asarray([], dtype=np.float64)
    recall = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    return e_sort, recall


def compute_auc(
    errors: List[float], thresholds: List[float], min_error: Optional[float] = 1e-3
) -> List[float]:
    """
    AUC@t = (1/t) * integral_0^t recall(x) dx, expressed in percentage.

    We build a stepwise recall function from sorted errors.
    """
    e, r = compute_recall_curve(errors)
    if len(e) == 0:
        return [0.0 for _ in thresholds]

    if min_error is not None:
        # Insert a small floor to avoid degenerate behavior at exactly 0
        idx0 = int(np.searchsorted(e, min_error, side="right"))
        base = idx0 / float(len(e))
        # Build step function arrays
        r2 = np.r_[base, base, r[idx0:]]
        e2 = np.r_[0.0, float(min_error), e[idx0:]]
    else:
        r2 = np.r_[0.0, r]
        e2 = np.r_[0.0, e]

    aucs = []
    for t in thresholds:
        t = float(t)
        if t <= 0:
            aucs.append(0.0)
            continue

        last = int(np.searchsorted(e2, t, side="right"))
        # Ensure at least one element before indexing last-1
        last = max(last, 1)

        # Clamp curve to exactly x=t
        rr = np.r_[r2[:last], r2[last - 1]]
        ee = np.r_[e2[:last], t]

        auc = float(np.trapz(rr, x=ee) / t) * 100.0
        aucs.append(auc)

    return aucs
