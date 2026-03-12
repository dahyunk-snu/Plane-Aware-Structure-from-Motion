from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Plane:
    plane_id: int
    n: np.ndarray  # (3,) unit
    d: float  # n·x + d = 0
    inlier_idx: np.ndarray  # indices into the input point array
    num_inliers: int
    area_proxy: float  # projected area proxy (hull or bbox area)


def _fit_plane_svd(X: np.ndarray) -> Tuple[np.ndarray, float]:
    c = X.mean(axis=0)
    Y = X - c
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n.astype(np.float64), d


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    x = float(
        np.clip(
            (a @ b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(x)))


def _plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # build orthonormal basis u,v on plane
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], np.float64)
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0], np.float64)
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def _area_proxy_uv(pts: np.ndarray, n: np.ndarray) -> float:
    # Project to (u,v). Use convex hull area if scipy exists; else bbox area.
    u, v = _plane_basis(n)
    uv = np.stack([pts @ u, pts @ v], axis=1)  # (M,2)

    try:
        from scipy.spatial import ConvexHull  # type: ignore

        hull = ConvexHull(uv)
        # ConvexHull.volume is area in 2D
        return float(hull.volume)
    except Exception:
        mn = uv.min(axis=0)
        mx = uv.max(axis=0)
        return float((mx[0] - mn[0]) * (mx[1] - mn[1]))


def extract_dominant_planes(
    X: np.ndarray,
    k_max: int,
    eps_plane: float,
    ransac_iters: int,
    min_inliers_abs: int,
    min_inliers_frac: float,
    merge_angle_deg: float,
    merge_offset: float,
    seed: int = 0,
    confidence: float = 0.99,
) -> Tuple[List[Plane], np.ndarray]:
    """
    Iterative RANSAC extraction on point cloud X (N,3).
    Returns:
      planes: list of Plane
      labels: (N,) int32, plane_id or -1
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 3:
        return [], np.full((N,), -1, np.int32)

    remaining = np.arange(N, dtype=np.int64)
    raw_planes: List[Plane] = []
    pid = 0

    for _ in range(k_max):
        if remaining.size < 3:
            break

        Xr = X[remaining]
        min_inliers = max(min_inliers_abs, int(min_inliers_frac * Xr.shape[0]))

        best_inl_local = None
        best_cnt = -1

        it = 0
        required_iters = ransac_iters
        while it < required_iters and it < ransac_iters:
            it += 1
            # 1. Randomly choose 3 points to estimate the plane
            idx = rng.choice(Xr.shape[0], size=3, replace=False)
            p1, p2, p3 = Xr[idx]

            # 2. Calculate a normal vector of the plane
            n = np.cross(p2 - p1, p3 - p1)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            d = -float(n @ p1)

            # 3. Search Inliers
            dist = np.abs(Xr @ n + d)
            inl = np.where(dist < eps_plane)[0]
            cnt = int(inl.size)

            # 4. Determine the sample time
            if cnt > best_cnt:
                best_cnt = cnt
                best_inl_local = inl

                w = best_cnt / float(Xr.shape[0])
                if 0.0 < w < 1.0:
                    den = np.log(1.0 - (w**3))
                    if den != 0.0:
                        required_iters = min(
                            required_iters, int(np.ceil(np.log(1.0 - confidence) / den))
                        )
                elif w == 1.0:
                    required_iters = 1

        if best_inl_local is None or best_cnt < min_inliers:
            break

        inlier_global = remaining[best_inl_local]
        n_refit, d_refit = _fit_plane_svd(X[inlier_global])
        area = _area_proxy_uv(X[inlier_global], n_refit)

        raw_planes.append(
            Plane(
                plane_id=pid,
                n=n_refit,
                d=d_refit,
                inlier_idx=inlier_global,
                num_inliers=int(inlier_global.size),
                area_proxy=area,
            )
        )
        pid += 1

        # remove inliers
        mask = np.ones((remaining.size,), dtype=bool)
        mask[best_inl_local] = False
        remaining = remaining[mask]

    # merge similar planes (to avoid duplicates)
    merged: List[Plane] = []
    used = [False] * len(raw_planes)
    new_id = 0

    for i, pi in enumerate(raw_planes):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(raw_planes)):
            if used[j]:
                continue
            pj = raw_planes[j]
            ang = _angle_deg(pi.n, pj.n)
            off = abs(pi.d - pj.d)
            if ang < merge_angle_deg and off < merge_offset:
                used[j] = True
                group.append(j)

        all_idx = np.concatenate([raw_planes[g].inlier_idx for g in group], axis=0)
        all_idx = np.unique(all_idx)

        n_refit, d_refit = _fit_plane_svd(X[all_idx])
        area = _area_proxy_uv(X[all_idx], n_refit)

        merged.append(
            Plane(
                plane_id=new_id,
                n=n_refit,
                d=d_refit,
                inlier_idx=all_idx,
                num_inliers=int(all_idx.size),
                area_proxy=area,
            )
        )
        new_id += 1

    # labels
    labels = np.full((N,), -1, np.int32)
    for pl in merged:
        labels[pl.inlier_idx] = pl.plane_id

    return merged, labels
