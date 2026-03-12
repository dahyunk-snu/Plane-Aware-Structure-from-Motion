import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap

from planes.plane_model import PlaneModel
from planes.utils import adjust_3d_points


def _collect_plane_points(
    recon: pycolmap.Reconstruction,
    feature_to_global_plane: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    feature_to_global_plane[image_id][point2D_idx] = global_plane_id(int)

    @return
      plane_to_points[gp] = (N, 3)
      plane_to_pids[gp]   = [pid, ...]
    """
    plane_to_points = collections.defaultdict(list)
    plane_to_pids = collections.defaultdict(list)

    for pid, p3d in recon.points3D.items():
        gp_set = set()
        for elem in p3d.track.elements:
            gp = feature_to_global_plane.get(elem.image_id, {}).get(elem.point2D_idx)
            if gp is not None:
                gp_set.add(gp)

        if len(gp_set) == 1:
            gp = next(iter(gp_set))
            plane_to_points[gp].append(p3d.xyz)
            plane_to_pids[gp].append(pid)

    for gp in list(plane_to_points.keys()):
        plane_to_points[gp] = np.asarray(plane_to_points[gp], dtype=np.float64)

    return plane_to_points, plane_to_pids


def refine_3d_planes(
    recon: pycolmap.Reconstruction,
    feature_to_global_plane: Dict[int, Dict[int, int]],
    thresh: float,
    confidence: float,
    min_global_inliers: int,
    max_iters: int,
) -> Tuple[Dict[int, PlaneModel], Dict[int, List[int]]]:
    plane_models = {}
    refined_plane_to_pids = {}

    plane_to_points, plane_to_pids = _collect_plane_points(
        recon, feature_to_global_plane
    )

    for gp, pts in plane_to_points.items():
        if pts.shape[0] < 3:
            continue

        model, inlier_idx = _fit_plane_ransac_indices(
            pts, thresh=thresh, confidence=confidence, max_iters=max_iters
        )
        if model is None or inlier_idx.size == 0:
            continue

        pid_list = plane_to_pids[gp]
        inlier_pids = [pid_list[i] for i in inlier_idx.tolist()]

        if len(inlier_pids) >= min_global_inliers:
            plane_models[gp] = model
            refined_plane_to_pids[gp] = inlier_pids

    return plane_models, refined_plane_to_pids


def refine_3d_plane_models(
    recon: pycolmap.Reconstruction,
    plane_models: Dict[int, PlaneModel],
    plane_to_pids: Dict[int, List[int]],
    thresh: float,
    confidence: float,
    min_global_inliers: int,
    max_iters: int,
    use_ransak: bool = False,
    snap_strength: float = 0.0,
    dynamic_snap: bool = True,
    snap_delta: float = 0.02,  # allow <= +2% reproj error increase
    snap_beta: float = 0.5,
    snap_gamma: float = 1.2,
    snap_alpha_min: float = 1e-6,
    snap_alpha_max: float = 0.2,
    snap_max_backtracks: int = 6,
) -> Tuple[Dict[int, PlaneModel], Dict[int, List[int]]]:
    """
    After BA:
      1) Re-fit each plane using current point positions (SVD).
      2) Remove outlier pids by MAD threshold on signed distance.
      3) Optionally snap inlier points onto the new plane (denoise).
    """
    new_models = {}
    new_plane_to_pids = {}
    is_new = False

    for gid, pids in plane_to_pids.items():
        pts = []  # List[(3,)]
        kept_pids = []  # List[int]

        for pid in pids:
            if pid not in recon.points3D:
                continue
            xyz = recon.points3D[pid].xyz
            if not isinstance(xyz, np.ndarray) or xyz.shape[0] != 3:
                continue
            pts.append(
                np.asarray(xyz, dtype=np.float64).reshape(
                    3,
                )
            )
            kept_pids.append(pid)

        if len(pts) < 3:
            continue

        pts_arr = np.stack(pts, axis=0)  # (M,3)
        if use_ransak:
            model, inlier_idx = _fit_plane_ransac_indices(
                pts_arr,
                thresh=thresh,
                confidence=confidence,
                max_iters=max_iters,
            )

            if model == None:
                continue
        else:
            ref_n = (
                plane_models.get(gid).n if gid in plane_models else None
            )  # (3,) or None
            model = _fit_plane_svd(pts_arr, ref_normal=ref_n)

            if model == None:
                continue

            dist = np.abs(pts_arr @ model.n + model.d)
            inlier_idx = np.where(dist < thresh)[0]
            if inlier_idx.size == 0:
                continue

        inlier_pids = [kept_pids[i] for i in inlier_idx.tolist()]

        if len(inlier_pids) < min_global_inliers:
            continue

        new_models[gid] = model
        new_plane_to_pids[gid] = inlier_pids

        alpha, accepted = adjust_3d_points(
            recon,
            new_models,
            new_plane_to_pids,
            snap_strength,
            dynamic_snap,
            snap_delta,
            snap_beta,
            snap_gamma,
            snap_alpha_min,
            snap_alpha_max,
            snap_max_backtracks,
        )
        is_new = True

    if is_new:
        return new_models, new_plane_to_pids, alpha, accepted
    else:
        return new_models, new_plane_to_pids, snap_strength, False


def _fit_plane_svd(
    points: np.ndarray, ref_normal: Optional[np.ndarray] = None
) -> PlaneModel:
    """
    points: (N, 3)
    ref_normal: (3,)  (optional)
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("Need at least 3 points of shape (N,3).")

    centroid = pts.mean(axis=0)  # (3,)
    centered = pts - centroid  # (N,3)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)  # vt: (3,3)
    n = vt[-1]  # (3,)
    n = n / np.linalg.norm(n)  # (3,)
    d = -float(n @ centroid)  # scalar

    if ref_normal is not None:
        ref = np.asarray(ref_normal, dtype=np.float64).reshape(
            3,
        )  # (3,)
        if float(n @ ref) < 0.0:
            n = -n
            d = -d

    return PlaneModel(n=n, d=d)


def _fit_plane_ransac_indices(
    points: np.ndarray,
    thresh: float,
    confidence: float,
    max_iters: int,
):
    N = points.shape[0]
    if N < 3:
        return None, np.empty((0,), dtype=int)

    best_inliers = np.empty((0,), dtype=int)
    best_plane = None
    required_iters = max_iters
    s = 3

    it = 0
    while it < required_iters and it < max_iters:
        it += 1
        # 1. Randomly choose 3 points to estimate the plane
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]

        # 2. Calculate a normal vector of the plane
        n = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n
        d = -float(np.dot(n, p1))

        # 3. Search Inliers
        dist = np.abs(points @ n + d)
        inliers = np.where(dist < thresh)[0]

        # 4. Determine the sample time
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_plane = (n, d)

            w = best_inliers.size / float(N)
            if 0.0 < w < 1.0:
                den = np.log(1.0 - (w**s))
                if den != 0.0:
                    required_iters = min(
                        required_iters, int(np.ceil(np.log(1.0 - confidence) / den))
                    )
            elif w == 1.0:
                required_iters = 1

    if best_plane is None:
        return None, np.empty((0,), dtype=int)

    # 5. Refine with least_squares
    final_inlier_pts = points[best_inliers]
    final_plane = _fit_plane_svd(final_inlier_pts)

    final_dist = np.abs(points @ final_plane.n + final_plane.d)
    final_inliers = np.where(final_dist < thresh)[0]

    return final_plane, final_inliers
