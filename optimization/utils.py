from typing import Dict, List
import numpy as np

import pycolmap


def _track_len(recon: pycolmap.Reconstruction, pid: int) -> int:
    try:
        return len(recon.points3D[pid].track.elements)
    except Exception:
        return 0


def project_xyz_to_plane(
    xyz: np.ndarray, n: np.ndarray, d: float, alpha: float
) -> np.ndarray:
    # X <- X - alpha * (n^T X + d) n
    r = float(n @ xyz + d)
    return xyz - float(alpha) * r * n


def _select_anchor_pids_for_plane(
    recon: pycolmap.Reconstruction,
    pids: List[int],
    max_anchors: int,
    min_track_len: int,
) -> List[int]:
    cand = [
        pid
        for pid in pids
        if pid in recon.points3D and _track_len(recon, pid) >= min_track_len
    ]
    if not cand:
        return []

    cand.sort(key=lambda pid: _track_len(recon, pid), reverse=True)
    cand = cand[: max(max_anchors * 5, max_anchors)]

    xyzs = {
        pid: np.asarray(recon.points3D[pid].xyz, np.float64).reshape(3) for pid in cand
    }
    selected = [cand[0]]
    if max_anchors == 1:
        return selected

    while len(selected) < max_anchors and len(selected) < len(cand):
        best_pid = None
        best_score = -1.0
        for pid in cand:
            if pid in selected:
                continue
            x = xyzs[pid]
            mind = min(float(np.linalg.norm(x - xyzs[sid])) for sid in selected)
            if mind > best_score:
                best_score = mind
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)

    return selected


def build_anchor_set(
    recon: pycolmap.Reconstruction,
    plane_to_pids: Dict[int, List[int]],
    max_anchors_per_plane: int,
    min_track_len: int,
) -> Dict[int, List[int]]:
    plane_to_anchors: Dict[int, List[int]] = {}
    for gid, pids in plane_to_pids.items():
        anchors = _select_anchor_pids_for_plane(
            recon, pids, max_anchors=max_anchors_per_plane, min_track_len=min_track_len
        )
        if anchors:
            plane_to_anchors[gid] = anchors
    return plane_to_anchors
