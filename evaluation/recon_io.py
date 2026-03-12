from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from utils import norm_image_name


@dataclass(frozen=True)
class ReconData:
    recon_dir: Path
    # image_id -> normalized image name
    image_id_to_name: Dict[int, str]
    # name -> camera center
    centers_by_name: Dict[str, np.ndarray]
    # pid -> xyz
    xyz_by_pid: Dict[int, np.ndarray]
    # pid -> list[(image_id, point2D_idx)]
    track_by_pid: Dict[int, List[Tuple[int, int]]]
    # (image_id, point2D_idx) -> (u,v)
    uv_lookup: Dict[Tuple[int, int], Tuple[float, float]]


def _camera_center_from_pycolmap_image(img) -> np.ndarray:
    R = img.cam_from_world().rotation.matrix()
    t = np.asarray(img.cam_from_world().translation, dtype=np.float64).reshape(3)
    return (-R.T @ t).astype(np.float64)


def load_pycolmap_recon(recon_dir: Path) -> ReconData:
    try:
        import pycolmap  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pycolmap is required to load recon: {e}")

    recon = pycolmap.Reconstruction(str(recon_dir))

    image_id_to_name: Dict[int, str] = {}
    centers_by_name: Dict[str, np.ndarray] = {}

    for img_id, img in recon.images.items():
        if not img.has_pose:
            continue
        nm = norm_image_name(img.name)
        image_id_to_name[int(img_id)] = nm
        centers_by_name[nm] = _camera_center_from_pycolmap_image(img)

    xyz_by_pid: Dict[int, np.ndarray] = {}
    track_by_pid: Dict[int, List[Tuple[int, int]]] = {}

    for pid, p3d in recon.points3D.items():
        pid_int = int(pid)
        xyz_by_pid[pid_int] = np.asarray(p3d.xyz, dtype=np.float64).reshape(3)
        tr = []
        for el in p3d.track.elements:
            tr.append((int(el.image_id), int(el.point2D_idx)))
        track_by_pid[pid_int] = tr

    # uv lookup from images.points2D
    uv_lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for img_id, img in recon.images.items():
        if not img.has_pose:
            continue
        pts2d = img.points2D
        # pts2d is list-like
        for j in range(len(pts2d)):
            p2d = pts2d[j]
            xy = np.asarray(p2d.xy, dtype=np.float64).reshape(2)
            uv_lookup[(int(img_id), int(j))] = (float(xy[0]), float(xy[1]))

    return ReconData(
        recon_dir=recon_dir,
        image_id_to_name=image_id_to_name,
        centers_by_name=centers_by_name,
        xyz_by_pid=xyz_by_pid,
        track_by_pid=track_by_pid,
        uv_lookup=uv_lookup,
    )
