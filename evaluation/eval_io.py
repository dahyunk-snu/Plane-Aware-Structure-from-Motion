from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray  # (N,3) float64
    rgb: Optional[np.ndarray]  # (N,3) uint8 or None


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # depends on model


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


def load_ply(p: Path) -> PointCloud:
    try:
        import open3d as o3d  # type: ignore

        pc = o3d.io.read_point_cloud(str(p))
        xyz = np.asarray(pc.points, dtype=np.float64)
        rgb = None
        if pc.has_colors():
            c = np.asarray(pc.colors, dtype=np.float64)
            rgb = np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)
        return PointCloud(xyz=xyz, rgb=rgb)
    except Exception as e:
        raise RuntimeError(f"[DEBUG] Failed to load PLY via Open3D: {p}\n{e}")


def save_ply(p: Path, cloud: PointCloud) -> None:
    try:
        import open3d as o3d  # type: ignore

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud.xyz.astype(np.float64))
        if cloud.rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(cloud.rgb.astype(np.float64) / 255.0)
        p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(p), pc)
    except Exception as e:
        raise RuntimeError(f"[DEBUG] Failed to save PLY via Open3D: {p}\n{e}")


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if voxel_size <= 0 or cloud.xyz.shape[0] == 0:
        return cloud
    try:
        import open3d as o3d  # type: ignore

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud.xyz.astype(np.float64))
        if cloud.rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(
                (cloud.rgb.astype(np.float64) / 255.0)
            )
        pc2 = pc.voxel_down_sample(voxel_size=float(voxel_size))
        xyz = np.asarray(pc2.points, dtype=np.float64)
        rgb = None
        if pc2.has_colors():
            c = np.asarray(pc2.colors, dtype=np.float64)
            rgb = np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)
        return PointCloud(xyz=xyz, rgb=rgb)
    except Exception:
        # Fallback: simple grid quantization (no color averaging)
        X = cloud.xyz
        key = np.floor(X / voxel_size).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        xyz = X[idx]
        rgb = cloud.rgb[idx] if cloud.rgb is not None else None
        return PointCloud(xyz=xyz, rgb=rgb)


def load_cameras_txt(p: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
        cams[cam_id] = Camera(
            camera_id=cam_id, model=model, width=width, height=height, params=params
        )
    if len(cams) == 0:
        raise RuntimeError(f"[DEBUG] No cameras parsed from {p}")
    return cams


def load_images_txt(p: Path) -> List[ImagePose]:
    poses: List[ImagePose] = []
    lines = p.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        poses.append(
            ImagePose(
                image_id=image_id,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                tx=tx,
                ty=ty,
                tz=tz,
                camera_id=camera_id,
                name=name,
            )
        )
        # skip points2D line
        if i < len(lines):
            i += 1
    if len(poses) == 0:
        raise RuntimeError(f"[DEBUG] No images parsed from {p}")
    return poses


def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # COLMAP quaternion: (qw, qx, qy, qz), world->cam rotation
    w, x, y, z = qw, qx, qy, qz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def get_intrinsics_K(cam: Camera) -> np.ndarray:
    if cam.model.upper() == "PINHOLE":
        # params: fx fy cx cy
        fx, fy, cx, cy = cam.params.tolist()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K
    raise NotImplementedError(f"[DEBUG] Camera model not supported: {cam.model}")


def world_to_cam(pose: ImagePose) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns R,t such that X_cam = R X_world + t
    """
    R = quat_to_R(pose.qw, pose.qx, pose.qy, pose.qz)
    t = np.array([pose.tx, pose.ty, pose.tz], dtype=np.float64)
    return R, t


def camera_center_from_pose(pose: ImagePose) -> np.ndarray:
    R, t = world_to_cam(pose)
    return -R.T @ t
