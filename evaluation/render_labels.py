from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_io import Camera, ImagePose, get_intrinsics_K, world_to_cam


@dataclass(frozen=True)
class RenderResult:
    label_map: np.ndarray  # (H,W) int32, -1 unknown
    depth_map: np.ndarray  # (H,W) float32, inf if unknown
    valid_mask: np.ndarray  # (H,W) uint8 0/1


def _save_png_label(p: Path, label: np.ndarray, unknown_code: int = 0) -> None:
    # Encode: unknown -> unknown_code, plane_id -> plane_id+1
    enc = np.where(label < 0, unknown_code, label + 1).astype(np.uint16)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(enc).save(str(p))
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(str(p), enc)


def _save_png_mask(p: Path, mask: np.ndarray) -> None:
    img = mask.astype(np.uint8) * 255
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(img).save(str(p))
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(str(p), img)


def render_plane_label_map(
    xyz_world: np.ndarray,  # (N,3)
    labels: np.ndarray,  # (N,) int32, -1 unknown
    cam: Camera,
    pose: ImagePose,
    splat_radius: int = 1,
) -> RenderResult:
    """
    Projects labeled scan points into the image using (K, R, t) where pose is world->cam.
    Uses z-buffer to keep nearest point per pixel.
    """
    H, W = cam.height, cam.width
    label_map = np.full((H, W), -1, np.int32)
    depth_map = np.full((H, W), np.inf, np.float32)

    # Filter valid labels first
    idx = np.where(labels >= 0)[0]
    if idx.size == 0:
        return RenderResult(
            label_map=label_map,
            depth_map=depth_map,
            valid_mask=np.zeros((H, W), np.uint8),
        )

    Xw = xyz_world[idx].astype(np.float64)
    lb = labels[idx].astype(np.int32)

    K = get_intrinsics_K(cam)
    R, t = world_to_cam(pose)

    # world -> cam
    Xc = (Xw @ R.T) + t.reshape(1, 3)  # (N,3), since X_cam = R X_world + t
    z = Xc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return RenderResult(
            label_map=label_map,
            depth_map=depth_map,
            valid_mask=np.zeros((H, W), np.uint8),
        )

    Xc = Xc[valid]
    z = z[valid]
    lb = lb[valid]

    # project
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    INT32_MAX = float(np.iinfo(np.int32).max)
    finite_uv = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    finite_uv &= (np.abs(u) < INT32_MAX) & (np.abs(v) < INT32_MAX)

    if not np.any(finite_uv):
        return RenderResult(label_map, depth_map, np.zeros((H, W), np.uint8))

    u = u[finite_uv]
    v = v[finite_uv]
    z = z[finite_uv].astype(np.float32)
    lb = lb[finite_uv]

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not np.any(in_img):
        return RenderResult(
            label_map=label_map,
            depth_map=depth_map,
            valid_mask=np.zeros((H, W), np.uint8),
        )

    ui = ui[in_img]
    vi = vi[in_img]
    z = z[in_img].astype(np.float32)
    lb = lb[in_img]

    # z-buffer update (optionally with splat)
    if splat_radius <= 0:
        # vectorized-ish: but need per-pixel min check
        for px_u, px_v, zz, pid in zip(ui, vi, z, lb):
            if zz < depth_map[px_v, px_u]:
                depth_map[px_v, px_u] = zz
                label_map[px_v, px_u] = int(pid)
    else:
        r = int(splat_radius)
        for px_u, px_v, zz, pid in zip(ui, vi, z, lb):
            u0 = max(0, px_u - r)
            u1 = min(W - 1, px_u + r)
            v0 = max(0, px_v - r)
            v1 = min(H - 1, px_v + r)
            # update a small window
            win_depth = depth_map[v0 : v1 + 1, u0 : u1 + 1]
            # only overwrite where closer
            closer = zz < win_depth
            if np.any(closer):
                win_depth[closer] = zz
                depth_map[v0 : v1 + 1, u0 : u1 + 1] = win_depth
                win_label = label_map[v0 : v1 + 1, u0 : u1 + 1]
                win_label[closer] = int(pid)
                label_map[v0 : v1 + 1, u0 : u1 + 1] = win_label

    valid_mask = (np.isfinite(depth_map)).astype(np.uint8)
    return RenderResult(label_map=label_map, depth_map=depth_map, valid_mask=valid_mask)


def _save_png_rgb_u8(p: Path, arr_rgb_u8: np.ndarray) -> None:
    """Save RGB uint8 image."""
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr_rgb_u8, mode="RGB").save(str(p))
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(str(p), arr_rgb_u8)


def _labels_to_vis_rgb(label_map: np.ndarray) -> np.ndarray:
    """
    Convert int32 label_map (-1 unknown, 0..K-1 plane_id) to RGB uint8 for visualization.
    - unknown: black
    - each plane_id: deterministic pseudo-random color (stable across runs)
    """
    H, W = label_map.shape[:2]
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    plane_ids = np.unique(label_map[label_map >= 0])
    if plane_ids.size == 0:
        return vis

    # Deterministic color per plane_id (fast LUT)
    colors = {}
    for pid in plane_ids.tolist():
        rng = np.random.default_rng(int(pid) * 9973 + 12345)
        colors[int(pid)] = rng.integers(0, 256, size=3, dtype=np.uint8)

    for pid, col in colors.items():
        m = label_map == pid
        if np.any(m):
            vis[m] = col

    return vis


def save_render_outputs(
    out_dir: Path,
    image_name: str,
    result: RenderResult,
    save_png: bool,
    save_depth_npy: bool,
    save_mask_png: bool,
    unknown_code: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    if save_png:
        _save_png_label(
            out_dir / f"{stem}_plane_id.png",
            result.label_map,
            unknown_code=unknown_code,
        )

        vis_rgb = _labels_to_vis_rgb(result.label_map)
        _save_png_rgb_u8(out_dir / f"{stem}_plane_id_vis.png", vis_rgb)

    if save_depth_npy:
        np.save(out_dir / f"{stem}_depth.npy", result.depth_map.astype(np.float32))

    if save_mask_png:
        _save_png_mask(out_dir / f"{stem}_valid_mask.png", result.valid_mask)
