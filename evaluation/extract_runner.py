from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import (
    ETH3D_ROOT,
    SCAN_ROOT,
    OUT_ROOT,
    CALIB_DIRNAME,
    SCAN_EVAL_DIRNAME,
    ScanConfig,
    PlaneConfig,
    RenderConfig,
    CompareConfig,
)
from progress import tqdm
from utils import save_json, norm_image_name
from parser import parse_scan_alignment_mlp
from scan_merge import merge_scans_to_global
from plane_ransac import extract_dominant_planes
from eval_io import load_cameras_txt, load_images_txt
from render_labels import render_plane_label_map, save_render_outputs


def _discover_scenes() -> List[str]:
    scenes: List[str] = []
    if not SCAN_ROOT.exists():
        return scenes

    for d in sorted(SCAN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        scan_eval_dir = d / SCAN_EVAL_DIRNAME
        if scan_eval_dir.exists() and scan_eval_dir.is_dir():
            scenes.append(d.name)
    return scenes


def _resolve_scene_paths(scene: str) -> Dict[str, Path]:
    """
    - eth3d calib: root/datasets/eth3d/<scene>/dslr_calibration_undistorted/{cameras,images}.txt
    - scan eval:   root/datasets/scan/<scene>/dslr_scan_eval/{scan_alignment.mlp,scan1.ply,scan2.ply}
    """
    calib_dir = ETH3D_ROOT / scene / CALIB_DIRNAME
    scan_eval_dir = SCAN_ROOT / scene / SCAN_EVAL_DIRNAME

    return {
        "scene": Path(scene),
        "calib_dir": calib_dir,
        "cameras_txt": calib_dir / "cameras.txt",
        "images_txt": calib_dir / "images.txt",
        "scan_eval_dir": scan_eval_dir,
        "mlp": scan_eval_dir / "scan_alignment.mlp",
        "out_dir": OUT_ROOT / scene,
    }


def _build_transforms(scan_eval_dir: Path, mlp_path: Path) -> Dict[str, np.ndarray]:
    if mlp_path.exists():
        return parse_scan_alignment_mlp(mlp_path)
    else:
        raise RuntimeError(f"[DEBUG] No scan_alignment.mlp found in {scan_eval_dir}")


def run_scene(
    scene: str, scan_cfg: ScanConfig, plane_cfg: PlaneConfig, render_cfg: RenderConfig
) -> None:
    P = _resolve_scene_paths(scene)
    out_dir = P["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- validate inputs ---
    if not P["cameras_txt"].exists() or not P["images_txt"].exists():
        save_json(
            out_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "missing cameras.txt or images.txt",
                "expected": [str(P["cameras_txt"]), str(P["images_txt"])],
            },
        )
        return

    if not P["scan_eval_dir"].exists():
        save_json(
            out_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "missing scan eval dir",
                "expected": str(P["scan_eval_dir"]),
            },
        )
        return

    # --- transforms ---
    transforms = _build_transforms(P["scan_eval_dir"], P["mlp"])
    save_json(
        out_dir / "scan_transforms.json", {k: v.tolist() for k, v in transforms.items()}
    )

    # --- merge scans to global ---
    merged_ply = out_dir / "scan_global_merged.ply"
    merged = merge_scans_to_global(
        scan_dir=P["scan_eval_dir"],
        transforms=transforms,
        voxel_size=scan_cfg.voxel_size,
        max_points=scan_cfg.max_points,
        out_ply=merged_ply,
    )

    save_json(
        out_dir / "scan_merge_stats.json",
        {
            "scene": scene,
            "per_scan_counts": merged.per_scan_counts,
            "merged_num_points": int(merged.cloud.xyz.shape[0]),
            "voxel_size": scan_cfg.voxel_size,
            "max_points": scan_cfg.max_points,
        },
    )

    X = merged.cloud.xyz
    if X.shape[0] < 3:
        save_json(
            out_dir / "status.json",
            {"scene": scene, "status": "skipped", "reason": "scan has <3 points"},
        )
        return

    # --- extract planes from scan ---
    planes, labels = extract_dominant_planes(
        X=X,
        k_max=plane_cfg.k_max,
        eps_plane=plane_cfg.eps_plane,
        ransac_iters=plane_cfg.ransac_iters,
        min_inliers_abs=plane_cfg.min_inliers_abs,
        min_inliers_frac=plane_cfg.min_inliers_frac,
        merge_angle_deg=plane_cfg.merge_angle_deg,
        merge_offset=plane_cfg.merge_offset,
        seed=plane_cfg.seed,
    )

    save_json(
        out_dir / "planes_from_scan.json",
        {
            "scene": scene,
            "num_planes": len(planes),
            "eps_plane": plane_cfg.eps_plane,
            "planes": [
                {
                    "plane_id": pl.plane_id,
                    "n": pl.n.tolist(),
                    "d": float(pl.d),
                    "num_inliers": int(pl.num_inliers),
                    "area_proxy": float(pl.area_proxy),
                }
                for pl in planes
            ],
            "label_stats": {
                "num_labeled_points": int(np.sum(labels >= 0)),
                "num_unlabeled_points": int(np.sum(labels < 0)),
            },
        },
    )

    np.savez_compressed(
        out_dir / "scan_labeled_points.npz",
        xyz=X.astype(np.float32),
        label=labels.astype(np.int32),
    )

    # --- load cameras / poses ---
    cams = load_cameras_txt(P["cameras_txt"])
    poses = load_images_txt(P["images_txt"])

    # --- cap render points (optional) ---
    idx_labeled = np.where(labels >= 0)[0]
    if (
        render_cfg.max_points_render > 0
        and idx_labeled.size > render_cfg.max_points_render
    ):
        rng = np.random.default_rng(0)
        idx_labeled = rng.choice(
            idx_labeled.size, size=render_cfg.max_points_render, replace=False
        )
        idx_labeled = np.sort(idx_labeled)

    X_render = X[idx_labeled]
    L_render = labels[idx_labeled]

    # --- render label maps for all images ---
    poses_to_render = []
    for i, pose in enumerate(poses):
        if render_cfg.image_stride > 1 and (i % render_cfg.image_stride != 0):
            continue
        if pose.camera_id not in cams:
            continue
        poses_to_render.append(pose)

    render_out = out_dir / "render"
    render_out.mkdir(parents=True, exist_ok=True)

    for pose in tqdm(
        poses_to_render,
        desc=f"[{scene}] Rendering label maps",
        total=len(poses_to_render),
    ):
        cam = cams[pose.camera_id]
        result = render_plane_label_map(
            xyz_world=X_render,
            labels=L_render,
            cam=cam,
            pose=pose,
            splat_radius=render_cfg.splat_radius,
        )
        save_render_outputs(
            out_dir=render_out,
            image_name=pose.name,
            result=result,
            save_png=render_cfg.save_png,
            save_depth_npy=render_cfg.save_depth_npy,
            save_mask_png=render_cfg.save_mask_png,
            unknown_code=render_cfg.unknown_code,
        )

    save_json(out_dir / "status.json", {"scene": scene, "status": "ok"})


def run_all_scenes() -> None:
    scan_cfg = ScanConfig()
    plane_cfg = PlaneConfig()
    render_cfg = RenderConfig()

    scenes = _discover_scenes()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    save_json(
        OUT_ROOT / "run_info.json",
        {
            "eth3d_root": str(ETH3D_ROOT),
            "scan_root": str(SCAN_ROOT),
            "out_root": str(OUT_ROOT),
            "calib_dirname": CALIB_DIRNAME,
            "scan_eval_dirname": SCAN_EVAL_DIRNAME,
            "num_scenes": len(scenes),
            "scenes": scenes,
            "configs": {
                "scan": scan_cfg.__dict__,
                "plane": plane_cfg.__dict__,
                "render": render_cfg.__dict__,
            },
        },
    )

    for scene in tqdm(scenes, desc="All scenes", total=len(scenes)):
        run_scene(scene, scan_cfg=scan_cfg, plane_cfg=plane_cfg, render_cfg=render_cfg)


if __name__ == "__main__":
    run_all_scenes()
