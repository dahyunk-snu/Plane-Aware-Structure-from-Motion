from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ETH3D_ROOT = PROJECT_ROOT / "datasets" / "eth3d"
SCAN_ROOT = PROJECT_ROOT / "datasets" / "scan"
PRED_ROOTS = PROJECT_ROOT / "outputs" / "PASfM"
OUT_ROOT = PROJECT_ROOT / "datasets" / "eval_scan_plane"

CALIB_DIRNAME = "dslr_calibration_undistorted"
SCAN_EVAL_DIRNAME = "dslr_scan_eval"
MODEL_FILES_BIN = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class ScanConfig:
    voxel_size: float = 0.01  # meters; 0 disables downsample
    max_points: int = 0  # 0 => no cap after merge


@dataclass(frozen=True)
class PlaneConfig:
    k_max: int = 5
    eps_plane: float = 0.02
    ransac_iters: int = 4000
    min_inliers_abs: int = 50_000
    min_inliers_frac: float = 0.02
    merge_angle_deg: float = 5.0
    merge_offset: float = 0.04
    seed: int = 0


@dataclass(frozen=True)
class RenderConfig:
    splat_radius: int = 1
    image_stride: int = 1
    max_points_render: int = 0
    save_png: bool = True
    save_depth_npy: bool = True
    save_mask_png: bool = True
    unknown_code: int = 0


@dataclass(frozen=True)
class CompareConfig:
    majority_ratio: float = 0.6  # observation majority for plane membership
    min_valid_obs: int = 2  # minimum labeled observations per 3D point
    tau_list_m: tuple[float, float, float] = (0.01, 0.02, 0.05)  # 1/2/5cm
    max_images_2d_maps: int = 6  # per scene
    tile_size_px: int = 32  # for 2D heatmap binning
