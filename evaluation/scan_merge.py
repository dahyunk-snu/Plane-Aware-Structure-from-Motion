from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from parser import apply_transform
from eval_io import PointCloud, load_ply, save_ply, voxel_downsample
from progress import tqdm


@dataclass(frozen=True)
class MergedScan:
    cloud: PointCloud
    per_scan_counts: Dict[str, int]


def merge_scans_to_global(
    scan_dir: Path,
    transforms: Dict[str, np.ndarray],  # basename -> 4x4
    voxel_size: float = 0.0,
    max_points: int = 0,
    out_ply: Optional[Path] = None,
) -> MergedScan:
    """
    Loads all PLYs referenced by transforms, applies transform to make them global, merges.
    """
    xyz_all: List[np.ndarray] = []
    rgb_all: List[np.ndarray] = []
    counts: Dict[str, int] = {}

    have_any_rgb = False

    for basename, M in tqdm(
        transforms.items(),
        total=len(transforms),
        desc="Merging scans",
    ):
        ply_path = scan_dir / basename
        cloud = load_ply(ply_path)
        Xg = apply_transform(cloud.xyz, M)
        xyz_all.append(Xg)
        counts[basename] = int(Xg.shape[0])

        if cloud.rgb is not None:
            have_any_rgb = True
            rgb_all.append(cloud.rgb)
        else:
            rgb_all.append(None)  # type: ignore

    xyz = (
        np.concatenate(xyz_all, axis=0)
        if len(xyz_all) > 0
        else np.zeros((0, 3), np.float64)
    )

    rgb = None
    if have_any_rgb:
        # If some scans have no rgb, fill zeros for them
        rgb_chunks = []
        for i, Xg in enumerate(xyz_all):
            c = rgb_all[i]
            if c is None:
                rgb_chunks.append(np.zeros((Xg.shape[0], 3), np.uint8))
            else:
                rgb_chunks.append(c)
        rgb = np.concatenate(rgb_chunks, axis=0)

    merged = PointCloud(xyz=xyz, rgb=rgb)

    # optional downsample
    if voxel_size > 0:
        merged = voxel_downsample(merged, voxel_size=voxel_size)

    # optional cap
    if max_points > 0 and merged.xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(merged.xyz.shape[0], size=max_points, replace=False)
        merged = PointCloud(
            xyz=merged.xyz[idx],
            rgb=(merged.rgb[idx] if merged.rgb is not None else None),
        )

    if out_ply is not None:
        save_ply(out_ply, merged)

    return MergedScan(cloud=merged, per_scan_counts=counts)
