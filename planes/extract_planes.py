import collections
from typing import Dict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import h5py
import pycolmap

from planes.utils import extract_image_name, save_planes_json
from planes.merge_planes import merge_planes
from planes.refine_planes import refine_3d_planes
from planes.config import PlaneExtractionConfig


def _observed_point3d_ids(img: pycolmap.Image) -> set[int]:
    pids = set()
    for p2d in img.points2D:
        if p2d.has_point3D():  # Point2D.has_point3D()
            pids.add(int(p2d.point3D_id))
    return pids


def is_pure_rotation(
    recon: pycolmap.Reconstruction,
    img_id0: int,
    img_id1: int,
    rot_angle_thresh_deg: float,
    min_common_points: int = 30,
    statistic: str = "median",  # "median" or "p25"
) -> bool:
    img0 = recon.images[img_id0]
    img1 = recon.images[img_id1]

    if (not img0.has_pose) or (not img1.has_pose):
        return False

    pids0 = _observed_point3d_ids(img0)
    pids1 = _observed_point3d_ids(img1)
    common = list(pids0.intersection(pids1))

    if len(common) < min_common_points:
        return False

    C0 = img0.projection_center()  # (3,1)
    C1 = img1.projection_center()  # (3,1)

    angles_deg = []
    for pid in common:
        X = recon.points3D[pid].xyz.reshape(3, 1)
        ang_rad = pycolmap.calculate_triangulation_angle(C0, C1, X)  # radians
        if np.isfinite(ang_rad) and ang_rad >= 0:
            angles_deg.append(np.degrees(ang_rad))

    if len(angles_deg) < min_common_points:
        return False

    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if statistic == "p25":
        score = float(np.percentile(angles_deg, 25))
    else:
        score = float(np.median(angles_deg))

    return score < rot_angle_thresh_deg


def extract_local_planes(
    recon: pycolmap.Reconstruction,
    feature_file: h5py.File,
    matches_file: h5py.File,
    ransac_thresh_px: float,
    min_pair_inliers: int,
    max_planes_per_pair: int,
    rot_angle_thresh_deg: float,
    exclude_pure_rotation: bool,
    features_group: str,
):
    """
    @return feature_to_planes[image_id][feat_idx] = local_plane_id List
    """
    feature_to_planes = collections.defaultdict(lambda: collections.defaultdict(list))

    # name_to_id[filtered_image_name] = image_id
    name_to_id: Dict[str, int] = {
        extract_image_name(img.name): img_id for img_id, img in recon.images.items()
    }

    # total match pair count (for tqdm)
    total_pairs = sum(
        1
        for mk1 in matches_file.keys()
        for mk2 in matches_file[mk1]
        if isinstance(matches_file[mk1][mk2], h5py.Group)
        and "matches0" in matches_file[mk1][mk2]
    )

    pbar = tqdm(total=total_pairs, desc="Local Plane Extraction")

    cnt_lp = 0
    # Iterate all match pairs
    for match_key1 in matches_file.keys():
        match_grp1 = matches_file[match_key1]
        if not isinstance(match_grp1, h5py.Group):
            continue

        for match_key2 in match_grp1.keys():
            match_grp2 = match_grp1[match_key2]
            if not isinstance(match_grp2, h5py.Group) or "matches0" not in match_grp2:
                continue
            pbar.update(1)

            # 1. Load valid matches
            try:
                matches0 = match_grp2["matches0"][()]
            except Exception as e:
                print(
                    f"[DEBUG] Error reading matches0 for ({match_key1}, {match_key2}): {e}"
                )
                continue

            valid = matches0 > -1
            if valid.sum() < min_pair_inliers:
                continue

            idx0 = np.where(valid)[0]
            idx1 = matches0[valid]
            matches = np.stack([idx0, idx1], axis=1)

            # 2. Load keypoints
            img_name0 = extract_image_name(match_key1)
            img_name1 = extract_image_name(match_key2)

            # Enforce lexicographic order to avoid duplicates
            if img_name0 > img_name1:
                continue

            if img_name0 not in name_to_id or img_name1 not in name_to_id:
                # print(
                #    f"[DEBUG] name_to_id missing: name0={img_name0 in name_to_id}, name1={img_name1 in name_to_id}"
                # )
                # print(
                #    f"[DEBUG] name_to_id missing: img_name0={img_name0}, img_name1={img_name1}"
                # )
                continue

            try:
                kp0_all = feature_file[features_group][img_name0]["keypoints"][()][
                    matches[:, 0]
                ]
                kp1_all = feature_file[features_group][img_name1]["keypoints"][()][
                    matches[:, 1]
                ]
            except KeyError as e:
                print(
                    f"[DEBUG] KeyError accessing features for ({img_name0}, {img_name1}): {e}"
                )
                continue

            # 3. Pure rotation filtering
            img_id0 = name_to_id[img_name0]
            img_id1 = name_to_id[img_name1]

            if exclude_pure_rotation and is_pure_rotation(
                recon,
                img_id0,
                img_id1,
                rot_angle_thresh_deg,
            ):
                continue

            # 4. Extract multiple planes per pair
            curr_kp0 = kp0_all.copy()
            curr_kp1 = kp1_all.copy()
            curr_idx = np.arange(len(matches))

            for plane_idx in range(max_planes_per_pair):
                if len(curr_kp0) < min_pair_inliers:
                    break

                H, mask = cv2.findHomography(
                    curr_kp0, curr_kp1, cv2.RANSAC, ransac_thresh_px
                )
                if H is None:
                    break

                mask = mask.ravel().astype(bool)
                inliers = mask.sum()
                if inliers < min_pair_inliers:
                    break

                local_plane_id = f"{img_name0}|{img_name1}|{plane_idx}"
                cnt_lp += 1

                # Assign inliers to plane
                inlier_match_indices = curr_idx[mask]
                for m in inlier_match_indices:
                    feat_idx0 = matches[m, 0]
                    feat_idx1 = matches[m, 1]

                    feature_to_planes[img_id0][feat_idx0].append(local_plane_id)
                    feature_to_planes[img_id1][feat_idx1].append(local_plane_id)

                # Remove inliers for next plane hypothesis
                curr_kp0 = curr_kp0[~mask]
                curr_kp1 = curr_kp1[~mask]
                curr_idx = curr_idx[~mask]

    pbar.close()
    print(f"[PASfM] Estimated local planes: # {cnt_lp}")
    return feature_to_planes


def extract_planes(
    recon: pycolmap.Reconstruction,
    feature_path: Path,
    match_path: Path,
    config: PlaneExtractionConfig = PlaneExtractionConfig(),
    save_path: Path = Path("planes.json"),
):

    with h5py.File(feature_path, "r") as feature_file, h5py.File(
        match_path, "r"
    ) as matches_file:

        print("[PASfM] Extracting local planes...")
        feat2local = extract_local_planes(
            recon=recon,
            feature_file=feature_file,
            matches_file=matches_file,
            ransac_thresh_px=config.ransac_thresh_px,
            min_pair_inliers=config.min_pair_inliers,
            max_planes_per_pair=config.max_planes_per_pair,
            rot_angle_thresh_deg=config.rot_angle_thresh_deg,
            exclude_pure_rotation=config.exclude_pure_rotation,
            features_group=config.features_group,
        )

    print("[PASfM] Extracting global planes...")
    feat2global = merge_planes(
        feat2local, config.min_overlap, config.min_global_inliers
    )

    print("[PASfM] Refining 3D Planes...")
    plane_models, plane_to_pids = refine_3d_planes(
        recon,
        feat2global,
        thresh=config.plane_ransac_thresh_m,
        confidence=config.plane_ransac_conf,
        min_global_inliers=config.min_global_inliers,
        max_iters=config.plane_ransac_max_iters,
    )

    print("[PASfM] Saving results...")
    result = {
        "feature_to_planes": feat2global,
        "plane_models": {
            int(pid): {
                "normal": plane_models[pid].n.tolist(),
                "d": float(plane_models[pid].d),
            }
            for pid in plane_models
        },
        "plane_to_pids": {
            int(pid): [int(x) for x in plane_to_pids[pid]] for pid in plane_to_pids
        },
    }
    save_planes_json(result, save_path)

    return plane_models, plane_to_pids
