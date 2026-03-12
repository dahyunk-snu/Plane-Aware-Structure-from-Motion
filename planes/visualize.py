import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Set, Optional

import cv2
import h5py
import numpy as np
import pycolmap


def _as_int(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    return int(str(x))


def _normalize_feature_to_global(result: dict) -> Dict[int, Dict[int, int]]:
    if "feature_to_global_plane" in result:
        src = result["feature_to_global_plane"]
    elif "feature_to_planes" in result:
        src = result["feature_to_planes"]
    else:
        raise KeyError(
            "result must contain 'feature_to_global_plane' or 'feature_to_planes'"
        )

    out: Dict[int, Dict[int, int]] = {}
    for img_id_k, fmap in src.items():
        img_id = _as_int(img_id_k)
        out_img: Dict[int, int] = {}
        for feat_k, gp_k in fmap.items():
            out_img[_as_int(feat_k)] = _as_int(gp_k)
        if out_img:
            out[img_id] = out_img
    return out


def _normalize_plane_to_pids(result: dict) -> Dict[int, list]:
    src = result.get("plane_to_pids", {})
    out: Dict[int, list] = {}
    for gp_k, pids in src.items():
        gp = _as_int(gp_k)
        out[gp] = [_as_int(pid) for pid in pids]
    return out


def _h5_resolve_image_key(h5_group: h5py.Group, recon_image_name: str) -> Optional[str]:
    cand = []
    cand.append(recon_image_name)
    cand.append(Path(recon_image_name).name)
    cand.append(Path(recon_image_name).stem)

    for k in cand:
        if k in h5_group:
            return k
    return None


def build_pre_features(
    feature_to_global: Dict[int, Dict[int, int]],
) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Returns: pre_by_plane[gp] = set((img_id, feat_idx), ...)
    """
    pre_by_plane: Dict[int, Set[Tuple[int, int]]] = {}
    for img_id, fmap in feature_to_global.items():
        for feat_idx, gp in fmap.items():
            pre_by_plane.setdefault(gp, set()).add((img_id, feat_idx))
    return pre_by_plane


def build_post_features(
    recon: pycolmap.Reconstruction, plane_to_pids: Dict[int, list]
) -> Dict[int, Set[Tuple[int, int]]]:
    post_by_plane: Dict[int, Set[Tuple[int, int]]] = {}
    for gp, pids in plane_to_pids.items():
        s: Set[Tuple[int, int]] = set()
        for pid in pids:
            if pid not in recon.points3D:
                continue
            p3d = recon.points3D[pid]
            for elem in p3d.track.elements:
                s.add((int(elem.image_id), int(elem.point2D_idx)))
        post_by_plane[gp] = s
    return post_by_plane


def select_top_images_for_plane(
    pre: Set[Tuple[int, int]],
    post: Set[Tuple[int, int]],
    max_images: int,
) -> list:
    counts: Dict[int, int] = {}
    for img_id, _ in pre | post:
        counts[img_id] = counts.get(img_id, 0) + 1
    return [
        img_id
        for img_id, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
            :max_images
        ]
    ]


def draw_points_on_image(
    img_bgr: np.ndarray,
    pts_xy: np.ndarray,
    color_bgr: Tuple[int, int, int],
    radius: int = 3,
    thickness: int = -1,
):
    if pts_xy.size == 0:
        return
    # pts_xy: (N,2)
    for x, y in pts_xy:
        cv2.circle(
            img_bgr,
            (int(round(x)), int(round(y))),
            radius,
            color_bgr,
            thickness,
            lineType=cv2.LINE_AA,
        )


def visualize_planes_pre_post(
    result: dict,
    recon: pycolmap.Reconstruction,
    feature_path: Path,
    image_root: Path,
    output_dir: Path,
    max_images_per_plane: int = 4,
    features_group_name: str = "dslr_images_undistorted",
    point_radius: int = 3,
):
    """
    Saves overlay images to:
      output_dir/plane_<gp>/<image_stem>.png
    """
    output_dir = Path(output_dir)
    image_root = Path(image_root)

    feature_to_global = _normalize_feature_to_global(result)
    plane_to_pids = _normalize_plane_to_pids(result)

    pre_by_plane = build_pre_features(feature_to_global)
    post_by_plane = build_post_features(recon, plane_to_pids)

    all_plane_ids = sorted(set(pre_by_plane.keys()) | set(post_by_plane.keys()))

    with h5py.File(feature_path, "r") as f:
        if features_group_name not in f:
            raise KeyError(f"H5 does not contain group '{features_group_name}'")
        feat_group = f[features_group_name]

        for gp in all_plane_ids:
            pre = pre_by_plane.get(gp, set())
            post = post_by_plane.get(gp, set())

            kept = pre & post
            removed = pre - post
            post_only = post - pre

            img_ids = select_top_images_for_plane(
                pre, post, max_images=max_images_per_plane
            )
            if not img_ids:
                continue

            plane_dir = output_dir / f"plane_{gp:04d}"
            plane_dir.mkdir(parents=True, exist_ok=True)

            summary_txt = (
                f"plane={gp}\n"
                f"pre={len(pre)} post={len(post)} kept={len(kept)} removed={len(removed)} post_only={len(post_only)}\n"
            )
            (plane_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")

            for img_id in img_ids:
                if img_id not in recon.images:
                    continue
                img = recon.images[img_id]
                recon_name = img.name 
                img_path = image_root / recon_name

                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    alt_path = image_root / Path(recon_name).name
                    img_bgr = cv2.imread(str(alt_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue

                h5_key = _h5_resolve_image_key(feat_group, recon_name)
                if h5_key is None:
                    continue
                kps = feat_group[h5_key]["keypoints"][()]  # (N,2) or (N, >=2)

                pre_idx = sorted([fi for (iid, fi) in pre if iid == img_id])
                kept_idx = sorted([fi for (iid, fi) in kept if iid == img_id])
                removed_idx = sorted([fi for (iid, fi) in removed if iid == img_id])
                post_only_idx = sorted([fi for (iid, fi) in post_only if iid == img_id])

                def idx_to_xy(idxs: list) -> np.ndarray:
                    if not idxs:
                        return np.empty((0, 2), dtype=np.float32)
                    xy = kps[np.asarray(idxs, dtype=np.int64)]
                    xy = np.asarray(xy, dtype=np.float32)
                    if xy.shape[1] > 2:
                        xy = xy[:, :2]
                    return xy

                pts_kept = idx_to_xy(kept_idx)
                pts_removed = idx_to_xy(removed_idx)
                pts_post_only = idx_to_xy(post_only_idx)

                draw_points_on_image(
                    img_bgr, pts_removed, color_bgr=(0, 0, 255), radius=point_radius
                )  # red
                draw_points_on_image(
                    img_bgr, pts_post_only, color_bgr=(255, 0, 0), radius=point_radius
                )  # blue
                draw_points_on_image(
                    img_bgr, pts_kept, color_bgr=(0, 255, 255), radius=point_radius
                )  # yellow

                title = f"plane {gp} | img_id {img_id}"
                stats = f"pre:{len(pre_idx)} kept:{len(kept_idx)} removed:{len(removed_idx)} post_only:{len(post_only_idx)}"
                cv2.putText(
                    img_bgr,
                    title,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img_bgr,
                    stats,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                out_name = f"{Path(recon_name).stem}_overlay.png"
                cv2.imwrite(str(plane_dir / out_name), img_bgr)


def _find_model_dir(sfm_dir: Path) -> Path:
    """
    Accepts:
      - <out>/sfm/0
      - <out>/sfm
    """
    sfm_dir = Path(sfm_dir)
    cands = [sfm_dir / "0", sfm_dir]
    for d in cands:
        if not d.exists():
            continue
        has_bin = (
            (d / "cameras.bin").exists()
            and (d / "images.bin").exists()
            and (d / "points3D.bin").exists()
        )
        has_txt = (
            (d / "cameras.txt").exists()
            and (d / "images.txt").exists()
            and (d / "points3D.txt").exists()
        )
        if has_bin or has_txt:
            return d
    raise FileNotFoundError(f"No COLMAP model found in: {sfm_dir} (checked {cands})")


def main():
    import shutil

    dataset_root = Path("./datasets/eth3d")
    pred_root = Path("./outputs/PASfM")
    vis_root = Path("./visualize")

    max_images_per_plane = 4
    features_group = "dslr_images_undistorted"
    point_radius = 5

    def _first_existing_dir(cands):
        for p in cands:
            p = Path(p)
            if p.exists() and p.is_dir():
                return p
        return None

    def _first_existing_file(cands):
        for p in cands:
            p = Path(p)
            if p.exists() and p.is_file():
                return p
        return None

    def _resolve_pred_base(scene: str) -> Path | None:
        cands = [
            pred_root / scene / "pasfm",
            pred_root / scene / "sfm",
            pred_root / scene,
        ]
        return _first_existing_dir(cands)

    def _resolve_scene_paths(scene_dir: Path):
        scene = scene_dir.name

        image_root = scene_dir / "images" / "dslr_images_undistorted"
        if not image_root.exists():
            print(f"[Skip:{scene}] image_root not found: {image_root}")
            return None

        pred_base = _resolve_pred_base(scene)
        if pred_base is None:
            print(f"[Skip:{scene}] pred_base not found under {pred_root}")
            return None

        planes_json = _first_existing_file(
            [
                pred_base / "planes.json",
                pred_base / "planes" / "planes.json",
            ]
        )
        if planes_json is None:
            print(f"[Skip:{scene}] planes.json not found under {pred_base}")
            return None

        feature_h5 = _first_existing_file(
            [
                pred_base / "features.h5",
                pred_base / "features_superpoint.h5",
                pred_base / "features-superpoint.h5",
            ]
        )
        if feature_h5 is None:
            feats = sorted(pred_base.glob("*features*.h5"))
            feature_h5 = feats[0] if feats else None
        if feature_h5 is None:
            print(f"[Skip:{scene}] features*.h5 not found under {pred_base}")
            return None

        sfm_dir = _first_existing_dir(
            [
                pred_base / "sfm",
                pred_base / "sparse",
                pred_base,
            ]
        )
        if sfm_dir is None:
            print(f"[Skip:{scene}] sfm_dir not found under {pred_base}")
            return None

        return planes_json, sfm_dir, feature_h5, image_root, pred_base

    vis_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders found under: {dataset_root}")

    ok_scenes = 0
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        resolved = _resolve_scene_paths(scene_dir)
        if resolved is None:
            continue

        planes_json, sfm_dir, feature_h5, image_root, pred_base = resolved

        # load COLMAP model
        try:
            model_dir = _find_model_dir(sfm_dir)
            recon = pycolmap.Reconstruction(str(model_dir))
        except Exception as e:
            print(f"[Skip:{scene}] load model failed: {sfm_dir} ({e})")
            continue

        # load planes json
        try:
            with planes_json.open("r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            print(f"[Skip:{scene}] read planes_json failed: {planes_json} ({e})")
            continue

        out_dir_scene = vis_root / scene
        out_dir_scene.mkdir(parents=True, exist_ok=True)

        try:
            visualize_planes_pre_post(
                result=result,
                recon=recon,
                feature_path=feature_h5,
                image_root=image_root,
                output_dir=out_dir_scene,
                max_images_per_plane=max_images_per_plane,
                features_group_name=features_group,
                point_radius=point_radius,
            )
            print(f"[OK:{scene}] Saved → {out_dir_scene}")
            ok_scenes += 1
        except Exception as e:
            print(f"[Skip:{scene}] visualize failed ({e})")
            continue

    zip_path = shutil.make_archive(str(vis_root), "zip", root_dir=str(vis_root))
    print(f"[OK] Scenes visualized: {ok_scenes}/{len(scene_dirs)}")
    print(f"[OK] Zipped → {zip_path}")


if __name__ == "__main__":
    main()
