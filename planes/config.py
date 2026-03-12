from dataclasses import dataclass


@dataclass(frozen=True)
class PlaneExtractionConfig:
    # 2D homography RANSAC (local planes)
    ransac_thresh_px: float = 1.5
    min_pair_inliers: int = 30
    max_planes_per_pair: int = 5
    rot_angle_thresh_deg: float = 1.0
    exclude_pure_rotation: bool = True

    # local -> global merging
    min_overlap: int = 100
    min_global_inliers: int = 50

    # 3D plane refine
    plane_ransac_thresh_m: float = 0.02
    plane_ransac_conf: float = 0.999
    plane_ransac_max_iters: int = 10_000

    # HDF5 group name
    features_group: str = "dslr_images_undistorted"
