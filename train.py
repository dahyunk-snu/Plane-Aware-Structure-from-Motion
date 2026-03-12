from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
from config import PASfMConfig
from planes.extract_planes import extract_planes
from planes.utils import load_planes_json
from optimization.optimization import (
    run_plane_aware_ba,
    run_plane_aware_ba_with_fixed_points,
    bundle_adjustment,
    weighted_bundle_adjustment,
)
from optimization.score import compute_point_weights
from utils import load_model


class PASfM:
    """
    Plane-Aware Structure-from-Motion pipeline
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        config: PASfMConfig = PASfMConfig(),
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)

        self.feature_conf = config.feature_conf
        self.matcher_conf = config.matcher_conf

        self.total_ba_iters = config.total_ba_iters
        self.stage_ba_iters = config.stage_ba_iters

        self.use_ba_with_fixed_points = config.use_ba_with_fixed_points
        self.use_weighted_ba = config.use_weighted_ba
        self.always_extract_planes = config.always_extract_planes
        self.compare_with_colmap = config.compare_with_colmap
        self.compute_only_colmap = config.compute_only_colmap

    def extract_features(self, image_dir: Path):
        feature_path = extract_features.main(
            self.feature_conf, image_dir, self.output_dir
        )
        return feature_path

    def match_features(self, feature_path: Path):
        pairs_path = self.output_dir / "pairs.txt"

        pairs_from_exhaustive.main(pairs_path, features=feature_path)
        match_path = match_features.main(
            self.matcher_conf,
            pairs_path,
            self.feature_conf["output"],
            self.output_dir,
        )
        return pairs_path, match_path

    def main(self):
        print(f"[PASfM] Processing scene: {self.dataset_dir.name}")

        image_dir = self.dataset_dir / "images"
        if not image_dir.exists():
            print(f"[Error] no images in {self.dataset_dir.name}")
            return

        # 0. Setup output folders
        sfm_dir = self.output_dir / "sfm"
        sfm_dir.mkdir(parents=True, exist_ok=True)

        pasfm_dir = self.output_dir / "pasfm"
        pasfm_dir.mkdir(parents=True, exist_ok=True)

        sfm_ba_dir = self.output_dir / "sfm+ba"
        sfm_ba_dir.mkdir(parents=True, exist_ok=True)

        sfm_w_ba_dir = self.output_dir / "sfm+wba"
        sfm_w_ba_dir.mkdir(parents=True, exist_ok=True)

        planes_path = self.output_dir / "planes.json"
        final_planes_path = self.output_dir / "final_planes.json"

        vis_path = self.output_dir / "vis"
        vis_path.mkdir(parents=True, exist_ok=True)

        # 1. Feature extraction + matching
        print(f"[PASfM] Extract and match image features...")
        feature_path = self.extract_features(image_dir)
        pairs_path, match_path = self.match_features(feature_path)

        # 2. COLMAP sparse reconstruction
        print(f"[PASfM] Reconstruct a 3D model...")
        recon = load_model(sfm_dir)
        if recon is None:
            recon = reconstruction.main(
                sfm_dir, image_dir, pairs_path, feature_path, match_path
            )

        # 3. Estimate planes
        print(f"[PASfM] Estimate planes...")
        plane_models, plane_to_pids = None, None

        if not self.always_extract_planes:
            plane_models, plane_to_pids = load_planes_json(planes_path)

        if plane_models is None or plane_to_pids is None:
            plane_models, plane_to_pids = extract_planes(
                recon, feature_path, match_path, save_path=planes_path
            )

        # 4. Plane-Aware BA
        print(f"[PASfM] Run Plane-Aware BA...")
        if not self.compute_only_colmap:
            if self.use_ba_with_fixed_points:
                run_plane_aware_ba_with_fixed_points(
                    recon=recon,
                    sfm_dir=pasfm_dir,
                    save_path=final_planes_path,
                    plane_models=plane_models,
                    plane_to_pids=plane_to_pids,
                    total_ba_iters=self.total_ba_iters,
                    stage_ba_iters=self.stage_ba_iters,
                    use_weighted_ba=self.use_weighted_ba,
                    match_path=match_path,
                )
            else:
                run_plane_aware_ba(
                    recon=recon,
                    sfm_dir=pasfm_dir,
                    save_path=final_planes_path,
                    plane_models=plane_models,
                    plane_to_pids=plane_to_pids,
                    total_ba_iters=self.total_ba_iters,
                    stage_ba_iters=self.stage_ba_iters,
                    use_weighted_ba=self.use_weighted_ba,
                    match_path=match_path,
                    vis_path=vis_path,
                )

        # 5. Compare with BA
        if self.compare_with_colmap or self.compute_only_colmap:
            iters = 0
            for _ in range(self.total_ba_iters + 1):
                iters += self.stage_ba_iters

            print(f"[PASfM] COLMAP + BA: {iters} iterations")
            recon = load_model(sfm_dir)
            bundle_adjustment(recon, max_num_iterations=iters)
            recon.write(str(sfm_ba_dir))

            print(f"[PASfM] COLMAP + Weighted BA: {iters} iterations")
            recon = load_model(sfm_dir)
            weights = compute_point_weights(recon, match_path)
            weighted_bundle_adjustment(recon, weights, max_num_iterations=iters)
            recon.write(str(sfm_w_ba_dir))


if __name__ == "__main__":
    dataset_dir = "./datasets/eth3d/living_room"
    output_dir = "./outputs/PASfM/living_room"

    model = PASfM(dataset_dir, output_dir)
    model.main()
