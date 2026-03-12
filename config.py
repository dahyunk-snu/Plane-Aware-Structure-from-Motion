from dataclasses import dataclass

from hloc import extract_features, match_features


@dataclass(frozen=True)
class PASfMConfig:
    feature_conf = extract_features.confs["superpoint_highres"]
    matcher_conf = match_features.confs["superglue"]

    total_ba_iters = 10
    stage_ba_iters = 30

    use_weighted_ba = False
    use_ba_with_fixed_points = False
    always_extract_planes = True
    compare_with_colmap = False
