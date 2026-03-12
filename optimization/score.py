import numpy as np
import pycolmap
import h5py
from typing import Dict
from pathlib import Path

import matplotlib.pyplot as plt

from planes.utils import extract_image_name


def compute_point_weights(
    recon: pycolmap.Reconstruction, matches_path: str, vis_path: Path
) -> Dict[int, float]:
    print("Computing point weights from hierarchical matches.h5...")

    feature_to_pid = {}
    for pid, point in recon.points3D.items():
        for track_el in point.track.elements:
            img_name = extract_image_name(recon.images[track_el.image_id].name)
            feature_to_pid[(img_name, track_el.point2D_idx)] = pid

    pid_max_scores: Dict[int, float] = {}

    with h5py.File(matches_path, "r") as f_match:
        keys1 = list(f_match.keys())

        for name1 in keys1:
            grp1 = f_match[name1]
            if not isinstance(grp1, h5py.Group):
                continue

            for name2 in grp1.keys():
                grp2 = grp1[name2]
                if not isinstance(grp2, h5py.Group):
                    continue

                if "matches0" not in grp2 or "matching_scores0" not in grp2:
                    continue

                matches = grp2["matches0"][()]  # Shape: (N, 2) or (N,)
                scores = grp2["matching_scores0"][()]  # Shape: (N,)

                matches = np.array(matches)
                scores = np.array(scores)

                valid_mask = matches > -1

                for idx1, idx2 in enumerate(matches):
                    if not valid_mask[idx1]:
                        continue

                    score = float(scores[idx1])

                    pid1 = feature_to_pid.get((extract_image_name(name1), idx1))
                    pid2 = feature_to_pid.get((extract_image_name(name2), idx2))

                    target_pids = set()
                    if pid1 is not None:
                        target_pids.add(pid1)
                    if pid2 is not None:
                        target_pids.add(pid2)

                    for pid in target_pids:
                        if pid not in pid_max_scores:
                            pid_max_scores[pid] = score
                        else:
                            if score > pid_max_scores[pid]:
                                pid_max_scores[pid] = score

    valid_scores = list(pid_max_scores.values())

    if len(valid_scores) > 0:
        fallback_weight = float(np.percentile(valid_scores, 5))

        print(
            f"Distribution Statistics - Max: {np.max(valid_scores):.4f}, Mean: {np.mean(valid_scores):.4f}, 5th Percentile: {fallback_weight:.4f}"
        )
    else:
        fallback_weight = 0.5
        print("Warning: No matching scores found in file. Using default weight 0.5.")

    weights = {}
    missing_count = 0

    for pid in recon.points3D:
        if pid in pid_max_scores:
            weights[pid] = pid_max_scores[pid]
        else:
            weights[pid] = fallback_weight
            missing_count += 1

    print(f"Computed weights for {len(weights)} points.")
    print(
        f" - {missing_count} points had no match score and were assigned the 5th percentile weight ({fallback_weight:.4f})."
    )

    vis_path.mkdir(parents=True, exist_ok=True)

    wvals = np.asarray(list(weights.values()), dtype=np.float64)
    wvals = wvals[np.isfinite(wvals)]

    if wvals.size > 0:
        plt.figure()
        plt.hist(wvals, bins=60)
        plt.xlabel("weight (max match score)")
        plt.ylabel("count")
        plt.title("Point weight distribution")
        plt.tight_layout()
        plt.savefig(vis_path / "weights_hist.png", dpi=200)
        plt.close()

        ws = np.sort(wvals)
        cdf = np.arange(1, ws.size + 1) / ws.size

        plt.figure()
        plt.plot(ws, cdf)
        plt.xlabel("weight (max match score)")
        plt.ylabel("CDF")
        plt.title("Point weight CDF")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(vis_path / "weights_cdf.png", dpi=200)
        plt.close()

        print(f"[VIS] Saved weight distribution plots to: {vis_path}")
    else:
        print("[VIS] No finite weights to plot.")

    return weights
