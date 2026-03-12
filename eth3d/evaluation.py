from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import math

import numpy as np
import pycolmap

from utils import norm_name, get_c2w_R_C
from error_metrics import (
    rotation_angle_error_deg,
    translation_direction_error_deg,
    compute_auc,
)

PRED_SUBDIRS = ["sfm", "pasfm", "sfm+ba"]
GT_SUBDIR = "dslr_calibration_undistorted"

THRESHOLDS = [1.0, 3.0, 5.0]
ALLOW_FLIP_T = True
PENALIZE_MISSING_PAIRS = True
MISSING_PAIR_ERROR = 180.0

WRITE_CSV = True

SCENE_SPLIT: Dict[str, str] = {
    # Training (DSLR, individual images)
    "courtyard": "outdoor",
    "delivery_area": "indoor",
    "electro": "outdoor",
    "facade": "outdoor",
    "kicker": "indoor",
    "meadow": "outdoor",
    "office": "indoor",
    "pipes": "indoor",
    "playground": "outdoor",
    "relief": "indoor",
    "relief_2": "indoor",
    "terrace": "outdoor",
    "terrains": "indoor",
    "botanical_garden": "indoor",
    "boulders": "outdoor",
    "bridge": "indoor",
    "door": "indoor",
    "exhibition_hall": "indoor",
    "lecture_room": "indoor",
    "living_room": "indoor",
    "lounge": "indoor",
    "observatory": "outdoor",
    "old_computer": "indoor",
    "statue": "indoor",
    "terrace_2": "outdoor",
    # Test (rig scenes appear on the same page; some people also place them under eth3d root)
    "lakeside": "outdoor",
    "sand_box": "outdoor",
    "storage_room": "indoor",
    "storage_room_2": "indoor",
    "tunnel": "outdoor",
}


def _guess_project_root() -> Path:
    """
    eth3d/evaluation.py 라는 전형적 위치를 가정하고 project root를 추정.
    - /proj/eth3d/evaluation.py  -> root=/proj
    """
    here = Path(__file__).resolve()
    if here.parent.name == "eth3d":
        return here.parent.parent
    return Path.cwd()


def _pick_existing(candidates: List[Path], fallback: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return fallback


PROJECT_ROOT = _guess_project_root()
GT_ROOT = _pick_existing(
    [
        PROJECT_ROOT / "datasets" / "eth3d",
        Path.cwd() / "datasets" / "eth3d",
        PROJECT_ROOT / "datasets",
    ],
    fallback=PROJECT_ROOT / "datasets" / "eth3d",
)
PRED_ROOT = _pick_existing(
    [
        PROJECT_ROOT / "outputs" / "PASfM",
        Path.cwd() / "outputs" / "PASfM",
        PROJECT_ROOT / "outputs",
    ],
    fallback=PROJECT_ROOT / "outputs" / "PASfM",
)


def build_pose_dict(
    recon: pycolmap.Reconstruction,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build {image_name: (R_wc, C_w)} from reconstruction images."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for _, img in recon.images.items():
        if not img.has_pose:
            continue
        key = norm_name(img.name)
        out[key] = get_c2w_R_C(img)
    return out


def relative_pose_from_c2w(
    Ri_wc: np.ndarray, Ci_w: np.ndarray, Rj_wc: np.ndarray, Cj_w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose from camera i to camera j in camera-j coordinates.

      R_ji = (Rj_wc)^T * (Ri_wc)
      t_ji = (Rj_wc)^T * (Ci - Cj)
    """
    R_jw = Rj_wc.T
    R_ji = R_jw @ Ri_wc
    t_ji = R_jw @ (Ci_w - Cj_w)
    return R_ji, t_ji


def evaluate_pairwise_auc(
    poses_gt: Dict[str, Tuple[np.ndarray, np.ndarray]],
    poses_pred: Dict[str, Tuple[np.ndarray, np.ndarray]],
    thresholds: List[float],
    allow_flip_t: bool = False,
    penalize_missing_pairs: bool = False,
    missing_pair_error: float = 180.0,
) -> Tuple[List[float], Dict[str, float], List[float]]:
    """
    Evaluate AUC@thresholds using pairwise relative pose errors.

    Returns:
      aucs, stats, errors
    """
    names_gt = sorted(list(poses_gt.keys()))
    names_pred = set(poses_pred.keys())
    common = sorted(list(set(poses_gt.keys()) & set(poses_pred.keys())))

    stats: Dict[str, float] = {
        "num_gt": float(len(names_gt)),
        "num_pred": float(len(poses_pred)),
        "num_common": float(len(common)),
        "success_ratio_common_over_gt": float(len(common) / max(1, len(names_gt))),
    }

    errors: List[float] = []

    if not penalize_missing_pairs:
        names = common
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ni, nj = names[a], names[b]

                Ri_gt, Ci_gt = poses_gt[ni]
                Rj_gt, Cj_gt = poses_gt[nj]
                Rrel_gt, trel_gt = relative_pose_from_c2w(Ri_gt, Ci_gt, Rj_gt, Cj_gt)

                Ri_pr, Ci_pr = poses_pred[ni]
                Rj_pr, Cj_pr = poses_pred[nj]
                Rrel_pr, trel_pr = relative_pose_from_c2w(Ri_pr, Ci_pr, Rj_pr, Cj_pr)

                eR = rotation_angle_error_deg(Rrel_gt, Rrel_pr)
                et = translation_direction_error_deg(
                    trel_gt, trel_pr, allow_flip=allow_flip_t
                )
                errors.append(max(eR, et))
    else:
        for a in range(len(names_gt)):
            for b in range(a + 1, len(names_gt)):
                ni, nj = names_gt[a], names_gt[b]
                if (ni in names_pred) and (nj in names_pred):
                    Ri_gt, Ci_gt = poses_gt[ni]
                    Rj_gt, Cj_gt = poses_gt[nj]
                    Rrel_gt, trel_gt = relative_pose_from_c2w(
                        Ri_gt, Ci_gt, Rj_gt, Cj_gt
                    )

                    Ri_pr, Ci_pr = poses_pred[ni]
                    Rj_pr, Cj_pr = poses_pred[nj]
                    Rrel_pr, trel_pr = relative_pose_from_c2w(
                        Ri_pr, Ci_pr, Rj_pr, Cj_pr
                    )

                    eR = rotation_angle_error_deg(Rrel_gt, Rrel_pr)
                    et = translation_direction_error_deg(
                        trel_gt, trel_pr, allow_flip=allow_flip_t
                    )
                    errors.append(max(eR, et))
                else:
                    errors.append(float(missing_pair_error))

    stats["num_pairs_used"] = float(len(errors))

    if len(errors) == 0:
        aucs = [float("nan")] * len(thresholds)
    else:
        aucs = compute_auc(errors, thresholds=thresholds, min_error=1e-3)

    return aucs, stats, errors


@dataclass
class MethodResult:
    aucs: List[float]
    stats: Dict[str, float]
    errors: List[float]
    ok: bool
    msg: str = ""


def _load_recon(model_dir: Path) -> pycolmap.Reconstruction:
    return pycolmap.Reconstruction(str(model_dir))


def _eval_one(pred_dir: Path, gt_dir: Path) -> MethodResult:
    try:
        recon_pred = _load_recon(pred_dir)
        recon_gt = _load_recon(gt_dir)

        poses_pred = build_pose_dict(recon_pred)
        poses_gt = build_pose_dict(recon_gt)

        aucs, stats, errors = evaluate_pairwise_auc(
            poses_gt=poses_gt,
            poses_pred=poses_pred,
            thresholds=THRESHOLDS,
            allow_flip_t=ALLOW_FLIP_T,
            penalize_missing_pairs=PENALIZE_MISSING_PAIRS,
            missing_pair_error=MISSING_PAIR_ERROR,
        )
        return MethodResult(aucs=aucs, stats=stats, errors=errors, ok=True)
    except Exception as e:
        return MethodResult(
            aucs=[float("nan")] * len(THRESHOLDS),
            stats={},
            errors=[],
            ok=False,
            msg=f"{type(e).__name__}: {e}",
        )


def _split_of(scene: str) -> str:
    return SCENE_SPLIT.get(scene, "unknown")


def _fmt_auc_triplet(aucs: List[float]) -> str:
    def f(x: float) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.4f}"

    return "/".join(f(a) for a in aucs)


def _mean_auc(aucs: List[float]) -> float:
    vals = [a for a in aucs if not (isinstance(a, float) and math.isnan(a))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _safe_auc_from_errors(errors: List[float]) -> List[float]:
    if len(errors) == 0:
        return [float("nan")] * len(THRESHOLDS)
    return compute_auc(errors, thresholds=THRESHOLDS, min_error=1e-3)


def main():
    print("============================================================")
    print("ETH3D Multi-Scene Evaluation (no argparse)")
    print("============================================================")
    print(f"- Project root guess: {PROJECT_ROOT}")
    print(f"- GT_ROOT  : {GT_ROOT}")
    print(f"- PRED_ROOT: {PRED_ROOT}")
    print(f"- GT_SUBDIR: {GT_SUBDIR}")
    print(f"- Methods  : {', '.join(PRED_SUBDIRS)}")
    print(
        f"- Thresholds(deg): {', '.join(str(int(t)) if t.is_integer() else str(t) for t in THRESHOLDS)}"
    )
    print(
        f"- allow_flip_t={ALLOW_FLIP_T}, penalize_missing_pairs={PENALIZE_MISSING_PAIRS}"
    )
    print("")

    if not GT_ROOT.exists():
        raise SystemExit(f"[ERROR] GT_ROOT not found: {GT_ROOT}")
    if not PRED_ROOT.exists():
        print(
            f"[WARN] PRED_ROOT not found yet: {PRED_ROOT} (missing predictions will show as MISSING)"
        )

    scene_names = sorted([p.name for p in GT_ROOT.iterdir() if p.is_dir()])

    all_scene_results: Dict[str, Dict[str, MethodResult]] = {}
    skipped: List[Tuple[str, str]] = []

    errors_by_split: Dict[str, Dict[str, List[float]]] = {}
    per_scene_auc_by_split: Dict[str, Dict[str, List[List[float]]]] = {}

    for split in ["indoor", "outdoor", "unknown"]:
        errors_by_split[split] = {m: [] for m in PRED_SUBDIRS}
        per_scene_auc_by_split[split] = {m: [] for m in PRED_SUBDIRS}

    for scene in scene_names:
        gt_dir = GT_ROOT / scene / GT_SUBDIR
        if not gt_dir.exists():
            skipped.append((scene, "missing_gt_subdir"))
            continue

        split = _split_of(scene)
        all_scene_results[scene] = {}

        for m in PRED_SUBDIRS:
            pred_dir = PRED_ROOT / scene / m
            if not pred_dir.exists():
                all_scene_results[scene][m] = MethodResult(
                    aucs=[float("nan")] * len(THRESHOLDS),
                    stats={},
                    errors=[],
                    ok=False,
                    msg="MISSING",
                )
                continue

            res = _eval_one(pred_dir, gt_dir)
            all_scene_results[scene][m] = res

            if res.ok:
                errors_by_split[split][m].extend(res.errors)
                per_scene_auc_by_split[split][m].append(res.aucs)

        best_m = None
        best_score = -1.0
        for m in PRED_SUBDIRS:
            score = _mean_auc(all_scene_results[scene][m].aucs)
            if (
                not (isinstance(score, float) and math.isnan(score))
                and score > best_score
            ):
                best_score = score
                best_m = m

        print(f"--- {scene} ({split}) ---")
        for m in PRED_SUBDIRS:
            r = all_scene_results[scene][m]
            tag = "★" if (best_m == m and r.ok) else " "
            if not r.ok:
                print(f"{tag} {m:<7}  AUC { _fmt_auc_triplet(r.aucs) }   [{r.msg}]")
            else:
                nc = int(r.stats.get("num_common", 0))
                npairs = int(r.stats.get("num_pairs_used", 0))
                succ = r.stats.get("success_ratio_common_over_gt", float("nan"))
                succ_s = "—" if math.isnan(succ) else f"{succ:.3f}"
                print(
                    f"{tag} {m:<7}  AUC { _fmt_auc_triplet(r.aucs) }   common={nc:<4} pairs={npairs:<6} succ={succ_s}"
                )

        # delta vs sfm
        if (
            all_scene_results[scene].get("sfm", None)
            and all_scene_results[scene]["sfm"].ok
        ):
            base = all_scene_results[scene]["sfm"].aucs
            for m in ["pasfm", "sfm+ba"]:
                if m in all_scene_results[scene] and all_scene_results[scene][m].ok:
                    cur = all_scene_results[scene][m].aucs
                    deltas = []
                    for t, c, b in zip(THRESHOLDS, cur, base):
                        if (isinstance(c, float) and math.isnan(c)) or (
                            isinstance(b, float) and math.isnan(b)
                        ):
                            deltas.append(f"AUC@{t:g}:—")
                        else:
                            deltas.append(f"AUC@{t:g}:{(c-b):+0.4f}")
                    print(f"    Δ ({m} - sfm): " + " ".join(deltas))
        print("")

    def _print_summary_block(title: str, split_key: str):
        print("============================================================")
        print(f"{title}")
        print("============================================================")
        header = f"{'method':<7} | {'macro(scene avg) AUC@1/3/5':<28} | {'global(pair-concat) AUC@1/3/5':<30} | {'#pairs':>7}"
        print(header)
        print("-" * len(header))
        for m in PRED_SUBDIRS:
            auc_list = per_scene_auc_by_split[split_key][m]
            if len(auc_list) == 0:
                macro = [float("nan")] * len(THRESHOLDS)
            else:
                macro = []
                for k in range(len(THRESHOLDS)):
                    vals = [
                        aucs[k]
                        for aucs in auc_list
                        if not (isinstance(aucs[k], float) and math.isnan(aucs[k]))
                    ]
                    macro.append(float(sum(vals) / len(vals)) if vals else float("nan"))

            # global: errors concat 후 AUC
            errs = errors_by_split[split_key][m]
            glob = _safe_auc_from_errors(errs)

            print(
                f"{m:<7} | {_fmt_auc_triplet(macro):<28} | {_fmt_auc_triplet(glob):<30} | {len(errs):>7}"
            )
        print("")

    _print_summary_block("SUMMARY: INDOOR", "indoor")
    _print_summary_block("SUMMARY: OUTDOOR", "outdoor")
    _print_summary_block("SUMMARY: UNKNOWN (not in split map)", "unknown")

    # Overall (all splits)
    overall_errors = {m: [] for m in PRED_SUBDIRS}
    overall_macro_aucs = {m: [] for m in PRED_SUBDIRS}
    for split in ["indoor", "outdoor", "unknown"]:
        for m in PRED_SUBDIRS:
            overall_errors[m].extend(errors_by_split[split][m])
            overall_macro_aucs[m].extend(per_scene_auc_by_split[split][m])

    print("============================================================")
    print("SUMMARY: OVERALL (all scenes)")
    print("============================================================")
    header = f"{'method':<7} | {'macro(scene avg) AUC@1/3/5':<28} | {'global(pair-concat) AUC@1/3/5':<30} | {'#pairs':>7}"
    print(header)
    print("-" * len(header))
    for m in PRED_SUBDIRS:
        auc_list = overall_macro_aucs[m]
        if len(auc_list) == 0:
            macro = [float("nan")] * len(THRESHOLDS)
        else:
            macro = []
            for k in range(len(THRESHOLDS)):
                vals = [
                    aucs[k]
                    for aucs in auc_list
                    if not (isinstance(aucs[k], float) and math.isnan(aucs[k]))
                ]
                macro.append(float(sum(vals) / len(vals)) if vals else float("nan"))

        glob = _safe_auc_from_errors(overall_errors[m])
        print(
            f"{m:<7} | {_fmt_auc_triplet(macro):<28} | {_fmt_auc_triplet(glob):<30} | {len(overall_errors[m]):>7}"
        )
    print("")

    if WRITE_CSV:
        out_csv = PROJECT_ROOT / "eth3d_eval_results.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "scene",
                    "split",
                    "method",
                    "auc@1",
                    "auc@3",
                    "auc@5",
                    "num_gt",
                    "num_pred",
                    "num_common",
                    "num_pairs_used",
                    "success_ratio_common_over_gt",
                    "status",
                    "msg",
                ]
            )
            for scene in sorted(all_scene_results.keys()):
                split = _split_of(scene)
                for m in PRED_SUBDIRS:
                    r = all_scene_results[scene][m]
                    a1, a3, a5 = r.aucs
                    w.writerow(
                        [
                            scene,
                            split,
                            m,
                            a1,
                            a3,
                            a5,
                            r.stats.get("num_gt", ""),
                            r.stats.get("num_pred", ""),
                            r.stats.get("num_common", ""),
                            r.stats.get("num_pairs_used", ""),
                            r.stats.get("success_ratio_common_over_gt", ""),
                            "OK" if r.ok else "FAIL",
                            r.msg,
                        ]
                    )
        print(f"[CSV] wrote: {out_csv}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
