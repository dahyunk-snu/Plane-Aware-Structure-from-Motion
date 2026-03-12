from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils import load_json, save_json
from label_maps import load_label_maps
from recon_io import load_pycolmap_recon
from membership import assign_points_to_planes
from metrics import PlaneEq


def _normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return np.full_like(v, np.nan, dtype=np.float64)
    return (v / n).astype(np.float64)


def _angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    n1u = _normalize_vec(n1)
    n2u = _normalize_vec(n2)
    if not (np.all(np.isfinite(n1u)) and np.all(np.isfinite(n2u))):
        return float("nan")
    c = float(np.clip(float(n1u @ n2u), -1.0, 1.0))
    c = abs(c)  # normal sign ambiguity
    return float(math.degrees(math.acos(c)))


def _has_colmap_model_files(d: Path) -> bool:
    cams_bin = d / "cameras.bin"
    imgs_bin = d / "images.bin"
    pts_bin = d / "points3D.bin"
    cams_txt = d / "cameras.txt"
    imgs_txt = d / "images.txt"
    pts_txt = d / "points3D.txt"
    return (cams_bin.exists() and imgs_bin.exists() and pts_bin.exists()) or (
        cams_txt.exists() and imgs_txt.exists() and pts_txt.exists()
    )


def find_recon_model_dir(recon_root: Path) -> Path:
    """
    recon_root examples:
      - outputs/PASfM/<scene>/pasfm
      - or already a COLMAP model dir
    """
    recon_root = recon_root.expanduser().resolve()

    if (
        recon_root.exists()
        and recon_root.is_dir()
        and _has_colmap_model_files(recon_root)
    ):
        return recon_root

    candidates = [
        recon_root,
        recon_root / "pasfm",
        recon_root / "sparse" / "0",
        recon_root / "sparse" / "model",
        recon_root / "sparse",
        recon_root / "colmap" / "sparse" / "0",
        recon_root / "pasfm" / "sparse" / "0",
    ]
    for c in candidates:
        if c.exists() and c.is_dir() and _has_colmap_model_files(c):
            return c

    # bounded search
    for depth in range(1, 6):
        for d in recon_root.glob("/".join(["*"] * depth)):
            if d.is_dir() and _has_colmap_model_files(d):
                return d

    raise RuntimeError(f"Could not find COLMAP model files under: {recon_root}")


def load_scan_planes(root_dir: Path) -> Dict[int, PlaneEq]:
    """
    Expects: <root_dir>/planes_from_scan.json
    """
    js = load_json(root_dir / "planes_from_scan.json")
    planes: Dict[int, PlaneEq] = {}
    for p in js.get("planes", []):
        pid = int(p["plane_id"])
        n = _normalize_vec(np.asarray(p["n"], np.float64).reshape(3))
        d = float(p["d"])
        planes[pid] = PlaneEq(n=n, d=d)
    return planes


def _as_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: List[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    try:
        return [int(x)]
    except Exception:
        return []


def parse_plane_to_pids(
    final_planes_json: Path, debug: bool = False
) -> Dict[int, List[int]]:
    js = load_json(final_planes_json)

    if debug:
        keys = list(js.keys()) if isinstance(js, dict) else []
        print(f"[DEBUG] final_planes_json keys: {keys}")

    if isinstance(js, dict) and "plane_to_pids" in js:
        ptp = js["plane_to_pids"]
        if isinstance(ptp, dict):
            return {int(k): _as_int_list(v) for k, v in ptp.items()}
        if isinstance(ptp, list):
            out: Dict[int, List[int]] = {}
            for i, v in enumerate(ptp):
                out[int(i)] = _as_int_list(v)
            return out

    if isinstance(js, dict) and "planes" in js and isinstance(js["planes"], list):
        out: Dict[int, List[int]] = {}
        for p in js["planes"]:
            if not isinstance(p, dict):
                continue
            pid = p.get("plane_id", p.get("id", None))
            if pid is None:
                continue
            pid_i = int(pid)
            pids = (
                p.get("plane_to_pids", None)
                or p.get("pids", None)
                or p.get("point3D_ids", None)
                or p.get("point3d_ids", None)
            )
            out[pid_i] = _as_int_list(pids)
        if out:
            return out

    raise RuntimeError(
        f"Unsupported final_planes.json format: {final_planes_json} "
        f"(expected keys: plane_to_pids or planes)"
    )


def _counter_topk(c: Counter, k: int = 10) -> List[Tuple[Any, int]]:
    return [(a, int(b)) for a, b in c.most_common(k)]

def eval_consistency(
    scene: str,
    final_planes_json: Path,
    recon_root: Path,
    scene_out_dir: Path,
    render_dir: Path,
    out_dir: Path,
    majority_ratio: float,
    min_valid_obs: int,
    angle_deg_parallel: float,
    min_second_ratio: float,
    debug: bool,
    debug_k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    planes_json = render_dir / "planes_from_scan.json"
    render_maps_dir = render_dir / "render"
    if debug:
        print("\n[DEBUG] ===== PATHS =====")
        print(f"[DEBUG] scene: {scene}")
        print(
            f"[DEBUG] scene_out_dir: {scene_out_dir} (exists={scene_out_dir.exists()})"
        )
        print(
            f"[DEBUG] planes_from_scan.json: {planes_json} (exists={planes_json.exists()})"
        )
        print(f"[DEBUG] render_dir: {render_dir} (exists={render_dir.exists()})")
        print(
            f"[DEBUG] render/maps dir: {render_maps_dir} (exists={render_maps_dir.exists()})"
        )
        print(
            f"[DEBUG] final_planes_json: {final_planes_json} (exists={final_planes_json.exists()})"
        )
        print(f"[DEBUG] recon_root: {recon_root} (exists={recon_root.exists()})")

    # Load GT planes
    gt_planes = load_scan_planes(render_dir)
    if not gt_planes:
        raise RuntimeError(f"No GT planes loaded from: {planes_json}")

    gt_plane_ids = sorted(gt_planes.keys())
    if debug:
        print("\n[DEBUG] ===== GT PLANES =====")
        print(f"[DEBUG] num_gt_planes: {len(gt_planes)}")
        print(f"[DEBUG] gt_plane_id range: {gt_plane_ids[0]} .. {gt_plane_ids[-1]}")
        print(
            f"[DEBUG] sample gt_plane_ids: {gt_plane_ids[: min(debug_k, len(gt_plane_ids))]}"
        )

    # Load label maps
    label_maps = load_label_maps(render_maps_dir)
    if debug:
        print("\n[DEBUG] ===== LABEL MAPS =====")
        plane_png = getattr(label_maps, "plane_png", None)
        mask_png = getattr(label_maps, "mask_png", None)

        if isinstance(plane_png, dict):
            print(f"[DEBUG] label_maps.plane_png count: {len(plane_png)}")
            print(
                f"[DEBUG] plane_png sample keys: {list(plane_png.keys())[: min(debug_k, len(plane_png))]}"
            )
        else:
            print(f"[DEBUG] label_maps.plane_png not a dict (type={type(plane_png)})")

        if isinstance(mask_png, dict):
            print(f"[DEBUG] label_maps.mask_png count: {len(mask_png)}")
        else:
            if mask_png is not None:
                print(f"[DEBUG] label_maps.mask_png type: {type(mask_png)}")

    # Load PASfM recon
    model_dir = find_recon_model_dir(recon_root)
    pas = load_pycolmap_recon(model_dir)
    if debug:
        print("\n[DEBUG] ===== RECON =====")
        print(f"[DEBUG] recon model_dir: {model_dir}")
        print(f"[DEBUG] num_images (image_id_to_name): {len(pas.image_id_to_name)}")
        print(f"[DEBUG] num_points (xyz_by_pid): {len(pas.xyz_by_pid)}")
        print(f"[DEBUG] num_tracks (track_by_pid): {len(pas.track_by_pid)}")
        uv_lookup = getattr(pas, "uv_lookup", None)
        print(f"[DEBUG] uv_lookup type: {type(uv_lookup)}")

    # --- Assign pid -> GT plane_id (observation voting)
    if debug:
        print("\n[DEBUG] ===== MEMBERSHIP ASSIGNMENT =====")
        print(f"[DEBUG] majority_ratio={majority_ratio}, min_valid_obs={min_valid_obs}")

    ass_any = assign_points_to_planes(
        track_by_pid=pas.track_by_pid,
        image_id_to_name=pas.image_id_to_name,
        uv_lookup=pas.uv_lookup,
        label_maps=label_maps,
        majority_ratio=majority_ratio,
        min_valid_obs=min_valid_obs,
    )

    pid_to_ass: Dict[int, Any] = {}
    pid_dup = 0
    for a in ass_any.values():
        pid = int(getattr(a, "pid"))
        if pid in pid_to_ass:
            pid_dup += 1
        pid_to_ass[pid] = a

    if debug:
        print(f"[DEBUG] raw assignments dict size: {len(ass_any)}")
        print(
            f"[DEBUG] pid_to_ass size: {len(pid_to_ass)} (duplicate pid overwrites={pid_dup})"
        )

        plane_id_counts = Counter()
        valid_in_gt = 0
        invalid_in_gt = 0
        neg_or_none = 0
        for a in pid_to_ass.values():
            pl = getattr(a, "plane_id", None)
            if pl is None:
                neg_or_none += 1
                continue
            pl_i = int(pl)
            plane_id_counts[pl_i] += 1
            if pl_i < 0:
                neg_or_none += 1
            elif pl_i in gt_planes:
                valid_in_gt += 1
            else:
                invalid_in_gt += 1

        print(
            f"[DEBUG] plane_id distribution top10: {_counter_topk(plane_id_counts, 10)}"
        )
        print(f"[DEBUG] assigned plane_id in GT planes: {valid_in_gt}")
        print(f"[DEBUG] assigned plane_id NOT in GT planes: {invalid_in_gt}")
        print(f"[DEBUG] plane_id <0 or None: {neg_or_none}")

        sample_pids = list(pid_to_ass.keys())[: min(debug_k, len(pid_to_ass))]
        print(f"[DEBUG] sample pid_to_ass pids: {sample_pids}")
        for sp in sample_pids[: min(5, len(sample_pids))]:
            a = pid_to_ass[sp]
            print(
                f"[DEBUG]  pid={sp}, plane_id={getattr(a, 'plane_id', None)}, key={getattr(a, 'key', None)}"
            )

    # --- Load PAS global plane_to_pids
    plane_to_pids = parse_plane_to_pids(final_planes_json, debug=debug)

    if debug:
        print("\n[DEBUG] ===== PAS PLANE_TO_PIDS =====")
        sizes = [(int(pid), len(pids)) for pid, pids in plane_to_pids.items()]
        sizes.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] num_pas_planes: {len(plane_to_pids)}")
        print(f"[DEBUG] top planes by size (top5): {sizes[:5]}")

        if sizes:
            top_plane_id = sizes[0][0]
            sample_pids = plane_to_pids[top_plane_id][
                : min(debug_k, len(plane_to_pids[top_plane_id]))
            ]
            print(
                f"[DEBUG] sample pids from largest PAS plane {top_plane_id}: {sample_pids}"
            )

            in_ass = sum(1 for x in sample_pids if int(x) in pid_to_ass)
            in_xyz = sum(1 for x in sample_pids if int(x) in pas.xyz_by_pid)
            print(
                f"[DEBUG] sample pids present in pid_to_ass: {in_ass}/{len(sample_pids)}"
            )
            print(
                f"[DEBUG] sample pids present in xyz_by_pid: {in_xyz}/{len(sample_pids)}"
            )

    per_plane_rows: List[Dict[str, Any]] = []
    evaluated = 0

    total_points_eval = 0  # sum(num_total) over evaluated planes
    total_points_with_gt = 0  # sum(num_with_gt) over evaluated planes
    total_top1 = 0  # sum(top_cnt) over evaluated planes

    top_ratio_total_list: List[float] = []
    debug_zero_planes: List[Dict[str, Any]] = []

    for pas_plane_id, pids in tqdm(
        sorted(plane_to_pids.items(), key=lambda kv: int(kv[0])),
        desc=f"[{scene}] PAS plane → GT consistency",
    ):
        if not pids:
            continue

        missing_ass = 0
        neg_plane = 0
        not_in_gt = 0

        gt_hist = Counter()
        for pid in pids:
            pid_i = int(pid)
            a = pid_to_ass.get(pid_i)
            if a is None:
                missing_ass += 1
                continue

            pl = getattr(a, "plane_id", None)
            if pl is None:
                neg_plane += 1
                continue

            pl_i = int(pl)
            if pl_i < 0:
                neg_plane += 1
                continue

            if pl_i not in gt_planes:
                not_in_gt += 1
                continue

            gt_hist[pl_i] += 1

        num_total = len(pids)
        num_with_gt = int(sum(gt_hist.values()))
        num_unassigned = num_total - num_with_gt

        if num_with_gt <= 0:
            row = {
                "pas_plane_id": int(pas_plane_id),
                "num_points_total": int(num_total),
                "num_points_with_gt": 0,
                "num_points_unassigned_or_invalid": int(num_unassigned),
                "assigned_fraction": 0.0,
                "reasons": {
                    "missing_assignment_lookup": int(missing_ass),
                    "plane_id_negative_or_none": int(neg_plane),
                    "plane_id_not_in_gt_planes": int(not_in_gt),
                },
                "gt_hist_topk": [],
                "top_gt_plane_id": None,
                "top_count": None,
                "top_ratio_over_total": None,
                "top_ratio_over_with_gt": None,
                "second_gt_plane_id": None,
                "second_count": None,
                "second_ratio_over_total": None,
                "second_ratio_over_with_gt": None,
                "top2_angle_deg": None,
                "top2_parallel": None,
                "top2_d_abs_diff": None,
                "mixed_parallel_flag": None,
            }
            per_plane_rows.append(row)

            if debug and len(debug_zero_planes) < min(10, debug_k):
                sample = [int(x) for x in pids[: min(debug_k, len(pids))]]
                sample_in_ass = [x for x in sample if x in pid_to_ass]
                sample_plane_ids = []
                for x in sample_in_ass[: min(10, len(sample_in_ass))]:
                    sample_plane_ids.append(
                        {
                            "pid": x,
                            "plane_id": getattr(pid_to_ass[x], "plane_id", None),
                            "key": getattr(pid_to_ass[x], "key", None),
                        }
                    )
                debug_zero_planes.append(
                    {
                        "pas_plane_id": int(pas_plane_id),
                        "sample_pids": sample,
                        "sample_pids_found_in_pid_to_ass": sample_in_ass[
                            : min(20, len(sample_in_ass))
                        ],
                        "sample_found_plane_ids": sample_plane_ids,
                        "reasons": row["reasons"],
                    }
                )
            continue

        # evaluate
        evaluated += 1

        top2 = gt_hist.most_common(2)
        top_gt, top_cnt = top2[0]
        second_gt, second_cnt = top2[1] if len(top2) > 1 else (None, 0)

        top_ratio_over_total = float(top_cnt / num_total)
        # diagnostic
        top_ratio_over_with_gt = float(top_cnt / num_with_gt)

        second_ratio_over_total = (
            float(second_cnt / num_total) if second_gt is not None else 0.0
        )
        second_ratio_over_with_gt = (
            float(second_cnt / num_with_gt) if second_gt is not None else 0.0
        )

        assigned_fraction = float(num_with_gt / num_total)

        # parallel mix analysis (top1 vs top2)
        top2_parallel = None
        top2_angle = None
        top2_dd = None
        if second_gt is not None:
            n1 = gt_planes[top_gt].n
            n2 = gt_planes[second_gt].n
            top2_angle = _angle_deg(n1, n2)
            top2_parallel = bool(
                np.isfinite(top2_angle) and top2_angle <= angle_deg_parallel
            )

            d1 = float(gt_planes[top_gt].d)
            d2 = float(gt_planes[second_gt].d)
            top2_dd = float(abs(d1 - d2))

        mixed_parallel = bool(top2_parallel) and (
            second_ratio_over_total >= min_second_ratio
        )

        per_plane_rows.append(
            {
                "pas_plane_id": int(pas_plane_id),
                "num_points_total": int(num_total),
                "num_points_with_gt": int(num_with_gt),
                "num_points_unassigned_or_invalid": int(num_unassigned),
                "assigned_fraction": float(assigned_fraction),
                "reasons": {
                    "missing_assignment_lookup": int(missing_ass),
                    "plane_id_negative_or_none": int(neg_plane),
                    "plane_id_not_in_gt_planes": int(not_in_gt),
                },
                "gt_hist_topk": [
                    (int(pid), int(c)) for pid, c in gt_hist.most_common(10)
                ],
                "top_gt_plane_id": int(top_gt),
                "top_count": int(top_cnt),
                "top_ratio_over_total": float(top_ratio_over_total),
                "top_ratio_over_with_gt": float(top_ratio_over_with_gt),
                "second_gt_plane_id": (
                    int(second_gt) if second_gt is not None else None
                ),
                "second_count": int(second_cnt),
                "second_ratio_over_total": float(second_ratio_over_total),
                "second_ratio_over_with_gt": float(second_ratio_over_with_gt),
                "top2_angle_deg": top2_angle,
                "top2_parallel": top2_parallel,
                "top2_d_abs_diff": top2_dd,
                "mixed_parallel_flag": bool(mixed_parallel),
            }
        )

        # aggregate (over evaluated planes only)
        total_points_eval += int(num_total)
        total_points_with_gt += int(num_with_gt)
        total_top1 += int(top_cnt)
        top_ratio_total_list.append(float(top_ratio_over_total))

    # metrics
    mean_top_ratio_weighted = (
        float(total_top1 / total_points_eval) if total_points_eval > 0 else None
    )
    mean_top_ratio_unweighted = (
        float(np.mean(top_ratio_total_list)) if top_ratio_total_list else None
    )

    parallel_mix_fraction = (
        float(
            sum(1 for r in per_plane_rows if r.get("mixed_parallel_flag", False))
            / evaluated
        )
        if evaluated > 0
        else None
    )
    global_assigned_fraction = (
        float(total_points_with_gt / total_points_eval)
        if total_points_eval > 0
        else None
    )

    # Save
    summary = {
        "scene": scene,
        "paths": {
            "scene_out_dir": str(scene_out_dir),
            "planes_from_scan_json": str(planes_json),
            "render_dir": str(render_dir),
            "final_planes_json": str(final_planes_json),
            "recon_root": str(recon_root),
            "recon_model_dir": str(model_dir),
            "out_dir": str(out_dir),
        },
        "membership_params": {
            "majority_ratio": float(majority_ratio),
            "min_valid_obs": int(min_valid_obs),
        },
        "parallel_mix_params": {
            "angle_deg_parallel": float(angle_deg_parallel),
            "min_second_ratio_over_total": float(min_second_ratio),
        },
        "counts": {
            "num_gt_planes": int(len(gt_planes)),
            "num_pas_planes": int(len(plane_to_pids)),
            "num_pas_planes_evaluated": int(evaluated),
            "total_points_total_in_eval_planes": int(total_points_eval),
            "total_points_with_gt_in_eval_planes": int(total_points_with_gt),
        },
        "metrics": {
            "mean_top_ratio_over_total_unweighted": mean_top_ratio_unweighted,
            "mean_top_ratio_over_total_weighted": mean_top_ratio_weighted,
            "global_assigned_fraction": global_assigned_fraction,
            "parallel_mix_fraction": parallel_mix_fraction,
        },
    }

    # Debug report (very useful when evaluated=0)
    debug_report = None
    if debug:
        all_pas_pids: List[int] = []
        for _, pp in plane_to_pids.items():
            all_pas_pids.extend([int(x) for x in pp])
        all_pas_pids = list(dict.fromkeys(all_pas_pids))  # unique preserve order

        n_in_ass = sum(1 for x in all_pas_pids if x in pid_to_ass)
        n_in_xyz = sum(1 for x in all_pas_pids if x in pas.xyz_by_pid)

        valid_gt_cnt = 0
        neg_cnt = 0
        not_in_gt_cnt = 0
        for x in all_pas_pids:
            a = pid_to_ass.get(x)
            if a is None:
                continue
            pl = getattr(a, "plane_id", None)
            if pl is None or int(pl) < 0:
                neg_cnt += 1
                continue
            if int(pl) not in gt_planes:
                not_in_gt_cnt += 1
                continue
            valid_gt_cnt += 1

        debug_report = {
            "path_check": {
                "scene_out_dir_exists": scene_out_dir.exists(),
                "planes_from_scan_exists": planes_json.exists(),
                "render_dir_exists": render_dir.exists(),
                "render_maps_dir_exists": render_maps_dir.exists(),
                "final_planes_exists": final_planes_json.exists(),
                "recon_root_exists": recon_root.exists(),
                "recon_model_dir": str(model_dir),
                "recon_model_has_files": _has_colmap_model_files(model_dir),
            },
            "gt_planes": {
                "count": len(gt_planes),
                "id_min": gt_plane_ids[0] if gt_plane_ids else None,
                "id_max": gt_plane_ids[-1] if gt_plane_ids else None,
            },
            "label_maps": {
                "plane_png_count": (
                    len(getattr(label_maps, "plane_png", {}))
                    if isinstance(getattr(label_maps, "plane_png", None), dict)
                    else None
                ),
                "mask_png_count": (
                    len(getattr(label_maps, "mask_png", {}))
                    if isinstance(getattr(label_maps, "mask_png", None), dict)
                    else None
                ),
            },
            "recon": {
                "num_images": len(pas.image_id_to_name),
                "num_points_xyz": len(pas.xyz_by_pid),
                "num_tracks": len(pas.track_by_pid),
            },
            "assignment": {
                "raw_dict_len": len(ass_any),
                "pid_to_ass_len": len(pid_to_ass),
                "plane_id_top10": _counter_topk(
                    Counter(
                        int(getattr(a, "plane_id", -9999)) for a in pid_to_ass.values()
                    ),
                    10,
                ),
            },
            "pas_planes": {
                "num_pas_planes": len(plane_to_pids),
                "unique_pas_pids": len(all_pas_pids),
                "pids_in_pid_to_ass": f"{n_in_ass}/{len(all_pas_pids)}",
                "pids_in_xyz_by_pid": f"{n_in_xyz}/{len(all_pas_pids)}",
                "among_in_pid_to_ass": {
                    "valid_gt_plane_id": valid_gt_cnt,
                    "plane_id_negative_or_none": neg_cnt,
                    "plane_id_not_in_gt_planes": not_in_gt_cnt,
                },
            },
            "zero_eval_planes_examples": debug_zero_planes,
            "aggregation": {
                "evaluated_planes": int(evaluated),
                "total_points_eval": int(total_points_eval),
                "total_points_with_gt": int(total_points_with_gt),
                "total_top1": int(total_top1),
            },
        }

    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "per_plane.json", per_plane_rows)
    if debug_report is not None:
        save_json(out_dir / "debug.json", debug_report)

    print(f"\n[{scene}] GT plane consistency finished.")
    print(f"  out_dir: {out_dir}")
    print(f"  planes evaluated: {evaluated} / {len(plane_to_pids)}")
    print(f"  mean top-ratio over TOTAL (unweighted): {mean_top_ratio_unweighted}")
    print(f"  mean top-ratio over TOTAL (weighted):   {mean_top_ratio_weighted}")
    print(f"  global assigned fraction:               {global_assigned_fraction}")
    print(f"  parallel-mix fraction:                  {parallel_mix_fraction}")
    if debug:
        print(f"  debug report: {out_dir / 'debug.json'}")
    print("")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=str, required=True)

    # optional overrides
    p.add_argument("--final_planes_json", type=str, default=None)
    p.add_argument("--recon_dir", type=str, default=None)
    p.add_argument("--scene_out_dir", type=str, default=None)
    p.add_argument("--render_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    # membership params
    p.add_argument("--majority_ratio", type=float, default=0.6)
    p.add_argument("--min_valid_obs", type=int, default=3)

    # parallel mix params
    p.add_argument("--angle_deg_parallel", type=float, default=10.0)
    # NOTE: now interpreted as "ratio over TOTAL points", not "over assigned points"
    p.add_argument("--min_second_ratio", type=float, default=0.2)

    # debug
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_k", type=int, default=20)

    return p


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    scene = str(args.scene)

    # defaults (so that `--scene terrace` alone works)
    default_scene_out_dir = Path("outputs") / "PASfM" / scene
    default_final_planes = default_scene_out_dir / "final_planes.json"
    default_recon_root = default_scene_out_dir / "pasfm"
    default_render_dir = Path("datasets") / "eval_scan_plane" / scene
    default_out_dir = default_scene_out_dir / "gt_plane_consistency"

    scene_out_dir = (
        Path(args.scene_out_dir).expanduser()
        if args.scene_out_dir
        else default_scene_out_dir
    )
    final_planes_json = (
        Path(args.final_planes_json).expanduser()
        if args.final_planes_json
        else default_final_planes
    )
    recon_root = (
        Path(args.recon_dir).expanduser() if args.recon_dir else default_recon_root
    )
    render_dir = (
        Path(args.render_dir).expanduser() if args.render_dir else default_render_dir
    )
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir

    return final_planes_json, recon_root, scene_out_dir, render_dir, out_dir


def main() -> None:
    args = build_argparser().parse_args()

    final_planes_json, recon_root, scene_out_dir, render_dir, out_dir = resolve_paths(
        args
    )

    eval_consistency(
        scene=str(args.scene),
        final_planes_json=final_planes_json,
        recon_root=recon_root,
        scene_out_dir=scene_out_dir,
        render_dir=render_dir,
        out_dir=out_dir,
        majority_ratio=float(args.majority_ratio),
        min_valid_obs=int(args.min_valid_obs),
        angle_deg_parallel=float(args.angle_deg_parallel),
        min_second_ratio=float(args.min_second_ratio),
        debug=bool(args.debug),
        debug_k=int(args.debug_k),
    )


if __name__ == "__main__":
    main()
