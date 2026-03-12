import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterable

from planes.plane_model import PlaneModel
import pycolmap
import math


def extract_image_name(key):
    key = key.split("/")[-1]
    key = key.split("-")[-1]
    return key


def _build_pid_to_plane_map(
    recon: pycolmap.Reconstruction,
    new_models: Dict[int, PlaneModel],
    new_plane_to_pids: Dict[int, List[int]],
) -> Dict[int, Tuple[np.ndarray, float]]:
    pid_to_plane: Dict[int, Tuple[np.ndarray, float]] = {}
    for gid, pids in new_plane_to_pids.items():
        model = new_models.get(gid, None)
        if model is None:
            continue
        n = np.asarray(model.n, dtype=np.float64).reshape(
            3,
        )
        d = float(model.d)
        for pid in pids:
            if pid in recon.points3D and pid not in pid_to_plane:
                pid_to_plane[pid] = (n, d)
    return pid_to_plane


def _mean_point_error(
    recon: pycolmap.Reconstruction,
    pids: Iterable[int],
) -> float:
    vals = []
    for pid in pids:
        if pid not in recon.points3D:
            continue
        e = float(recon.points3D[pid].error)
        if e >= 0.0 and (not math.isnan(e)) and (not math.isinf(e)):
            vals.append(e)
    return float(np.mean(vals)) if vals else float("nan")


def _apply_fixed_snap(
    recon: pycolmap.Reconstruction,
    pid_to_plane: Dict[int, Tuple[np.ndarray, float]],
    alpha: float,
) -> None:
    alpha = float(alpha)
    if alpha <= 0.0:
        return
    for pid, (n, d) in pid_to_plane.items():
        xyz = recon.points3D[pid].xyz
        r = float(n @ xyz + d)
        recon.points3D[pid].xyz = xyz - alpha * r * n


def _apply_dynamic_snap(
    recon: pycolmap.Reconstruction,
    pid_to_plane: Dict[int, Tuple[np.ndarray, float]],
    alpha_init: float,
    snap_delta: float,
    snap_beta: float,
    snap_alpha_min: float,
    snap_alpha_max: float,
    snap_max_backtracks: int,
) -> Tuple[bool, float]:
    """
    Find the largest alpha that does NOT worsen reprojection error too much,
    then apply snap once with that alpha.

    accept if new_err <= base_err * (1 + snap_delta)
    strategy:
      - if accepted -> try to grow alpha (alpha /= snap_beta) repeatedly
      - if rejected -> shrink alpha (alpha *= snap_beta) until accepted (or give up)
    returns: (accepted, best_alpha_or_last_alpha)
    """
    if len(pid_to_plane) == 0:
        return False, 0.0

    # clamp & early exit
    alpha = float(np.clip(alpha_init, 0.0, snap_alpha_max))
    if alpha <= snap_alpha_min:
        return False, alpha

    # baseline reprojection error
    try:
        recon.update_point_3d_errors()
    except Exception:
        print("[DEBUG] Fail to update point errors.")
        pass
    base_err = _mean_point_error(recon, pid_to_plane.keys())

    # if reproj error is not reliable -> fallback to fixed snap
    if isinstance(base_err, float) and math.isnan(base_err):
        print("[DEBUG] base_err is NaN. Fallback to fixed snap.")
        _apply_fixed_snap(recon, pid_to_plane, alpha)
        return True, alpha

    # save original xyz (we will evaluate candidates always from this baseline)
    old_xyz = {pid: recon.points3D[pid].xyz.copy() for pid in pid_to_plane.keys()}

    accepted = False
    best_alpha = 0.0

    # growth factor: inverse of shrink (e.g., beta=0.5 -> grow x2)
    grow = 1.0 / max(float(snap_beta), 1e-12)

    def _restore():
        for pid, xyz0 in old_xyz.items():
            recon.points3D[pid].xyz = xyz0.copy()

    # try up to N trials (reuse snap_max_backtracks as "max trials")
    for _ in range(int(snap_max_backtracks) + 1):
        if alpha <= snap_alpha_min:
            break

        # evaluate candidate alpha from baseline
        _restore()
        _apply_fixed_snap(recon, pid_to_plane, alpha)

        try:
            recon.update_point_3d_errors()
        except Exception:
            print("[DEBUG] base_err is NaN. Fallback to fixed snap.")
            pass
        new_err = _mean_point_error(recon, pid_to_plane.keys())

        if (not math.isnan(new_err)) and (
            new_err <= base_err * (1.0 + float(snap_delta))
        ):
            accepted = True
            best_alpha = float(alpha)

            # try to push alpha higher
            next_alpha = min(float(alpha) * float(grow), float(snap_alpha_max))
            if next_alpha <= alpha + 1e-12:
                break
            alpha = next_alpha
        else:
            # if we already found a feasible alpha, stop growing (best_alpha is maximal under our search)
            if accepted:
                break
            # otherwise, shrink and keep trying
            alpha *= float(snap_beta)

    # finally apply the best feasible alpha (once)
    _restore()
    if accepted and best_alpha > snap_alpha_min:
        _apply_fixed_snap(recon, pid_to_plane, best_alpha)
        try:
            recon.update_point_3d_errors()
        except Exception:
            print("[DEBUG] base_err is NaN. Fallback to fixed snap.")
            pass
        return True, float(best_alpha)

    # nothing feasible found
    return False, float(alpha)


def adjust_3d_points(
    recon: pycolmap.Reconstruction,
    new_models: Dict[int, PlaneModel],
    new_plane_to_pids: Dict[int, List[int]],
    snap_strength: float,
    dynamic_snap: bool,
    snap_delta: float,
    snap_beta: float,
    snap_gamma: float,
    snap_alpha_min: float,
    snap_alpha_max: float,
    snap_max_backtracks: int,
) -> None:
    """
    - dynamic_snap=False: fixed snap (alpha=snap_strength or snap_state['alpha'])
    - dynamic_snap=True : reproj-guarded dynamic snap
    """
    if snap_strength <= 0.0 or len(new_plane_to_pids) == 0:
        return

    pid_to_plane = _build_pid_to_plane_map(recon, new_models, new_plane_to_pids)
    if len(pid_to_plane) == 0:
        return

    alpha = float(np.clip(snap_strength, 0.0, snap_alpha_max))

    if not dynamic_snap:
        _apply_fixed_snap(recon, pid_to_plane, alpha)
        return alpha

    accepted, alpha_used = _apply_dynamic_snap(
        recon,
        pid_to_plane,
        alpha_init=alpha,
        snap_delta=snap_delta,
        snap_beta=snap_beta,
        snap_alpha_min=snap_alpha_min,
        snap_alpha_max=snap_alpha_max,
        snap_max_backtracks=snap_max_backtracks,
    )

    if accepted:
        return (
            float(min(alpha_used * float(snap_gamma), float(snap_alpha_max))),
            accepted,
        )
    else:
        return float(max(alpha_used, float(snap_alpha_min))), accepted


def jsonify(obj):
    """Make obj JSON-serializable (supports numpy, Path, dict keys)."""
    if obj is None:
        return None

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # mappings
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}

    # sequences
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]

    # sets
    if isinstance(obj, set):
        return [jsonify(v) for v in sorted(obj, key=lambda x: str(x))]

    return obj


def save_planes_json(result: dict, save_path: Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(jsonify(result), f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved plane results → {save_path}")


def load_planes_json(
    save_path: str,
) -> Tuple[Optional[Dict[int, PlaneModel]], Optional[Dict[int, List[int]]]]:
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

    if "plane_models" not in data or "plane_to_pids" not in data:
        return None, None

    try:
        plane_models_raw = data["plane_models"]
        plane_to_pids_raw = data["plane_to_pids"]

        plane_models: Dict[int, PlaneModel] = {}
        for pid, v in plane_models_raw.items():
            pid_i = int(pid)
            normal = np.asarray(v["normal"], dtype=np.float64).reshape(-1)
            if normal.size != 3:
                raise ValueError(f"plane_models[{pid}] normal size != 3")
            d = float(v["d"])
            plane_models[pid_i] = PlaneModel(n=normal, d=d)

        plane_to_pids: Dict[int, List[int]] = {
            int(pid): [int(x) for x in xs] for pid, xs in plane_to_pids_raw.items()
        }
    except Exception:
        return None, None

    return plane_models, plane_to_pids
