from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np

import pyceres
import pycolmap
import pycolmap.cost_functions as cf

from planes.plane_model import PlaneModel
from planes.refine_planes import refine_3d_plane_models
from planes.config import PlaneExtractionConfig
from planes.utils import save_planes_json
from optimization.utils import build_anchor_set, project_xyz_to_plane
from optimization.cost import PlaneDistanceCost
from optimization.config import OptimizationConfig
from optimization.score import compute_point_weights


def _make_ba_config(
    recon: pycolmap.Reconstruction,
    constant_pids: Optional[set[int]] = None,
) -> pycolmap.BundleAdjustmentConfig:
    cfg = pycolmap.BundleAdjustmentConfig()
    constant_pids = constant_pids or set()

    # optimize all registered images
    for image_id, img in recon.images.items():
        if img.has_pose:
            cfg.add_image(image_id)

    # optimize all points
    for pid in recon.points3D.keys():
        if pid in constant_pids:
            cfg.add_constant_point(pid)
        else:
            cfg.add_variable_point(pid)

    # fix 2 cameras
    cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

    return cfg


def _make_ba_options(
    max_num_iterations: int,
    loss: str = "CAUCHY",
) -> pycolmap.BundleAdjustmentOptions:
    opt = pycolmap.BundleAdjustmentOptions()
    # loss_function_type: TRIVIAL / SOFT_L1 / CAUCHY
    opt.loss_function_type = getattr(pycolmap.LossFunctionType, loss)
    opt.solver_options.max_num_iterations = int(max_num_iterations)
    opt.solver_options.minimizer_progress_to_stdout = True

    opt.print_summary = False
    opt.solver_options.minimizer_progress_to_stdout = False

    # use gpu
    opt.use_gpu = True
    opt.gpu_index = "0"

    return opt


def bundle_adjustment(
    recon: pycolmap.Reconstruction,
    plane_models: Optional[Dict[int, PlaneModel]] = None,
    plane_to_pids: Optional[Dict[int, List[int]]] = None,
    plane_lambda: float = 0.0,
    max_num_iterations: int = 50,
    constant_pids: Optional[set[int]] = None,
) -> pyceres.SolverSummary:
    """
    Run BA. If plane_models/plane_to_pids given and plane_lambda>0,
    add plane distance residuals for constrained points.
    """
    cfg = _make_ba_config(recon, constant_pids=constant_pids)
    opt = _make_ba_options(max_num_iterations=max_num_iterations)

    ba = pycolmap.create_default_bundle_adjuster(opt, cfg, recon)

    if plane_models is not None and plane_to_pids is not None and plane_lambda > 0:
        sqrt_w = float(np.sqrt(plane_lambda))
        prob = ba.problem  # ceres::Problem
        added = 0
        for gid, pids in plane_to_pids.items():
            if gid not in plane_models:
                continue
            pm = plane_models[gid]
            for pid in pids:
                if pid not in recon.points3D:
                    continue
                xyz = recon.points3D[pid].xyz
                if not (
                    isinstance(xyz, np.ndarray)
                    and xyz.shape == (3,)
                    and xyz.flags.writeable
                ):
                    raise RuntimeError(
                        f"points3D[{pid}].xyz is not a writable numpy view (needed for shared parameter block)."
                    )
                cost = PlaneDistanceCost(pm.n, pm.d, sqrt_w)
                prob.add_residual_block(cost, None, [xyz])
                added += 1
        print(f"[Plane-Aware BA] added residual blocks: {added}")

    summary = ba.solve()
    return summary


import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions as cf


def weighted_bundle_adjustment(
    recon: pycolmap.Reconstruction,
    weights,
    max_num_iterations: int = 50,
    constant_pids=None,
    fix_two_cams: bool = True,
    fix_intrinsics: bool = True,
    sigma0_px: float = 1.0,
    w_clip=(0.2, 5.0), 
    robust_cauchy: float | None = 1.0, 
):
    constant_pids = constant_pids or set()

    prob = pyceres.Problem()

    for cam_id, camera in recon.cameras.items():
        prob.add_parameter_block(camera.params, int(np.asarray(camera.params).size))
        if fix_intrinsics:
            prob.set_parameter_block_constant(camera.params)

    qbuf = {}  # img_id -> (4,) wxyz
    tbuf = {}  # img_id -> (3,)

    added_points = set()
    num_residuals = 0

    reg_img_ids = [iid for iid, img in recon.images.items() if img.has_pose]
    if len(reg_img_ids) == 0:
        return pyceres.SolverSummary()

    ws = (
        np.asarray(list(weights.values()), dtype=np.float64)
        if weights
        else np.array([], dtype=np.float64)
    )
    ws = ws[np.isfinite(ws)]
    w_med = float(np.median(ws)) if ws.size > 0 else 1.0
    if not np.isfinite(w_med) or w_med <= 0.0:
        w_med = 1.0

    anchor1 = reg_img_ids[0]
    anchor2 = reg_img_ids[1] if (fix_two_cams and len(reg_img_ids) > 1) else None

    anchor_pid = None

    loss = None
    if robust_cauchy is not None and hasattr(pyceres, "CauchyLoss"):
        loss = pyceres.CauchyLoss(float(robust_cauchy))

    for img_id in reg_img_ids:
        image = recon.images[img_id]
        camera = recon.cameras[image.camera_id]

        camera_model_id = camera.model

        c2w = image.cam_from_world()

        # quat: (x,y,z,w) -> (w,x,y,z)
        q_xyzw = np.asarray(c2w.rotation.quat, dtype=np.float64).reshape(4)
        q_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
        )
        q_wxyz /= np.linalg.norm(q_wxyz) + 1e-12

        t = np.asarray(c2w.translation, dtype=np.float64).reshape(3)

        qbuf[img_id] = np.ascontiguousarray(q_wxyz)
        tbuf[img_id] = np.ascontiguousarray(t)

        qvec = qbuf[img_id]
        tvec = tbuf[img_id]

        prob.add_parameter_block(qvec, 4)
        prob.add_parameter_block(tvec, 3)

        if hasattr(pyceres, "QuaternionManifold") and hasattr(prob, "set_manifold"):
            prob.set_manifold(qvec, pyceres.QuaternionManifold())

        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pid = int(p2d.point3D_id)
            if pid not in recon.points3D:
                continue

            p3d = recon.points3D[pid]

            if pid not in added_points:
                prob.add_parameter_block(p3d.xyz, 3)
                if pid in constant_pids:
                    prob.set_parameter_block_constant(p3d.xyz)
                added_points.add(pid)

                if anchor_pid is None and pid not in constant_pids:
                    anchor_pid = pid

            w = float(weights.get(pid, 1.0))
            if (not np.isfinite(w)) or w <= 0.0:
                w = 1.0

            w = w / w_med
            w = float(np.clip(w, w_clip[0], w_clip[1]))

            sigma2 = (sigma0_px * sigma0_px) / w
            cov = sigma2 * np.eye(2, dtype=np.float64)

            xy = np.asarray(p2d.xy, dtype=np.float64).reshape(2, 1)
            cost = cf.ReprojErrorCost(camera_model_id, cov, xy)

            prob.add_residual_block(cost, loss, [qvec, tvec, p3d.xyz, camera.params])
            num_residuals += 1

    prob.set_parameter_block_constant(qbuf[anchor1])
    prob.set_parameter_block_constant(tbuf[anchor1])

    if anchor2 is not None:
        prob.set_parameter_block_constant(qbuf[anchor2])
        prob.set_parameter_block_constant(tbuf[anchor2])

    if anchor_pid is not None and anchor_pid not in constant_pids:
        prob.set_parameter_block_constant(recon.points3D[anchor_pid].xyz)

    options = pyceres.SolverOptions()
    options.max_num_iterations = int(max_num_iterations)
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    for img_id in reg_img_ids:
        image = recon.images[img_id]
        q = qbuf[img_id]  # wxyz
        t = tbuf[img_id]

        # wxyz -> xyzw
        q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64).reshape(4, 1)
        rot = pycolmap.Rotation3d(q_xyzw)
        cam_from_world_new = pycolmap.Rigid3d(rot, t.reshape(3, 1))

        image.frame.set_cam_from_world(image.camera_id, cam_from_world_new)

    print(
        f"[Weighted BA] Added {num_residuals} residuals. w_med={w_med:.3g}, clip={w_clip}, sigma0={sigma0_px}"
    )
    return summary


def run_plane_aware_ba(
    recon: pycolmap.Reconstruction,
    sfm_dir: Path,
    save_path: Path,
    plane_models: Dict[int, PlaneModel],
    plane_to_pids: Dict[int, List[int]],
    total_ba_iters: int = 10,
    stage_ba_iters: int = 50,
    use_weighted_ba: bool = False,
    match_path: Path = None,
    vis_path: Path = None,
    plane_config: PlaneExtractionConfig = PlaneExtractionConfig(),
    opt_config: OptimizationConfig = OptimizationConfig(),
) -> Tuple[Dict[int, PlaneModel], Dict[int, List[int]]]:
    pm = plane_models
    p2p = plane_to_pids

    total_pts = sum(len(v) for v in plane_to_pids.values())
    print(
        f"\n[PASfM] initial_planes={len(plane_models)} initial_plane_points={total_pts}"
    )
    if len(plane_models) == 0:
        print("[PASfM] Skip optimization stage because no planes were found.")
        return pm, p2p

    if use_weighted_ba:
        weights = compute_point_weights(recon, match_path, vis_path)

    curr_alpha = opt_config.snap_strength
    for _ in range(total_ba_iters):
        if use_weighted_ba:
            s = weighted_bundle_adjustment(
                recon,
                weights=weights,
                max_num_iterations=stage_ba_iters,
            )
        else:
            s = bundle_adjustment(
                recon,
                plane_models=pm,
                plane_to_pids=p2p,
                plane_lambda=float(0.0),
                max_num_iterations=stage_ba_iters,
            )
        # print(s.BriefReport())

        pm, p2p, curr_alpha, accepted = refine_3d_plane_models(
            recon,
            plane_models=pm,
            plane_to_pids=p2p,
            thresh=plane_config.plane_ransac_thresh_m,
            confidence=plane_config.plane_ransac_conf,
            min_global_inliers=plane_config.min_global_inliers,
            max_iters=plane_config.plane_ransac_max_iters,
            use_ransak=opt_config.use_ransak,
            snap_strength=curr_alpha,
            dynamic_snap=opt_config.dynamic_snap,
            snap_delta=opt_config.snap_delta,
            snap_beta=opt_config.snap_beta,
            snap_gamma=opt_config.snap_gamma,
            snap_alpha_min=opt_config.snap_alpha_min,
            snap_alpha_max=opt_config.snap_alpha_max,
            snap_max_backtracks=opt_config.snap_max_backtracks,
        )

        if len(pm) == 0:
            print(
                "[PASfM] Skip optimization stage while refining the plane models because no planes were found."
            )
            return pm, p2p

        total_pts = sum(len(v) for v in p2p.values())
        print(
            f"[PASfM] (accepted {accepted}) alpha={curr_alpha:.3e}  planes={len(pm)} constrained_points={total_pts}"
        )

    s = bundle_adjustment(
        recon,
        plane_models=pm,
        plane_to_pids=p2p,
        plane_lambda=0.0,
        max_num_iterations=stage_ba_iters,
    )
    # print(s.BriefReport())

    recon.write(str(sfm_dir))
    result = {
        "plane_models": {
            int(pid): {
                "normal": pm[pid].n.tolist(),
                "d": float(pm[pid].d),
            }
            for pid in pm
        },
        "plane_to_pids": {int(pid): [int(x) for x in p2p[pid]] for pid in p2p},
    }
    save_planes_json(result, save_path)
    return pm, p2p


def run_plane_aware_ba_with_fixed_points(
    recon: pycolmap.Reconstruction,
    sfm_dir: Path,
    save_path: Path,
    plane_models: Dict[int, PlaneModel],
    plane_to_pids: Dict[int, List[int]],
    total_ba_iters: int = 10,
    stage_ba_iters: int = 50,
    use_weighted_ba: bool = False,
    match_path: Path = None,
    anchor_max_per_plane: int = 30,
    anchor_min_track_len: int = 4,
    anchor_ba_iters: int = 50,
    release_ba_iters: int = 50,
    plane_config: PlaneExtractionConfig = PlaneExtractionConfig(),
    opt_config: OptimizationConfig = OptimizationConfig(),
) -> Tuple[Dict[int, PlaneModel], Dict[int, List[int]]]:
    pm = plane_models
    p2p = plane_to_pids

    for _ in range(total_ba_iters):
        s = bundle_adjustment(
            recon,
            plane_models=pm,
            plane_to_pids=p2p,
            plane_lambda=0.0,
            max_num_iterations=stage_ba_iters,
        )
        # print(s.BriefReport())

        pm, p2p = refine_3d_plane_models(
            recon,
            plane_models=pm,
            plane_to_pids=p2p,
            thresh=plane_config.plane_ransac_thresh_m,
            min_global_inliers=plane_config.min_global_inliers,
            snap_strength=0.0,
        )

        plane_to_anchors = build_anchor_set(
            recon,
            plane_to_pids=p2p,
            max_anchors_per_plane=anchor_max_per_plane,
            min_track_len=anchor_min_track_len,
        )

        constant_pids: set[int] = set()
        for gid, anchors in plane_to_anchors.items():
            if gid not in pm:
                continue
            n = pm[gid].n
            d = float(pm[gid].d)
            alpha = opt_config.snap_strength
            for pid in anchors:
                if pid not in recon.points3D:
                    continue
                xyz = np.asarray(recon.points3D[pid].xyz, np.float64).reshape(3)
                recon.points3D[pid].xyz = project_xyz_to_plane(xyz, n, d, alpha=alpha)
                constant_pids.add(pid)

        s_anchor = bundle_adjustment(
            recon,
            plane_models=None,
            plane_to_pids=None,
            plane_lambda=0.0,
            max_num_iterations=anchor_ba_iters,
            constant_pids=constant_pids,
        )

        s_rel = bundle_adjustment(
            recon,
            plane_models=None,
            plane_to_pids=None,
            plane_lambda=0.0,
            max_num_iterations=release_ba_iters,
            constant_pids=None,
        )

        total_pts = sum(len(v) for v in p2p.values())
        print(
            f"[PASfM] anchors: {len(constant_pids)} planes={len(pm)} plane_points={total_pts}"
        )

    recon.write(str(sfm_dir))
    result = {
        "plane_models": {
            int(pid): {
                "normal": pm[pid].n.tolist(),
                "d": float(pm[pid].d),
            }
            for pid in pm
        },
        "plane_to_pids": {int(pid): [int(x) for x in p2p[pid]] for pid in p2p},
    }
    save_planes_json(result, save_path)
    return pm, p2p
