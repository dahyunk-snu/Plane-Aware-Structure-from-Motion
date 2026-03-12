from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import (
    ETH3D_ROOT,
    OUT_ROOT,
    CALIB_DIRNAME,
    PRED_ROOTS,
    MODEL_FILES_BIN,
    CompareConfig,
)
from progress import tqdm
from utils import save_json, load_json, norm_image_name
from eval_io import load_images_txt, camera_center_from_pose
from label_maps import load_label_maps
from recon_io import load_pycolmap_recon
from sim3 import umeyama_sim3, rmse
from membership import assign_points_to_planes
from metrics import PlaneEq, point_plane_dist, summarize_dist
from vis import (
    save_cdf_plot,
    save_delta_hist,
    save_tile_heatmaps,
    save_hist_with_mean_std,
    save_mean_std_bar,
    save_per_plane_mean_std_bars,
)


def _has_model_files(d: Path) -> bool:
    if all((d / f).exists() for f in MODEL_FILES_BIN):
        return True
    return False


def _normalize_plane(n: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    n = np.asarray(n, np.float64).reshape(-1)
    nn = float(np.linalg.norm(n))
    if nn <= 0:
        return n, float("nan")
    return (n / nn), (float(d) / nn)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    # points: (N,3), returns unit normal n and offset d in n^T x + d = 0
    P = np.asarray(points, np.float64)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 3:
        return np.full(3, np.nan), float("nan")
    c = P.mean(axis=0)
    A = P - c
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    n = Vt[-1]
    n, d = _normalize_plane(n, -float(n @ c))
    return n, d


def find_recon_dir(scene: str, kind: str) -> Optional[Path]:
    """
    kind: "colmap" or "pasfm"
    heuristics:
      - check roots/<scene>/<kind>...
    """
    kind_lower = kind.lower()
    if not PRED_ROOTS.exists():
        print(f"[DEBUG] PRED_ROOTS not found at {PRED_ROOTS}.")
        return None

    p = PRED_ROOTS / scene / kind_lower
    if p.exists() and p.is_dir() and _has_model_files(p):
        return p
    else:
        print(f"[DEBUG] Reconstruction files not found in {p}.")
        return None


def load_scan_planes(scene_out_dir: Path) -> Dict[int, PlaneEq]:
    js = load_json(scene_out_dir / "planes_from_scan.json")
    planes = {}
    for p in js.get("planes", []):
        pid = int(p["plane_id"])
        n = np.array(p["n"], dtype=np.float64).reshape(3)
        n = n / (np.linalg.norm(n) + 1e-12)
        d = float(p["d"])
        planes[pid] = PlaneEq(n=n, d=d)
    return planes


def load_gt_centers(scene: str) -> Dict[str, np.ndarray]:
    images_txt = ETH3D_ROOT / scene / CALIB_DIRNAME / "images.txt"
    poses = load_images_txt(images_txt)
    out: Dict[str, np.ndarray] = {}
    for p in poses:
        out[norm_image_name(p.name)] = camera_center_from_pose(p)
    return out


def _build_center_pairs(
    gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    keys = sorted(set(gt.keys()) & set(pred.keys()))
    if len(keys) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    A = np.stack([pred[k] for k in keys], axis=0)
    B = np.stack([gt[k] for k in keys], axis=0)
    return A, B, keys


def run_scene_compare(scene: str, cfg: CompareConfig) -> None:
    scene_out_dir = OUT_ROOT / scene
    compare_dir = scene_out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    # prerequisites: scan planes + render maps
    planes_json = scene_out_dir / "planes_from_scan.json"
    render_dir = scene_out_dir / "render"
    if not planes_json.exists() or not render_dir.exists():
        save_json(
            compare_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "missing scan planes/render",
            },
        )
        return

    planes = load_scan_planes(scene_out_dir)
    if not planes:
        save_json(
            compare_dir / "status.json",
            {"scene": scene, "status": "skipped", "reason": "no planes"},
        )
        return

    label_maps = load_label_maps(render_dir)

    # find recon dirs
    col_dir = find_recon_dir(scene, "sfm")
    pas_dir = find_recon_dir(scene, "pasfm")
    if col_dir is None or pas_dir is None:
        print(f"[DEBUG] Reconstruction files not found in col/pasfm dirs.")
        save_json(
            compare_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "missing recon(s)",
                "found": {
                    "colmap": str(col_dir) if col_dir else None,
                    "pasfm": str(pas_dir) if pas_dir else None,
                },
            },
        )
        return

    # load recons
    col = load_pycolmap_recon(col_dir)
    pas = load_pycolmap_recon(pas_dir)

    # Sim3 align to GT (scan/global)
    gt_centers = load_gt_centers(scene)
    A_c, B_c, keys_c = _build_center_pairs(gt_centers, col.centers_by_name)
    A_p, B_p, keys_p = _build_center_pairs(gt_centers, pas.centers_by_name)
    if A_c.shape[0] < 3 or A_p.shape[0] < 3:
        save_json(
            compare_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "insufficient center matches",
            },
        )
        return

    sim_c = umeyama_sim3(A_c, B_c)
    sim_p = umeyama_sim3(A_p, B_p)

    rmse_c = rmse(sim_c.apply(A_c), B_c)
    rmse_p = rmse(sim_p.apply(A_p), B_p)

    save_json(
        compare_dir / "sim3.json",
        {
            "scene": scene,
            "colmap_dir": str(col_dir),
            "pasfm_dir": str(pas_dir),
            "num_center_pairs_colmap": int(A_c.shape[0]),
            "num_center_pairs_pasfm": int(A_p.shape[0]),
            "center_rmse_colmap": rmse_c,
            "center_rmse_pasfm": rmse_p,
            "sim3_colmap": {"s": sim_c.s, "R": sim_c.R.tolist(), "t": sim_c.t.tolist()},
            "sim3_pasfm": {"s": sim_p.s, "R": sim_p.R.tolist(), "t": sim_p.t.tolist()},
        },
    )

    # membership (observation-based, using scan label maps)
    ass_c = assign_points_to_planes(
        track_by_pid=col.track_by_pid,
        image_id_to_name=col.image_id_to_name,
        uv_lookup=col.uv_lookup,
        label_maps=label_maps,
        majority_ratio=cfg.majority_ratio,
        min_valid_obs=cfg.min_valid_obs,
    )
    ass_p = assign_points_to_planes(
        track_by_pid=pas.track_by_pid,
        image_id_to_name=pas.image_id_to_name,
        uv_lookup=pas.uv_lookup,
        label_maps=label_maps,
        majority_ratio=cfg.majority_ratio,
        min_valid_obs=cfg.min_valid_obs,
    )

    # key intersection for matching
    key_to_c = {a.key: a for a in ass_c.values()}
    key_to_p = {a.key: a for a in ass_p.values()}
    keys = sorted(set(key_to_c.keys()) & set(key_to_p.keys()))
    if not keys:
        save_json(
            compare_dir / "status.json",
            {"scene": scene, "status": "skipped", "reason": "no matched keys"},
        )
        return

    # build matched pairs, enforce same plane_id
    pid_pairs: List[Tuple[int, int, int]] = []
    for k in keys:
        ac = key_to_c[k]
        ap = key_to_p[k]
        if ac.plane_id != ap.plane_id:
            continue
        if ac.plane_id not in planes:
            continue
        pid_pairs.append((ac.pid, ap.pid, ac.plane_id))

    if not pid_pairs:
        save_json(
            compare_dir / "status.json",
            {
                "scene": scene,
                "status": "skipped",
                "reason": "no matched pairs after plane agreement",
            },
        )
        return

    # compute distances
    dc_all = []
    dp_all = []
    delta_all = []
    per_plane: Dict[int, Dict[str, list]] = {
        pid: {"dc": [], "dp": [], "delta": [], "Xc": [], "Xp": []}
        for pid in planes.keys()
    }

    for pid_c, pid_p, plane_id in tqdm(
        pid_pairs, desc=f"[{scene}] Computing plane distances", total=len(pid_pairs)
    ):
        Xc = col.xyz_by_pid.get(pid_c)
        Xp = pas.xyz_by_pid.get(pid_p)
        if Xc is None or Xp is None:
            continue
        Xc_g = sim_c.apply(Xc.reshape(1, 3))[0]
        Xp_g = sim_p.apply(Xp.reshape(1, 3))[0]
        pe = planes[plane_id]
        dc = float(abs(pe.n @ Xc_g + pe.d))
        dp = float(abs(pe.n @ Xp_g + pe.d))
        delta = dc - dp

        dc_all.append(dc)
        dp_all.append(dp)
        delta_all.append(delta)
        if plane_id in per_plane:
            per_plane[plane_id]["dc"].append(dc)
            per_plane[plane_id]["dp"].append(dp)
            per_plane[plane_id]["delta"].append(delta)
            per_plane[plane_id]["Xc"].append(Xc_g)
            per_plane[plane_id]["Xp"].append(Xp_g)

    dc_all = np.asarray(dc_all, np.float64)
    dp_all = np.asarray(dp_all, np.float64)
    delta_all = np.asarray(delta_all, np.float64)

    # metrics
    tau = cfg.tau_list_m
    per_plane_metrics = {}
    for pid, arrs in per_plane.items():
        dc = np.asarray(arrs["dc"], np.float64)
        dp = np.asarray(arrs["dp"], np.float64)
        delta = np.asarray(arrs["delta"], np.float64)

        pe_gt = planes[pid]
        n_gt_u, d_gt_u = _normalize_plane(pe_gt.n, pe_gt.d)

        Xc_pts = np.asarray(arrs["Xc"], np.float64)  # (Nc,3)
        Xp_pts = np.asarray(arrs["Xp"], np.float64)  # (Np,3)

        n_c_u, d_c_u = _fit_plane_svd(Xc_pts)
        n_p_u, d_p_u = _fit_plane_svd(Xp_pts)

        # align fitted normals to GT normal (avoid sign ambiguity)
        if (
            np.all(np.isfinite(n_c_u))
            and np.all(np.isfinite(n_gt_u))
            and float(n_c_u @ n_gt_u) < 0
        ):
            n_c_u, d_c_u = -n_c_u, -d_c_u
        if (
            np.all(np.isfinite(n_p_u))
            and np.all(np.isfinite(n_gt_u))
            and float(n_p_u @ n_gt_u) < 0
        ):
            n_p_u, d_p_u = -n_p_u, -d_p_u

        cos_c = (
            float(np.clip(n_c_u @ n_gt_u, -1.0, 1.0))
            if np.all(np.isfinite(n_c_u))
            else float("nan")
        )
        cos_p = (
            float(np.clip(n_p_u @ n_gt_u, -1.0, 1.0))
            if np.all(np.isfinite(n_p_u))
            else float("nan")
        )
        dd_c = (
            float(abs(d_c_u - d_gt_u))
            if np.isfinite(d_c_u) and np.isfinite(d_gt_u)
            else float("nan")
        )
        dd_p = (
            float(abs(d_p_u - d_gt_u))
            if np.isfinite(d_p_u) and np.isfinite(d_gt_u)
            else float("nan")
        )

        per_plane_metrics[pid] = {
            "gt_plane": {
                "n_unit": n_gt_u.tolist(),
                "d_unit": float(d_gt_u),
            },
            "colmap_plane_fit": {
                "n_unit": n_c_u.tolist(),
                "d_unit": float(d_c_u),
                "cosine_to_gt": cos_c,
                "abs_d_diff_to_gt": dd_c,
                "num_points": int(Xc_pts.shape[0]) if Xc_pts.ndim == 2 else 0,
            },
            "pasfm_plane_fit": {
                "n_unit": n_p_u.tolist(),
                "d_unit": float(d_p_u),
                "cosine_to_gt": cos_p,
                "abs_d_diff_to_gt": dd_p,
                "num_points": int(Xp_pts.shape[0]) if Xp_pts.ndim == 2 else 0,
            },
            "colmap": summarize_dist(dc, tau),
            "pasfm": summarize_dist(dp, tau),
            "delta": {
                "n": int(delta.size),
                "median": (
                    float(np.percentile(delta, 50)) if delta.size else float("nan")
                ),
                "p90": float(np.percentile(delta, 90)) if delta.size else float("nan"),
                "improve_rate": (
                    float(np.mean(delta > 0)) if delta.size else float("nan")
                ),
            },
        }

    # --- add: cosine sim summary (plane-fit normal vs GT) ---
    cos_c_list = []
    cos_p_list = []
    dd_c_abs_list = []
    dd_p_abs_list = []

    for m in per_plane_metrics.values():
        cc = m.get("colmap_plane_fit", {}).get("cosine_to_gt", float("nan"))
        cp = m.get("pasfm_plane_fit", {}).get("cosine_to_gt", float("nan"))
        d_gt = m.get("gt_plane", {}).get("d_unit", float("nan"))
        d_c = m.get("colmap_plane_fit", {}).get("d_unit", float("nan"))
        d_p = m.get("pasfm_plane_fit", {}).get("d_unit", float("nan"))

        if np.isfinite(cc):
            cos_c_list.append(float(cc))
        if np.isfinite(cp):
            cos_p_list.append(float(cp))
        if np.isfinite(d_gt) and np.isfinite(d_c):
            dd_c_abs_list.append(float(abs(d_c - d_gt)))
        if np.isfinite(d_gt) and np.isfinite(d_p):
            dd_p_abs_list.append(float(abs(d_p - d_gt)))

    cos_c = np.asarray(cos_c_list, np.float64)
    cos_p = np.asarray(cos_p_list, np.float64)
    dd_c_abs = np.asarray(dd_c_abs_list, np.float64)
    dd_p_abs = np.asarray(dd_p_abs_list, np.float64)

    cos_c_mean = float(np.mean(cos_c)) if cos_c.size else float("nan")
    cos_p_mean = float(np.mean(cos_p)) if cos_p.size else float("nan")
    dd_c_mean = float(np.mean(dd_c_abs)) if dd_c_abs.size else float("nan")
    dd_p_mean = float(np.mean(dd_p_abs)) if dd_p_abs.size else float("nan")

    # 표준편차는 n=0이면 NaN, n=1이면 0.0으로 두는 방식
    cos_c_std = float(np.std(cos_c, ddof=0)) if cos_c.size else float("nan")
    cos_p_std = float(np.std(cos_p, ddof=0)) if cos_p.size else float("nan")
    dd_c_std = float(np.std(dd_c_abs, ddof=0)) if dd_c_abs.size else float("nan")
    dd_p_std = float(np.std(dd_p_abs, ddof=0)) if dd_p_abs.size else float("nan")

    summary = {
        "scene": scene,
        "matched_pairs": int(len(pid_pairs)),
        "colmap": summarize_dist(dc_all, tau),
        "pasfm": summarize_dist(dp_all, tau),
        "delta": {
            "n": int(delta_all.size),
            "median": (
                float(np.percentile(delta_all, 50)) if delta_all.size else float("nan")
            ),
            "p90": (
                float(np.percentile(delta_all, 90)) if delta_all.size else float("nan")
            ),
            "improve_rate": (
                float(np.mean(delta_all > 0)) if delta_all.size else float("nan")
            ),
        },
        "normal_cosine_to_gt": {
            "colmap": {"n": int(cos_c.size), "mean": cos_c_mean, "std": cos_c_std},
            "pasfm": {"n": int(cos_p.size), "mean": cos_p_mean, "std": cos_p_std},
        },
        "plane_offset_abs_diff_to_gt": {
            "colmap": {"n": int(dd_c_abs.size), "mean": dd_c_mean, "std": dd_c_std},
            "pasfm": {"n": int(dd_p_abs.size), "mean": dd_p_mean, "std": dd_p_std},
        },
    }

    save_json(compare_dir / "metrics_summary.json", summary)
    save_json(compare_dir / "metrics_per_plane.json", per_plane_metrics)

    # visualizations: CDF + delta hist
    plots_dir = compare_dir / "plots"
    save_cdf_plot(
        plots_dir / "cdf_all.png", dc_all, dp_all, title=f"{scene} - CDF (all planes)"
    )
    save_delta_hist(
        plots_dir / "delta_hist_all.png",
        delta_all,
        title=f"{scene} - Δ histogram (all planes)",
    )
    if dc_all.size > 0 and dp_all.size > 0:
        save_hist_with_mean_std(
            plots_dir / "hist_all_colmap.png",
            dc_all,
            title=f"{scene} - COLMAP dist hist (mean±std)",
            xlabel="Point-to-plane distance (m)",
        )
        save_hist_with_mean_std(
            plots_dir / "hist_all_pasfm.png",
            dp_all,
            title=f"{scene} - PASfM dist hist (mean±std)",
            xlabel="Point-to-plane distance (m)",
        )

        save_mean_std_bar(
            plots_dir / "mean_std_all.png",
            mean_c=float(np.mean(dc_all)),
            std_c=float(np.std(dc_all)),
            mean_p=float(np.mean(dp_all)),
            std_p=float(np.std(dp_all)),
            title=f"{scene} - Mean ± Std (all planes)",
        )

    if delta_all.size > 0:
        save_hist_with_mean_std(
            plots_dir / "hist_all_delta.png",
            delta_all,
            title=f"{scene} - Δ hist (mean±std), Δ=d_colmap-d_pasfm",
            xlabel="Δ (m)",
        )

    plane_stats = []
    for pid, arrs in per_plane.items():
        n = len(arrs["dc"])
        if n <= 0:
            continue
        plane_stats.append((pid, n))
    plane_stats.sort(key=lambda x: x[1], reverse=True)

    topK = min(6, len(plane_stats))
    if topK > 0:
        top_ids = [pid for pid, _ in plane_stats[:topK]]
        mean_c = [
            float(np.mean(np.asarray(per_plane[pid]["dc"], np.float64)))
            for pid in top_ids
        ]
        std_c = [
            float(np.std(np.asarray(per_plane[pid]["dc"], np.float64)))
            for pid in top_ids
        ]
        mean_p = [
            float(np.mean(np.asarray(per_plane[pid]["dp"], np.float64)))
            for pid in top_ids
        ]
        std_p = [
            float(np.std(np.asarray(per_plane[pid]["dp"], np.float64)))
            for pid in top_ids
        ]

        save_per_plane_mean_std_bars(
            plots_dir / "mean_std_top_planes.png",
            plane_ids=top_ids,
            mean_c=mean_c,
            std_c=std_c,
            mean_p=mean_p,
            std_p=std_p,
            title=f"{scene} - Mean±Std (top {topK} planes by samples)",
        )

    for pid, arrs in per_plane.items():
        dc = np.asarray(arrs["dc"], np.float64)
        dp = np.asarray(arrs["dp"], np.float64)
        if dc.size == 0 or dp.size == 0:
            continue
        save_cdf_plot(
            plots_dir / f"cdf_plane_{pid}.png",
            dc,
            dp,
            title=f"{scene} - CDF (plane {pid})",
        )

    from label_maps import _read_png  # 내부 PNG 로더

    # 1) matched pid_c에 대해 (plane_id, dc, dp, delta) 미리 계산
    pidc_info: Dict[int, Tuple[int, float, float, float]] = {}
    for pid_c, pid_p, plane_id in pid_pairs:
        Xc = col.xyz_by_pid.get(pid_c)
        Xp = pas.xyz_by_pid.get(pid_p)
        if Xc is None or Xp is None:
            continue
        Xc_g = sim_c.apply(Xc.reshape(1, 3))[0]
        Xp_g = sim_p.apply(Xp.reshape(1, 3))[0]
        pe = planes.get(plane_id)
        if pe is None:
            continue
        dc = float(abs(pe.n @ Xc_g + pe.d))
        dp = float(abs(pe.n @ Xp_g + pe.d))
        de = dc - dp
        pidc_info[pid_c] = (plane_id, dc, dp, de)

    # 2) 관측을 이미지별로 모으기: stem -> [(ui,vi,plane_id,dc,dp,delta), ...]
    obs_by_stem: Dict[str, List[Tuple[int, int, int, float, float, float]]] = {}
    for pid_c, track in tqdm(
        col.track_by_pid.items(),
        desc=f"[{scene}] Collect obs for heatmaps",
        total=len(col.track_by_pid),
    ):
        info = pidc_info.get(pid_c)
        if info is None:
            continue
        plane_id, dc, dp, de = info

        for img_id, p2d_idx in track:
            nm = col.image_id_to_name.get(img_id)
            if nm is None:
                continue
            stem = nm.rsplit(".", 1)[0]
            if stem not in label_maps.plane_png or stem not in label_maps.mask_png:
                continue
            uv = col.uv_lookup.get((img_id, p2d_idx))
            if uv is None:
                continue
            ui = int(round(uv[0]))
            vi = int(round(uv[1]))
            obs_by_stem.setdefault(stem, []).append((ui, vi, plane_id, dc, dp, de))

    # 3) stem마다 PNG 1회 로드 후 벡터화로 필터링 + top-N 선정
    per_img_obs: Dict[str, List[Tuple[float, float, float, float, float]]] = {}
    img_count: Dict[str, int] = {}

    for stem, obs in tqdm(
        obs_by_stem.items(),
        desc=f"[{scene}] Filter obs by scan label (batched)",
        total=len(obs_by_stem),
    ):
        plane_path = label_maps.plane_png.get(stem)
        mask_path = label_maps.mask_png.get(stem)
        if plane_path is None or mask_path is None:
            continue

        plane = _read_png(plane_path)  # (H,W) uint16
        mask = _read_png(mask_path)  # (H,W) uint8 or (H,W,3)
        mask2 = mask[:, :, 0] if mask.ndim == 3 else mask

        H, W = plane.shape[0], plane.shape[1]
        arr = np.asarray(obs, dtype=np.float64)  # (M,6)

        ui = arr[:, 0].astype(np.int64)
        vi = arr[:, 1].astype(np.int64)
        pl = arr[:, 2].astype(np.int64)
        dc = arr[:, 3]
        dp = arr[:, 4]
        de = arr[:, 5]

        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(inb):
            continue

        ui = ui[inb]
        vi = vi[inb]
        pl = pl[inb]
        dc = dc[inb]
        dp = dp[inb]
        de = de[inb]

        # scan 커버리지(valid mask) + plane 일치(png 값==plane_id+1) 필터
        mval = mask2[vi, ui] > 0
        pval = plane[vi, ui].astype(np.int64)
        ok = mval & (pval == (pl + 1))
        if not np.any(ok):
            continue

        ui = ui[ok].astype(np.float64)
        vi = vi[ok].astype(np.float64)
        dc = dc[ok]
        dp = dp[ok]
        de = de[ok]

        per_img_obs[stem] = list(
            zip(ui.tolist(), vi.tolist(), dc.tolist(), dp.tolist(), de.tolist())
        )
        img_count[stem] = int(len(ui))

    top = sorted(img_count.items(), key=lambda kv: kv[1], reverse=True)[
        : cfg.max_images_2d_maps
    ]
    heat_dir = compare_dir / "heatmaps"

    for stem, _cnt in top:
        obs = per_img_obs.get(stem, [])
        if len(obs) < 50:
            continue
        uv = np.array([[o[0], o[1]] for o in obs], np.float64)
        ec = np.array([o[2] for o in obs], np.float64)
        ep = np.array([o[3] for o in obs], np.float64)
        dl = np.array([o[4] for o in obs], np.float64)

        # image size from label png
        plane_png = label_maps.plane_png.get(stem)
        if plane_png is None:
            continue
        img = _read_png(plane_png)
        H, W = img.shape[0], img.shape[1]

        save_tile_heatmaps(
            out_dir=heat_dir / stem,
            W=W,
            H=H,
            tile=cfg.tile_size_px,
            uv=uv,
            err_c=ec,
            err_p=ep,
            delta=dl,
            title_prefix=f"{scene}/{stem}",
        )
    save_json(compare_dir / "status.json", {"scene": scene, "status": "ok"})
