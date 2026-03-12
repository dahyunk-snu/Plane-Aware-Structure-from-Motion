from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from progress import tqdm
from label_maps import LabelMaps, _read_png


@dataclass(frozen=True)
class PointAssignment:
    pid: int
    plane_id: int
    majority: float
    n_obs: int
    key: str  # canonical key for matching across recons


def _canonical_key_from_track(
    track: List[Tuple[int, int]], image_id_to_name: Dict[int, str]
) -> Optional[str]:
    pairs: List[Tuple[str, int]] = []
    for img_id, p2d_idx in track:
        nm = image_id_to_name.get(img_id)
        if nm is None:
            continue
        pairs.append((nm, int(p2d_idx)))
    if not pairs:
        return None
    nm, idx = sorted(pairs)[0]
    return f"{nm}:{idx}"


def assign_points_to_planes(
    track_by_pid: Dict[int, List[Tuple[int, int]]],
    image_id_to_name: Dict[int, str],
    uv_lookup: Dict[Tuple[int, int], Tuple[float, float]],
    label_maps: LabelMaps,
    majority_ratio: float,
    min_valid_obs: int,
) -> Dict[int, PointAssignment]:
    """
    1) 모든 관측을 image_stem별로 모음 (obs_by_image)
    2) image_stem마다 plane_id.png / valid_mask.png를 딱 1번 로드
    3) 해당 이미지의 모든 (u,v) 관측을 한 번에 라벨 조회
    4) pid별 (plane_id 카운트, valid_obs 카운트) 누적
    5) majority vote로 최종 plane_id 결정
    """
    # --- 1) pid별 canonical key + image별 관측 모으기 ---
    pid_to_key: Dict[int, str] = {}
    obs_by_image: Dict[str, List[Tuple[int, int, int]]] = (
        {}
    )  # stem -> [(pid, ui, vi), ...]

    for pid, track in tqdm(
        track_by_pid.items(),
        desc="Collecting observations by image",
        total=len(track_by_pid),
    ):
        key = _canonical_key_from_track(track, image_id_to_name)
        if key is None:
            continue
        pid = int(pid)
        pid_to_key[pid] = key

        for img_id, p2d_idx in track:
            nm = image_id_to_name.get(img_id)
            if nm is None:
                continue
            stem = nm.rsplit(".", 1)[0]  # filename stem

            # 라벨맵이 없는 이미지는 스킵 (디스크 I/O 자체를 줄임)
            if stem not in label_maps.plane_png or stem not in label_maps.mask_png:
                continue

            uv = uv_lookup.get((img_id, p2d_idx))
            if uv is None:
                continue

            ui = int(round(uv[0]))
            vi = int(round(uv[1]))
            obs_by_image.setdefault(stem, []).append((pid, ui, vi))

    if not obs_by_image:
        return {}

    # --- 2) pid별 누적 카운터 (plane별 카운트 + valid obs 수) ---
    pid_valid_obs: Dict[int, int] = {}
    pid_plane_counts: Dict[int, Dict[int, int]] = {}

    # --- 3) 이미지별로 plane/mask를 1번만 로드해서 배치 조회 ---
    for stem, obs in tqdm(
        obs_by_image.items(),
        desc="Assigning points to scan planes (batched by image)",
        total=len(obs_by_image),
    ):
        plane_path = label_maps.plane_png.get(stem)
        mask_path = label_maps.mask_png.get(stem)
        if plane_path is None or mask_path is None:
            continue

        plane = _read_png(plane_path)  # uint16 (H,W) expected
        mask = _read_png(mask_path)  # uint8  (H,W) or (H,W,3)

        if mask.ndim == 3:
            mask2 = mask[:, :, 0]
        else:
            mask2 = mask

        H, W = plane.shape[0], plane.shape[1]

        arr = np.asarray(obs, dtype=np.int64)  # (M,3): pid, ui, vi
        pids = arr[:, 0].astype(np.int64)
        ui = arr[:, 1].astype(np.int64)
        vi = arr[:, 2].astype(np.int64)

        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(inb):
            continue

        pids = pids[inb]
        ui = ui[inb]
        vi = vi[inb]

        mval = mask2[vi, ui].astype(np.int64)
        pval = plane[vi, ui].astype(np.int64)  # 0=unknown, >0 => plane_id+1

        valid = (mval > 0) & (pval > 0)
        if not np.any(valid):
            continue

        pids_v = pids[valid]
        plane_id_v = (pval[valid] - 1).astype(np.int64)

        # (a) valid obs count per pid
        uniq_pids, cnts = np.unique(pids_v, return_counts=True)
        for pid_i, c in zip(uniq_pids.tolist(), cnts.tolist()):
            pid_valid_obs[pid_i] = pid_valid_obs.get(pid_i, 0) + int(c)

        # (b) plane_id count per pid (pair unique)
        pairs = np.stack([pids_v, plane_id_v], axis=1)
        uniq_pairs, pair_cnts = np.unique(pairs, axis=0, return_counts=True)
        for (pid_i, pl_i), c in zip(uniq_pairs.tolist(), pair_cnts.tolist()):
            d = pid_plane_counts.get(pid_i)
            if d is None:
                d = {}
                pid_plane_counts[pid_i] = d
            d[int(pl_i)] = d.get(int(pl_i), 0) + int(c)

    # --- 4) majority vote로 최종 assignment ---
    out: Dict[int, PointAssignment] = {}
    for pid, key in tqdm(
        pid_to_key.items(), desc="Finalizing majority vote", total=len(pid_to_key)
    ):
        nobs = pid_valid_obs.get(pid, 0)
        if nobs < min_valid_obs:
            continue
        counts = pid_plane_counts.get(pid)
        if not counts:
            continue

        plane_id, cnt = max(counts.items(), key=lambda kv: kv[1])
        maj = cnt / max(1, nobs)
        if maj < majority_ratio:
            continue

        out[pid] = PointAssignment(
            pid=int(pid),
            plane_id=int(plane_id),
            majority=float(maj),
            n_obs=int(nobs),
            key=key,
        )

    return out
