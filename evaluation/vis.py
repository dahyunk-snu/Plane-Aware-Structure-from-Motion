from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def save_cdf_plot(out_png: Path, dc: np.ndarray, dp: np.ndarray, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for arr, lab in [(dc, "COLMAP"), (dp, "PASfM")]:
        if arr.size == 0:
            continue
        x = np.sort(arr)
        y = np.arange(1, x.size + 1) / x.size
        plt.plot(x, y, label=lab)
    plt.xlabel("Point-to-plane distance (m)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_delta_hist(out_png: Path, delta: np.ndarray, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if delta.size > 0:
        plt.hist(delta, bins=80)
    plt.xlabel("Δ = d_colmap - d_pasfm (m)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_tile_heatmaps(
    out_dir: Path,
    W: int,
    H: int,
    tile: int,
    uv: np.ndarray,  # (M,2) float
    err_c: np.ndarray,  # (M,)
    err_p: np.ndarray,  # (M,)
    delta: np.ndarray,  # (M,)
    title_prefix: str,
) -> None:
    import matplotlib.colors as mcolors

    out_dir.mkdir(parents=True, exist_ok=True)

    tw = (W + tile - 1) // tile
    th = (H + tile - 1) // tile

    def agg(values: np.ndarray) -> np.ndarray:
        s = np.zeros((th, tw), np.float64)
        c = np.zeros((th, tw), np.int32)
        ui = np.clip((uv[:, 0] / tile).astype(np.int32), 0, tw - 1)
        vi = np.clip((uv[:, 1] / tile).astype(np.int32), 0, th - 1)
        for u0, v0, val in zip(ui, vi, values):
            s[v0, u0] += float(val)
            c[v0, u0] += 1
        out = np.full((th, tw), np.nan, np.float64)
        m = c > 0
        out[m] = s[m] / c[m]
        return out

    M_c = agg(err_c)
    M_p = agg(err_p)
    M_d = agg(delta)

    def _p99(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 1.0
        return float(np.percentile(x, 99))

    vmax_c = max(1e-12, _p99(M_c))
    vmax_p = max(1e-12, _p99(M_p))

    d_abs = np.abs(M_d[np.isfinite(M_d)])
    max_abs = float(np.percentile(d_abs, 99)) if d_abs.size else 1.0
    max_abs = max(max_abs, 1e-12)

    cmap_c = plt.get_cmap("Reds").copy()
    cmap_p = plt.get_cmap("Blues").copy()

    cmap_d = mcolors.LinearSegmentedColormap.from_list(
        "rby", [(0.0, "red"), (0.5, "blue"), (1.0, "yellow")]
    )

    cmap_c.set_bad((0, 0, 0, 0))
    cmap_p.set_bad((0, 0, 0, 0))
    cmap_d.set_bad((0, 0, 0, 0))

    norm_c = mcolors.Normalize(vmin=0.0, vmax=vmax_c)
    norm_p = mcolors.Normalize(vmin=0.0, vmax=vmax_p)
    norm_d = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    maps = [
        ("err_colmap", M_c, cmap_c, norm_c),
        ("err_pasfm", M_p, cmap_p, norm_p),
        ("delta", M_d, cmap_d, norm_d),
    ]

    for name, M, cmap, norm in maps:
        plt.figure()
        Mm = np.ma.masked_invalid(M)
        plt.imshow(Mm, origin="upper", interpolation="nearest", cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title(f"{title_prefix} - {name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close()


def save_hist_with_mean_std(
    out_png: Path, arr: np.ndarray, title: str, xlabel: str
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if arr.size > 0:
        plt.hist(arr, bins=80)
        m = float(np.mean(arr))
        s = float(np.std(arr))
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_mean_std_bar(
    out_png: Path, mean_c: float, std_c: float, mean_p: float, std_p: float, title: str
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    labels = ["COLMAP", "PASfM"]
    means = [mean_c, mean_p]
    stds = [std_c, std_p]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel("Distance (m)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_per_plane_mean_std_bars(
    out_png: Path,
    plane_ids: List[int],
    mean_c: List[float],
    std_c: List[float],
    mean_p: List[float],
    std_p: List[float],
    title: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    x = np.arange(len(plane_ids))
    w = 0.4
    plt.bar(x - w / 2, mean_c, yerr=std_c, capsize=4, width=w, label="COLMAP")
    plt.bar(x + w / 2, mean_p, yerr=std_p, capsize=4, width=w, label="PASfM")
    plt.xticks(x, [str(pid) for pid in plane_ids])
    plt.xlabel("Plane ID")
    plt.ylabel("Distance (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
