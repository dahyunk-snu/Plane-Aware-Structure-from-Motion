from pathlib import Path
import pycolmap


def _find_recon_dir(sfm_dir: Path) -> Path | None:
    sfm_dir = Path(sfm_dir)

    candidates = [sfm_dir / "0", sfm_dir]
    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        has_bin = (
            (d / "cameras.bin").exists()
            and (d / "images.bin").exists()
            and (d / "points3D.bin").exists()
        )
        has_txt = (
            (d / "cameras.txt").exists()
            and (d / "images.txt").exists()
            and (d / "points3D.txt").exists()
        )
        if has_bin or has_txt:
            return d
    return None


def load_model(sfm_dir: Path):
    model_dir = _find_recon_dir(sfm_dir)
    if model_dir is not None:
        print(f"[PASfM] Found existing sparse model → {model_dir}")
        return pycolmap.Reconstruction(str(model_dir))

    return None
