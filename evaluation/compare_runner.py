from __future__ import annotations
from pathlib import Path
from typing import List

from config import OUT_ROOT, SCAN_ROOT, SCAN_EVAL_DIRNAME, CompareConfig
from progress import tqdm
from utils import save_json
from compare_scene import run_scene_compare


def _discover_scenes() -> List[str]:
    scenes: List[str] = []
    if not SCAN_ROOT.exists():
        return scenes
    for d in sorted(SCAN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        scan_eval_dir = d / SCAN_EVAL_DIRNAME
        if scan_eval_dir.exists():
            scenes.append(d.name)
    return scenes


def run_all_scenes_compare() -> None:
    cfg = CompareConfig()
    scenes = _discover_scenes()

    save_json(
        OUT_ROOT / "compare_run_info.json",
        {
            "out_root": str(OUT_ROOT),
            "num_scenes": len(scenes),
            "scenes": scenes,
            "compare_cfg": cfg.__dict__,
        },
    )

    for scene in tqdm(scenes, desc="Compare (all scenes)", total=len(scenes)):
        run_scene_compare(scene, cfg)


if __name__ == "__main__":
    run_all_scenes_compare()
