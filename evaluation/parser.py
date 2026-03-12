from __future__ import annotations
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET

import numpy as np


def _parse_transformtion_matrix(text: str) -> np.ndarray:
    # MeshLab MLP: <MLMatrix44> a b c ... </MLMatrix44>
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != 16:
        raise ValueError(f"[DEBUG] MLMatrix44 must have 16 floats, got {len(vals)}")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def parse_scan_alignment_mlp(mlp_path: Path) -> Dict[str, np.ndarray]:
    """
    Returns mapping: scan ply filename (basename) -> 4x4 matrix.
    Assumption: matrix converts local scan coords to global/aligned coords.
    """
    tree = ET.parse(str(mlp_path))
    root = tree.getroot()

    out: Dict[str, np.ndarray] = {}
    # Typical structure:
    # <MeshLabProject>
    #   <MeshGroup>
    #     <MLMesh label="scan1.ply" filename="scan1.ply">
    #       <MLMatrix44> ...16 floats... </MLMatrix44>
    #     </MLMesh>
    # ...
    for mlmesh in root.iter("MLMesh"):
        fname = mlmesh.attrib.get("filename")
        if fname is None:
            raise ValueError(f"[DEBUG] Missing filename in {mlp_path}")
        fname = Path(fname).name
        mat_node = mlmesh.find("MLMatrix44")
        if mat_node is None or mat_node.text is None:
            raise ValueError(f"[DEBUG] Missing MLMatrix44 for {fname} in {mlp_path}")
        out[fname] = _parse_transformtion_matrix(mat_node.text)

    if len(out) == 0:
        raise RuntimeError(f"[DEBUG] No MLMesh entries found in {mlp_path}")
    return out


def apply_transform(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    X: (N,3)
    M: (4,4)
    returns: (N,3) where X' = (M * [X,1])[:3]
    """
    if X.size == 0:
        return X
    ones = np.ones((X.shape[0], 1), dtype=np.float64)  # (N,1)
    Xh = np.concatenate([X.astype(np.float64), ones], axis=1)  # (N,4)
    Yh = Xh @ M.T  # (N,4)
    return Yh[:, :3]
