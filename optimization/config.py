from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    snap_strength: float = 0.1
    dynamic_snap: bool = True
    snap_delta: float = 0.02
    snap_beta: float = 0.5
    snap_gamma: float = 1.2
    snap_alpha_min: float = 1e-6
    snap_alpha_max: float = 0.2
    snap_max_backtracks: int = 20

    use_ransak: bool = False
