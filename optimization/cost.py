import pyceres
import numpy as np


class PlaneDistanceCost(pyceres.CostFunction):
    def __init__(self, n: np.ndarray, d: float, sqrt_w: float):
        super().__init__()
        n = np.asarray(n, dtype=np.float64).reshape(
            3,
        )
        n_norm = np.linalg.norm(n)
        if not np.isfinite(n_norm) or n_norm <= 0:
            raise ValueError("Plane normal is invalid.")
        self.n = n / n_norm
        self.d = float(d)
        self.sqrt_w = float(sqrt_w)

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3])

    def Evaluate(self, parameters, residuals, jacobians):
        # parameters[0] is a (3,) block
        X = parameters[0]
        r = self.n[0] * X[0] + self.n[1] * X[1] + self.n[2] * X[2] + self.d
        residuals[0] = self.sqrt_w * r

        if jacobians is not None:
            # jacobians[0] is row-major: (num_residuals x block_size) = (1 x 3)
            jacobians[0][0] = self.sqrt_w * self.n[0]
            jacobians[0][1] = self.sqrt_w * self.n[1]
            jacobians[0][2] = self.sqrt_w * self.n[2]
        return True
