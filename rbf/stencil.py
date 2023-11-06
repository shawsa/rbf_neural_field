"""
Stencil class for RBF interpolation.
"""

import numpy as np
import numpy.linalg as la
from .poly_utils import poly_basis_dim, poly_powers, poly_apply
from .rbf import RBF


class Stencil:
    def __init__(self, points: np.ndarray[float], validate=True):
        shape = points[0].shape
        if validate:
            shape = points[0].shape
            assert len(shape) == 1
            for point in points[1:]:
                assert point.shape == shape
        self.dim = shape[0]
        self.points = points

    def pairwise_diff(self) -> np.ndarray[float]:
        return [[la.norm(x - y) for x in self.points] for y in self.points]

    def dist_mat(self) -> np.ndarray[float]:
        return np.abs(self.pairwise_diff())

    def _rbf_mat(self, rbf: RBF):
        return rbf(self.dist_mat())

    def poly_mat(self, deg: int) -> np.ndarray[float]:
        num_cols = poly_basis_dim(self.dim, deg)
        P = np.ones((len(self.points), num_cols))
        for index, pows in enumerate(poly_powers(self.dim, max_deg=deg)):
            P[:, index] = poly_apply(self.points, pows)
        return P

    def _rbf_poly_mat(self, rbf: RBF, deg: int) -> np.ndarray[float]:
        P = self.poly_mat(deg=deg)
        num_poly = P.shape[1]
        return np.block(
            [[self._rbf_mat(rbf), P], [P.T, np.zeros((num_poly, num_poly))]]
        )

    def rbf_mat(self, rbf: RBF, deg: int = None) -> np.ndarray[float]:
        return self._rbf_poly_mat(rbf, deg=deg)

    def interpolation_weights(
        self, fs: np.ndarray, rbf: RBF, deg: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(fs) == len(self.points)
        ys = np.zeros(len(self.points) + poly_basis_dim(self.dim, deg))
        ys[: len(self.points)] = fs
        weights = la.solve(self.rbf_mat(rbf, deg), ys)
        rbf_weights = weights[: len(self.points)]
        poly_weights = weights[len(self.points) :]
        return rbf_weights, poly_weights
