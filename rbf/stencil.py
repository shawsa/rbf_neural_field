"""
Stencil class for RBF interpolation.
"""

from functools import cache
import numpy as np
import numpy.linalg as la
from .poly_utils import poly_basis_dim, poly_powers, poly_apply
from .rbf import RBF


class Stencil:
    def __init__(
        self,
        points: np.ndarray[float],
        validate=True,
    ):
        shape = points[0].shape
        if validate:
            assert len(shape) == 1
            for point in points[1:]:
                assert point.shape == shape
        self.dim = shape[0]
        self.points = points

        self.upper_bounds = np.max(points, axis=0)
        self.lower_bounds = np.min(points, axis=0)

    @property
    @cache
    def num_points(self):
        return len(self.points)

    def shift_points(self, points: np.ndarray) -> np.ndarray:
        mids = (self.upper_bounds + self.lower_bounds) / 2
        widths = self.upper_bounds - self.lower_bounds
        return (points - mids) / widths

    @property
    @cache
    def scaled_points(self):
        return self.shift_points(self.points)

    @property
    def pairwise_diff(self) -> np.ndarray[float]:
        points = self.scaled_points
        return [[la.norm(x - y) for x in points] for y in points]

    @property
    def dist_mat(self) -> np.ndarray[float]:
        return np.abs(self.pairwise_diff)

    def rbf_mat(self, rbf: RBF):
        return rbf(self.dist_mat)

    def poly_mat(self, poly_deg: int) -> np.ndarray[float]:
        P = np.ones((self.num_points, poly_basis_dim(self.dim, poly_deg)))
        for index, pows in enumerate(poly_powers(self.dim, max_deg=poly_deg)):
            P[:, index] = poly_apply(self.scaled_points, pows)
        return P

    def interpolation_matrix(self, rbf: RBF, poly_deg: int = -1) -> np.ndarray[float]:
        P = self.poly_mat(poly_deg=poly_deg)
        basis_size = P.shape[1]
        zeros = np.zeros((basis_size, basis_size))
        return np.block([[self.rbf_mat(rbf=rbf), P], [P.T, zeros]])
