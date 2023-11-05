"""
Stencil class for RBF interpolation.
"""

import numpy as np
import numpy.linalg as la
from .rbf import RBF


class Stencil:
    def __init__(self, points: np.ndarray[float], validate=True):
        if validate:
            shape = points[0].shape
            assert len(shape) == 1
            for point in points[1:]:
                assert point.shape == shape
        self.points = points

    def pairwise_diff(self) -> np.ndarray[float]:
        return [[la.norm(x-y) for x in self.points] for y in self.points]

    def dist_mat(self) -> np.ndarray[float]:
        return np.abs(self.pairwise_diff())

    def _rbf_mat(self, rbf: RBF):
        return rbf(self.dist_mat())

    def poly_mat(self, deg: int) -> np.ndarray[float]:
        # assume 2D for now. Fix this
        # aslo assume deg 1 for now
        assert self.points[0].shape == (2,)
        assert deg == 1
        P = np.ones((len(self.points), 3))
        P[:, 1] = self.points[:, 0]
        P[:, 2] = self.points[:, 1]
        return P

    def _rbf_poly_mat(self, rbf: RBF, deg: int) -> np.ndarray[float]:
        P = self.poly_mat(deg=deg)
        num_poly = P.shape[1]
        return np.block([
            [self._rbf_mat(rbf), P],
            [P.T, np.zeros((num_poly, num_poly))]
        ])

    def rbf_mat(self, rbf: RBF, poly_deg: int = None):
        if poly_deg is None:
            return self._rbf_mat(rbf)
        return self._rbf_poly_mat(rbf, deg=poly_deg)
