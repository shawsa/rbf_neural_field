
import numpy as np
from rbf import RBF, PHS
from rbf.linear_operator import LinearOperator, OperatorStencil
from rbf.poly_utils import Monomial


class Derivative1D(LinearOperator):

    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        return d * rbf.dr_div_r(r)

    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        return poly.diff(d, 1)


points = np.array([0, 1, 2, -1, -2], dtype=float)
stencil = OperatorStencil(points)
rbf = PHS(3)
poly_deg = 2
weights = stencil.weights(rbf, Derivative1D(), poly_deg)
print(weights)


def foo(x):
    return np.sin(x)


def d_foo(x):
    return np.cos(x)

print(foo(points))
print(d_foo(points[0]))
print(weights@foo(points))
