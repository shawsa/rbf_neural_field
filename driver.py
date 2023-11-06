import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from rbf.rbf import PHS
from rbf.stencil import Stencil, poly_powers, poly_apply

# only working without polynomials right now

xs, ys = np.meshgrid(*2 * (np.linspace(0, 1, 11),))
points = np.array([xs.ravel(), ys.ravel()]).T
rbf = PHS(3)
deg = 2
stencil = Stencil(points)


# two test functions
def Frankes_function(x, y):
    """A common function to test multidimensional interpolation."""
    return (
        0.75 * np.exp(-1 / 4 * ((9 * x - 2) ** 2 + (9 * y - 2) ** 2))
        + 0.75 * np.exp(-1 / 49 * ((9 * x + 12) ** 2 + (9 * y + 1) ** 2))
        + 0.5 * np.exp(-1 / 4 * ((9 * x - 7) ** 2 + (9 * y - 3) ** 2))
        + 0.2 * np.exp(-((9 * x - 4) ** 2 + (9 * y - 7) ** 2))
    )


def poly_test(x, y):
    """A polynomial test function."""
    return 4 - 3 * x + 2 * y - x**2 + x * y - y**2


# test_func = poly_test
test_func = Frankes_function

xs_dense, ys_dense = np.meshgrid(*2 * (np.linspace(0, 1, 201),))
fs_dense = test_func(xs_dense, ys_dense)

plt.figure("Test Function")
plt.pcolormesh(xs_dense, ys_dense, fs_dense)
plt.plot(points[:, 0], points[:, 1], "k.")

fs = test_func(points[:, 0], points[:, 1])
rbf_weights, poly_weights = stencil.interpolation_weights(fs, rbf, deg)


def eval_f(x, y):
    z = np.array([x, y])
    return sum(
        w * rbf(la.norm(z - point)) for w, point in zip(rbf_weights, points)
    ) + sum(
        w * poly_apply(z, pows)
        for w, pows in zip(poly_weights, poly_powers(dim=2, max_deg=deg))
    )


eval_point = 0.21, 0.23
print(eval_f(*eval_point))
print(test_func(*eval_point))

errors = np.empty(ys_dense.shape)
for index, (x, y) in tqdm(
    enumerate(zip(xs_dense.ravel(), ys_dense.ravel())), total=len(xs_dense.ravel())
):
    errors.ravel()[index] = abs(eval_f(x, y) - test_func(x, y))

print(np.max(errors))

plt.figure()
plt.pcolormesh(xs_dense, ys_dense, errors)
