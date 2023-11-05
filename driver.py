import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from rbf.rbf import PHS
from rbf.stencil import Stencil

# only working without polynomials right now

xs, ys = np.meshgrid(*2*(np.linspace(0, 1, 11),))
points = np.array([xs.ravel(), ys.ravel()]).T
rbf = PHS(3)
deg = None
stencil = Stencil(points)


def Frankes_function(x, y):
    return (
        .75*np.exp(-1/4*((9*x - 2)**2 + (9*y - 2)**2)) +
        .75*np.exp(-1/49*((9*x + 12)**2 + (9*y + 1)**2)) +
        .5*np.exp(-1/4*((9*x - 7)**2 + (9*y - 3)**2)) +
        .2*np.exp(-((9*x - 4)**2 + (9*y - 7)**2))
    )


xs_dense, ys_dense = np.meshgrid(*2*(np.linspace(0, 1, 201),))
Fs = Frankes_function(xs_dense, ys_dense)

plt.pcolormesh(xs_dense, ys_dense, Fs)
plt.plot(points[:, 0], points[:, 1], 'k.')


ys = Frankes_function(points[:, 0], points[:, 1])
# ys = np.zeros(len(points) + 3)
# ys[:-3] = Frankes_function(points[:, 0], points[:, 1])
weights = la.solve(stencil.rbf_mat(rbf, poly_deg=deg), ys)


def eval_f(x, y):
    z = np.array([x, y])
    return sum(w*rbf(la.norm(z - point)) for w, point in zip(weights, points))

eval_point = 0.21, 0.23
print(eval_f(*eval_point))
print(Frankes_function(*eval_point))

errors = np.empty(Fs.shape)
for index, (x, y) in tqdm(enumerate(zip(xs_dense.ravel(), ys_dense.ravel())), total=len(xs_dense.ravel())):
    errors.ravel()[index] = abs(eval_f(x, y) - Frankes_function(x, y))

print(np.max(errors))

plt.figure()
plt.pcolormesh(xs_dense, ys_dense, errors)
