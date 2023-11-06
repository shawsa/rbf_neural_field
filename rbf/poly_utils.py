"""
Utilities for multidimensional polynomials.

RBF methods can included polynomial basis terms to increase accuracy.
Polynomial basis terms are also necessary to ensure invertability of
RBF matrices based on polyharmonic spline RBFs.
"""

from functools import cache, reduce
from math import comb
import numpy as np
import operator
from typing import Generator


PolyPowGen = Generator[tuple[int], None, None]


@cache
def poly_basis_dim(dim: int, deg: int) -> int:
    """
    The number of polynomial basis terms up to a given degree for a given
    dimension.
    """
    return comb(dim + deg, deg)


def poly_powers_of_deg(dim: int, deg: int) -> PolyPowGen:
    """
    Find all combinations of exponents of polynomials of the specified degree for
    the specified dimension. Return a generator that yields tuples of the exponents.
    """
    if dim == 1:
        yield (deg,)
    else:
        for d in range(deg + 1):
            yield from ((*pows, d) for pows in poly_powers_of_deg(dim - 1, deg - d))


def poly_powers_gen(dim: int, max_deg: int) -> PolyPowGen:
    """
    Find all combinations of exponents of polynomials of at most the specified degree
    for the specified dimension. Return a generator that yields tuples of the exponents.
    """
    for deg in range(max_deg + 1):
        yield from poly_powers_of_deg(dim, deg)


@cache
def poly_powers(dim: int, max_deg: int) -> tuple[tuple[int]]:
    """
    Cached version of poly_powers_gen that returns tuple.
    More efficient for repeated calls with same inputs.
    """
    return tuple(poly_powers_gen(dim, max_deg))


def poly_apply(points: np.ndarray[float], pows: tuple[int]) -> np.ndarray[float]:
    """
    Apply the polynomial powers to an array of points, pointwise.
    """
    return reduce(operator.mul, (x**p for x, p in zip(points.T, pows)))
