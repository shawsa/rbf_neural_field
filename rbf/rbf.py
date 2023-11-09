"""
Classes for radial basis functions.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np


class RBF(ABC):
    @abstractmethod
    def __call__(r):
        ...


class OddPHS(RBF):
    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        assert deg % 2 == 1
        self.deg = deg

    def __call__(self, r):
        return r**self.deg

    def dr(self, r):
        ####################################
        return 


class EvenPHS(RBF):
    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        self.deg = deg
        self.even = deg % 2 == 0

    def __call__(self, r):
        # account for removeable singularity at r = 0
        ret = np.empty(r.shape)
        mask = ret == 0
        ret[mask] = 0
        ret[~mask] = r[~mask]**self.deg * np.log(r[~mask])
        return ret


def PHS(deg: int) -> RBF:
    if deg % 2 == 0:
        return EvenPHS(deg)
    return OddPHS(deg)
