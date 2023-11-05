"""
Classes for radial basis functions.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np


class RBF(ABC):
    @abstractmethod
    def __call__(r):
        ...


class PHS(RBF):

    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        self.deg = deg
        self.even = deg % 2 == 0

    def __call__(self, r):
        if self.even:
            ret = np.empty(r.shape)
            mask = ret == 0
            ret[mask] = 0
            ret[~mask] = r[~mask]**self.deg * np.log(r[~mask])
            return ret
        return r**self.deg
