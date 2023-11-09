from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
from .rbf import RBF
from .stencil import Stencil


class LinearOperator(ABC):
    def __init__(self, rbf: RBF):
        self.rbf = rbf

    @abstractmethod
    def rbf_op(self, r: float, d: np.ndarray[float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def poly_op(self, d: np.ndarray[float]) -> float:
        raise NotImplementedError


class OperatorStencil(Stencil):
    def __init__(
        self,
        center: np.ndarray[float],
        points: np.ndarray[float],
        validate=True,
    ):
        assert center in points
        super(Stencil, self).__init__(points, validate)
        self.center = center

    def weights(self, rbf: RBF, op: LinearOperator, poly_deg: int):
        pass
