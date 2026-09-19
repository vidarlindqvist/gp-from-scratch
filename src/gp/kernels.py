"""Covariance functions, the prior over functions a Gaussian process starts from.

Takes two sets of input points, each shaped (n_points, n_features).
Returns the covariance between every pair, shaped (len(X1), len(X2)).

A kernel answers one question: how strongly should the function values at two
inputs be correlated? RBF and Matern 3/2 answer it with distance, so nearby
inputs give similar outputs. Linear answers it with the inner product, which
gives straight line fits instead. Every kernel carries a variance setting the
overall scale, and the two distance based ones carry a length scale setting how
far apart inputs must be before they stop informing each other.
"""

from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    """Anything that turns two sets of inputs into a covariance matrix.

    Subclasses implement __call__ and inherit the shared validation and
    distance helpers.
    """

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...

    def _validate_inputs(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """Both inputs must be 2D and agree on the number of features."""
        if X1.ndim != 2:
            raise ValueError("X1 must be a 2D array")

        if X2.ndim != 2:
            raise ValueError("X2 must be a 2D array")

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same number of features")

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared Euclidean distance between every pair of rows.

        Broadcasting does the pairing: X1 is reshaped to (n1, 1, features) and
        X2 to (1, n2, features), so subtracting them gives every pair at once
        with no Python loop. Summing over the last axis collapses the features.
        """
        X1_expanded = np.expand_dims(X1, axis=1)
        X2_expanded = np.expand_dims(X2, axis=0)
        squared_differences = (X1_expanded - X2_expanded) ** 2
        return np.sum(squared_differences, axis=-1)


class RBF(Kernel):
    """Radial basis function kernel, also called squared exponential.

        k(x, x') = variance * exp(-||x - x'||^2 / (2 * length_scale^2))

    The usual default. It is infinitely differentiable, so the posterior mean
    comes out very smooth. A small length_scale lets the fit wiggle between
    points, a large one flattens it towards a straight line.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        if length_scale <= 0:
            raise ValueError("length_scale must be greater than 0")

        if variance < 0:
            raise ValueError("variance must be non-negative")

        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        self._validate_inputs(X1, X2)
        squared_distances = self._squared_distance(X1, X2)
        return self.variance * np.exp(-squared_distances / (2 * self.length_scale**2))


class Linear(Kernel):
    """Linear kernel, k(x, x') = variance * (x . x').

    A GP with this kernel is Bayesian linear regression in disguise: the
    posterior mean is a straight line, so unlike the distance based kernels it
    keeps extrapolating instead of reverting to the prior away from the data.
    """

    def __init__(self, variance: float = 1.0) -> None:
        if variance < 0:
            raise ValueError("variance must be non-negative")

        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        self._validate_inputs(X1, X2)
        return self.variance * (X1 @ X2.T)


class Matern32(Kernel):
    """Matern kernel with nu = 3/2, where r is the distance ||x - x'||.

        k(r) = variance * (1 + sqrt(3) r / length_scale)
                        * exp(-sqrt(3) r / length_scale)

    It uses r rather than r^2, which leaves it only once differentiable and so
    rougher than RBF. That is usually the better assumption for real data,
    where RBF's smoothness is stronger than anything actually measured.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        if length_scale <= 0:
            raise ValueError("length_scale must be greater than 0")

        if variance < 0:
            raise ValueError("variance must be non-negative")

        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        self._validate_inputs(X1, X2)
        fraction = np.sqrt(3) * self._radius(X1, X2) / self.length_scale
        return self.variance * (1 + fraction) * np.exp(-fraction)

    def _radius(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Euclidean distance between every pair of rows. Matern needs r, not r^2."""
        return np.sqrt(self._squared_distance(X1, X2))
