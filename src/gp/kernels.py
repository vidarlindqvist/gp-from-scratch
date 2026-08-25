import numpy as np


class Kernel:
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RBF(Kernel):
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        if length_scale <= 0:
            raise ValueError("length_scale must be greater than 0")

        if variance < 0:
            raise ValueError("variance must be non-negative")

        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if X1.ndim != 2:
            raise ValueError("X1 must be a 2D array")

        if X2.ndim != 2:
            raise ValueError("X2 must be a 2D array")

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same number of features")

        squared_distances = self._squared_distance(X1, X2)
        return self.variance * np.exp(-squared_distances / (2 * self.length_scale**2))

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_expanded = np.expand_dims(X1, axis=1)
        X2_expanded = np.expand_dims(X2, axis=0)
        squared_differences = (X1_expanded - X2_expanded) ** 2
        return np.sum(squared_differences, axis=-1)
