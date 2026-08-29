import numpy as np

from gp.kernels import Kernel


class GaussianProcess:
    def __init__(self, kernel: Kernel, noise: float) -> None:
        if noise < 0:
            raise ValueError("noise must be non-negative")

        self.kernel = kernel
        self.noise = noise

    def fit(self, X, y):
        # validate training data
        # construct K(X, X)
        # add observation noise
        # factorize the covariance matrix
        # solve for the quantities needed during predicti, noise):
        # store kernel and noise
        # validate parameters
        ...

    def predict(self, X):
        # validate that the GP has been fitted
        # validate input dimensions
        # construct K(X, X_train)
        # calculate predictive variance
        # return predictions
        ...

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations")

    def _compute_covariance_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel(X1, X2)
