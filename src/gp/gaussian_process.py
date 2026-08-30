import numpy as np

from gp.kernels import Kernel


class GaussianProcess:
    def __init__(self, kernel: Kernel, noise: float) -> None:
        if noise < 0:
            raise ValueError("noise must be non-negative")

        self.kernel = kernel
        self.noise = noise
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_training_data(X, y)
        self.X_train = X
        self.y_train = y
        self.K_train = self._compute_covariance_matrix(self.X_train, self.X_train)
        self.K_y = self.K_train + self.noise * np.eye(self.K_train.shape[0])
        self.L = np.linalg.cholesky(self.K_y)
        v = _forward_substitution(self.L, self.y_train)
        self.alpha = _backward_substitution(self.L.T, v)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("model must be fitted")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "X and training data must have the same number of features"
            )

        K_star = self._compute_covariance_matrix(X, self.X_train)
        K_star_star = self._compute_covariance_matrix(X, X)
        V = _forward_substitution(self.L, K_star.T)
        return K_star @ self.alpha, K_star_star - V.T @ V

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations")

    def _compute_covariance_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.kernel(X1, X2)


def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    x = np.zeros_like(b)

    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    return x


def _backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros_like(b)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]

    return x
