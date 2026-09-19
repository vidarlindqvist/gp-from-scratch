"""Gaussian process regression: fit on observed data, predict with uncertainty.

Takes a kernel and the variance of the observation noise.
Returns, from predict, the posterior mean and covariance at the test inputs.

Follows algorithm 2.1 of Rasmussen and Williams. Everything hangs off a single
Cholesky factorisation K_y = L L^T computed once in fit and then reused by the
predictive mean, the predictive covariance and the log marginal likelihood, so
the O(n^3) work happens one time rather than on every call.
"""

import numpy as np

from gp.kernels import Kernel
from gp.likelihood import log_marginal_likelihood


class GaussianProcess:
    """A GP regressor over a fixed kernel and a fixed observation noise level.

    fit stores the training data and factorises the covariance matrix, and both
    predict and log_marginal_likelihood then reuse that factorisation. Nothing
    here optimises the hyperparameters; the kernel is taken as given.
    """

    def __init__(self, kernel: Kernel, noise_variance: float) -> None:
        """noise_variance is sigma_n^2, the amount added to the diagonal of K.

        It is stored as self.noise_variance. Set it to zero only for noiseless
        data, and expect the Cholesky to complain if K is then singular.
        """
        if noise_variance < 0:
            raise ValueError("noise_variance must be non-negative")

        self.kernel = kernel
        self.noise_variance = noise_variance
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Factorise the training covariance and solve for alpha = K_y^{-1} y.

        K_y = K + noise_variance * I is symmetric positive definite, so it
        has a Cholesky factorisation K_y = L L^T. Rather than inverting K_y,
        alpha is found by solving the two triangular systems

            L v = y        then        L^T alpha = v

        which is the whole point of the project: the substitutions are written
        out at the bottom of this file instead of going through np.linalg.solve.
        """
        self._validate_training_data(X, y)
        self.X_train = X
        self.y_train = y
        self.K_train = self._compute_covariance_matrix(self.X_train, self.X_train)
        self.K_y = self.K_train + self.noise_variance * np.eye(self.K_train.shape[0])
        self.L = np.linalg.cholesky(self.K_y)
        v = _forward_substitution(self.L, self.y_train)
        self.alpha = _backward_substitution(self.L.T, v)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance at X, equations 2.25 and 2.26.

            mean       = K_star @ alpha
            covariance = K_star_star - V^T V,   where  L V = K_star^T

        The mean costs nothing beyond a matrix product because alpha was
        already solved for in fit. The subtracted term in the covariance is
        what the training data removed from the prior uncertainty, so the
        covariance shrinks near observed points and relaxes back to the prior
        far away from them.

        Returns the full covariance matrix, not just its diagonal, so take
        np.diag of it for the per point predictive variance.
        """
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

    def log_marginal_likelihood(self) -> float:
        """How well this kernel and noise level explain the training data.

        Reuses alpha and L from fit, so it is nearly free once fitted. Higher
        is better; compare it across kernels or hyperparameters to choose
        between them. See gp.likelihood for the formula.
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted")

        return log_marginal_likelihood(self.y_train, self.alpha, self.L)

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """X must be 2D, y 1D, and the two must agree on the observation count."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations")

    def _compute_covariance_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """The kernel evaluated between two sets of inputs."""
        return self.kernel(X1, X2)


def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L x = b for lower triangular L, working from the top row down.

    Row i reads L[i, :i] @ x[:i] + L[i, i] * x[i] = b[i], and everything before
    x[i] is already known, so each unknown falls straight out. b may be a
    matrix, in which case every column is solved at once.
    """
    n = L.shape[0]
    x = np.zeros_like(b)

    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    return x


def _backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve U x = b for upper triangular U, working from the bottom row up.

    The mirror of the forward pass: row i now depends only on x[i + 1:], which
    the previous iterations have already produced.
    """
    n = U.shape[0]
    x = np.zeros_like(b)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]

    return x
