import numpy as np


def log_marginal_likelihood(
    y: np.ndarray,
    alpha: np.ndarray,
    L: np.ndarray,
) -> float:
    n = y.shape[0]

    data_fit = y @ alpha
    log_determinant = 2.0 * np.sum(np.log(np.diag(L)))

    return -0.5 * (data_fit + log_determinant + n * np.log(2.0 * np.pi))
