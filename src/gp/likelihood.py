"""The log marginal likelihood, the score used to compare kernels and noise levels.

Takes the training targets y, the solved alpha = K_y^{-1} y, and the Cholesky
factor L of K_y, all of which fit has already produced.
Returns log p(y | X) as a single float.

Equation 2.30 of Rasmussen and Williams,

    log p(y | X) = -0.5 y^T K_y^{-1} y - 0.5 log|K_y| - 0.5 n log(2 pi)

Both awkward terms come free from the factorisation. The quadratic form is just
y @ alpha, and because K_y = L L^T its determinant is the product of the squared
diagonal of L, so log|K_y| = 2 * sum(log(diag(L))).

Summing logs of the diagonal rather than forming the determinant is the part
worth remembering: det(K_y) underflows to zero for even a moderate number of
training points, while the sum of logs stays perfectly well behaved.
"""

import numpy as np


def log_marginal_likelihood(
    y: np.ndarray,
    alpha: np.ndarray,
    L: np.ndarray,
) -> float:
    """log p(y | X) for the training targets, given the factorisation from fit.

    The three terms pull against each other. data_fit rewards explaining the
    observations, log_determinant penalises a model flexible enough to explain
    anything at all, and the last term is a constant that only depends on how
    many points there are. That built in trade off is why this can be maximised
    directly to pick hyperparameters, with no validation set and no cross
    validation split.

    Note that this is the marginal likelihood of the training targets, so it is
    only comparable between models fitted on the same y.
    """
    n = y.shape[0]

    data_fit = y @ alpha
    log_determinant = 2.0 * np.sum(np.log(np.diag(L)))

    return -0.5 * (data_fit + log_determinant + n * np.log(2.0 * np.pi))
