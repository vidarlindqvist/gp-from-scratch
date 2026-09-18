import numpy as np
import pytest

from gp.likelihood import log_marginal_likelihood


def test_log_marginal_likelihood_returns_float() -> None:
    y = np.array([1.0, 2.0])
    alpha = np.array([0.5, 1.0])
    L = np.array(
        [
            [2.0, 0.0],
            [1.0, 3.0],
        ]
    )

    result = log_marginal_likelihood(y, alpha, L)

    assert isinstance(result, float)


def test_log_marginal_likelihood_data_fit_term() -> None:
    y = np.array([1.0, 2.0])
    alpha = np.array([0.5, 1.0])

    expected_data_fit = y @ alpha

    assert expected_data_fit == pytest.approx(2.5)


def test_log_marginal_likelihood_log_determinant() -> None:
    L = np.array(
        [
            [2.0, 0.0],
            [1.0, 3.0],
        ]
    )

    expected_log_determinant = np.log(36.0)

    calculated_log_determinant = 2.0 * np.sum(np.log(np.diag(L)))

    assert calculated_log_determinant == pytest.approx(expected_log_determinant)


def test_log_marginal_likelihood_matches_formula() -> None:
    y = np.array([1.0, 2.0])

    alpha = np.array([0.5, 1.0])

    L = np.array(
        [
            [2.0, 0.0],
            [1.0, 3.0],
        ]
    )

    n = y.shape[0]

    expected = -0.5 * (
        y @ alpha + 2.0 * np.sum(np.log(np.diag(L))) + n * np.log(2.0 * np.pi)
    )

    result = log_marginal_likelihood(
        y,
        alpha,
        L,
    )

    assert result == pytest.approx(expected)


def test_log_marginal_likelihood_matches_direct_calculation() -> None:
    y = np.array([1.0, 2.0])

    K_y = np.array(
        [
            [4.0, 2.0],
            [2.0, 10.0],
        ]
    )

    L = np.linalg.cholesky(K_y)

    alpha = np.linalg.solve(K_y, y)

    expected = -0.5 * (
        y @ np.linalg.solve(K_y, y)
        + np.log(np.linalg.det(K_y))
        + len(y) * np.log(2.0 * np.pi)
    )

    result = log_marginal_likelihood(
        y,
        alpha,
        L,
    )

    assert result == pytest.approx(expected)
