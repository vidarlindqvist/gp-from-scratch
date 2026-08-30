import numpy as np
import pytest

from gp.gaussian_process import GaussianProcess, _forward_substitution
from gp.kernels import RBF


def test_gaussian_process_stores_kernel_and_noise():
    kernel = RBF()
    noise = 0.1

    gp = GaussianProcess(kernel, noise)

    assert gp.kernel is kernel
    assert gp.noise == noise


def test_gaussian_process_accepts_zero_noise():
    gp = GaussianProcess(RBF(), noise=0.0)

    assert gp.noise == 0.0


def test_gaussian_process_rejects_negative_noise():
    with pytest.raises(ValueError):
        GaussianProcess(RBF(), noise=-1.0)


# Covariance matrix tests


def test_validate_training_data_accepts_valid_data():
    gp = GaussianProcess(RBF(), noise=0.1)

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    gp._validate_training_data(X, y)


def test_validate_training_data_rejects_non_2d_X():
    gp = GaussianProcess(RBF(), noise=0.1)

    X = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        gp._validate_training_data(X, y)


def test_validate_training_data_rejects_non_1d_y():
    gp = GaussianProcess(RBF(), noise=0.1)

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    y = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    with pytest.raises(ValueError):
        gp._validate_training_data(X, y)


def test_validate_training_data_rejects_mismatched_observations():
    gp = GaussianProcess(RBF(), noise=0.1)

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        gp._validate_training_data(X, y)


# Cholesky tests


def test_fit_cholesky_factorization():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X, y)

    assert np.allclose(gp.L @ gp.L.T, gp.K_y)


def test_fit_stores_training_data():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X, y)

    assert np.array_equal(gp.X_train, X)
    assert np.array_equal(gp.y_train, y)


def test_fit_constructs_training_covariance():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    kernel = RBF()
    gp = GaussianProcess(kernel, noise=0.1)
    gp.fit(X, y)

    expected = kernel(X, X)

    assert np.allclose(gp.K_train, expected)


def test_fit_adds_observation_noise():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    noise = 0.1
    kernel = RBF()
    gp = GaussianProcess(kernel, noise=noise)
    gp.fit(X, y)

    expected = kernel(X, X) + noise * np.eye(X.shape[0])

    assert np.allclose(gp.K_y, expected)


def test_fit_computes_alpha():
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0])

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X, y)

    assert np.allclose(gp.K_y @ gp.alpha, gp.y_train)


# Predict tests


def test_predict_rejects_unfitted_model():
    gp = GaussianProcess(RBF(), noise=0.1)

    X = np.array([[0.0]])

    with pytest.raises(ValueError, match="model must be fitted"):
        gp.predict(X)


def test_predict_rejects_non_2d_X():
    gp = GaussianProcess(RBF(), noise=0.1)

    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([1.0, 2.0, 3.0])

    gp.fit(X_train, y_train)

    X = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="X must be a 2D array"):
        gp.predict(X)


def test_predict_rejects_wrong_number_of_features():
    gp = GaussianProcess(RBF(), noise=0.1)

    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    gp.fit(X_train, y_train)

    X = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="X and training data must have the same number of features",
    ):
        gp.predict(X)


def test_predict_accepts_single_prediction_point():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array([[1.0]])

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    mean, _ = gp.predict(X_predict)

    assert mean.shape == (1,)


def test_predict_accepts_multiple_prediction_points():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    mean, _ = gp.predict(X_predict)

    assert mean.shape == (5,)


def test_predict_mean_has_correct_shape():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    mean, _ = gp.predict(X_predict)

    assert mean.shape == (2,)


def test_predict_covariance_has_correct_shape():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    _, covariance = gp.predict(X_predict)

    assert covariance.shape == (2, 2)


def test_predict_mean_matches_gp_equation():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    mean, _ = gp.predict(X_predict)

    K_star = gp.kernel(X_predict, X_train)
    expected_mean = K_star @ gp.alpha

    assert np.allclose(mean, expected_mean)


def test_predict_covariance_matches_gp_equation():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    _, covariance = gp.predict(X_predict)

    K_star = gp.kernel(X_predict, X_train)
    K_star_star = gp.kernel(X_predict, X_predict)
    V = _forward_substitution(gp.L, K_star.T)

    expected_covariance = K_star_star - V.T @ V

    assert np.allclose(covariance, expected_covariance)


def test_predict_covariance_is_symmetric():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.0],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    _, covariance = gp.predict(X_predict)

    assert np.allclose(covariance, covariance.T)


def test_predict_covariance_has_nonnegative_diagonal():
    X_train = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0])

    X_predict = np.array(
        [
            [0.5],
            [1.0],
            [1.5],
        ]
    )

    gp = GaussianProcess(RBF(), noise=0.1)
    gp.fit(X_train, y_train)

    _, covariance = gp.predict(X_predict)

    variances = np.diag(covariance)

    assert np.all(variances >= 0)
