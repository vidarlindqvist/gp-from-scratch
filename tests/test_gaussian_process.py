import numpy as np
import pytest

from gp.gaussian_process import GaussianProcess
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


