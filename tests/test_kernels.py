import numpy as np
import pytest

from gp.kernels import RBF, Kernel


def test_rbf_output_shape():
    X1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    X2 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    kernel = RBF()
    K = kernel(X1, X2)

    assert K.shape == (3, 5)


def test_rbf_diagonal_equals_variance():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    variance = 2.5
    kernel = RBF(variance=variance)

    K = kernel(X, X)

    assert np.allclose(np.diag(K), variance)


def test_rbf_is_symmetric():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    kernel = RBF()
    K = kernel(X, X)

    assert np.allclose(K, K.T)


def test_rbf_known_value():
    X1 = np.array([[0.0]])
    X2 = np.array([[1.0]])

    kernel = RBF()

    K = kernel(X1, X2)

    expected = np.exp(-0.5)

    assert np.isclose(K[0, 0], expected)


def test_rbf_rejects_zero_length_scale():
    with pytest.raises(ValueError):
        RBF(length_scale=0)


def test_rbf_rejects_negative_length_scale():
    with pytest.raises(ValueError):
        RBF(length_scale=-1)


def test_rbf_rejects_negative_variance():
    with pytest.raises(ValueError):
        RBF(variance=-1)


def test_rbf_rejects_1d_x1():
    X1 = np.array([0.0, 1.0])

    X2 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    kernel = RBF()

    with pytest.raises(ValueError):
        kernel(X1, X2)


def test_rbf_rejects_1d_x2():
    X1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    X2 = np.array([0.0, 1.0])

    kernel = RBF()

    with pytest.raises(ValueError):
        kernel(X1, X2)


def test_rbf_rejects_incompatible_features():
    X1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    X2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    kernel = RBF()

    with pytest.raises(ValueError):
        kernel(X1, X2)

def test_kernel_is_abstract():
    with pytest.raises(TypeError):
        Kernel()
