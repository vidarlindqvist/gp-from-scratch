import numpy as np
import pytest

from gp.kernels import Kernel, RBF, Linear, Matern32

# RBF tests

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

# Kernel ABC tests

def test_kernel_is_abstract():
    with pytest.raises(TypeError):
        Kernel()

# Linear kernel tests

def test_linear_output_shape():
    X1 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    X2 = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    kernel = Linear()
    K = kernel(X1, X2)

    assert K.shape == (3, 2)


def test_linear_known_values():
    X1 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    X2 = np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    kernel = Linear()
    K = kernel(X1, X2)

    expected = np.array(
        [
            [17.0, 23.0],
            [39.0, 53.0],
        ]
    )

    assert np.allclose(K, expected)


def test_linear_variance_scales_kernel():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    variance = 2.5
    kernel = Linear(variance=variance)

    K = kernel(X, X)

    expected = variance * (X @ X.T)

    assert np.allclose(K, expected)


def test_linear_is_symmetric():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    kernel = Linear()
    K = kernel(X, X)

    assert np.allclose(K, K.T)


def test_linear_rejects_negative_variance():
    with pytest.raises(ValueError):
        Linear(variance=-1)


def test_linear_rejects_1d_x1():
    X1 = np.array([1.0, 2.0])

    X2 = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    kernel = Linear()

    with pytest.raises(ValueError):
        kernel(X1, X2)


def test_linear_rejects_1d_x2():
    X1 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    X2 = np.array([1.0, 2.0])

    kernel = Linear()

    with pytest.raises(ValueError):
        kernel(X1, X2)


def test_linear_rejects_incompatible_features():
    X1 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    X2 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    kernel = Linear()

    with pytest.raises(ValueError):
        kernel(X1, X2)

# Matern32 kernel tests

def test_matern32_output_shape():
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

    kernel = Matern32()
    K = kernel(X1, X2)

    assert K.shape == (3, 5)


def test_matern32_diagonal_equals_variance():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    variance = 2.5
    kernel = Matern32(variance=variance)

    K = kernel(X, X)

    assert np.allclose(np.diag(K), variance)


def test_matern32_is_symmetric():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    kernel = Matern32()
    K = kernel(X, X)

    assert np.allclose(K, K.T)


def test_matern32_known_value():
    X1 = np.array([[0.0]])
    X2 = np.array([[1.0]])

    kernel = Matern32()

    K = kernel(X1, X2)

    expected = (1 + np.sqrt(3)) * np.exp(-np.sqrt(3))

    assert np.isclose(K[0, 0], expected)


def test_matern32_variance_scales_kernel():
    X = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    kernel = Matern32(variance=3.0)
    K = kernel(X, X)

    unit_kernel = Matern32(variance=1.0)
    expected = 3.0 * unit_kernel(X, X)

    assert np.allclose(K, expected)


def test_matern32_rejects_zero_length_scale():
    with pytest.raises(ValueError):
        Matern32(length_scale=0)


def test_matern32_rejects_negative_length_scale():
    with pytest.raises(ValueError):
        Matern32(length_scale=-1)


def test_matern32_rejects_negative_variance():
    with pytest.raises(ValueError):
        Matern32(variance=-1)
