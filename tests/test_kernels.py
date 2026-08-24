import numpy as np

from gp.kernels import RBF


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
