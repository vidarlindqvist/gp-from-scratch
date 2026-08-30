import matplotlib.pyplot as plt
import numpy as np

from gp.gaussian_process import GaussianProcess
from gp.kernels import RBF

X_train = np.array(
    [
        [-2.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
    ]
)

y_train = np.sin(X_train[:, 0])

gp = GaussianProcess(
    kernel=RBF(),
    noise=0.1,
)

gp.fit(X_train, y_train)

X_predict = np.linspace(-4.0, 4.0, 200).reshape(-1, 1)

mean, covariance = gp.predict(X_predict)

variances = np.diag(covariance)
std = np.sqrt(variances)

variances = np.diag(covariance)
std = np.sqrt(variances)

upper = mean + 2 * std
lower = mean - 2 * std

plt.plot(X_predict[:, 0], mean, label="GP mean")
plt.scatter(X_train[:, 0], y_train, label="Training data")

plt.fill_between(
    X_predict[:, 0],
    lower,
    upper,
    alpha=0.2,
    label="95% uncertainty",
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()

plt.show()
