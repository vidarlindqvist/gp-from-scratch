# gp-from-scratch

A NumPy implementation of Gaussian Process Regression, built from scratch as a
learning project. No scikit-learn, no SciPy solvers -- the point is to
actually implement the linear algebra rather than call a library function
for it.

## What's actually implemented

- Kernels: **RBF**, **Linear**, **Matern 3/2**
- `GaussianProcess` with `fit(X, y)` / `predict(X)`
- **Log marginal likelihood**, for scoring a kernel and noise level
- Observation noise
- Multi-dimensional inputs
- Input validation (shapes, negative hyperparameters, etc.)
- 55 tests, all passing

## The one deliberate design choice worth mentioning

Fitting a GP normally involves solving `K_y^{-1} y`. Instead of forming that
inverse, this does a Cholesky factorization `K_y = L L^T` and solves two
triangular systems:

```
L v = y
L^T alpha = v
```

Both `_forward_substitution` and `_backward_substitution` are hand-written in
`gaussian_process.py` rather than calling `np.linalg.solve` -- that's the
actual point of the project, seeing the substitution step instead of hiding
it behind a library call.

## Log marginal likelihood

`likelihood.py` implements equation 2.30 of Rasmussen and Williams, the
probability of the training targets with the function values integrated out:

```
log p(y | X) = -0.5 y^T K_y^{-1} y  -  0.5 log|K_y|  -  0.5 n log(2 pi)
```

This is the score to maximise when choosing a kernel, a length scale or a
noise level. It needs no validation set, because the middle term already
penalises a model flexible enough to fit anything: the first term rewards
explaining the data, the second charges for the flexibility that took, and the
third is a constant. Higher is better, and values are only comparable between
models fitted on the same `y`.

The implementation reuses what `fit` already computed, so it costs almost
nothing on top:

- `y^T K_y^{-1} y` is just `y @ alpha`, since `alpha` already solves
  `K_y alpha = y`.
- `log|K_y|` comes from the Cholesky factor. Because `K_y = L L^T`, the
  determinant is the product of the squared diagonal of `L`, so the log
  determinant is `2 * sum(log(diag(L)))`.

That second point is the part worth not forgetting. Summing the logs of the
diagonal avoids ever forming `det(K_y)`, which underflows to zero for even a
moderate number of training points and would take the log with it. Working in
logs the whole way keeps it stable.

`GaussianProcess.log_marginal_likelihood()` wraps the free function and passes
in the stored `y_train`, `alpha` and `L`, so it only works after `fit`.

## Project structure

```
gp-from-scratch/
├── src/gp/
│   ├── __init__.py
│   ├── kernels.py
│   ├── gaussian_process.py
│   └── likelihood.py
├── tests/
│   ├── test_kernels.py
│   ├── test_gaussian_process.py
│   └── test_likelihood.py
├── examples/
│   └── gpr_plot.py
└── pyproject.toml
```

## Setup

```bash
git clone <repository-url>
cd gp-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -e . --group dev
```

## Example

```python
import numpy as np
from gp.gaussian_process import GaussianProcess
from gp.kernels import RBF

X_train = np.array([[0.0], [1.0], [2.0]])
y_train = np.array([1.0, 2.0, 3.0])
X_test = np.array([[0.5], [1.5]])

gp = GaussianProcess(kernel=RBF(), noise_variance=0.1)
gp.fit(X_train, y_train)

mean, covariance = gp.predict(X_test)
```

`mean` is the predictive mean at each test point; `np.diag(covariance)` gives
the predictive variance at each point.

To compare two kernels, fit both and keep the higher score:

```python
for length_scale in [0.5, 1.0, 2.0]:
    gp = GaussianProcess(kernel=RBF(length_scale=length_scale), noise_variance=0.1)
    gp.fit(X_train, y_train)
    print(length_scale, gp.log_marginal_likelihood())
```

`examples/gpr_plot.py` plots the posterior mean with a two standard deviation
band around it.

## Tests

```bash
pytest
```

Covers kernel correctness (shape, symmetry, known values, input validation),
the Cholesky factorization itself, correctness of `alpha`, the predictive
mean/covariance equations against a direct re-derivation in the test, and the
log marginal likelihood against both its own formula and a direct calculation
using `np.linalg.solve` and `np.linalg.det`.

## Status

Early / learning project. Implemented so far: the three kernels above, the
core fit/predict pipeline, and the log marginal likelihood. Not yet
implemented, in no particular order: kernel composition, hyperparameter
optimization (the likelihood is there, nothing maximises it yet), posterior
sampling, gradients of the likelihood.

No license file yet -- treat as all-rights-reserved until one's added.
