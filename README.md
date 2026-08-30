# gp-from-scratch

A NumPy implementation of Gaussian Process Regression, built from scratch as a
learning project. No scikit-learn, no SciPy solvers -- the point is to
actually implement the linear algebra rather than call a library function
for it.

## What's actually implemented

- Kernels: **RBF**, **Linear**, **Matern 3/2**
- `GaussianProcess` with `fit(X, y)` / `predict(X)`
- Observation noise
- Multi-dimensional inputs
- Input validation (shapes, negative hyperparameters, etc.)
- 50 tests, all passing

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

## Project structure

```
gp-from-scratch/
├── src/gp/
│   ├── __init__.py
│   ├── kernels.py
│   └── gaussian_process.py
├── tests/
│   ├── test_kernels.py
│   └── test_gaussian_process.py
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

gp = GaussianProcess(kernel=RBF(), noise=0.1)
gp.fit(X_train, y_train)

mean, covariance = gp.predict(X_test)
```

`mean` is the predictive mean at each test point; `np.diag(covariance)` gives
the predictive variance at each point.

## Tests

```bash
pytest
```

Covers kernel correctness (shape, symmetry, known values, input validation),
the Cholesky factorization itself, correctness of `alpha`, and the predictive
mean/covariance equations against a direct re-derivation in the test.

## Status

Early / learning project. Implemented so far: the three kernels above and the
core fit/predict pipeline. Not yet implemented, in no particular order:
kernel composition, log marginal likelihood, hyperparameter optimization,
posterior sampling, plotting.

No license file yet -- treat as all-rights-reserved until one's added.

