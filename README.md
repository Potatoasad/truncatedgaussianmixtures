# Truncated Gaussian Mixtures
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13999618.svg)](https://doi.org/10.5281/zenodo.13999618)
[![PyPI version](https://badge.fury.io/py/truncatedgaussianmixtures.svg)](https://badge.fury.io/py/truncatedgaussianmixtures)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://potatoasad.github.io/truncatedgaussianmixtures/index.html)

Fit data to mixtures of *truncated* multivariate gaussians

```math
   p(x) = \sum_k w_k\  \phi_{[{\bf a}, {\bf b}]}({\bf x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
```

where ${\bf a}$ and ${\bf b}$ are the bounding corners of the hypercube we would like to truncate the gaussian in.
This package supports

- Full, Diagonal and Block diagonal covariances for each component
- Inteface to allow one to perform the fit in some latent space given by a user-defined transformation, and carry along any target labels
- Sampling and pdf evaluation from the resultant fit


Quick Start
===============================================================
You can install this library using

```bash
pip install truncatedgaussianmixtures
```

You can then import truncatedgaussianmixtures. At the first import it may take a while, since this will create a local
julia installation. This will only happen once at the first ever import. 

The main function to use is `fit_gmm`. The following is a simple example use case. 

```python
import pandas as pd
from truncatedgaussianmixtures import fit_gmm
import numpy as np

# Generate some data
df = pd.DataFrame(np.random.randn(80_000, 2), columns=["x", "y"])

# Truncated it to the unit square
cond = (df['x'] < 1) & (df['x'] > 0)  
cond &= (df['y'] < 1) & (df['y'] > 0)
df = df[cond]

# Fit it a truncated gaussian mixture model to it
gmm = fit_gmm(data = df,      # data to fit to
             N    = 1,       # Number of components of the mixture model
             a    = [0,0],   # lower corner of the truncation
             b    = [1,1],   # upper corner of the truncation
             cov  = "diag"   # covariance structure: any of ("diag", "full")
      )

# Sample from the gmm
df_fit = gmm.sample(len(df));

gmm.pdf(np.array([0, 0]).reshape(1,2)) ## Dimension index needs to be the last index
```

## Target distributions
`TruncatedMultivariateNormal` and `TruncatedGaussianMixture` wrap the Julia
`TruncatedMvNormal` / `MixtureModel` and expose `sample`, `logpdf` and `pdf`,
so a known target can be built, sampled and refitted:
```python
from truncatedgaussianmixtures import TruncatedMultivariateNormal, TruncatedGaussianMixture, fit_gmm

target = TruncatedGaussianMixture(
    [TruncatedMultivariateNormal([0.2, 0.8], [[0.10, 0.03], [0.03, 0.05]], [0, 0], [1, 1]),
     TruncatedMultivariateNormal([0.7, 0.2], [0.02, 0.04], [0, 0], [1, 1])],   # variances -> diagonal
    weights=[0.6, 0.4], cols=["x", "y"])
df = target.sample(50_000, seed=1)                  # DataFrame; as_dataframe=False for an array
fit = fit_gmm(df, 2, [0, 0], [1, 1], cov="full", backend="jax")
fit.distribution                                    # the fitted TruncatedGaussianMixture
fit.distribution.logpdf(df.values).mean() - target.logpdf(df.values).mean()
```
* `TruncatedGaussianMixture.from_arrays(means, covariances, weights, lower, upper)`
  and `.from_julia(mixture)` are alternative constructors; components may have
  different boxes; `sample(..., return_components=True)` also returns the
  (0-based) component of each draw.
* `mean` / `cov` are the parameters of the untruncated normal;
  `log_normalizer` is the log of its mass inside the box.
* `fit.distribution` lives in the fitted space (after any transformation);
  `fit.sample()` maps draws back through the inverse transformation.

## GPU backend (JAX E-step + Julia M-step)

`backend="jax"` is the default. JAX computes the responsibilities,
the observed log likelihood and the weighted sufficient statistics on the GPU,
and TruncatedGaussianMixtures.jl performs the truncated-moment M-step on the CPU
in one batched, threaded call per iteration. The data and the responsibilities
stay in JAX device (GPU) memory; only `O(K d²)` numbers are copied between the
GPU and the host (CPU) per iteration.

```python
# pip install "truncatedgaussianmixtures[cuda]"   (or [jax] for CPU-only JAX)
fit = fit_gmm(df, 20, a, b, cov="full", block_structure=[0, 0, 0, 1, 1, 1],
              backend="jax", device="gpu", seed=1)
fit.diagnostics.timing_summary()      # compile / GPU / Julia / transfer times
fit.diagnostics.average_loglikelihood # per-iteration average weighted log likelihood
```

* `backend="julia"` is the original fitter and remains the reference.
* Supported with `backend="jax"`: DataFrame or `[N, d]` array input, sample
  weights, diagonal / full / block covariances (blocks of any size), annealing
  schedules, transformations and ignored columns (the Julia transformation runs
  once before fitting), boundary unbiasing, `seed=` and `initial_parameters=`.
* `method="em"` (default), `method="gradient"` (L-BFGS on the observed
  likelihood with analytic gradients from the Julia backend) or
  `method="hybrid"` (EM, then a one-way switch to L-BFGS when an M-step is
  unhealthy, e.g. the moment-matching fallback for dimensions where the data
  are nearly flat, or when EM progress is slow while the gradient is not small;
  see `fit.diagnostics.switch_reason`). Only `dtype="float64"`.
* `progress=True` (the `fit_gmm` default) shows tqdm bars for the EM (or
  annealing) and L-BFGS stages with the current average log likelihood and its
  change; `progress=False` turns them off.
* Initialization defaults to weighted k-means in JAX (`init="kmeans"`);
  `init="julia"` uses the legacy Julia k-means.
* The result is the same `TGMM` object. Arrays (`means`, `covariances`,
  `weights`, `log_normalizers`) are its canonical state; the Julia mixture
  (`fit.gmm`) is built lazily. Responsibilities are computed lazily from
  `fit.data`, or returned by the final GPU E-step with
  `return_responsibilities=True` (8·N·K bytes).
* Convergence is judged on the average weighted observed log likelihood
  (`tol`, at least `min_iterations`, `n_consecutive` successes).
