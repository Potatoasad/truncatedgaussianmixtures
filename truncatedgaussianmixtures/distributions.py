"""Python-side distribution objects that wrap the Julia ones.

``TruncatedMultivariateNormal`` wraps one Julia ``TruncatedMvNormal`` and
``TruncatedGaussianMixture`` wraps a Julia ``MixtureModel`` of them. Both
expose ``sample``, ``logpdf`` and ``pdf`` on NumPy arrays (densities are
evaluated in NumPy, using the Julia truncation mass), so a target can be built,
sampled and handed to ``fit_gmm`` in a few lines:

    target = TruncatedGaussianMixture.from_arrays(means, covariances, weights, lower, upper)
    df = target.sample(50_000, seed=1)
    fit = fit_gmm(df, target.K, lower, upper, cov="full", backend="jax")
    fit.distribution.logpdf(x) - target.logpdf(x)

``mean`` and ``cov`` are the parameters of the underlying (untruncated) normal,
not the moments of the truncated distribution.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .julia_import import jl, TruncatedGaussianMixtures as _TGMM

jl.seval("import Random")

__all__ = ["TruncatedMultivariateNormal", "TruncatedGaussianMixture"]


def _logsumexp(u, axis):
	m = np.max(u, axis=axis, keepdims=True)
	m = np.where(np.isfinite(m), m, 0.0)
	return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(u - m), axis=axis))


def _as_points(x, d):
	"""Points as ``[N, d]`` float64 (accepts a DataFrame, a single point or a scalar when d == 1)."""
	if isinstance(x, pd.DataFrame):
		x = x.values
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 0:
		x = x.reshape(1, 1)
	elif x.ndim == 1:
		x = x.reshape(1, -1) if x.shape[0] == d else x.reshape(-1, 1)
	if x.shape[1] != d:
		raise ValueError(f"expected points with {d} coordinates, got shape {x.shape}")
	return x


def _julia_rng(rng: np.random.Generator):
	return jl.Random.Xoshiro(int(rng.integers(0, 2**63 - 1)))


def _default_cols(d):
	return [f"x_{i}" for i in range(d)]


def _to_output(X, cols, as_dataframe):
	return pd.DataFrame(X, columns=list(cols)) if as_dataframe else X


class TruncatedMultivariateNormal:
	"""A multivariate normal ``N(mean, cov)`` truncated to the box ``lower ≤ x ≤ upper``.

	``cov`` is a ``[d, d]`` covariance matrix, or a length-``d`` vector of
	variances for a diagonal covariance (``diagonal=True`` also forces the
	diagonal form). Missing bounds are infinite. The Julia object is available
	as ``julia_object``.
	"""

	def __init__(self, mean, cov, lower=None, upper=None, cols: Optional[Sequence[str]] = None,
				 diagonal: Optional[bool] = None, log_normalizer: Optional[float] = None, *, _julia_object=None):
		self.mean = np.array(mean, dtype=np.float64).reshape(-1)
		d = self.mean.shape[0]
		cov = np.array(cov, dtype=np.float64)
		if cov.ndim == 1:
			diagonal = True if diagonal is None else diagonal
			cov = np.diag(cov)
		if cov.shape != (d, d):
			raise ValueError(f"cov must be [{d}, {d}] or [{d}], got shape {cov.shape}")
		self.diagonal = bool(diagonal) or d == 1
		self.cov = np.diag(np.diag(cov)) if self.diagonal else 0.5 * (cov + cov.T)
		self.lower = np.full(d, -np.inf) if lower is None else np.array(lower, dtype=np.float64).reshape(d)
		self.upper = np.full(d, np.inf) if upper is None else np.array(upper, dtype=np.float64).reshape(d)
		if np.any(self.lower >= self.upper):
			raise ValueError("every lower bound must be below the corresponding upper bound")
		self.cols = _default_cols(d) if cols is None else list(cols)
		self._L = np.linalg.cholesky(self.cov)
		if _julia_object is None:
			_julia_object = _TGMM.truncated_mvnormal(
				self.mean, self.cov, self.lower, self.upper,
				covariance_type=jl.Symbol("diag" if self.diagonal else "full"),
				log_normalizer=jl.nothing if log_normalizer is None else float(log_normalizer))
		self.julia_object = _julia_object
		self.log_normalizer = float(_julia_object.logtp)

	@classmethod
	def from_julia(cls, obj, cols: Optional[Sequence[str]] = None):
		"""Wrap an existing Julia ``TruncatedMvNormal`` (no copy of the Julia object)."""
		cov = np.asarray(jl.Matrix(obj.normal.Σ), dtype=np.float64)
		diagonal = bool(jl.isa(obj.normal.Σ, jl.Distributions.PDMats.PDiagMat))
		return cls(np.asarray(obj.normal.μ, dtype=np.float64), cov, np.asarray(obj.a), np.asarray(obj.b),
				   cols=cols, diagonal=diagonal, _julia_object=obj)

	@property
	def d(self) -> int:
		return self.mean.shape[0]

	def logpdf(self, x) -> np.ndarray:
		"""``log p(x)`` for points ``x[N, d]`` (``-inf`` outside the box)."""
		x = _as_points(x, self.d)
		z = np.linalg.solve(self._L, (x - self.mean).T)
		out = (-0.5 * np.sum(z * z, axis=0) - np.sum(np.log(np.diag(self._L)))
			   - 0.5 * self.d * np.log(2 * np.pi) - self.log_normalizer)
		inside = np.all((x >= self.lower) & (x <= self.upper), axis=1)
		return np.where(inside, out, -np.inf)

	def pdf(self, x) -> np.ndarray:
		return np.exp(self.logpdf(x))

	def sample(self, N: int = 1000, seed: Optional[int] = None, as_dataframe: bool = True, rng=None):
		"""``N`` draws (Julia sampler) as a DataFrame with ``cols``, or an ``[N, d]`` array."""
		rng = np.random.default_rng(seed) if rng is None else rng
		X = np.array(np.asarray(jl.rand(_julia_rng(rng), self.julia_object, int(N))).T, dtype=np.float64)
		return _to_output(X.reshape(int(N), self.d), self.cols, as_dataframe)

	def __repr__(self):
		kind = "diagonal" if self.diagonal else "full"
		return f"TruncatedMultivariateNormal(d={self.d}, {kind} covariance, log_normalizer={self.log_normalizer:.4g})"


class TruncatedGaussianMixture:
	"""A mixture ``Σ_k w_k TMVN_k`` of truncated multivariate normals.

	Build it from a list of ``TruncatedMultivariateNormal`` (each may have its
	own box), from arrays with ``from_arrays``, or from a Julia ``MixtureModel``
	with ``from_julia``. The Julia ``MixtureModel`` is built lazily
	(``julia_object``). Weights are normalized.
	"""

	def __init__(self, components: Sequence[TruncatedMultivariateNormal], weights=None,
				 cols: Optional[Sequence[str]] = None, *, _julia_object=None):
		self.components: List[TruncatedMultivariateNormal] = list(components)
		if not self.components:
			raise ValueError("a mixture needs at least one component")
		d = self.components[0].d
		if any(c.d != d for c in self.components):
			raise ValueError("all components must have the same dimension")
		K = len(self.components)
		w = np.ones(K) if weights is None else np.array(weights, dtype=np.float64).reshape(-1)
		if w.shape[0] != K or np.any(w < 0) or w.sum() <= 0:
			raise ValueError(f"weights must be {K} non-negative numbers with a positive sum")
		self.weights = w / w.sum()
		self.cols = list(self.components[0].cols) if cols is None else list(cols)
		self._julia_object = _julia_object

	@classmethod
	def from_arrays(cls, means, covariances, weights, lower=None, upper=None, cols: Optional[Sequence[str]] = None,
					cov: str = "full", log_normalizers=None):
		"""From ``means[K, d]``, ``covariances[K, d, d]`` (or ``[K, d]`` variances), ``weights[K]``.

		``lower``/``upper`` are one box ``[d]`` shared by all components, or ``[K, d]``.
		"""
		means = np.atleast_2d(np.asarray(means, dtype=np.float64))
		K, d = means.shape
		covariances = np.asarray(covariances, dtype=np.float64)
		lower = np.broadcast_to(np.full(d, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64), (K, d))
		upper = np.broadcast_to(np.full(d, np.inf) if upper is None else np.asarray(upper, dtype=np.float64), (K, d))
		diagonal = cov == "diag" or covariances.ndim == 2
		comps = [TruncatedMultivariateNormal(means[k], covariances[k], lower[k], upper[k], cols=cols, diagonal=diagonal,
											 log_normalizer=None if log_normalizers is None else log_normalizers[k])
				 for k in range(K)]
		return cls(comps, weights, cols=cols)

	@classmethod
	def from_julia(cls, mix, cols: Optional[Sequence[str]] = None):
		"""Wrap an existing Julia ``MixtureModel`` of ``TruncatedMvNormal``s."""
		comps = [TruncatedMultivariateNormal.from_julia(c, cols=cols) for c in mix.components]
		return cls(comps, np.asarray(mix.prior.p, dtype=np.float64), cols=cols, _julia_object=mix)

	# ------------------------------------------------------------------ parameters

	@property
	def K(self) -> int:
		return len(self.components)

	@property
	def d(self) -> int:
		return self.components[0].d

	@property
	def means(self) -> np.ndarray:
		return np.stack([c.mean for c in self.components])

	@property
	def covariances(self) -> np.ndarray:
		return np.stack([c.cov for c in self.components])

	@property
	def lower(self) -> np.ndarray:
		"""Per-component lower bounds ``[K, d]``."""
		return np.stack([c.lower for c in self.components])

	@property
	def upper(self) -> np.ndarray:
		"""Per-component upper bounds ``[K, d]``."""
		return np.stack([c.upper for c in self.components])

	@property
	def log_normalizers(self) -> np.ndarray:
		return np.array([c.log_normalizer for c in self.components])

	@property
	def julia_object(self):
		if self._julia_object is None:
			self._julia_object = _TGMM.mixture_from_components([c.julia_object for c in self.components], self.weights)
		return self._julia_object

	# ------------------------------------------------------------------ densities

	def component_logpdf(self, x) -> np.ndarray:
		"""``log p_k(x_n)`` as ``[N, K]``."""
		x = _as_points(x, self.d)
		return np.stack([c.logpdf(x) for c in self.components], axis=1)

	def logpdf(self, x) -> np.ndarray:
		with np.errstate(divide="ignore"):
			return _logsumexp(np.log(self.weights)[None, :] + self.component_logpdf(x), axis=1)

	def pdf(self, x) -> np.ndarray:
		return np.exp(self.logpdf(x))

	def responsibilities(self, x) -> np.ndarray:
		"""Posterior component probabilities ``[N, K]`` for points ``x[N, d]``."""
		with np.errstate(divide="ignore"):
			u = np.log(self.weights)[None, :] + self.component_logpdf(x)
		return np.exp(u - _logsumexp(u, axis=1)[:, None])

	# ------------------------------------------------------------------ sampling

	def sample(self, N: int = 1000, seed: Optional[int] = None, as_dataframe: bool = True,
			   return_components: bool = False):
		"""``N`` draws as a DataFrame with ``cols`` (or an ``[N, d]`` array).

		With ``return_components=True`` also returns the (0-based) component of
		every draw. ``seed`` makes the draws reproducible.
		"""
		rng = np.random.default_rng(seed)
		labels = rng.choice(self.K, size=int(N), p=self.weights)
		X = np.empty((int(N), self.d))
		for k, comp in enumerate(self.components):
			rows = np.flatnonzero(labels == k)
			if rows.size:
				X[rows] = comp.sample(rows.size, as_dataframe=False, rng=rng)
		out = _to_output(X, self.cols, as_dataframe)
		return (out, labels) if return_components else out

	def __repr__(self):
		return f"TruncatedGaussianMixture(K={self.K}, d={self.d}, cols={self.cols})"
