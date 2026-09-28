"""Batched, array-level calls into TruncatedGaussianMixtures.jl.

Python keeps the row-major convention (``means[K, d]``, ``covariances[K, d, d]``,
packed block second moments ``[K, P]``); Julia works column-major (``d × K``).
Arrays are handed over as Fortran-ordered transposes, which juliacall exposes
to Julia without copying, and results are copied back once per call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .julia_import import jl, TruncatedGaussianMixtures as _TGMM


def _f64(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _to_julia_dk(x: np.ndarray) -> np.ndarray:
    """[K, d] (C order) -> F-ordered view that Julia sees as d × K."""
    return np.asfortranarray(_f64(x).T)


def _to_julia_ddk(x: np.ndarray) -> np.ndarray:
    """[K, d, d] -> F-ordered array that Julia sees as d × d × K."""
    return np.asfortranarray(_f64(x).transpose(2, 1, 0))


def _from_julia_dk(x) -> np.ndarray:
    return np.array(np.asarray(x).T, dtype=np.float64, order="C")


def _from_julia_ddk(x) -> np.ndarray:
    return np.array(np.asarray(x).transpose(2, 1, 0), dtype=np.float64, order="C")


def _from_julia_vec(x) -> np.ndarray:
    return np.array(np.asarray(x), dtype=np.float64)


class JuliaBackendError(RuntimeError):
    """A Julia call failed; the message carries the component/iteration context."""


@dataclass
class EvaluationResult:
    log_normalizers: np.ndarray        # [K]
    model_means: np.ndarray            # [K, d]
    model_second_moments: np.ndarray   # [K, P] packed blocks
    status: list
    success: bool
    elapsed: float


@dataclass
class MStepResult:
    means: np.ndarray                  # [K, d]
    covariances: np.ndarray            # [K, d, d]
    mixture_weights: np.ndarray        # [K]
    log_normalizers: np.ndarray        # [K]
    model_means: np.ndarray
    model_second_moments: np.ndarray
    status: list
    success: bool
    elapsed: float                     # wall time of the Python -> Julia -> Python call
    elapsed_julia: dict = field(default_factory=dict)


class JuliaBackend:
    """Julia backend: owns one Julia ``OracleContext`` (plan, per-thread workspaces, caches) per fit."""

    def __init__(self, d: int, K: int, block_labels: Optional[Sequence[int]], cov: str,
                 threaded: bool = True, seed: int = 0, qmc_points_per_dim: int = 1000,
                 cdf_rtol: float = 0.05):
        labels = jl.nothing if block_labels is None else jl.Vector[jl.Int64]([int(x) for x in block_labels])
        self.d, self.K, self.cov = d, K, cov
        self.ctx = _TGMM.OracleContext(int(d), int(K), labels, jl.Symbol(cov), threaded=threaded,
                                       seed=int(seed), qmc_points_per_dim=int(qmc_points_per_dim),
                                       cdf_rtol=float(cdf_rtol))

    def evaluate(self, means, covariances, lower, upper) -> EvaluationResult:
        t0 = time.perf_counter()
        try:
            r = _TGMM.component_oracle_b(self.ctx, _to_julia_dk(means), _to_julia_ddk(covariances),
                                         _f64(lower), _f64(upper))
        except Exception as e:  # juliacall.JuliaError
            raise JuliaBackendError(f"Julia backend evaluation failed: {e}") from e
        return EvaluationResult(
            log_normalizers=_from_julia_vec(r.log_normalizers),
            model_means=_from_julia_dk(r.model_means),
            model_second_moments=_from_julia_dk(r.model_second_moments),
            status=[str(s) for s in r.status],
            success=bool(r.success),
            elapsed=time.perf_counter() - t0,
        )

    def m_step(self, Nk, S1, S2, means, covariances, lower, upper, *, moment_matching: bool,
               n_data: int, variance_floor: float = 0.0, iteration: Optional[int] = None) -> MStepResult:
        t0 = time.perf_counter()
        try:
            r = _TGMM.em_m_step_from_statistics_b(
                self.ctx, _f64(Nk), _to_julia_dk(S1), _to_julia_dk(S2), _to_julia_dk(means),
                _to_julia_ddk(covariances), _f64(lower), _f64(upper),
                moment_matching=bool(moment_matching), n_data=int(n_data), variance_floor=float(variance_floor))
        except Exception as e:
            where = "" if iteration is None else f" at iteration {iteration}"
            raise JuliaBackendError(f"Julia M-step failed{where}: {e}") from e
        return MStepResult(
            means=_from_julia_dk(r.means),
            covariances=_from_julia_ddk(r.covariances),
            mixture_weights=_from_julia_vec(r.mixture_weights),
            log_normalizers=_from_julia_vec(r.log_normalizers),
            model_means=_from_julia_dk(r.model_means),
            model_second_moments=_from_julia_dk(r.model_second_moments),
            status=[str(s) for s in r.status],
            success=bool(r.success),
            elapsed=time.perf_counter() - t0,
            elapsed_julia=dict(old_moments=float(r.elapsed_old_moments), update=float(r.elapsed_update),
                               new_moments=float(r.elapsed_new_moments)),
        )


def julia_initialize(X: np.ndarray, K: int, lower, upper, cov: str, weights=None, seed: Optional[int] = None):
    """Run the legacy Julia (k-means) initializer on ``X[N, d]``.

    Returns host arrays ``means[K, d]``, ``covariances[K, d, d]``, ``weights[K]``.
    """
    if seed is not None:
        jl.seval("import Random")
        jl.Random.seed_b(int(seed))
    Xjl = jl.Matrix[jl.Float64](_to_julia_dk(X))
    w = jl.nothing if weights is None else _TGMM.StatsBase.Weights(jl.Vector[jl.Float64](_f64(weights)))
    mix = _TGMM.initialize(Xjl, int(K), jl.Vector[jl.Float64](_f64(lower)), jl.Vector[jl.Float64](_f64(upper)),
                           cov=jl.Symbol(cov), weights=w)
    return arrays_from_julia_mixture(mix)


def arrays_from_julia_mixture(mix):
    """Extract ``(means[K,d], covariances[K,d,d], weights[K], log_normalizers[K], lower[d], upper[d])``."""
    comps = list(mix.components)
    means = np.stack([np.asarray(c.normal.μ, dtype=np.float64) for c in comps])
    covs = np.stack([np.asarray(jl.Matrix(c.normal.Σ), dtype=np.float64) for c in comps])
    weights = np.asarray(mix.prior.p, dtype=np.float64).copy()
    logZ = np.array([float(c.logtp) for c in comps])
    lower = np.asarray(comps[0].a, dtype=np.float64).copy()
    upper = np.asarray(comps[0].b, dtype=np.float64).copy()
    return means, covs, weights, logZ, lower, upper


def julia_mixture_from_arrays(means, covariances, weights, lower, upper, log_normalizers=None, cov="full"):
    """Build the legacy Julia ``MixtureModel`` (used lazily for sampling etc.)."""
    logZ = jl.nothing if log_normalizers is None else _f64(log_normalizers)
    return _TGMM.mixture_from_arrays(_to_julia_dk(means), _to_julia_ddk(covariances), _f64(weights),
                                     _f64(lower), _f64(upper), log_normalizers=logZ,
                                     covariance_type=jl.Symbol(cov))


def julia_boundary_pseudo_moments(X: np.ndarray, lower, upper, structure, bandwidth_scale=1.0):
    """Pseudo first moments ``M1[N, d]`` and diagonal pseudo variances ``V[N, d]``."""
    M1, V = _TGMM.boundary_pseudo_moments(_to_julia_dk(X), _f64(lower), _f64(upper),
                                          jl.Vector[jl.Int64]([int(s) for s in structure]),
                                          float(bandwidth_scale))
    return np.array(np.asarray(M1), dtype=np.float64, order="C"), np.array(np.asarray(V), dtype=np.float64, order="C")
