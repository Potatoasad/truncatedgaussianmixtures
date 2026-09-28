"""Tests for the Python-side TruncatedMultivariateNormal / TruncatedGaussianMixture wrappers."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from truncatedgaussianmixtures import (TruncatedMultivariateNormal, TruncatedGaussianMixture, TGMM, fit_gmm)
from truncatedgaussianmixtures.julia_import import jl


def julia_logpdf(obj, X):
    return np.array([float(jl.logpdf(obj, jl.Vector[jl.Float64](x))) for x in X])


def two_component_target():
    return TruncatedGaussianMixture(
        [TruncatedMultivariateNormal([0.2, 0.8], [[0.1, 0.03], [0.03, 0.05]], [0, 0], [1, 1]),
         TruncatedMultivariateNormal([0.7, 0.2], [[0.02, -0.01], [-0.01, 0.04]], [0, 0], [1, 1])],
        [0.6, 0.4], cols=["a", "b"])


@pytest.mark.parametrize("cov", [[[0.1, 0.03], [0.03, 0.05]], [0.1, 0.05]])
def test_tmvn_logpdf_matches_julia(cov):
    t = TruncatedMultivariateNormal([0.2, 0.8], cov, [0, 0], [1, 1])
    assert t.diagonal == (np.ndim(cov) == 1)
    X = t.sample(20, seed=0, as_dataframe=False)
    assert np.allclose(t.logpdf(X), julia_logpdf(t.julia_object, X), atol=1e-12)
    assert np.isneginf(t.logpdf([1.5, 0.5])).all()


def test_tmvn_one_dimensional_matches_scipy():
    mu, s, a, b = 0.3, 0.4, 0.0, 1.0
    t = TruncatedMultivariateNormal([mu], [[s * s]], [a], [b])
    x = np.linspace(0.01, 0.99, 25)
    ref = stats.truncnorm((a - mu) / s, (b - mu) / s, loc=mu, scale=s)
    assert np.allclose(t.logpdf(x[:, None]), ref.logpdf(x), atol=1e-10)
    draws = t.sample(40_000, seed=1, as_dataframe=False)[:, 0]
    assert abs(draws.mean() - ref.mean()) < 5 * ref.std() / np.sqrt(len(draws))


def test_tmvn_sampling_is_reproducible_and_inside_the_box():
    t = TruncatedMultivariateNormal([2.0, -1.0], [[1.0, 0.5], [0.5, 1.0]], [0, -0.5], [1, 0.5], cols=["u", "v"])
    df = t.sample(500, seed=3)
    assert list(df.columns) == ["u", "v"]
    assert df.equals(t.sample(500, seed=3))
    X = df.values
    assert np.all((X >= t.lower) & (X <= t.upper))


def test_unbounded_tmvn_is_a_normal():
    t = TruncatedMultivariateNormal([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]])
    assert t.log_normalizer == pytest.approx(0.0, abs=1e-6)
    x = np.array([[0.3, -0.4], [2.0, 1.0]])
    assert np.allclose(t.logpdf(x), stats.multivariate_normal([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]]).logpdf(x))


def test_mixture_densities_and_normalization():
    mix = two_component_target()
    X = mix.sample(30, seed=0, as_dataframe=False)
    assert np.allclose(mix.logpdf(X), julia_logpdf(mix.julia_object, X), atol=1e-12)
    r = mix.responsibilities(X)
    assert r.shape == (30, 2) and np.allclose(r.sum(axis=1), 1.0)
    g = (np.arange(400) + 0.5) / 400
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    assert mix.pdf(G).mean() == pytest.approx(1.0, abs=5e-3)


def test_mixture_sampling_labels_and_per_component_boxes():
    mix = TruncatedGaussianMixture(
        [TruncatedMultivariateNormal([0.0], [[1.0]], [-1.0], [0.0]),
         TruncatedMultivariateNormal([0.0], [[1.0]], [0.0], [2.0])], [0.25, 0.75])
    df, labels = mix.sample(20_000, seed=5, return_components=True)
    assert abs(labels.mean() - 0.75) < 0.02
    x = df.values[:, 0]
    assert np.all(x[labels == 0] <= 0.0) and np.all(x[labels == 1] >= 0.0)
    assert mix.lower.shape == (2, 1)


def test_from_arrays_and_from_julia_round_trip():
    mix = two_component_target()
    again = TruncatedGaussianMixture.from_arrays(mix.means, mix.covariances, mix.weights, [0, 0], [1, 1], cols=mix.cols)
    assert np.allclose(again.log_normalizers, mix.log_normalizers, atol=1e-3)
    wrapped = TruncatedGaussianMixture.from_julia(mix.julia_object, cols=mix.cols)
    assert np.allclose(wrapped.means, mix.means) and np.allclose(wrapped.covariances, mix.covariances)
    assert np.allclose(wrapped.weights, mix.weights)
    diag = TruncatedGaussianMixture.from_arrays(mix.means, [[0.1, 0.05], [0.02, 0.04]], [1, 1], [0, 0], [1, 1])
    assert all(c.diagonal for c in diag.components)
    assert all(c.diagonal for c in TruncatedGaussianMixture.from_julia(diag.julia_object).components)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        TruncatedMultivariateNormal([0.0, 0.0], np.eye(3))
    with pytest.raises(ValueError):
        TruncatedMultivariateNormal([0.0], [[1.0]], [1.0], [0.0])
    t = TruncatedMultivariateNormal([0.0], [[1.0]])
    with pytest.raises(ValueError):
        TruncatedGaussianMixture([t, t], [1.0])
    with pytest.raises(ValueError):
        TruncatedGaussianMixture([t, TruncatedMultivariateNormal([0.0, 0.0], np.eye(2))])


def test_fit_recovers_target_and_carries_distribution():
    target = two_component_target()
    df = target.sample(40_000, seed=3)
    fit = fit_gmm(df, 2, [0, 0], [1, 1], cov="full", backend="jax", seed=0)
    dist = fit.distribution
    assert isinstance(dist, TruncatedGaussianMixture) and dist.cols == ["a", "b"]
    X = df.values
    assert dist.logpdf(X).mean() == pytest.approx(target.logpdf(X).mean(), abs=5e-3)
    order = np.argsort(dist.means[:, 0])
    assert np.allclose(dist.means[order], target.means[[0, 1]], atol=0.05)
    assert np.allclose(dist.weights[order], target.weights, atol=0.02)
    assert np.allclose(fit.logpdf(X[:200]), dist.logpdf(X[:200]))
    assert fit.julia_gmm is dist.julia_object
    s = fit.sample(100, seed=1)
    assert list(s.columns) == ["a", "b"] and s.equals(fit.sample(100, seed=1))


def test_legacy_fit_distribution_wraps_the_julia_mixture():
    df = two_component_target().sample(3000, seed=4)
    fit = fit_gmm(df, 2, [0, 0], [1, 1], cov="full", MAX_REPS=20, backend="julia")
    assert np.allclose(fit.distribution.means, fit.means)
    assert np.allclose(fit.distribution.logpdf(df.values[:50]), fit.logpdf(df.values[:50]))
