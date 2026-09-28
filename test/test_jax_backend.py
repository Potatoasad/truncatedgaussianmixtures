"""Tests for the hybrid JAX (E-step) + Julia (M-step) backend."""
import numpy as np
import pandas as pd
import pytest

from truncatedgaussianmixtures import fit_gmm, TGMM, Transformation, AnnealingSchedule
from truncatedgaussianmixtures.julia_import import jl, TruncatedGaussianMixtures as TGMMjl
from truncatedgaussianmixtures.julia_backend import JuliaBackend, julia_mixture_from_arrays

import jax
import jax.numpy as jnp
from truncatedgaussianmixtures.jax_backend import BlockStructure, component_logpdf, make_estep, fit_gmm_jax


def random_params(rng, K, d, bs, lo, hi, outside=False):
    means = rng.uniform(lo, hi, size=(K, d))
    if outside:
        means[0] = hi + 0.3 * (hi - lo)       # a component centered outside the box
    covs = np.zeros((K, d, d))
    for k in range(K):
        for idx in bs.blocks:
            n = len(idx)
            A = rng.normal(size=(n, n))
            covs[k][np.ix_(idx, idx)] = (A @ A.T / n + 0.3 * np.eye(n)) * (0.05 * (hi[0] - lo[0]) ** 2)
    w = rng.dirichlet(np.ones(K))
    w[0] *= 1e-3                              # highly unequal weights
    return means, covs, w / w.sum()


LAYOUTS = [("diag", None, 3), ("full", None, 3), ("full", [0, 0, 1, 1, 1, 2], 6), ("full", [0] * 5, 5)]


@pytest.mark.parametrize("cov,labels,d", LAYOUTS)
def test_block_structure_pack_roundtrip(cov, labels, d):
    bs = BlockStructure.from_spec(d, cov, labels)
    rng = np.random.default_rng(0)
    dense = bs.project(rng.normal(size=(4, d, d)))
    assert np.array_equal(bs.unpack(bs.pack(dense)), dense)
    assert bs.pack(dense).shape == (4, bs.packed_length)


def test_block_order_is_first_appearance():
    bs = BlockStructure.from_spec(5, "full", [7, 3, 7, 3, 1])
    assert bs.blocks == ((0, 2), (1, 3), (4,))
    assert bs.labels == [0, 1, 0, 1, 2]


@pytest.mark.parametrize("quadratic", ["cholesky", "precision"])
@pytest.mark.parametrize("cov,labels,d", LAYOUTS)
def test_kernel_matches_numpy_and_julia(cov, labels, d, quadratic):
    rng = np.random.default_rng(1)
    K, N = 5, 400
    lo, hi = np.zeros(d), np.ones(d)
    bs = BlockStructure.from_spec(d, cov, labels)
    means, covs, w = random_params(rng, K, d, bs, lo, hi, outside=True)
    X = rng.uniform(lo, hi, size=(N, d))
    X[:5] = lo          # observations on the boundary
    X[5:10] = hi
    logZ = JuliaBackend(d, K, bs.labels, cov).evaluate(means, covs, lo, hi).log_normalizers

    lp = np.asarray(component_logpdf(jnp.asarray(X), jnp.asarray(means), jnp.asarray(bs.pack(covs)),
                                     jnp.asarray(logZ), bs, quadratic))
    fit = TGMM(means=means, covariances=covs, weights=w, lower=lo, upper=hi, cov=cov,
               block_structure=labels, log_normalizers=logZ, data=None)
    ref = fit.component_logpdf(X)
    tol = 1e-10 if quadratic == "cholesky" else 1e-8
    np.testing.assert_allclose(lp, ref, rtol=tol, atol=tol)

    # Julia mixture density (built with the same log normalizers)
    mix = julia_mixture_from_arrays(means, covs, w, lo, hi, log_normalizers=logZ, cov=cov)
    jl_lp = np.array(TGMMjl.Distributions.logpdf(mix, jl.Matrix[jl.Float64](np.asfortranarray(X.T))))
    np.testing.assert_allclose(fit.logpdf(X), jl_lp, rtol=1e-10, atol=1e-10)


def numpy_statistics(X, w, means, covs, pi, logZ, bs, beta, fit):
    lp = fit.component_logpdf(X)
    u1 = np.log(pi) + lp
    lse1 = np.logaddexp.reduce(u1, axis=1)
    ub = np.log(pi) + beta * lp
    r = np.exp(ub - np.logaddexp.reduce(ub, axis=1)[:, None])
    A = r * w[:, None]
    S2 = bs.pack(np.einsum("nk,ni,nj->kij", A, X, X))
    return A.sum(0), A.T @ X, S2, np.sum(w * lse1), r


@pytest.mark.parametrize("beta", [1.0, 0.4, 1.3])
@pytest.mark.parametrize("cov,labels,d", LAYOUTS)
def test_statistics_match_numpy(cov, labels, d, beta):
    rng = np.random.default_rng(2)
    K, N = 4, 1000
    lo, hi = np.zeros(d), np.ones(d)
    bs = BlockStructure.from_spec(d, cov, labels)
    means, covs, pi = random_params(rng, K, d, bs, lo, hi)
    X = rng.uniform(lo, hi, size=(N, d))
    w = rng.uniform(0.1, 3.0, size=N)                     # non-unit sum
    logZ = JuliaBackend(d, K, bs.labels, cov).evaluate(means, covs, lo, hi).log_normalizers
    fit = TGMM(means=means, covariances=covs, weights=pi, lower=lo, upper=hi, cov=cov,
               block_structure=labels, log_normalizers=logZ, data=None)
    Nk, S1, S2, ll, r = numpy_statistics(X, w, means, covs, pi, logZ, bs, beta, fit)
    params = jax.device_put((means, bs.pack(covs), np.log(pi), logZ))
    for chunk in (None, 128):
        es = make_estep(bs, X, w, params, chunk_size=chunk)
        out = jax.device_get(es(params, np.float64(beta)))
        np.testing.assert_allclose(out[0], Nk, rtol=1e-10)
        np.testing.assert_allclose(out[1], S1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[2], S2, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[3], ll, rtol=1e-11)
    np.testing.assert_allclose(np.asarray(es.responsibilities(params))[:N],
                               fit.compute_responsibilities(X), rtol=1e-10, atol=1e-14)


def legacy_single_update(X, K, a, b, means, covs, w, cov):
    """One legacy Julia `update!` from explicit initial parameters."""
    mix = julia_mixture_from_arrays(means, covs, w, a, b, cov=cov)
    N, d = X.shape
    data = TGMMjl.StandardData(TGMMjl.mat2vecs(jl.Matrix[jl.Float64](np.asfortranarray(X.T))))
    znk = jl.seval("(N, K) -> [zeros(K) for _ in 1:N]")(N, K)
    wts = TGMMjl.StatsBase.Weights(jl.Vector[jl.Float64](np.full(N, 1.0 / N)))
    EM = TGMMjl.ExpectationMaximization(data, mix, 1e-6, znk, jl.Vector[jl.Int64]([0] * d), wts)
    TGMMjl.update_b(EM)
    comps = list(EM.mix.components)
    return (np.stack([np.asarray(c.normal.μ) for c in comps]),
            np.stack([np.asarray(jl.Matrix(c.normal.Σ)) for c in comps]),
            np.asarray(EM.mix.prior.p))


@pytest.mark.parametrize("cov", ["diag", "full"])
def test_one_iteration_matches_legacy_update(cov):
    rng = np.random.default_rng(3)
    d, K, N = 2, 3, 3000
    a, b = np.zeros(d), np.array([1.0, 2.0])
    X = rng.uniform(a, b, size=(N, d)) ** 1.5 / np.array([1.0, 2.0 ** 0.5])
    bs = BlockStructure.from_spec(d, cov, None)
    means, covs, w = random_params(rng, K, d, bs, a, b)
    w = np.full(K, 1.0 / K)
    res = fit_gmm_jax(X, K, a, b, cov=cov, initial_parameters=dict(means=means, covariances=covs, weights=w),
                      MAX_REPS=1, min_iterations=10)
    lm, lc, lw = legacy_single_update(X, K, a, b, means, covs, w, cov)
    # diagonal moments are deterministic in both; full 2D blocks use QMC in the legacy code
    tol = 1e-8 if cov == "diag" else 2e-3
    np.testing.assert_allclose(res.means, lm, atol=tol)
    np.testing.assert_allclose(res.covariances, lc, atol=tol)
    np.testing.assert_allclose(res.weights, lw, atol=1e-10 if cov == "diag" else 1e-4)


@pytest.fixture(scope="module")
def synthetic_df():
    rng = np.random.default_rng(4)
    n = 6000
    X = np.concatenate([rng.normal([0.25, 0.3, 0.6], [0.08, 0.1, 0.15], size=(n, 3)),
                        rng.normal([0.7, 0.65, 0.3], [0.12, 0.1, 0.1], size=(n, 3))])
    X = X[np.all((X > 0) & (X < 1), axis=1)]
    return pd.DataFrame(X, columns=["x", "y", "z"])


def test_end_to_end_matches_julia_backend(synthetic_df):
    kw = dict(cov="full", block_structure=[0, 0, 1], seed=5, progress=False)
    fj = fit_gmm(synthetic_df, 3, [0, 0, 0], [1, 1, 1], backend="julia", **kw)
    fx = fit_gmm(synthetic_df, 3, [0, 0, 0], [1, 1, 1], backend="jax", **kw)
    X = synthetic_df.values
    ll_j, ll_x = fj.logpdf(X).mean(), fx.logpdf(X).mean()
    assert abs(ll_j - ll_x) < 2e-2
    assert fx.diagnostics.converged
    assert fx.diagnostics.n_likelihood_decreases == 0
    np.testing.assert_allclose(fx.diagnostics.final_average_loglikelihood, ll_x, rtol=1e-8)
    # symmetric KL between the fitted densities, estimated on the data sample
    assert abs(np.mean(fj.logpdf(X) - fx.logpdf(X))) < 5e-2


def test_array_input_weights_and_responsibilities():
    rng = np.random.default_rng(6)
    X = rng.beta(2, 5, size=(3000, 2))
    wts = rng.uniform(0.5, 2, size=len(X))
    fit = fit_gmm(X, 3, [0, 0], [1, 1], cov="diag", weights=wts, backend="jax", seed=1,
                  return_responsibilities=True)
    assert fit.responsibilities.shape == (3000, 3)
    np.testing.assert_allclose(fit.responsibilities.sum(1), 1.0)
    np.testing.assert_allclose(fit.responsibilities, fit.compute_responsibilities(X), rtol=1e-8, atol=1e-12)
    assert fit.cols == ["x_0", "x_1"]


def test_transformation_ignore_columns_and_save_load(tmp_path):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({'x': rng.uniform(0.05, 1, 3000), 'y': rng.uniform(0, 1, 3000),
                       'label': rng.integers(0, 4, 3000).astype(float)})
    T = Transformation(['x', 'y'], "(x,y) -> (log(x), y)", ['lx', 'y'], "(lx,y) -> (exp(lx), y)",
                       ignore_columns=['label'])
    fit = fit_gmm(df, 3, [np.log(0.05), 0], [0, 1], cov="full", transformation=T, backend="jax", seed=2)
    assert set(['x', 'y', 'label', 'components']).issubset(fit.data.columns)
    np.testing.assert_allclose(fit.data['x'].values, df['x'].values, rtol=1e-10)
    assert fit.data['components'].between(1, 3).all()
    s = fit.sample(100)
    assert list(s.columns[:2]) == ['x', 'y']
    fn = tmp_path / "fit.h5"
    fit.save(fn)
    g = TGMM.load(fn)
    np.testing.assert_allclose(g.means, fit.means)
    np.testing.assert_allclose(g.log_normalizers, fit.log_normalizers)
    assert g.backend == "jax"
    np.testing.assert_allclose(g.logpdf(np.array([[-1.0, 0.5]])), fit.logpdf(np.array([[-1.0, 0.5]])))


def test_annealing_schedule_runs_every_beta(synthetic_df):
    betas = list(np.linspace(0.2, 1.0, 9)) + [1.0] * 6
    fit = fit_gmm(synthetic_df, 3, [0, 0, 0], [1, 1, 1], cov="diag", backend="jax", seed=3,
                  annealing_schedule=AnnealingSchedule(betas))
    assert fit.diagnostics.n_iterations == len(betas)
    np.testing.assert_allclose([it.beta for it in fit.diagnostics.iterations], betas)


def test_boundary_unbiasing_runs(synthetic_df):
    fit = fit_gmm(synthetic_df.iloc[:2000], 2, [0, 0, 0], [1, 1, 1], cov="full", backend="jax", seed=1,
                  boundary_unbiasing={'structure': [1, 1, 1], 'bandwidth_scale': 0.01}, block_structure=[0, 0, 0])
    assert np.all(np.isfinite(fit.means))
    assert any("boundary unbiasing" in n for n in fit.diagnostics.notes)


def test_explicit_errors(synthetic_df):
    with pytest.raises(ValueError, match="method"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="jax", method="newton")
    with pytest.raises(ValueError, match="float64"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="jax", dtype="float32")
    with pytest.raises(ValueError, match="backend"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="auto")
    with pytest.raises(ValueError, match="outside the truncation box"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [0.5, 1, 1], backend="jax")
    with pytest.raises(ValueError, match="device"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="julia", device="gpu")


def test_no_implicit_transfers_during_fit():
    """Only explicit device_put/device_get cross the host/device boundary."""
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 1, size=(2000, 3))
    with jax.transfer_guard("disallow"):
        res = fit_gmm_jax(X, 3, np.zeros(3), np.ones(3), cov="full", block_structure=[0, 0, 1], seed=1,
                          MAX_REPS=5, min_iterations=10, return_responsibilities=True)
    assert res.responsibilities.shape == (2000, 3)


@pytest.mark.parametrize("order", ["jax_first", "julia_first"])
def test_import_order_clean_process(order):
    import subprocess, sys, textwrap
    first = "import jax" if order == "jax_first" else "import truncatedgaussianmixtures"
    code = textwrap.dedent(f"""
        {first}
        import numpy as np
        import truncatedgaussianmixtures as t
        X = np.random.default_rng(0).uniform(0, 1, size=(500, 2))
        f = t.fit_gmm(X, 2, [0, 0], [1, 1], cov="diag", backend="jax", seed=1, MAX_REPS=12)
        print("OK", f.weights.sum())
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout


# ---------------------------------------------------------------- k-means, gradient, hybrid

def test_jax_kmeans_initialization_matches_legacy_convention():
    from truncatedgaussianmixtures.jax_backend.initialize import jax_kmeans_initialize
    rng = np.random.default_rng(9)
    centers = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]])
    X = np.concatenate([c + 0.03 * rng.normal(size=(1000, 2)) for c in centers])
    w = rng.uniform(0.5, 2.0, len(X))
    m, c, f, info = jax_kmeans_initialize(X, 3, "full", weights=w, seed=0)
    order = np.argsort(m[:, 0] + 10 * m[:, 1])
    np.testing.assert_allclose(m[order], centers[np.argsort(centers[:, 0] + 10 * centers[:, 1])], atol=0.01)
    np.testing.assert_allclose(f, 1 / 3, atol=1e-12)          # unweighted cluster fractions
    lab = np.argmin(((X[:, None] - m[None]) ** 2).sum(-1), 1)
    for k in range(3):
        R = X[lab == k] - np.average(X[lab == k], axis=0, weights=w[lab == k])
        np.testing.assert_allclose(c[k], (w[lab == k, None, None] * R[:, :, None] * R[:, None, :]).sum(0) / w[lab == k].sum(), rtol=1e-10)


@pytest.mark.parametrize("cov,labels", [("diag", None), ("full", [0, 0, 1]), ("full", [0, 0, 0])])
def test_gradient_matches_finite_differences(cov, labels):
    from truncatedgaussianmixtures.jax_backend.gradient import CholeskyParameterization, ObservedLikelihood
    rng = np.random.default_rng(10)
    d, K, N = 3, 3, 2000
    lo, hi = np.array([0.0, -1.0, 0.0]), np.array([1.0, 2.0, 0.5])
    X = rng.uniform(lo, hi, size=(N, d)); w = rng.uniform(0.5, 2, N)
    bs = BlockStructure.from_spec(d, cov, labels)
    means, covs, _ = random_params(rng, K, d, bs, lo, hi)
    means[0, 0] = 1.4                                            # outside the box
    pi = np.array([0.2, 0.3, 0.5])
    param = CholeskyParameterization(bs, K)
    julia = JuliaBackend(d, K, bs.labels, cov, qmc_points_per_dim=100_000)
    es = make_estep(bs, X, w, jax.device_put((means, bs.pack(covs), np.log(pi), np.zeros(K))))
    obj = ObservedLikelihood(es, julia, param, lo, hi, w.sum(), None)
    th = param.pack(means, covs, pi)
    f, g = obj(th)
    h = 1e-5
    fd = np.array([(obj(th + h * e)[0] - obj(th - h * e)[0]) / (2 * h) for e in np.eye(len(th))])
    tol = {"diag": 1e-8, "full": 1e-5}[cov]                      # 2D/3D blocks: QMC floor of mvnormcdf
    assert np.max(np.abs(fd - g)) <= tol * np.max(np.abs(g))


def test_gradient_method_improves_on_em_start():
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, size=(4000, 2)) ** np.array([1.0, 2.0])
    em = fit_gmm_jax(X, 3, [0, 0], [1, 1], cov="full", seed=1, MAX_REPS=15, min_iterations=100)
    gr = fit_gmm_jax(X, 3, [0, 0], [1, 1], cov="full", method="gradient", seed=1,
                     initial_parameters=dict(means=em.means, covariances=em.covariances, weights=em.weights))
    assert gr.diagnostics.final_average_loglikelihood > em.diagnostics.final_average_loglikelihood
    lls = [i.average_loglikelihood for i in gr.diagnostics.iterations]
    assert np.all(np.diff(lls) >= -1e-10)                        # accepted L-BFGS steps never decrease


def test_hybrid_switch_on_moment_matching_fallback():
    rng = np.random.default_rng(12)
    X = np.column_stack([rng.uniform(0, 1, 6000), rng.beta(2, 5, 6000)])   # flat first coordinate
    r = fit_gmm_jax(X, 4, [0, 0], [1, 1], cov="diag", method="hybrid", seed=1, MAX_REPS=300)
    d = r.diagnostics
    assert d.switch_reason is not None
    assert any(i.method == "lbfgs" for i in d.iterations)
    assert all(s == "ok" for s in d.component_status)
    em_part = [i for i in d.iterations if i.method == "em"]
    assert d.final_average_loglikelihood >= em_part[-1].average_loglikelihood - 1e-12


def test_hybrid_slow_em_switch_and_public_api(synthetic_df):
    fit = fit_gmm(synthetic_df, 3, [0, 0, 0], [1, 1, 1], cov="full", backend="jax", method="hybrid", seed=2,
                  switch_tol=1e-3, switch_patience=2, grad_tol=1e-8)
    assert fit.diagnostics.method == "hybrid"
    assert fit.diagnostics.switch_reason.startswith("slow EM progress")
    with pytest.raises(ValueError, match="requires backend='jax'"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="julia", method="hybrid")
    with pytest.raises(ValueError, match="annealing"):
        fit_gmm(synthetic_df, 2, [0, 0, 0], [1, 1, 1], backend="jax", method="gradient",
                annealing_schedule=AnnealingSchedule([0.5, 1.0]))


def test_parameterization_roundtrip_full_blocks():
    from truncatedgaussianmixtures.jax_backend.gradient import CholeskyParameterization
    rng = np.random.default_rng(13)
    bs = BlockStructure.from_spec(6, "full", [0, 0, 1, 1, 1, 2])
    means, covs, pi = random_params(rng, 4, 6, bs, np.zeros(6), np.ones(6))
    param = CholeskyParameterization(bs, 4)
    m, c, p, _ = param.unpack(param.pack(means, covs, pi))
    np.testing.assert_allclose(m, means); np.testing.assert_allclose(c, covs, atol=1e-14); np.testing.assert_allclose(p, pi)


def test_sigma_cap_keeps_flat_directions_flat():
    rng = np.random.default_rng(14)
    X = np.column_stack([rng.uniform(0, 2, 5000), 0.5 + 0.1 * rng.normal(size=5000)])
    X = X[(X[:, 1] > 0) & (X[:, 1] < 1)]
    cap = 20.0
    r = fit_gmm_jax(X, 1, [0, 0], [2, 1], cov="full", method="gradient", seed=1, max_sigma_factor=cap)
    sig = np.sqrt(np.diag(r.covariances[0]))
    assert sig[0] <= cap * 2 * (1 + 1e-9)                 # the cap binds exactly
    # a flat direction stays flat: the fitted marginal varies by < 0.5% across the box
    xs = np.linspace(0, 2, 101)
    dens = np.exp(-0.5 * ((xs - r.means[0, 0]) / sig[0]) ** 2)
    assert dens.max() / dens.min() - 1 < 5e-3
    assert abs(sig[1] - 0.1) < 0.01                       # the peaked direction is unaffected
    assert any("at the cap" in n for n in r.diagnostics.notes)
    capped = fit_gmm_jax(X, 1, [0, 0], [2, 1], cov="full", method="gradient", seed=1, max_sigma_factor=2.0)
    assert np.sqrt(capped.covariances[0, 0, 0]) <= 2.0 * 2 * (1 + 1e-9)
