"""Outer EM loop of the hybrid backend.

Per iteration:

    GPU (JAX):   responsibilities -> (Nk, S1, S2 blocks, log likelihood, entropy)
    GPU -> CPU:  one device_get of that small pytree
    CPU (Julia): one batched M-step call -> new parameters + their log normalizers
    CPU -> GPU:  one device_put of the new small parameter bundle

``X`` and the responsibilities stay in JAX device (GPU) memory and are never
copied to host (CPU) memory. CPU copies of the parameters are kept so they are
never read back from the GPU.
"""
from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import jax

from .blocks import BlockStructure
from .state import FitDiagnostics, FitIteration
from .statistics import make_estep
from .initialize import jax_kmeans_initialize
from .gradient import CholeskyParameterization, ObservedLikelihood, gradient_norm
from ..julia_backend import (JuliaBackend, julia_boundary_pseudo_moments, julia_initialize)
from .progress import progress_bar as _progress_bar, finish_bar as _finish_bar

# statuses that describe a legitimate (if special) component update
_NONFATAL = {"ok", "empty_component", "moment_match_legacy_fix", "moment_match_legacy_fallback",
             "repaired_nonpositive_covariance"}
# ... of which these are not genuine EM updates (the hybrid method switches on them)
_UNHEALTHY = {"moment_match_legacy_fix", "moment_match_legacy_fallback", "repaired_nonpositive_covariance"}


def _run_lbfgs(estep, julia, param, means, covs, mix_w, lo, hi, W, dev, tol, grad_tol, maxiter, diag, verbose,
               max_sigma_factor=None, window=50, progress=False):
    """L-BFGS on the observed likelihood; records one FitIteration per accepted step."""
    from scipy.optimize import minimize

    obj = ObservedLikelihood(estep, julia, param, lo, hi, W, dev)
    bounds, lb, ub = param.bounds(lo, hi, max_sigma_factor)
    theta0 = np.clip(param.pack(means, covs, mix_w), lb, ub)
    seen = {}
    wrapped = obj

    def fun(theta):
        f, g = wrapped(theta)
        seen[theta.tobytes()] = f
        return f, g

    f0, _ = fun(theta0)
    start_it = len(diag.iterations)
    state = dict(prev=-f0, c=dict(obj.counters.__dict__))

    def callback(xk):
        f = seen.get(xk.tobytes())
        if f is None:
            f, _ = fun(xk)
        c = obj.counters
        prev = state["c"]
        ll = -f
        diag.iterations.append(FitIteration(
            len(diag.iterations), 1.0, ll, ll - state["prev"], "lbfgs",
            f"evaluations:{c.evaluations - prev['evaluations']}",
            c.elapsed_gpu - prev["elapsed_gpu"], c.elapsed_julia - prev["elapsed_julia"],
            c.elapsed_transfer - prev["elapsed_transfer"]))
        bar.update(1)
        bar.set_postfix_str(f"loglik={ll:.6f} Δ={ll - state['prev']:.1e} evals={c.evaluations}", refresh=False)
        if verbose:
            bar.write(f"[jax-tgmm] lbfgs it={len(diag.iterations) - start_it:4d} avg_loglik={ll:.8f} "
                      f"delta={ll - state['prev']:.3e} evals={c.evaluations - prev['evaluations']}")
        state["prev"] = ll
        state["c"] = dict(c.__dict__)
        seen.clear()
        # windowed stopping rule, same meaning as EM's tol: average gain per accepted step
        hist = state.setdefault("hist", [])
        hist.append(ll)
        if len(hist) > window and (hist[-1] - hist[-1 - window]) / window < tol:
            state["stopped"] = (f"average log-likelihood gain over the last {window} steps "
                                f"< tol={tol:g}")
            raise StopIteration

    bar = _progress_bar(progress, maxiter, "TGMM L-BFGS")
    try:
        res = minimize(fun, theta0, jac=True, method="L-BFGS-B", callback=callback, bounds=bounds,
                       # scipy's own relative-reduction test fires on single tiny steps; use the windowed rule instead
                       options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-15, gtol=grad_tol, maxcor=20))
    except BaseException:
        bar.close()
        raise
    theta = res.x if np.isfinite(res.fun) else obj.best[1]
    if obj.best[1] is not None and obj.best[0] < res.fun:
        theta = obj.best[1]
    means, covs, mix_w, _ = param.unpack(theta)
    n_steps = len(diag.iterations) - start_it
    stopped = state.get("stopped")
    diag.lbfgs_message = stopped if stopped else str(res.message)
    diag.n_gradient_evaluations = obj.counters.evaluations
    diag.n_failed_evaluations = obj.counters.failed_evaluations
    diag.evaluations_per_step = obj.counters.evaluations / max(n_steps, 1)
    if res.jac is not None:
        g = np.where((res.x >= ub - 1e-12) & (res.jac < 0), 0.0, res.jac)      # projected gradient
        g = np.where((res.x <= lb + 1e-12) & (g > 0), 0.0, g)
        diag.final_gradient_norm = float(np.max(np.abs(g)))
    n_cap = int(np.sum(res.x[param.n_mean:param.n_mean + param.n_logsig] >= ub[param.n_mean:param.n_mean + param.n_logsig] - 1e-9))
    if max_sigma_factor is not None:
        diag.notes.append(f"{n_cap} of {param.n_logsig} per-dimension σ's end at the cap "
                          f"({max_sigma_factor:g}× the box width): flat directions")
    success = bool(stopped) or (bool(res.success) and "PROJ" in str(res.message).upper())
    _finish_bar(bar, "converged" if success else "not converged")
    return means, covs, mix_w, dict(success=success, message=diag.lbfgs_message)


def resolve_device(device: Optional[str]):
    """``None`` -> JAX default device; ``"gpu"``/``"cuda"`` or ``"cpu"`` -> first such device."""
    if device is None:
        return jax.devices()[0]
    if not isinstance(device, str):
        return device
    kind = {"cuda": "gpu"}.get(device.lower(), device.lower())
    try:
        return jax.devices(kind)[0]
    except RuntimeError as e:
        raise RuntimeError(f"device={device!r} requested but JAX has no {kind} backend: {e}") from e


@dataclass
class JAXFitResult:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    log_normalizers: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    responsibilities: Optional[np.ndarray]
    diagnostics: FitDiagnostics


def _validate_data(X, a, b):
    if not np.all(np.isfinite(X)):
        raise ValueError("data contains non-finite values")
    outside = np.any((X < a) | (X > b), axis=1)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} observations lie outside the truncation box [a, b]; "
                         "the truncated likelihood is zero there")


def _repair_initial_covariances(covs, bs: BlockStructure, lower, upper, N, notes):
    """Project onto the blocks and inflate non-positive-definite initial blocks (legacy repair)."""
    covs = bs.project(0.5 * (covs + np.swapaxes(covs, 1, 2)))
    repair = ((upper - lower) / max(N, 1)) ** 2
    repair = np.where(np.isfinite(repair), repair, 1.0)
    for k in range(covs.shape[0]):
        for idx in bs.blocks:
            B = np.ix_(idx, idx)
            tries = 0
            while tries < 60:
                try:
                    np.linalg.cholesky(covs[k][B])
                    break
                except np.linalg.LinAlgError:
                    covs[k][B] += np.diag(repair[list(idx)] * (4.0 ** tries))
                    tries += 1
            if tries:
                notes.append(f"initial covariance of component {k} (block {idx}) was not positive definite; "
                             f"inflated its diagonal")
    return covs


def fit_gmm_jax(X: np.ndarray, K: int, a, b, *, cov: str = "full", block_structure: Optional[Sequence] = None,
                sample_weights: Optional[np.ndarray] = None, tol: float = 1e-6, MAX_REPS: int = 200,
                min_iterations: int = 10, n_consecutive: int = 2, parameter_tol: Optional[float] = None,
                beta_schedule: Optional[Sequence[float]] = None, initial_parameters: Optional[dict] = None,
                seed: Optional[int] = None, boundary_unbiasing: Optional[dict] = None,
                return_responsibilities: bool = False, device=None, quadratic: str = "cholesky",
                chunk_size: Optional[int] = None, julia_threaded: bool = True, init: str = "kmeans",
                qmc_points_per_dim: int = 1000, variance_floor: float = 0.0,
                method: str = "em", switch_tol: float = 1e-5, switch_patience: int = 3,
                grad_tol: float = 1e-5, gradient_maxiter: int = 1000, switch_on_unhealthy: bool = True,
                max_sigma_factor: Optional[float] = 20.0,
                verbose: bool = False, progress: bool = False) -> JAXFitResult:
    """Fit a truncated Gaussian mixture with the JAX E-step and the batched Julia M-step.

    ``X`` is ``[N, d]``. Iteration semantics follow the legacy Julia fitter:
    without an annealing schedule at least ``min_iterations`` updates are made,
    then EM stops once the average weighted observed log likelihood changes by
    less than ``tol`` for ``n_consecutive`` iterations (a decrease beyond ``tol``
    never counts as convergence), or after ``MAX_REPS`` updates. With a
    ``beta_schedule`` exactly one update per β is made, as in the legacy fitter.

    ``method="gradient"``/``"hybrid"``: the L-BFGS stage caps every per-dimension
    σ at ``max_sigma_factor`` × (b_i - a_i) (default 20, i.e. ≲0.03% density
    variation across the box: flat directions stay flat) and keeps the means
    within that many box widths of the box. ``None`` removes the cap.

    ``progress=True`` shows tqdm progress bars (one for EM, one for L-BFGS)
    with the current average log likelihood and its change.
    """
    t_start = time.perf_counter()
    if method not in ("em", "gradient", "hybrid"):
        raise ValueError(f"method must be 'em', 'gradient' or 'hybrid', got {method!r}")
    if method != "em" and boundary_unbiasing is not None:
        raise ValueError("boundary unbiasing changes the EM statistics, not the likelihood; it is only "
                         "supported with method='em'")
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array [N, d]")
    N, d = X.shape
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != (d,) or b.shape != (d,):
        raise ValueError(f"bounds must have length {d}")
    if not np.all(a < b):
        raise ValueError("every lower bound must be smaller than the upper bound")
    _validate_data(X, a, b)
    w = np.ones(N) if sample_weights is None else np.ascontiguousarray(sample_weights, dtype=np.float64).reshape(-1)
    if w.shape != (N,) or np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("sample weights must be finite, non-negative, of length N and not all zero")
    W = float(w.sum())

    dev = resolve_device(device)
    bs = BlockStructure.from_spec(d, cov, block_structure)
    diag = FitDiagnostics(backend="jax", device=str(dev), method=method)

    # ---- initialization (legacy Julia initializer unless explicit parameters are given)
    if initial_parameters is not None:
        means0 = np.array(initial_parameters["means"], dtype=np.float64).reshape(K, d)
        covs0 = np.array(initial_parameters["covariances"], dtype=np.float64).reshape(K, d, d)
        wts0 = np.array(initial_parameters["weights"], dtype=np.float64).reshape(K)
    elif init == "kmeans":
        t_init = time.perf_counter()
        means0, covs0, wts0, info = jax_kmeans_initialize(X, K, cov, weights=sample_weights, seed=seed, device=dev)
        diag.notes.append(f"JAX k-means initialization: {info['iterations']} Lloyd iterations, "
                          f"{time.perf_counter() - t_init:.2f} s")
    elif init == "julia":
        means0, covs0, wts0, _, _, _ = julia_initialize(X, K, a, b, cov, weights=sample_weights, seed=seed)
    else:
        raise ValueError(f"init must be 'kmeans' (JAX) or 'julia' (legacy Julia k-means), got {init!r}")
    wts0 = wts0 / wts0.sum()

    # ---- work in coordinates centered on the box (exact translation equivariance;
    #      keeps the raw second moments well conditioned)
    center = np.where(np.isfinite(a) & np.isfinite(b), 0.5 * (a + b), np.average(X, axis=0, weights=w))
    Xs = X - center
    lo, hi = a - center, b - center
    means = means0 - center
    covs = _repair_initial_covariances(covs0, bs, lo, hi, N, diag.notes)
    mix_w = wts0

    Y = V = None
    if boundary_unbiasing is not None:
        M1, V = julia_boundary_pseudo_moments(X, a, b, boundary_unbiasing["structure"],
                                              boundary_unbiasing.get("bandwidth_scale", 1.0))
        Y = M1 - center
        diag.notes.append("boundary unbiasing: statistics use pseudo moments; the reported likelihood uses "
                          "the raw positions and is a diagnostic that need not increase monotonically")

    julia = JuliaBackend(d, K, bs.labels, cov, threaded=julia_threaded, seed=0 if seed is None else seed,
                               qmc_points_per_dim=qmc_points_per_dim)
    init = julia.evaluate(means, covs, lo, hi)
    if not all(s == "ok" for s in init.status):
        bad = {k: s for k, s in enumerate(init.status) if s != "ok"}
        raise RuntimeError(f"the Julia backend could not evaluate the initial parameters: {bad}")
    logZ = init.log_normalizers

    def upload(means, covs, mix_w, logZ):
        with np.errstate(divide="ignore"):
            log_pi = np.log(mix_w)
        return jax.device_put((means, bs.pack(covs), log_pi, logZ), dev)

    params = upload(means, covs, mix_w, logZ)
    estep = make_estep(bs, Xs, w, params, Y=Y, V=V, quadratic=quadratic, chunk_size=chunk_size, device=dev)
    diag.compile_time = estep.compile_time

    # ---- iteration plan
    if beta_schedule is not None:
        if method == "gradient":
            raise ValueError("method='gradient' optimizes the observed likelihood (β = 1); "
                             "use method='hybrid' to run an annealing schedule with EM first")
        betas = [float(x) for x in beta_schedule]
        annealed = True
    else:
        betas = None
        annealed = False
    max_updates = 0 if method == "gradient" else (len(betas) if annealed else int(MAX_REPS))
    log_K = np.log(K) if K > 1 else np.nan
    param = CholeskyParameterization(bs, K)
    _, sig_lb, sig_ub = param.bounds(lo, hi, max_sigma_factor)
    model_means, model_second = init.model_means, init.model_second_moments

    prev_ll = None
    prev_state = None
    streak = 0
    slow_streak = 0
    param_change_ok = True
    status = ["ok"] * K
    converged = False
    switch_reason = None
    it = 0
    bar = _progress_bar(progress and max_updates > 0, max_updates, "TGMM annealing" if annealed else "TGMM EM")
    for it in range(max_updates):
        beta = betas[it] if annealed else 1.0
        t0 = time.perf_counter()
        Nk, S1, S2, ll_sum, ent_sum = jax.device_get(estep(params, jax.device_put(np.float64(beta), dev)))
        t_gpu = time.perf_counter() - t0
        avg_ll = float(ll_sum) / W
        entropy = float(ent_sum) / (N * log_K)
        delta = None if prev_ll is None else avg_ll - prev_ll
        bar.update(1)
        bar.set_postfix_str(f"loglik={avg_ll:.6f}" + ("" if delta is None else f" Δ={delta:.1e}")
                            + (f" β={beta:.3f}" if annealed else ""), refresh=False)
        if delta is not None and delta < -tol:
            diag.n_likelihood_decreases += 1
            if method == "hybrid" and not annealed and prev_state is not None:
                # the last EM update lowered the likelihood: undo it and switch
                means, covs, mix_w, logZ, model_means, model_second = prev_state
                params = upload(means, covs, mix_w, logZ)
                switch_reason = f"likelihood decreased by {-delta:.3e} at iteration {it}"
                break

        # convergence / switching is judged on the parameters that produced this E-step
        if not annealed and delta is not None and it >= min_iterations:
            small = -tol <= delta < tol
            streak = streak + 1 if small else 0
            if method == "hybrid":
                slow_streak = slow_streak + 1 if (0 <= delta < switch_tol) else 0
                if slow_streak >= switch_patience or streak >= n_consecutive:
                    gn = gradient_norm(param, means, covs, mix_w, Nk, S1, S2, model_means, model_second, W,
                                       sig_lb, sig_ub)
                    if gn > grad_tol:
                        switch_reason = (f"slow EM progress (Δ={delta:.2e} < {switch_tol:g} for {slow_streak} "
                                         f"iterations) with gradient norm {gn:.2e} > {grad_tol:g}")
                        diag.iterations.append(FitIteration(it, beta, avg_ll, delta, "em", "switch", t_gpu,
                                                            0.0, 0.0, entropy, False))
                        break
                    if streak >= n_consecutive:
                        converged = True
                        diag.iterations.append(FitIteration(it, beta, avg_ll, delta, "em", "converged", t_gpu,
                                                            0.0, 0.0, entropy, False))
                        break
            elif streak >= n_consecutive and param_change_ok:
                converged = True
                diag.iterations.append(FitIteration(it, beta, avg_ll, delta, "em", "converged", t_gpu, 0.0, 0.0,
                                                    entropy, False))
                break

        mm = (cov == "diag") and bool(entropy > 0.5)
        upd = julia.m_step(Nk, S1, S2, means, covs, lo, hi, moment_matching=mm, n_data=N,
                            variance_floor=variance_floor, iteration=it)
        fatal = {k: s for k, s in enumerate(upd.status) if s not in _NONFATAL}
        unhealthy = {k: s for k, s in enumerate(upd.status) if s in _UNHEALTHY or s not in _NONFATAL}
        if method == "hybrid" and switch_on_unhealthy and unhealthy and not annealed:
            # reject the update: these components could not be updated by a valid EM step
            switch_reason = (f"unhealthy M-step at iteration {it}: "
                             + ", ".join(f"{s}×{c}" for s, c in Counter(unhealthy.values()).items()))
            diag.iterations.append(FitIteration(it, beta, avg_ll, delta, "em", "switch", t_gpu, upd.elapsed, 0.0,
                                                entropy, mm))
            break
        if fatal:
            raise RuntimeError(f"Julia M-step produced invalid components at iteration {it}: {fatal}")
        status = upd.status
        if parameter_tol is not None:
            scale = np.sqrt(np.maximum(np.einsum("kii->ki", upd.covariances), 1e-300))
            change = np.max(np.abs(upd.means - means) / scale)
            param_change_ok = change < parameter_tol
        else:
            param_change_ok = True
        prev_state = (means, covs, mix_w, logZ, model_means, model_second)
        means, covs, mix_w, logZ = upd.means, upd.covariances, upd.mixture_weights, upd.log_normalizers
        model_means, model_second = upd.model_means, upd.model_second_moments
        t1 = time.perf_counter()
        params = upload(means, covs, mix_w, logZ)
        t_up = time.perf_counter() - t1
        summary = ",".join(f"{s}:{c}" for s, c in Counter(status).items())
        diag.iterations.append(FitIteration(it, beta, avg_ll, delta, "em", summary, t_gpu, upd.elapsed, t_up,
                                            entropy, mm))
        if verbose:
            bar.write(f"[jax-tgmm] it={it:4d} beta={beta:.3f} avg_loglik={avg_ll:.8f} "
                      f"delta={'' if delta is None else f'{delta:.3e}'} gpu={t_gpu*1e3:.1f}ms "
                      f"julia={upd.elapsed*1e3:.1f}ms")
        prev_ll = avg_ll
    else:
        if method == "hybrid":
            switch_reason = ("annealing schedule finished" if annealed
                             else f"EM iteration budget (MAX_REPS={MAX_REPS}) exhausted")
    _finish_bar(bar, "converged" if converged else "→ L-BFGS" if switch_reason is not None
                else "" if annealed else "not converged")

    # ---- direct observed-likelihood optimization (method="gradient", or after the hybrid switch)
    if method == "gradient" or (method == "hybrid" and switch_reason is not None):
        diag.switch_reason = "method='gradient'" if method == "gradient" else switch_reason
        diag.switch_iteration = len(diag.iterations)
        if verbose:
            print(f"[jax-tgmm] switching to L-BFGS: {diag.switch_reason}")
        means, covs, mix_w, lb = _run_lbfgs(estep, julia, param, means, covs, mix_w, lo, hi, W, dev, tol,
                                            grad_tol, gradient_maxiter, diag, verbose, max_sigma_factor,
                                            progress=progress)
        converged = lb["success"]
        ev = julia.evaluate(means, covs, lo, hi)
        if not all(s == "ok" for s in ev.status):
            raise RuntimeError(f"the Julia backend could not evaluate the optimized parameters: {ev.status}")
        logZ = ev.log_normalizers
        status = ev.status
        params = upload(means, covs, mix_w, logZ)

    # ---- final E-step so that the reported likelihood (and responsibilities)
    #      belong to the returned parameters
    Nk, S1, S2, ll_sum, _ = jax.device_get(estep(params, jax.device_put(np.float64(1.0), dev)))
    diag.final_average_loglikelihood = float(ll_sum) / W
    resp = None
    if return_responsibilities:
        resp = np.asarray(jax.device_get(estep.responsibilities(params)))[:N]

    diag.converged = converged
    diag.n_iterations = len(diag.iterations)
    diag.component_status = list(status)
    diag.status_counts = dict(Counter(status))
    diag.total_time = time.perf_counter() - t_start
    if not converged and not (annealed and method == "em"):
        warnings.warn("JAX backend did not converge " + (f"within MAX_REPS={MAX_REPS} iterations" if method == "em"
                      else f"(L-BFGS: {diag.lbfgs_message})"), RuntimeWarning)

    return JAXFitResult(means=means + center, covariances=covs, weights=mix_w, log_normalizers=logZ,
                        lower=a, upper=b, responsibilities=resp, diagnostics=diag)
