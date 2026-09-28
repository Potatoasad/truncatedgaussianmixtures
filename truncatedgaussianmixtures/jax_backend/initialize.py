"""Weighted k-means initialization in JAX (replaces the Julia k-means for large N).

The initial mixture follows the legacy Julia `initialize` exactly, only the
clustering is different:

    means        = k-means centers (weighted cluster means)
    covariances  = weighted within-cluster covariance (diagonal for cov="diag")
    weights      = unweighted cluster fractions  n_k / N
"""
from __future__ import annotations

import time
from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


def _pad_rows(a, n_chunks, chunk):
    pad = n_chunks * chunk - a.shape[0]
    if pad:
        a = jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)])
    return a.reshape((n_chunks, chunk) + a.shape[1:])


def _assign(X, C, chunk):
    """Nearest center and squared distance for every row, in chunks of `chunk` rows."""
    N = X.shape[0]
    cn = jnp.sum(C * C, axis=1)

    def one(Xc):
        d2 = jnp.sum(Xc * Xc, axis=1)[:, None] - 2.0 * Xc @ C.T + cn[None, :]
        return jnp.argmin(d2, axis=1), jnp.maximum(jnp.min(d2, axis=1), 0.0)

    if chunk >= N:
        return one(X)
    n_chunks = -(-N // chunk)
    lab, dist = jax.lax.map(one, _pad_rows(X, n_chunks, chunk))
    return lab.reshape(-1)[:N], dist.reshape(-1)[:N]


@partial(jax.jit, static_argnames=("K",))
def _kmeanspp(key, Xs, ws, K):
    """k-means++ seeding (D² · w sampling) on a subsample."""
    n = Xs.shape[0]
    key, k0 = jax.random.split(key)
    first = jax.random.categorical(k0, jnp.log(ws))
    C = jnp.zeros((K, Xs.shape[1]), Xs.dtype).at[0].set(Xs[first])
    d2 = jnp.sum((Xs - Xs[first]) ** 2, axis=1)

    def body(i, carry):
        C, d2, key = carry
        key, sub = jax.random.split(key)
        logits = jnp.log(ws) + jnp.log(jnp.maximum(d2, 1e-300))
        j = jax.random.categorical(sub, logits)
        C = C.at[i].set(Xs[j])
        d2 = jnp.minimum(d2, jnp.sum((Xs - Xs[j]) ** 2, axis=1))
        return C, d2, key

    C, _, _ = jax.lax.fori_loop(1, K, body, (C, d2, key))
    return C


@partial(jax.jit, static_argnames=("max_iter", "chunk"))
def _lloyd(X, w, C0, tol, max_iter, chunk):
    K = C0.shape[0]

    def objective_and_update(C):
        lab, dist = _assign(X, C, chunk)
        ws = jax.ops.segment_sum(w, lab, num_segments=K)
        sums = jax.ops.segment_sum(w[:, None] * X, lab, num_segments=K)
        newC = jnp.where(ws[:, None] > 0, sums / jnp.maximum(ws, 1e-300)[:, None], C)   # keep empty clusters
        return jnp.sum(w * dist), newC

    def cond(s):
        it, _, obj, prev = s
        return (it < max_iter) & (jnp.abs(prev - obj) > tol * jnp.abs(obj))

    def body(s):
        it, C, obj, _ = s
        new_obj, newC = objective_and_update(C)
        return it + 1, newC, new_obj, obj

    obj0, C1 = objective_and_update(C0)
    it, C, obj, _ = jax.lax.while_loop(cond, body, (jnp.array(1), C1, obj0, jnp.inf))
    lab, _ = _assign(X, C, chunk)
    return C, lab, it, obj


@partial(jax.jit, static_argnames=("K", "diag"))
def _cluster_moments(X, w, lab, C, K, diag):
    n = jax.ops.segment_sum(jnp.ones_like(w), lab, num_segments=K)
    ws = jax.ops.segment_sum(w, lab, num_segments=K)
    mean = jax.ops.segment_sum(w[:, None] * X, lab, num_segments=K) / jnp.maximum(ws, 1e-300)[:, None]
    R = X - mean[lab]
    if diag:
        v = jax.ops.segment_sum(w[:, None] * R * R, lab, num_segments=K) / jnp.maximum(ws, 1e-300)[:, None]
        cov = jax.vmap(jnp.diag)(v)
    else:
        d = X.shape[1]
        RR = (R[:, :, None] * R[:, None, :]).reshape(X.shape[0], d * d)
        cov = (jax.ops.segment_sum(w[:, None] * RR, lab, num_segments=K) / jnp.maximum(ws, 1e-300)[:, None]).reshape(K, d, d)
    return C, cov, n / X.shape[0]


def jax_kmeans_initialize(X: np.ndarray, K: int, cov: str, weights: Optional[np.ndarray] = None,
                          seed: Optional[int] = None, device=None, max_iter: int = 300, tol: float = 1e-6,
                          seed_sample: int = 50_000, chunk: int = 262_144):
    """Legacy-style initial mixture from weighted k-means run with JAX.

    Returns host arrays ``means[K, d]``, ``covariances[K, d, d]``, ``weights[K]``
    and a small info dict.
    """
    # one-time setup: small implicit transfers (PRNG key, scalars) are fine here; the
    # transfer guard is meant for the per-iteration fit loop
    with jax.transfer_guard("allow"):
        return _jax_kmeans_initialize(X, K, cov, weights, seed, device, max_iter, tol, seed_sample, chunk)


def _jax_kmeans_initialize(X, K, cov, weights, seed, device, max_iter, tol, seed_sample, chunk):
    t0 = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[0]
    w = np.ones(N) if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    key = jax.random.PRNGKey(0 if seed is None else int(seed))
    key, ksub, kpp = jax.random.split(key, 3)
    Xd, wd = jax.device_put((X, w), device)
    # seeding on a (uniform) subsample; the D²·w sampling uses the weights
    if N > seed_sample:
        idx = np.asarray(jax.random.choice(ksub, N, (seed_sample,), replace=False))
        Xs, ws = jax.device_put((X[idx], w[idx]), device)
    else:
        Xs, ws = Xd, wd
    C0 = _kmeanspp(kpp, Xs, ws, K)
    C, lab, it, obj = _lloyd(Xd, wd, C0, tol, max_iter, min(chunk, N))
    means, covs, frac = jax.device_get(_cluster_moments(Xd, wd, lab, C, K, cov == "diag"))
    info = dict(iterations=int(it), objective=float(obj), elapsed=time.perf_counter() - t0)
    return np.asarray(means), np.asarray(covs), np.asarray(frac), info
