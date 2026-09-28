"""Responsibilities, observed likelihood and sufficient statistics in one compiled pass."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, xlogy

from .blocks import BlockStructure
from .likelihood import component_logpdf


def _block_second_moments(A, Y, V, bs: BlockStructure):
    """Packed ``[K, P]`` block second moments ``Σ_n A_nk y_n y_nᵀ`` (+ diag(Σ_n A_nk v_n))."""
    parts = [None] * len(bs.blocks)
    single_pos = [b for b, idx in enumerate(bs.blocks) if len(idx) == 1]
    if single_pos:
        s_idx = np.array([bs.blocks[b][0] for b in single_pos])
        Ys = Y[:, s_idx]
        S = A.T @ (Ys * Ys)
        if V is not None:
            S = S + A.T @ V[:, s_idx]
        for j, b in enumerate(single_pos):
            parts[b] = S[:, j:j + 1]
    for b, idx in enumerate(bs.blocks):
        n = len(idx)
        if n == 1:
            continue
        ia = np.array(idx)
        YB = Y[:, ia]
        # column-major within the block; the product is symmetric either way
        YY = (YB[:, None, :] * YB[:, :, None]).reshape(Y.shape[0], n * n)
        S = A.T @ YY
        if V is not None:
            diag_cols = np.array([i + i * n for i in range(n)])
            S = S.at[:, diag_cols].add(A.T @ V[:, ia])
        parts[b] = S
    return jnp.concatenate(parts, axis=1)


def _chunk_statistics(X, w, valid, Y, V, means, cov_packed, log_pi, log_norm, beta, bs, quadratic):
    logp = component_logpdf(X, means, cov_packed, log_norm, bs, quadratic)
    joint = log_pi[None, :] + logp
    lse1 = logsumexp(joint, axis=1)                               # observed likelihood (β = 1)
    tempered = log_pi[None, :] + beta * logp                      # legacy convention r ∝ π p^β
    r = jnp.exp(tempered - logsumexp(tempered, axis=1, keepdims=True))
    A = r * w[:, None]
    Nk = jnp.sum(A, axis=0)
    S1 = A.T @ Y
    S2 = _block_second_moments(A, Y, V, bs)
    loglik = jnp.sum(w * lse1)
    entropy = -jnp.sum(valid[:, None] * xlogy(r, r))
    return Nk, S1, S2, loglik, entropy


@dataclass
class EStep:
    """Ahead-of-time compiled E-step + statistics for one static problem shape.

    ``__call__(params, beta)`` returns device arrays ``(Nk, S1, S2, loglik_sum,
    entropy_sum)``; ``responsibilities(params)`` returns ``[N, K]`` at β = 1.
    """

    bs: BlockStructure
    quadratic: str
    chunk_size: Optional[int]
    data: tuple
    compiled: Any
    compile_time: float
    _resp: Any = None

    def __call__(self, params, beta):
        return self.compiled(*self.data, *params, beta)

    def responsibilities(self, params):
        if self._resp is None:
            bs, quadratic = self.bs, self.quadratic

            def resp(X, means, cov_packed, log_pi, log_norm):
                joint = log_pi[None, :] + component_logpdf(X, means, cov_packed, log_norm, bs, quadratic)
                return jnp.exp(joint - logsumexp(joint, axis=1, keepdims=True))

            self._resp = jax.jit(resp)
        X = self.data[0]
        if self.chunk_size is not None:
            X = X.reshape(-1, X.shape[-1])
        return self._resp(X, *params)


def make_estep(bs: BlockStructure, X, w, params, *, Y=None, V=None, quadratic: str = "cholesky",
               chunk_size: Optional[int] = None, device=None) -> EStep:
    """Upload the data once and compile the E-step for its static shape.

    ``X``, ``w`` (and optional ``Y``/``V`` for boundary-unbiased statistics)
    are host arrays; ``params = (means, cov_packed, log_pi, log_norm)`` are
    device arrays used only for tracing shapes.
    """
    N, d = X.shape
    has_Y = Y is not None
    has_V = V is not None
    valid = np.ones(N)
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        n_chunks = -(-N // chunk_size)
        pad = n_chunks * chunk_size - N

        def padded(a, fill=0.0):
            if a is None:
                return None
            a = np.concatenate([a, np.full((pad,) + a.shape[1:], fill)]) if pad else a
            return a.reshape((n_chunks, chunk_size) + a.shape[1:])

        # padding rows sit at the origin of the (shifted, box-centered) coordinates with zero weight
        X, w, valid, Y, V = padded(X), padded(w), padded(valid), padded(Y), padded(V)

    put = (lambda a: None if a is None else jax.device_put(np.ascontiguousarray(a, dtype=np.float64), device))
    data = (put(X), put(w), put(valid), put(Y) if has_Y else None, put(V) if has_V else None)

    def full(X, w, valid, Y, V, means, cov_packed, log_pi, log_norm, beta):
        return _chunk_statistics(X, w, valid, Y if has_Y else X, V if has_V else None,
                                 means, cov_packed, log_pi, log_norm, beta, bs, quadratic)

    def chunked(X, w, valid, Y, V, means, cov_packed, log_pi, log_norm, beta):
        K = means.shape[0]
        zero = (jnp.zeros(K), jnp.zeros((K, d)), jnp.zeros((K, bs.packed_length)), jnp.zeros(()), jnp.zeros(()))

        def body(carry, xs):
            Xc, wc, vc, Yc, Vc = xs
            out = _chunk_statistics(Xc, wc, vc, Yc if has_Y else Xc, Vc if has_V else None,
                                    means, cov_packed, log_pi, log_norm, beta, bs, quadratic)
            return jax.tree_util.tree_map(jnp.add, carry, out), None

        xs = (X, w, valid, Y if has_Y else X, V if has_V else X)
        carry, _ = jax.lax.scan(body, zero, xs)
        return carry

    fn = chunked if chunk_size is not None else full
    beta0 = jax.device_put(np.float64(1.0), device)
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*data, *params, beta0).compile()
    compile_time = time.perf_counter() - t0
    return EStep(bs=bs, quadratic=quadratic, chunk_size=chunk_size, data=data, compiled=compiled,
                 compile_time=compile_time)
