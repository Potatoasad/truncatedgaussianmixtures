"""Blockwise truncated-Gaussian component log densities (JAX)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .blocks import BlockStructure

LOG2PI = float(np.log(2.0 * np.pi))
QUADRATIC_FORMS = ("cholesky", "precision")


def block_covariances(cov_packed, bs: BlockStructure):
    """Per-block ``[K, n, n]`` covariances from the packed ``[K, P]`` array."""
    out = []
    for n, off in zip(bs.sizes, bs.offsets):
        S = cov_packed[:, off:off + n * n].reshape(-1, n, n)
        out.append(0.5 * (S + jnp.swapaxes(S, -1, -2)))
    return out


def component_logpdf(X, means, cov_packed, log_normalizers, bs: BlockStructure, quadratic: str = "cholesky"):
    """``log p_k(x_n)`` for all observations and components, ``[N, K]``.

    The covariance is used block by block (never as a dense ``d × d`` matrix).
    The static tuple of blocks is unrolled at trace time. Singleton blocks
    (diagonal covariances) are handled together in one vectorized term.

    ``quadratic="cholesky"`` whitens with the triangular inverse of each block
    Cholesky factor (numerically preferable); ``"precision"`` expands
    ``xᵀPx - 2xᵀPμ + μᵀPμ`` as matrix products (GEMM-friendly, fewer
    intermediates, but loses precision when ``|x|, |μ| >> σ``).
    """
    if quadratic not in QUADRATIC_FORMS:
        raise ValueError(f"quadratic must be one of {QUADRATIC_FORMS}")
    N = X.shape[0]
    K = means.shape[0]
    out = jnp.broadcast_to(-(0.5 * bs.d * LOG2PI) - log_normalizers[None, :], (N, K))

    covs = block_covariances(cov_packed, bs)
    singles = [(idx[0], S) for idx, S in zip(bs.blocks, covs) if len(idx) == 1]
    if singles:
        s_idx = np.array([i for i, _ in singles])
        var = jnp.stack([S[:, 0, 0] for _, S in singles], axis=1)   # [K, ns]
        Xs = X[:, s_idx]
        Ms = means[:, s_idx]
        if quadratic == "cholesky":
            diff = Xs[:, None, :] - Ms[None, :, :]                  # [N, K, ns]
            quad = jnp.sum(diff * diff / var[None], axis=-1)
        else:
            iv = 1.0 / var
            quad = (Xs * Xs) @ iv.T - 2.0 * Xs @ (Ms * iv).T + jnp.sum(Ms * Ms * iv, axis=1)[None]
        out = out - 0.5 * (quad + jnp.sum(jnp.log(var), axis=1)[None])

    for idx, S in zip(bs.blocks, covs):
        n = len(idx)
        if n == 1:
            continue
        ia = np.array(idx)
        XB = X[:, ia]
        MB = means[:, ia]
        L = jnp.linalg.cholesky(S)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
        Linv = solve_triangular(L, jnp.broadcast_to(jnp.eye(n, dtype=S.dtype), S.shape), lower=True)
        if quadratic == "cholesky":
            Z = jnp.einsum("kij,nj->nki", Linv, XB) - jnp.einsum("kij,kj->ki", Linv, MB)[None]
            quad = jnp.sum(Z * Z, axis=-1)
        else:
            P = jnp.einsum("kji,kjl->kil", Linv, Linv)                # Σ⁻¹ = L⁻ᵀ L⁻¹
            XX = (XB[:, :, None] * XB[:, None, :]).reshape(N, n * n)
            Pm = jnp.einsum("kij,kj->ki", P, MB)
            quad = XX @ P.reshape(K, n * n).T - 2.0 * XB @ Pm.T + jnp.sum(MB * Pm, axis=1)[None]
        out = out - 0.5 * (quad + logdet[None, :])
    return out
