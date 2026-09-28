"""Direct observed-likelihood optimization (phase two of the brief).

Objective (per unit weight):   f(θ) = -(1/W) Σ_n w_n log Σ_k π_k p_k(x_n)

For the current parameters the Julia backend returns the truncated model moments
m_k = E_k[X], M_k = E_k[XXᵀ] (per covariance block) and log Z_k; JAX returns
Nk, S1, S2 and the log likelihood. The exact gradients are

    ∂ℓ/∂μ_k = Σ_k⁻¹ (S1_k - N_k m_k)
    ∂ℓ/∂Σ_k = ½ Σ_k⁻¹ [D_k - N_k C_k] Σ_k⁻¹
        D_k = S2_k - μ S1ᵀ - S1 μᵀ + N_k μμᵀ,   C_k = M_k - μ mᵀ - m μᵀ + μμᵀ
    ∂ℓ/∂h_k = N_k - W π_k                         (π = softmax(h))

applied block by block. Optimizer coordinates are: means, log σ per coordinate,
an unconstrained correlation factor per covariance block (every trial
covariance is positive definite and block diagonal, for any block size), and
the mixture logits. σ can be capped at a multiple of the box width.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax

from .blocks import BlockStructure

_MIN_WEIGHT = 1e-12


class CholeskyParameterization:
    """Packing of (means, covariances, weights) <-> unconstrained vector θ.

    Every covariance block is written as Σ_B = D R D with D = diag(σ) and R a
    correlation matrix. θ holds, per component: the means, log σ for every
    coordinate, the strictly-lower entries u of each block's correlation factor
    (row i of the Cholesky factor of R is (u_i, 1)/‖(u_i, 1)‖, so R has a unit
    diagonal and is positive definite for any u), and the mixture logits.
    Because σ is a coordinate, ``max_sigma`` caps each σ_i exactly (a box
    bound for L-BFGS-B), for diagonal and full blocks alike.
    """

    def __init__(self, bs: BlockStructure, K: int):
        self.bs, self.K, self.d = bs, K, bs.d
        self.strict = [np.tril_indices(len(idx), -1) for idx in bs.blocks]
        self.n_mean = K * self.d
        self.n_logsig = K * self.d
        self.n_corr = K * sum(len(r) for r, _ in self.strict)
        self.size = self.n_mean + self.n_logsig + self.n_corr + K

    # ---- layout helpers
    def _split(self, theta):
        K, d = self.K, self.d
        o = 0
        means = theta[o:o + K * d].reshape(K, d); o += K * d
        logsig = theta[o:o + K * d].reshape(K, d); o += K * d
        corr = []
        for r, _ in self.strict:
            corr.append(theta[o:o + K * len(r)].reshape(K, len(r))); o += K * len(r)
        return means, logsig, corr, theta[o:o + K]

    def bounds(self, lower, upper, max_sigma_factor):
        """L-BFGS-B bounds: σ_i ≤ f·(b_i - a_i), means within f box widths of the box."""
        lb = np.full(self.size, -np.inf)
        ub = np.full(self.size, np.inf)
        if max_sigma_factor is not None:
            width = np.asarray(upper, float) - np.asarray(lower, float)
            fin = np.isfinite(width)
            cap = np.where(fin, np.log(max_sigma_factor * np.where(fin, width, 1.0)), np.inf)
            mlo = np.where(fin, lower - max_sigma_factor * np.where(fin, width, 0.0), -np.inf)
            mhi = np.where(fin, upper + max_sigma_factor * np.where(fin, width, 0.0), np.inf)
            K, d = self.K, self.d
            lb[:K * d] = np.tile(mlo, K)
            ub[:K * d] = np.tile(mhi, K)
            ub[K * d:2 * K * d] = np.tile(cap, K)
        return list(zip(np.where(np.isfinite(lb), lb, None), np.where(np.isfinite(ub), ub, None))), lb, ub

    def pack(self, means, covs, weights):
        K, d = self.K, self.d
        sig = np.sqrt(np.einsum("kii->ki", covs))
        parts = [np.asarray(means, dtype=np.float64).reshape(-1), np.log(sig).reshape(-1)]
        for idx, (r, c) in zip(self.bs.blocks, self.strict):
            ix = np.array(idx)
            S = covs[:, ix][:, :, ix]
            s = sig[:, ix]
            R = S / (s[:, :, None] * s[:, None, :])
            L = np.linalg.cholesky(0.5 * (R + np.swapaxes(R, 1, 2)))
            U = L / np.einsum("kii->ki", L)[:, :, None]           # rows scaled so the diagonal is 1
            parts.append(U[:, r, c].reshape(-1))
        w = np.maximum(np.asarray(weights, dtype=np.float64), _MIN_WEIGHT)
        h = np.log(w / w.sum())
        parts.append(h - h.mean())
        return np.concatenate(parts)

    def unpack(self, theta):
        K, d = self.K, self.d
        means, logsig, corr, h = self._split(theta)
        sig = np.exp(logsig)
        covs = np.zeros((K, d, d))
        aux = []
        for idx, (r, c), u in zip(self.bs.blocks, self.strict, corr):
            n = len(idx)
            ix = np.array(idx)
            V = np.zeros((K, n, n))
            V[:, np.arange(n), np.arange(n)] = 1.0
            V[:, r, c] = u
            norm = np.sqrt(np.sum(V * V, axis=2))                  # ‖(u_i, 1)‖ per row
            L = V / norm[:, :, None]
            R = L @ np.swapaxes(L, 1, 2)
            s = sig[:, ix]
            covs[np.ix_(np.arange(K), ix, ix)] = s[:, :, None] * R * s[:, None, :]
            aux.append((L, norm, R))
        pi = np.exp(h - h.max())
        return means, covs, pi / pi.sum(), (sig, aux)

    def gradient(self, theta, means, covs, pi, extra, Nk, S1, S2, model_means, model_second, W):
        """Gradient of ℓ (not yet normalized) in θ coordinates."""
        bs = self.bs
        sig, aux = extra
        S2d = bs.unpack(S2)
        Md = bs.unpack(model_second)
        g_mu = np.zeros_like(means)
        g_logsig = np.zeros_like(means)
        g_corr = []
        outer = lambda u, v: u[:, :, None] * v[:, None, :]
        for idx, (r, c), (L, norm, R) in zip(bs.blocks, self.strict, aux):
            ix = np.array(idx)
            Sig = covs[:, ix][:, :, ix]
            mu, s1, m = means[:, ix], S1[:, ix], model_means[:, ix]
            s2, M = S2d[:, ix][:, :, ix], Md[:, ix][:, :, ix]
            g_mu[:, ix] = np.linalg.solve(Sig, (s1 - Nk[:, None] * m)[..., None])[..., 0]
            D = s2 - outer(mu, s1) - outer(s1, mu) + Nk[:, None, None] * outer(mu, mu)
            C = M - outer(mu, m) - outer(m, mu) + outer(mu, mu)
            Sinv = np.linalg.inv(Sig)
            G = 0.5 * Sinv @ (D - Nk[:, None, None] * C) @ Sinv
            G = 0.5 * (G + np.swapaxes(G, 1, 2))                   # dℓ = tr(G dΣ)
            s = sig[:, ix]
            # Σ = D R D:  dℓ/dσ_i = 2 Σ_j G_ij R_ij σ_j ;  dℓ/dR = D G D
            g_logsig[:, ix] = s * 2.0 * np.einsum("kij,kij,kj->ki", G, R, s)
            if len(r):
                GR = s[:, :, None] * G * s[:, None, :]
                gL = 2.0 * GR @ L                                  # R = L Lᵀ
                # L_i = v_i / ‖v_i‖:  dℓ/dv_i = (g_i - (g_i·L_i) L_i) / ‖v_i‖  (lower-triangular entries)
                n = len(idx)
                tri = np.tril(np.ones((n, n)))
                gL = gL * tri
                proj = np.sum(gL * L, axis=2, keepdims=True)
                gV = (gL - proj * L) / norm[:, :, None]
                g_corr.append(gV[:, r, c].reshape(-1))
        g_h = Nk - W * pi
        return np.concatenate([g_mu.reshape(-1), g_logsig.reshape(-1)] + g_corr + [g_h])


@dataclass
class ObjectiveCounters:
    evaluations: int = 0
    failed_evaluations: int = 0
    elapsed_julia: float = 0.0
    elapsed_gpu: float = 0.0
    elapsed_transfer: float = 0.0


class ObservedLikelihood:
    """f(θ) and ∇f(θ): one batched Julia backend call + one JAX pass per evaluation."""

    def __init__(self, estep, julia, param: CholeskyParameterization, lower, upper, W, device):
        self.estep, self.julia, self.param = estep, julia, param
        self.lower, self.upper, self.W, self.device = lower, upper, W, device
        self.counters = ObjectiveCounters()
        self.last = None
        self.best = (np.inf, None)

    def evaluate_state(self, means, covs, pi):
        """Julia backend + E-step at explicit parameters. Returns (ok, dict)."""
        t0 = time.perf_counter()
        ev = self.julia.evaluate(means, covs, self.lower, self.upper)
        t1 = time.perf_counter()
        self.counters.elapsed_julia += t1 - t0
        if not all(s == "ok" for s in ev.status):
            return False, dict(status=ev.status)
        params = jax.device_put((means, self.param.bs.pack(covs), np.log(pi), ev.log_normalizers), self.device)
        beta = jax.device_put(np.float64(1.0), self.device)
        t2 = time.perf_counter()
        self.counters.elapsed_transfer += t2 - t1
        Nk, S1, S2, ll, _ = jax.device_get(self.estep(params, beta))
        self.counters.elapsed_gpu += time.perf_counter() - t2
        return True, dict(Nk=Nk, S1=S1, S2=S2, ll=float(ll), log_normalizers=ev.log_normalizers,
                          model_means=ev.model_means, model_second=ev.model_second_moments, status=ev.status)

    def __call__(self, theta):
        self.counters.evaluations += 1
        means, covs, pi, extra = self.param.unpack(theta)
        ok, st = self.evaluate_state(means, covs, pi)
        if not ok or not np.isfinite(st["ll"]):
            # an invalid trial point (e.g. zero truncation mass): reject it so the line search backtracks
            self.counters.failed_evaluations += 1
            return np.inf, np.zeros_like(theta)
        g = self.param.gradient(theta, means, covs, pi, extra, st["Nk"], st["S1"], st["S2"],
                                st["model_means"], st["model_second"], self.W)
        f = -st["ll"] / self.W
        self.last = (theta.copy(), f, st)
        if f < self.best[0]:
            self.best = (f, theta.copy())
        return f, -g / self.W


def gradient_norm(param: CholeskyParameterization, means, covs, pi, Nk, S1, S2, model_means, model_second, W,
                  lb=None, ub=None):
    """Per-unit-weight ∞-norm of the (bound-projected) ∇ℓ in optimizer coordinates."""
    theta = param.pack(means, covs, pi)
    if lb is not None:
        theta = np.clip(theta, lb, ub)
    m, c, p, extra = param.unpack(theta)
    g = param.gradient(theta, m, c, p, extra, Nk, S1, S2, model_means, model_second, W) / W
    if lb is not None:
        # ascent direction on ℓ is +g: components pushing past an active bound do not count
        g = np.where((theta >= ub - 1e-12) & (g > 0), 0.0, g)
        g = np.where((theta <= lb + 1e-12) & (g < 0), 0.0, g)
    return float(np.max(np.abs(g)))
