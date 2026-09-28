"""Small fit-state and diagnostic dataclasses (host-side, plain NumPy/Python)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional

import numpy as np


@dataclass
class JAXFitState:
    """JAX arrays kept in memory on the JAX device (GPU, or CPU if device="cpu") across EM iterations."""

    X: Any                    # [N, d]   observations (shifted), used for the likelihood
    sample_weights: Any       # [N]
    Y: Any                    # [N, d]   positions used for the statistics (X, or boundary pseudo means)
    V: Any                    # [N, d]   diagonal pseudo variances, or None
    means: Any                # [K, d]
    covariances: Any          # [K, P]   packed blocks
    log_mixture_weights: Any  # [K]
    log_normalizers: Any      # [K]
    iteration: int = 0
    beta: float = 1.0


@dataclass
class FitIteration:
    iteration: int
    beta: float
    average_loglikelihood: float
    delta_loglikelihood: Optional[float]
    method: str
    julia_status: str
    elapsed_gpu: float          # E-step + statistics, including the device synchronization
    elapsed_julia: float        # Python -> Julia M-step call (dispatch + compute)
    elapsed_transfer: float     # host -> device parameter upload
    entropy: float = float("nan")
    moment_matching: bool = False


@dataclass
class FitDiagnostics:
    backend: str
    device: str
    method: str = "em"
    iterations: List[FitIteration] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    final_average_loglikelihood: float = float("nan")
    n_likelihood_decreases: int = 0
    component_status: List[str] = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)
    compile_time: float = float("nan")
    total_time: float = float("nan")
    notes: List[str] = field(default_factory=list)
    # direct / hybrid optimization
    switch_iteration: Optional[int] = None
    switch_reason: Optional[str] = None
    lbfgs_message: Optional[str] = None
    n_gradient_evaluations: int = 0
    n_failed_evaluations: int = 0
    evaluations_per_step: float = float("nan")
    final_gradient_norm: float = float("nan")

    @property
    def average_loglikelihood(self) -> np.ndarray:
        return np.array([it.average_loglikelihood for it in self.iterations])

    def timing_summary(self) -> dict:
        its = self.iterations[1:] if len(self.iterations) > 1 else self.iterations
        tot = lambda name: float(np.sum([getattr(it, name) for it in its]))
        return dict(
            n_iterations=len(self.iterations),
            gpu_estep=tot("elapsed_gpu"),
            julia_mstep=tot("elapsed_julia"),
            transfer=tot("elapsed_transfer"),
            compile_time=self.compile_time,
            total_time=self.total_time,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["iterations"] = {k: np.array([getattr(it, k) for it in self.iterations])
                           for k in FitIteration.__dataclass_fields__
                           if k not in ("method", "julia_status")}
        d["iterations"]["julia_status"] = [it.julia_status for it in self.iterations]
        return d
