"""Hybrid backend: JAX (E-step, likelihood, sufficient statistics) + Julia (M-step).

Import policy (enforced here, in one place):

* ``truncatedgaussianmixtures`` imports juliacall first (see ``julia_import``);
  this subpackage is only imported when ``backend="jax"`` is requested.
* 64-bit JAX is enabled before any JAX array is created.
* Unless the user configured it, JAX does not preallocate most of the GPU
  memory (``XLA_PYTHON_CLIENT_PREALLOCATE=false``): workstation GPUs are
  commonly shared, and the fit needs only O(N·K) device memory.
"""
import os
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .blocks import BlockStructure  # noqa: E402
from .state import FitDiagnostics, FitIteration, JAXFitState  # noqa: E402
from .likelihood import component_logpdf  # noqa: E402
from .statistics import EStep, make_estep  # noqa: E402
from .fit import fit_gmm_jax, resolve_device  # noqa: E402

__all__ = [
    "BlockStructure", "FitDiagnostics", "FitIteration", "JAXFitState",
    "component_logpdf", "EStep", "make_estep", "fit_gmm_jax", "resolve_device",
]
