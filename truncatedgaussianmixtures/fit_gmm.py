from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import pandas as pd
import numpy as np
from .conversions import *
from .gmm import *
from .transformations import *


def convert_array_or_pandas(data):
	if isinstance(data, pd.DataFrame):
		return pandas_to_jl(data)
	else:
		return jl_array(data)


def make_statsbase_weights(data):
	jl.seval("using StatsBase: Weights")
	return jl.Weights(jl_array(data))


BACKENDS = ("julia", "jax")


def fit_gmm(data : Union[pd.DataFrame, np.array],
			N : int, a : Union[np.array, List], b : Union[np.array, List],
			cov : str = "diag", tol : float = 1e-6, MAX_REPS : int = 200,
			verbose : bool = False, progress : bool = True,
			block_structure : Optional[Union[np.array, List]] = None,
			boundary_unbiasing : Optional[Any] = None,
			weights : Optional[Union[np.array, List]] = None,
			transformation : Optional[Any] = None,
			annealing_schedule : Optional[Any] = None,
			ignore_columns : Optional[List] = None,
			backend : str = "jax",
			device : Optional[str] = None,
			dtype : str = "float64",
			method : str = "em",
			return_responsibilities : bool = False,
			seed : Optional[int] = None,
			initial_parameters : Optional[dict] = None,
			**backend_options
			):
	"""Fit a truncated Gaussian mixture with ``N`` components inside the box ``[a, b]``.

	``backend="jax"`` (default) runs the E-step, likelihood and sufficient statistics with
	JAX (``device="gpu"``/``"cpu"``, default: JAX's default device) and one
	batched Julia M-step per iteration. Extra ``backend_options`` for the JAX
	backend: ``init`` (``"kmeans"`` JAX k-means, default, or ``"julia"``),
	``chunk_size``, ``quadratic`` (``"cholesky"``/``"precision"``),
	``min_iterations``, ``n_consecutive``, ``parameter_tol``,
	``qmc_points_per_dim``, ``julia_threaded``, ``variance_floor``, and for
	``method="hybrid"``/``"gradient"``: ``switch_tol``, ``switch_patience``,
	``grad_tol``, ``gradient_maxiter``, ``switch_on_unhealthy``.

	``method="em"``: EM (JAX E-step + Julia M-step). ``method="gradient"``:
	L-BFGS on the observed likelihood with analytic gradients from the Julia
	backend. ``method="hybrid"``: EM, switching one way to L-BFGS when an
	M-step is unhealthy (e.g. the moment-matching fallback) or EM progress is
	slow while the gradient is not small.

	``backend="julia"`` runs the original TruncatedGaussianMixtures.jl fitter.

	``return_responsibilities=True`` stores the final ``[n_obs, N]``
	responsibilities on the result (8·n_obs·N bytes); otherwise they are
	computed lazily from ``data`` on first access.
	"""
	from . import julia_import
	if julia_import._USE_REVISE:
		julia_import.revise()
	if backend not in BACKENDS:
		raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
	if dtype != "float64":
		raise ValueError("only dtype='float64' is supported")
	if method not in ("em", "gradient", "hybrid"):
		raise ValueError(f"method must be 'em', 'gradient' or 'hybrid', got {method!r}")
	if backend == "julia" and method != "em":
		raise ValueError(f"method={method!r} requires backend='jax'")
	if backend == "jax":
		backend_options["method"] = method
		return _fit_gmm_jax(data, N, a, b, cov=cov, tol=tol, MAX_REPS=MAX_REPS, verbose=verbose, progress=progress,
							block_structure=block_structure, boundary_unbiasing=boundary_unbiasing,
							weights=weights, transformation=transformation,
							annealing_schedule=annealing_schedule, ignore_columns=ignore_columns,
							device=device, return_responsibilities=return_responsibilities, seed=seed,
							initial_parameters=initial_parameters, **backend_options)
	if device is not None:
		raise ValueError("device= is only meaningful for backend='jax'")
	if initial_parameters is not None:
		raise ValueError("initial_parameters= is only supported by backend='jax'")
	if backend_options:
		raise TypeError(f"unexpected options for backend='julia': {sorted(backend_options)}")
	if seed is not None:
		jl.seval("import Random")
		jl.Random.seed_b(int(seed))

	if ignore_columns is not None:
		cols = [col for col in data.columns if col not in ignore_columns]
		ignored_cols = ignore_columns
	else:
		cols = list(data.columns)
		ignored_cols = []

	if (ignore_columns is not None) and (transformation is None):
		transformation = Transformation(cols, "(x...) -> identity(x)", cols, "(x...) -> identity(x)", ignored_cols)

	if transformation is not None:
		ignored_cols = getattr(transformation, 'ignore_columns', [])
		if ignored_cols is None:
			ignored_cols = []
	else:
		ignored_cols = []

	if isinstance(data, pd.DataFrame):
		#print(type(ignored_cols), ignore_columns)
		cols = [col for col in data.columns if col not in ignored_cols]

	data = convert_array_or_pandas(data)
	a = jl_convert(jl.Vector[jl.Float64], a); b = jl_convert(jl.Vector[jl.Float64], b);
	#weights = jl_convert(jl.Vector[jl.Float64], weights)


	kwargs = dict(cov=cov, verbose=verbose, tol=tol, MAX_REPS=MAX_REPS, progress=progress)

	if boundary_unbiasing is not None:
		boundary_unbiasing_jl = jl.TruncatedGaussianMixtures.BoundaryUnbiasing(boundary_unbiasing["structure"],
												  boundary_unbiasing.get("bandwidth_scale", 1.0))
		kwargs['unbiasing'] = boundary_unbiasing_jl

	if block_structure is not None:
		kwargs["block_structure"] =  jl_convert(jl.Vector[jl.Int64], block_structure)

	if weights is not None:
		kwargs["weights"] = make_statsbase_weights(weights)

	common = dict(block_structure=block_structure, cov=cov, backend="julia",
				  sample_weights=None if weights is None else np.asarray(weights, dtype=np.float64))

	if (transformation is None) and (annealing_schedule is None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, **kwargs)
		fit = TGMM.from_julia(gmm, cols, data=jl_to_pandas(data), **common)

	elif (transformation is None) and (annealing_schedule is not None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, annealing_schedule, **kwargs)
		fit = TGMM.from_julia(gmm, cols, data=jl_to_pandas(data), **common)

	elif (transformation is not None) and (annealing_schedule is None):
		gmm, df = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation.julia_object, **kwargs)
		fit = TGMM.from_julia(gmm, cols, transformation=transformation, data=jl_to_pandas(df), **common)

	else:
		gmm, df =TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation.julia_object, annealing_schedule, **kwargs)
		fit = TGMM.from_julia(gmm, cols, transformation=transformation, data=jl_to_pandas(df), **common)

	if return_responsibilities:
		_ = fit.responsibilities
	return fit


def _beta_list(annealing_schedule):
	if annealing_schedule is None:
		return None
	if isinstance(annealing_schedule, (list, tuple, np.ndarray)):
		return [float(x) for x in annealing_schedule]
	return [float(x) for x in np.asarray(annealing_schedule.β_list)]


def _fit_gmm_jax(data, N, a, b, *, cov, tol, MAX_REPS, verbose, progress, block_structure, boundary_unbiasing,
				 weights, transformation, annealing_schedule, ignore_columns, device,
				 return_responsibilities, seed, initial_parameters, **options):
	try:
		from .jax_backend import fit_gmm_jax
	except ImportError as e:
		raise ImportError("backend='jax' (the default) needs JAX: install truncatedgaussianmixtures[cuda] "
						  "(or [jax] for CPU-only), or pass backend='julia'") from e

	# ---- select fitted and ignored columns exactly as the Julia path does
	is_df = isinstance(data, pd.DataFrame)
	if not is_df:
		if transformation is not None or ignore_columns is not None:
			raise ValueError("transformations / ignore_columns require a pandas DataFrame input")
		X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
		if X.ndim != 2:
			raise ValueError("array input must be [n_observations, n_dimensions]")
		cols = [f"x_{i}" for i in range(X.shape[1])]
		df_in = pd.DataFrame(X, columns=cols)
	else:
		cols = [col for col in data.columns if col not in (ignore_columns or [])]
		if (ignore_columns is not None) and (transformation is None):
			transformation = Transformation(cols, "(x...) -> identity(x)", cols, "(x...) -> identity(x)", list(ignore_columns))
		if transformation is not None:
			ignored = getattr(transformation, 'ignore_columns', None) or []
			cols = [col for col in data.columns if col not in ignored]
		df_in = data

	# ---- a Julia transformation runs once, on CPU, before fitting
	if transformation is not None:
		fwd = jl_to_pandas(jl.forward(transformation.julia_object, pandas_to_jl(df_in)))
		image_cols = [str(c) for c in transformation.julia_object.image_columns]
		X = np.ascontiguousarray(fwd[image_cols].values, dtype=np.float64)
	elif is_df:
		X = np.ascontiguousarray(df_in[cols].values, dtype=np.float64)

	w = None if weights is None else np.asarray(weights, dtype=np.float64)
	need_resp = return_responsibilities or transformation is not None
	res = fit_gmm_jax(X, N, a, b, cov=cov, block_structure=block_structure, sample_weights=w, tol=tol,
					  MAX_REPS=MAX_REPS, beta_schedule=_beta_list(annealing_schedule),
					  initial_parameters=initial_parameters, seed=seed, boundary_unbiasing=boundary_unbiasing,
					  return_responsibilities=need_resp, device=device, verbose=verbose, progress=progress,
					  **options)

	common = dict(means=res.means, covariances=res.covariances, weights=res.weights, lower=res.lower,
				  upper=res.upper, log_normalizers=res.log_normalizers, block_structure=block_structure,
				  cov=cov, sample_weights=w, diagnostics=res.diagnostics, backend="jax")

	if transformation is None:
		fit = TGMM(cols=cols, data=df_in if is_df else df_in, **common)
		if return_responsibilities:
			fit.responsibilities = res.responsibilities
		return fit

	# ---- mirror the Julia transformation path: assign components, restore
	#      ignored columns, and map the data back through the inverse transformation
	assignments = TGMM._sample_components(res.responsibilities)
	df_out = pd.DataFrame(X, columns=image_cols)
	df_out["components"] = assignments
	for col in transformation.julia_object.ignore_columns:
		df_out[str(col)] = data[str(col)].values
	df_out = jl_to_pandas(jl.inverse(transformation.julia_object, pandas_to_jl(df_out)))
	df_out["components"] = assignments
	fit = TGMM(cols=cols, transformation=transformation, data=df_out, **common)
	if return_responsibilities:
		fit.responsibilities = res.responsibilities
	return fit
