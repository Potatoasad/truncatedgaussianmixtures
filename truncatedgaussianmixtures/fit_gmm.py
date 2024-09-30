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


def fit_gmm(data : Union[pd.DataFrame, np.array],
			N : int, a : Union[np.array, List], b : Union[np.array, List],
			cov : str = "diag", tol : float = 1e-6, MAX_REPS : int = 200,
			verbose : bool = False, progress : bool = True,
			block_structure : Optional[Union[np.array, List]] = None,
			boundary_unbiasing : Optional[Any] = None,
			weights : Optional[Union[np.array, List]] = None,
			transformation : Optional[Any] = None,
			annealing_schedule : Optional[Any] = None,
			ignore_columns : Optional[List] = None
			):
	if ignore_columns is not None:
		cols = [col for col in data.columns if col not in ignore_columns]
		ignored_cols = ignore_columns
	else:
		cols = list(data.columns)
		ignored_cols = []

	if (ignore_columns is not None) and (transformation is None):
		transformation = Transformation(cols, "(x...) -> identity(x)", cols, "(x...) -> identity(x)", ignore_columns)

	if transformation is not None:
		ignored_cols = getattr(transformation, 'ignore_columns', [])
	else:
		ignored_cols = []

	if isinstance(data, pd.DataFrame):
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

	if (transformation is None) and (annealing_schedule is None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, **kwargs)
		return TGMM(gmm, cols, data=jl_to_pandas(data), block_structure=block_structure, cov=cov)

	if (transformation is None) and (annealing_schedule is not None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, annealing_schedule, **kwargs)
		return TGMM(gmm, cols, data=jl_to_pandas(data), block_structure=block_structure, cov=cov)

	if (transformation is not None) and (annealing_schedule is None):
		gmm, df = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation.julia_object, **kwargs)
		return TGMM(gmm, cols, transformation=transformation, data=jl_to_pandas(df), block_structure=block_structure, cov=cov)

	if (transformation is not None) and (annealing_schedule is not None):
		gmm, df =TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation.julia_object, annealing_schedule, **kwargs)
		return TGMM(gmm, cols, transformation=transformation, data=jl_to_pandas(df), block_structure=block_structure, cov=cov)
