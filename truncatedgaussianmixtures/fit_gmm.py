from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert 
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import pandas as pd
import numpy as np
from .conversions import *
from .gmm import *


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
			weights : Optional[Union[np.array, List]] = None,
			transformation : Optional[Any] = None,
			annealing_schedule : Optional[Any] = None
			):
	cols = None
	if isinstance(data, pd.DataFrame):
		ignored_cols = getattr(transformation, 'ignore_columns', [])
		cols = [col for col in data.columns if col not in ignored_cols]
	data = convert_array_or_pandas(data)
	a = jl_convert(jl.Vector[jl.Float64], a); b = jl_convert(jl.Vector[jl.Float64], b);


	kwargs = dict(cov=cov, verbose=verbose, tol=tol, MAX_REPS=MAX_REPS, progress=progress)
	if block_structure is not None:
		kwargs["block_structure"] = block_structure

	if weights is not None:
		kwargs["weights"] = make_statsbase_weights(weights)

	if (transformation is None) and (annealing_schedule is None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, **kwargs)
		return TGMM(gmm, cols)

	if (transformation is None) and (annealing_schedule is not None):
		gmm = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, annealing_schedule, **kwargs)
		return TGMM(gmm, cols)

	if (transformation is not None) and (annealing_schedule is None):
		gmm, df = TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation, **kwargs)
		return TGMM(gmm, cols, transformation=transformation), jl_to_pandas(df)

	if (transformation is not None) and (annealing_schedule is not None):
		gmm, df =TruncatedGaussianMixtures.fit_gmm(data, N, a, b, transformation, annealing_schedule, **kwargs)
		return TGMM(gmm, cols, transformation=transformation), jl_to_pandas(df)
