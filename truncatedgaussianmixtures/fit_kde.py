from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert 
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import pandas as pd
import numpy as np
from .conversions import *
from .gmm import *
from .kde import *
from .transformations import *


def convert_array_or_pandas(data):
	if isinstance(data, pd.DataFrame):
		return pandas_to_jl(data)
	else:
		return jl_array(data)


def make_statsbase_weights(data):
	jl.seval("using StatsBase: Weights")
	return jl.Weights(jl_array(data))


def fit_kde(df, a, b, subsampling=None, full_subsampling=None, weights=None, fix=False, progress=True, bandwidth_scale=None):
	kwargs = dict(
		subsampling=subsampling or jl.nothing, 
		full_subsampling=full_subsampling or jl.nothing, 
		weights=weights or jl.nothing, 
		fix=fix, 
		progress=progress, 
		bandwidth_scale = bandwidth_scale or jl.nothing
	)
	data = convert_array_or_pandas(df)
	a = jl_convert(jl.Vector[jl.Float64], a); b = jl_convert(jl.Vector[jl.Float64], b);
	gmm, indices = TruncatedGaussianMixtures.fit_kde(data, a, b, **kwargs)
	return KDE(gmm, cols=list(df.columns), data=jl_to_pandas(data))