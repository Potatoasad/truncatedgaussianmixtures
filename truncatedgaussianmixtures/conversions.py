from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert 
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd


def pandas_to_jl(df):
	jl.seval("using DataFrames")
	dff = jl.DataFrame(jl_array(df.values), jl.seval(":auto"))
	getattr(jl,"rename!")(dff, jl.convert(jl.seval("Vector{String}"),list(map(str,df.columns))))
	return dff

def jl_to_pandas(df):
	return pd.DataFrame({col : getattr(df,col) for col in jl.names(df)})


