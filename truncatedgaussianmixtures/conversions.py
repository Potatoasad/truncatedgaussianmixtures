from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert 
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd


def pandas_to_jl(df):
	jl.seval("using DataFrames")
	return jl.DataFrame(df)

def jl_to_pandas(df):
	return pd.DataFrame({col : getattr(df,col) for col in jl.names(df)})


