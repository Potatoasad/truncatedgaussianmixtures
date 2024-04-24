from .julia_import import jl, TruncatedGaussianMixtures
from juliacall import convert as jl_convert 
from dataclasses import dataclass, field
from typing import List, Dict, Optional

def Transformation(
	input_columns : List[str],
	forward_transformation : str,
	transformed_columns : List[str],
	inverse_transformation : str,
	ignore_columns : Optional[List[str]] = []):
		input_columns_jl = jl_convert(jl.Vector[jl.Symbol], input_columns)
		forward_transformation_jl = jl.seval(forward_transformation)
		transformed_columns_jl = jl_convert(jl.Vector[jl.Symbol], transformed_columns)
		inverse_transformation_jl = jl.seval(inverse_transformation)
		transformation_jl_out = TruncatedGaussianMixtures.Transformation(input_columns_jl, forward_transformation_jl, transformed_columns_jl, inverse_transformation_jl, ignore_columns=ignore_columns)
		return transformation_jl_out