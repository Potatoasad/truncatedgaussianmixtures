from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from .conversions import jl_to_pandas, pandas_to_jl
import pandas as pd
from juliacall import convert as jl_convert 
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class QuantileTransformation:
	dataframe : pd.DataFrame
	quantiled_columns : Optional[List[str]] = None
	n_quantiles : Optional[int] = 1_000

@dataclass
class Transformation:
	input_columns : List[str]
	forward_transformation : str
	transformed_columns : List[str]
	inverse_transformation : str
	ignore_columns : Optional[List[str]] = None
	extra_funcs : Optional[List[str]] = None ## list of strings of julia functions used in the transformation
	quantile_transformation : Optional[QuantileTransformation] = None

	def __post_init__(self):
		if self.ignore_columns is None:
			self.ignore_columns = [];
		if self.extra_funcs is not None:
			if type(self.extra_funcs) == type([]):
				for s in self.extra_funcs:
					jl.seval(s)
			elif type(self.extra_funcs) == type("blah"):
				jl.seval(s)
		input_columns_jl = jl_convert(jl.Vector[jl.Symbol], self.input_columns)
		forward_transformation_jl = jl.seval(self.forward_transformation)
		transformed_columns_jl = jl_convert(jl.Vector[jl.Symbol], self.transformed_columns)
		inverse_transformation_jl = jl.seval(self.inverse_transformation)
		transformation_jl_out = TruncatedGaussianMixtures.Transformation(input_columns_jl, forward_transformation_jl, transformed_columns_jl, inverse_transformation_jl, ignore_columns=self.ignore_columns)
		transformation_jl_out_no_ignore = TruncatedGaussianMixtures.Transformation(input_columns_jl, forward_transformation_jl, transformed_columns_jl, inverse_transformation_jl)
		
		if self.quantile_transformation is not None:
			quantile_df = self.quantile_transformation.dataframe
			self.quantile_transformation_indices_ignored = [i+1 for i,x in enumerate(self.input_columns) if x not in self.quantile_transformation.quantiled_columns]
			self.julia_object = jl.add_quantile_transformation(transformation_jl_out, pandas_to_jl(quantile_df), ignore_quantile_columns=jl_array(self.quantile_transformation_indices_ignored), n_quantiles=self.quantile_transformation.n_quantiles)
			self.julia_object_no_ignore = jl.add_quantile_transformation(transformation_jl_out_no_ignore, pandas_to_jl(quantile_df), ignore_quantile_columns=jl_array(self.quantile_transformation_indices_ignored), n_quantiles=self.quantile_transformation.n_quantiles)
		else:
			self.julia_object = transformation_jl_out
			self.julia_object_no_ignore = transformation_jl_out_no_ignore

	def forward(self, df):
		return jl_to_pandas(jl.forward(self.julia_object, pandas_to_jl(df)))

	def inverse(self, df):
		return jl_to_pandas(jl.inverse(self.julia_object, pandas_to_jl(df)))


