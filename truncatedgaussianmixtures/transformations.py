from .julia_import import jl, TruncatedGaussianMixtures
from juliacall import convert as jl_convert 
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Transformation:
	input_columns : List[str]
	forward_transformation : str
	transformed_columns : List[str]
	inverse_transformation : str
	ignore_columns : Optional[List[str]] = None
	extra_funcs : Optional[List[str]] = None ## list of strings of julia functions used in the transformation

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
		self.julia_object = transformation_jl_out
		self.julia_object_no_ignore = transformation_jl_out_no_ignore