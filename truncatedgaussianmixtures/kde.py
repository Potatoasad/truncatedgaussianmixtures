from .julia_import import jl
from dataclasses import dataclass
from .conversions import jl_to_pandas, pandas_to_jl
from .julia_helpers import jl_array
import juliacall
import numpy as np
from typing import Any, List, Optional

@dataclass
class KDE:
	gmm :  Any
	cols : Optional[List[str]] = None
	data : Optional[Any] = None

	def __post_init__(self):
		jl.seval("using LinearAlgebra")
		jl.seval("using Distributions")
		self.d = jl.length(self.gmm)
		if self.cols is None:
			if self.data is None:
				self.cols = [f'x_{i}' for i in len(self.d)]
			else:
				self.cols = list(self.data.columns)

		self.bandwidth = np.sqrt(np.array(jl.diag(self.gmm.components[1].normal.Σ)))

	def logpdf(self, x):
		if isinstance(x, np.ndarray):
			if len(x.shape) == 1:
				return jl.logpdf(self.gmm, jl_array(x))#jl_array(x.reshape(1, len(x))))
			else:
				return jl.logpdf(self.gmm, jl_array(x.transpose()))
		elif isinstance(x, float) or isinstance(x, int):
			return jl.logpdf(self.gmm, jl_array(np.array([x])))
		else:
			raise ValueError("KDE can only accept a numpy array or a float")

	def pdf(self, x):
		if isinstance(x, np.ndarray):
			if len(x.shape) == 1:
				return jl.pdf(self.gmm, jl_array(x))#jl_array(x.reshape(1, len(x))))
			else:
				return jl.pdf(self.gmm, jl_array(x.transpose()))
		elif isinstance(x, float) or isinstance(x, int):
			return jl.pdf(self.gmm, jl_array(np.array([x])))
		else:
			raise ValueError("KDE can only accept a numpy array or a float")

	def sample(self, N=1000):
		X = jl_array(jl.rand(self.gmm, N))
		return jl_to_pandas(jl.DataFrame(jl.collect(jl.transpose(X)), self.cols))




	