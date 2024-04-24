from .julia_import import jl
from dataclasses import dataclass
from .conversions import jl_to_pandas, pandas_to_jl
from .julia_helpers import jl_array
import juliacall
import numpy as np
from typing import Any, List, Optional

@dataclass
class TGMM:
	gmm :  Any
	cols : Optional[List[str]] = None
	domain_cols : Optional[List[str]] = None
	image_cols : Optional[List[str]] = None
	transformation : Optional[Any] = None

	def __post_init__(self):
		self._means = np.stack([jl_array(a.normal.μ) for a in self.gmm.components])
		self._covariances = np.stack([jl_array(a.normal.Σ) for a in self.gmm.components])
		jl.seval("using LinearAlgebra")
		self._std_deviations = np.sqrt(np.stack([jl_array(jl.diag(a.normal.Σ)) for a in self.gmm.components]))
		self._weights = np.array(self.gmm.prior.p)
		self.d = self.means.shape[-1]

		if self.transformation is not None:
			self.domain_cols = [str(x) for x in self.transformation.domain_columns]
			self.image_cols = [str(x) for x in self.transformation.image_columns]
			self.cols = self.image_cols
		elif self.cols is None:
			self.cols = [f"x_{i}" for i in range(self.d)]
			self.domain_cols = self.cols
			self.image_cols = self.image_cols

	@property
	def means(self):
		return self._means

	@property
	def covariances(self):
		return self._covariances

	@property
	def std_deviations(self):
		return self._std_deviations

	@property
	def weights(self):
		return self._weights

	def data_product(self, df, analytic_columns, sampled_columns, N=1000):
		df = df.copy()
		if isinstance(df, juliacall.AnyValue):
			df = jl_to_pandas(df)

		cols = [col for col in df.columns if col not in ["components"]]
		indices = {cols[i] : i for i in range(len(cols))}
		analytic_indices = [indices[col] for col in analytic_columns]
		sampled_indices = [indices[col] for col in sampled_columns]

		components = range(len(self.weights))

		dfs = []

		for component in components:
			df_component = df[(df["components"] == (component + 1))].sample(N, replace=True)
			dfs.append(df_component)

		data = {col : np.stack([dfs[i][col].values for i in components]) for col in cols}

		for i,k in enumerate(analytic_indices):
			data[analytic_columns[i] + "_mu_kernel"] = self.means[:,k]
			data[analytic_columns[i] + "_sigma_kernel"] = self.std_deviations[:,k]

		data["weights"] = self.weights

		return data

	def sample(self, N=1000):
		X = jl_array(jl.rand(self.gmm, N))
		if self.transformation is not None:
			df_in = jl.DataFrame(jl.collect(jl.transpose(X)), self.image_cols)
			df_out = jl.TruncatedGaussianMixtures.inverse(self.transformation, df_in)
			return jl_to_pandas(df_out)
		else:
			return jl_to_pandas(jl.DataFrame(jl.collect(jl.transpose(X)), self.cols))

	def sample_with_fixed_columns(self, df, analytic_columns, sampled_columns):
		df_out = df.copy()
		analytic_columns_transformed = analytic_columns.copy()
		if self.transformation is not None:
			df_out = jl_to_pandas(jl.TruncatedGaussianMixtures.forward(self.transformation, pandas_to_jl(df_out[self.domain_cols])))
			df_out["components"] = df["components"]
			domain_cols_to_image_cols = {k:v for k,v in zip(self.domain_cols, self.image_cols)}
			analytic_columns_transformed = [domain_cols_to_image_cols[a] for a in analytic_columns]
		cols = [col for col in df.columns if col not in ["components"]]
		indices = {cols[i] : i for i in range(len(cols))}
		analytic_indices = [indices[col] for col in analytic_columns]
		components = range(len(self.weights))

		for component in components:
			for i,k in enumerate(analytic_indices):
				in_component = (df_out["components"] == (component + 1))
				N = in_component.sum()
				df_out.loc[in_component, analytic_columns_transformed[i]] = jl_array(jl.rand(self.gmm.components[component], N)[k, :])

		if self.transformation is not None:
			df_out = jl_to_pandas(jl.TruncatedGaussianMixtures.inverse(self.transformation, pandas_to_jl(df_out[self.image_cols])))
			df_out["components"] = df["components"]

		return df_out




	