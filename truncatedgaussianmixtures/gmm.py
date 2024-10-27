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
	responsibilities : Optional[Any] = None
	block_structure : Optional[Any] = None
	cov : str = "full"
	data : Optional[Any] = None

	def __post_init__(self):
		self._means = np.stack([jl_array(a.normal.μ) for a in self.gmm.components])
		self._covariances = np.stack([jl_array(a.normal.Σ) for a in self.gmm.components])
		jl.seval("using LinearAlgebra")
		jl.seval("using Distributions")
		self._std_deviations = np.sqrt(np.stack([jl_array(jl.diag(a.normal.Σ)) for a in self.gmm.components]))
		self._weights = np.array(self.gmm.prior.p)
		self.d = self.means.shape[-1]

		if self.transformation is not None:
			self.domain_cols = [str(x) for x in self.transformation.julia_object.domain_columns]
			self.image_cols = [str(x) for x in self.transformation.julia_object.image_columns]
			self.cols = self.image_cols
		elif self.cols is None:
			self.cols = [f"x_{i}" for i in range(self.d)]
			self.domain_cols = self.cols
			self.image_cols = self.image_cols

		if self.cov == "full":
			if self.block_structure is None:
				self.block_structure = [0 for _ in range(len(self.cols))]

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

	def logpdf(self, x):
		if isinstance(x, np.ndarray):
			if len(x.shape) == 1:
				return jl.logpdf(self.gmm, jl_array(x.reshape(1, len(x))))
			else:
				return jl.logpdf(self.gmm, jl_array(x.transpose()))
		elif isinstance(x, float) or isinstance(x, int):
			return jl.logpdf(self.gmm, jl_array(np.array([x])))
		else:
			raise ValueError("KDE can only accept a numpy array or a float")

	def pdf(self, x):
		if isinstance(x, np.ndarray):
			if len(x.shape) == 1:
				return jl.pdf(self.gmm, jl_array(x.reshape(1, len(x))))
			else:
				return jl.pdf(self.gmm, jl_array(x.transpose()))
		elif isinstance(x, float) or isinstance(x, int):
			return jl.pdf(self.gmm, jl_array(np.array([x])))
		else:
			raise ValueError("KDE can only accept a numpy array or a float")

	def data_product(self, analytic_columns, sampled_columns, N=1000):
		df = self.data
		df = df.copy()
		if isinstance(df, juliacall.AnyValue):
			df = jl_to_pandas(df)

		cols = [col for col in df.columns if col not in ["components"]]
		indices = {cols[i] : i for i in range(len(cols))}
		analytic_indices = [indices[col] for col in analytic_columns]
		sampled_indices = [indices[col] for col in sampled_columns]

		components = range(len(self.weights))

		df = self.sample_with_fixed_columns(analytic_columns, sampled_columns)

		dfs = []

		for component in components:
			df_component = df[(df["components"] == (component + 1))].sample(N, replace=True)
			dfs.append(df_component)

		data = {col : np.stack([dfs[i][col].values for i in components]) for col in cols}

		for i,k in enumerate(analytic_indices):
			data[analytic_columns[i] + "_mu_kernel"] = self.means[:,k]
			data[analytic_columns[i] + "_sigma_kernel"] = self.std_deviations[:,k]
			for j,l in enumerate(analytic_indices):
				if self.cov == "full":
					if (k != l) and (self.block_structure[k] == self.block_structure[l]):
						data[analytic_columns[i] + "_rho_kernel"] = self._covariances[:, k, l] / np.sqrt(self._covariances[:, l, l] * self._covariances[:, k, k])
						data[analytic_columns[j] + "_rho_kernel"] = self._covariances[:, k, l] / np.sqrt(self._covariances[:, l, l] * self._covariances[:, k, k])

		data["weights"] = self.weights

		return data

	def sample(self, N=1000):
		X = jl_array(jl.rand(self.gmm, N))
		if self.transformation is not None:
			df_in = jl.DataFrame(jl.collect(jl.transpose(X)), self.image_cols)
			df_out = jl.TruncatedGaussianMixtures.inverse(self.transformation.julia_object_no_ignore, df_in)
			return jl_to_pandas(df_out)
		else:
			return jl_to_pandas(jl.DataFrame(jl.collect(jl.transpose(X)), self.cols))

	def generate_component_assignment(self, analytic_columns, sampled_columns, ignore_columns=[], data=None):
		if data is None:
			df = self.data;
		else:
			df = data
		df_out = df.copy()
		analytic_columns_transformed = analytic_columns.copy()
		all_cols = analytic_columns + sampled_columns
		true_columns = [col for col in df.columns if col in all_cols]
		jl.seval("import Distributions")
		jl.seval("get_znk(gmm,df) = [TruncatedGaussianMixtures.Zⁿ(gmm, collect(x)) for x in eachrow(df)]")
		jl.seval("get_comps(znk) = [rand(Distributions.Categorical(z)) for z in znk]")
		jl.seval("get_comps(gmm::Distributions.MixtureModel, df) = get_comps(get_znk(gmm,df))")
		if self.responsibilities is not None:
			df_out['components'] = jl.get_comps(self.responsibilities)
		else:
			df_out['components'] = jl.get_comps(self.gmm, pandas_to_jl(df_out[true_columns]))
			self.responsibilities = jl.get_znk(self.gmm,pandas_to_jl(df_out[true_columns]))
		for col in df.columns:
			if col not in df_out.columns:
				df_out[col] = df[col]
		return df_out

	def sample_with_fixed_columns(self, analytic_columns, sampled_columns, ignore_columns=[]):
		df = self.data;
		df_out = df.copy()
		analytic_columns_transformed = analytic_columns.copy()
		sampled_columns_transformed = sampled_columns.copy()
		ignored_cols = ignore_columns + ['components']
		cols = [col for col in df.columns if col not in ignored_cols]

		if self.transformation is not None:
			df_out = jl_to_pandas(jl.TruncatedGaussianMixtures.forward(self.transformation.julia_object, pandas_to_jl(df_out[self.domain_cols + self.transformation.julia_object.ignore_columns])))
			domain_cols_to_image_cols = {k:v for k,v in zip(self.domain_cols, self.image_cols)}
			analytic_columns_transformed = [domain_cols_to_image_cols[a] for a in analytic_columns_transformed]
			sampled_columns_transformed = [domain_cols_to_image_cols[a] for a in sampled_columns_transformed]

		if 'components' not in df_out.columns:
			df_out = self.generate_component_assignment(analytic_columns_transformed, sampled_columns_transformed, ignore_columns=ignored_cols, data=df_out)

		indices = {cols[i] : i for i in range(len(cols))}
		analytic_indices = [indices[col] for col in analytic_columns]
		components = range(len(self.weights))

		for component in components:
			for i,k in enumerate(analytic_indices):
				in_component = (df_out["components"] == (component + 1))
				N = in_component.sum()
				df_out.loc[in_component, analytic_columns_transformed[i]] = jl_array(jl.rand(self.gmm.components[component], N)[k, :])

		if self.transformation is not None:
			df_out = jl_to_pandas(jl.TruncatedGaussianMixtures.inverse(self.transformation.julia_object, pandas_to_jl(df_out[self.image_cols + self.transformation.julia_object.ignore_columns])))
			df_out["components"] = df["components"]

		return df_out




	