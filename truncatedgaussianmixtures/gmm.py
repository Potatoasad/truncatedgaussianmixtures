from .julia_import import jl
from dataclasses import dataclass, field
from .conversions import jl_to_pandas, pandas_to_jl
from .julia_helpers import jl_array
import juliacall
import numpy as np
import pandas as pd
from typing import Any, List, Optional
from .save_load import save_dict_h5, load_dict_h5
from .transformations import Transformation, QuantileTransformation
from .julia_backend import arrays_from_julia_mixture, JuliaBackend
from .distributions import TruncatedGaussianMixture


@dataclass
class TGMM:
	"""A fitted truncated Gaussian mixture.

	The numeric arrays are the canonical state (``means[K, d]``,
	``covariances[K, d, d]``, ``weights[K]``, ``lower``/``upper[d]``).
	``distribution`` is the fitted mixture as a Python
	``TruncatedGaussianMixture`` (in the fitted space, i.e. after any
	transformation); its Julia ``MixtureModel`` is ``julia_gmm`` (alias ``gmm``). Responsibilities are computed lazily from
	``data`` when first accessed, or returned by the fit with
	``return_responsibilities=True``.
	"""
	means : np.ndarray
	covariances : np.ndarray
	weights : np.ndarray
	lower : np.ndarray
	upper : np.ndarray
	cols : Optional[List[str]] = None
	domain_cols : Optional[List[str]] = None
	image_cols : Optional[List[str]] = None
	transformation : Optional[Any] = None
	responsibilities : Optional[Any] = field(default=None, repr=False, compare=False)
	block_structure : Optional[Any] = None
	cov : str = "full"
	data : Optional[Any] = None
	sample_weights : Any = None
	log_normalizers : Optional[np.ndarray] = None
	diagnostics : Optional[Any] = None
	backend : str = "julia"
	_julia_gmm : Any = field(default=None, repr=False)

	def __post_init__(self):
		self.means = np.asarray(self.means, dtype=np.float64)
		self.covariances = np.asarray(self.covariances, dtype=np.float64)
		self.weights = np.asarray(self.weights, dtype=np.float64)
		self.lower = np.asarray(self.lower, dtype=np.float64)
		self.upper = np.asarray(self.upper, dtype=np.float64)
		self.d = self.means.shape[-1]
		self._std_deviations = np.sqrt(np.einsum("kii->ki", self.covariances))
		self._transformed_data = None

		if self.transformation is not None:
			self.domain_cols = [str(x) for x in self.transformation.julia_object.domain_columns]
			self.image_cols = [str(x) for x in self.transformation.julia_object.image_columns]
			self.cols = self.image_cols
		elif self.cols is None:
			self.cols = [f"x_{i}" for i in range(self.d)]
		if self.transformation is None:
			self.domain_cols = self.domain_cols or list(self.cols)
			self.image_cols = self.image_cols or list(self.cols)

		if self.data is None:
			print("Warning: No data output to the result object")

		if self.cov == "full":
			if self.block_structure is None:
				self.block_structure = [0 for _ in range(len(self.cols))]

	# ------------------------------------------------------------------ construction

	@classmethod
	def from_julia(cls, gmm, cols=None, **kwargs):
		"""Wrap a Julia ``MixtureModel`` (legacy fitting backend)."""
		means, covs, weights, logZ, lower, upper = arrays_from_julia_mixture(gmm)
		kwargs.setdefault("log_normalizers", logZ)
		return cls(means=means, covariances=covs, weights=weights, lower=lower, upper=upper, cols=cols,
				   _julia_gmm=gmm, **kwargs)

	@property
	def distribution(self) -> TruncatedGaussianMixture:
		"""The fitted mixture as a ``TruncatedGaussianMixture`` (``sample``, ``logpdf``, ``pdf``, ...)."""
		dist = self.__dict__.get("_distribution")
		if dist is None:
			if self._julia_gmm is not None:
				dist = TruncatedGaussianMixture.from_julia(self._julia_gmm, cols=self.cols)
			else:
				dist = TruncatedGaussianMixture.from_arrays(self.means, self.covariances, self.weights, self.lower,
															self.upper, cols=self.cols, cov=self.cov,
															log_normalizers=self._ensure_log_normalizers())
			self.__dict__["_distribution"] = dist
		return dist

	@property
	def julia_gmm(self):
		if self._julia_gmm is None:
			self._julia_gmm = self.distribution.julia_object
		return self._julia_gmm

	@property
	def gmm(self):
		"""The Julia ``MixtureModel`` (backward compatible name for ``julia_gmm``)."""
		return self.julia_gmm

	# ------------------------------------------------------------------ arrays

	@property
	def std_deviations(self):
		return self._std_deviations

	def _ensure_log_normalizers(self):
		if self.log_normalizers is None:
			K, d = self.means.shape
			labels = None if self.cov == "diag" else self.block_structure
			julia = JuliaBackend(d, K, labels, "diag" if self.cov == "diag" else "full")
			r = julia.evaluate(self.means, self.covariances, self.lower, self.upper)
			self.log_normalizers = r.log_normalizers
		return self.log_normalizers

	def component_logpdf(self, x):
		"""``log p_k(x_n)`` as ``[N, K]`` (``-inf`` outside the truncation box)."""
		return self.distribution.component_logpdf(x)

	def logpdf(self, x):
		if isinstance(x, (float, int)):
			x = np.array([[x]], dtype=np.float64)
		elif not isinstance(x, np.ndarray):
			raise ValueError("logpdf can only accept a numpy array or a float")
		elif x.ndim == 1:
			x = x.reshape(1, len(x))
		return self.distribution.logpdf(x)

	def pdf(self, x):
		return np.exp(self.logpdf(x))

	def compute_responsibilities(self, x):
		"""Posterior component probabilities ``[N, K]`` for points ``x[N, d]``."""
		return self.distribution.responsibilities(x)

	@property
	def transformed_data(self):
		if self._transformed_data is None and self.data is not None and self.transformation is not None:
			self._transformed_data = jl_to_pandas(jl.forward(self.transformation.julia_object, pandas_to_jl(self.data)))
		return self._transformed_data

	# ------------------------------------------------------------------ legacy helpers

	@staticmethod
	def _sample_components(resp, rng=None):
		"""Draw one (1-based, as in Julia) component label per row of ``resp``."""
		rng = np.random.default_rng() if rng is None else rng
		c = np.cumsum(resp, axis=1)
		u = rng.random(resp.shape[0])[:, None] * c[:, -1:]
		return np.minimum((u > c).sum(axis=1), resp.shape[1] - 1) + 1

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

		if len(sampled_columns) != 0:
			df = self.sample_with_fixed_columns(analytic_columns, sampled_columns)

			dfs = []

			for component in components:
				df_component = df[(df["components"] == (component + 1))].sample(N, replace=True)
				dfs.append(df_component)

			data = {col : np.stack([dfs[i][col].values for i in components]) for col in cols}
		else:
			data = {}

		for i,k in enumerate(analytic_indices):
			data[analytic_columns[i] + "_mu_kernel"] = self.means[:,k]
			data[analytic_columns[i] + "_sigma_kernel"] = self.std_deviations[:,k]
			for j,l in enumerate(analytic_indices):
				if self.cov == "full":
					if (k != l) and (self.block_structure[k] == self.block_structure[l]):
						data[analytic_columns[i] + "_rho_kernel"] = self.covariances[:, k, l] / np.sqrt(self.covariances[:, l, l] * self.covariances[:, k, k])
						data[analytic_columns[j] + "_rho_kernel"] = self.covariances[:, k, l] / np.sqrt(self.covariances[:, l, l] * self.covariances[:, k, k])

		data["weights"] = self.weights

		return data

	def sample(self, N=1000, seed=None):
		"""``N`` draws as a DataFrame, mapped back through the inverse transformation if there is one."""
		if self.transformation is None:
			return self.distribution.sample(N, seed=seed)
		X = self.distribution.sample(N, seed=seed, as_dataframe=False)
		df_in = pandas_to_jl(pd.DataFrame(X, columns=self.image_cols))
		df_out = jl.TruncatedGaussianMixtures.inverse(self.transformation.julia_object_no_ignore, df_in)
		return jl_to_pandas(df_out)

	def generate_component_assignment(self, analytic_columns, sampled_columns, ignore_columns=[], data=None):
		if data is None:
			df = self.data;
		else:
			df = data
		df_out = df.copy()
		all_cols = analytic_columns + sampled_columns
		true_columns = [col for col in df.columns if col in all_cols]
		resp = self.__dict__.get("_responsibilities")
		if resp is None or resp.shape[0] != len(df_out):
			resp = self.compute_responsibilities(np.asarray(df_out[true_columns].values, dtype=np.float64))
			self.__dict__["_responsibilities"] = resp
		df_out['components'] = self._sample_components(resp)
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
				df_out.loc[in_component, analytic_columns_transformed[i]] = jl_array(jl.rand(self.julia_gmm.components[component], N)[k, :])

		if self.transformation is not None:
			df_out = jl_to_pandas(jl.TruncatedGaussianMixtures.inverse(self.transformation.julia_object, pandas_to_jl(df_out[self.image_cols + self.transformation.julia_object.ignore_columns])))
			df_out["components"] = df["components"]

		return df_out

	# ------------------------------------------------------------------ save / load

	def save(self, filename):
		fit2 = self;
		if fit2.transformation is not None:
			T = fit2.transformation
			transformation_save = {
				'input_columns' : T.input_columns,
				'forward_transformation' : T.forward_transformation,
				'transformed_columns' : T.transformed_columns,
				'inverse_transformation' : T.inverse_transformation,
				'ignore_columns' : T.ignore_columns,
				'extra_funcs' : T.extra_funcs,
				'quantile_transformation' : None
				}
			if T.quantile_transformation is not None:
				q_transform = {'dataframe' : T.quantile_transformation.dataframe,
							   'quantiled_columns' : T.quantile_transformation.quantiled_columns}
				transformation_save['quantile_transformation'] = q_transform
		else:
			transformation_save = None

		has_data = fit2.data is not None
		if has_data and 'components' in fit2.data.columns:
			thedata = fit2.data.drop('components', axis=1);
		else:
			thedata = fit2.data

		resp = fit2.responsibilities
		tgmm_save = {
			'cols' : fit2.cols,
			'domain_cols' : fit2.domain_cols,
			'image_cols' : fit2.image_cols,
			'responsibilities' : None if resp is None else np.asarray(resp),
			'block_structure' : fit2.block_structure,
			'cov' : fit2.cov,
			'data' : thedata,
			'backend' : fit2.backend,
			'gmm' : {
				'means' : fit2.means,
				'covariances' : fit2.covariances,
				'weights' : fit2.weights,
				'a' : fit2.lower,
				'b' : fit2.upper,
				'log_normalizers' : fit2.log_normalizers,
			}

		}
		if fit2.sample_weights is not None:
			tgmm_save['sample_weights'] = np.asarray(fit2.sample_weights, dtype=np.float64)
		if fit2.diagnostics is not None:
			dd = fit2.diagnostics.to_dict()
			tgmm_save['diagnostics'] = {
				'backend' : dd['backend'], 'device' : dd['device'], 'method' : dd['method'],
				'converged' : bool(dd['converged']), 'n_iterations' : int(dd['n_iterations']),
				'final_average_loglikelihood' : float(dd['final_average_loglikelihood']),
				'average_loglikelihood' : np.asarray(dd['iterations']['average_loglikelihood'], dtype=np.float64),
				'beta' : np.asarray(dd['iterations']['beta'], dtype=np.float64),
				'component_status' : list(dd['component_status']) or None,
			}

		tgmm_save['transformation'] = transformation_save
		if has_data and 'components' in fit2.data.columns:
			tgmm_save['components'] = fit2.data['components'].values

		save_dict_h5(filename, tgmm_save)

	@classmethod
	def load(cls, filename):
		result = load_dict_h5(filename)
		if (result.get('transformation', None) is not None):
			if result['transformation'].get('quantile_transformation', None) is not None:
				transformation_result = result['transformation']['quantile_transformation'];
				result['transformation']['quantile_transformation'] = QuantileTransformation(dataframe=transformation_result['dataframe'], quantiled_columns=transformation_result['quantiled_columns'])
			T = Transformation(**result['transformation'])
		else:
			T = None

		if ('components' in result.keys()) and (result.get('data',None) is not None):
			result['data']['components'] = result['components']

		g = result['gmm']
		obj = cls(means = np.asarray(g['means']),
			covariances = np.asarray(g['covariances']),
			weights = np.asarray(g['weights']),
			lower = np.asarray(g['a']),
			upper = np.asarray(g['b']),
			log_normalizers = None if g.get('log_normalizers', None) is None else np.asarray(g['log_normalizers']),
			cols = result['cols'],
			domain_cols= result['domain_cols'],
			image_cols= result['image_cols'],
			transformation= T,
			responsibilities= result.get('responsibilities', None),
			block_structure= result.get('block_structure',None),
			cov = result.get('cov',None),
			data = result.get('data',None),
			sample_weights = result.get('sample_weights',None),
			backend = result.get('backend', 'julia'))
		obj.loaded_diagnostics = result.get('diagnostics', None)
		return obj


# `responsibilities` is a dataclass field for construction, but reads are lazy:
# attach the property after @dataclass has generated __init__.
def _get_responsibilities(self):
	resp = self.__dict__.get("_responsibilities")
	if resp is None and self.data is not None:
		if self.transformation is not None:
			x = self.transformed_data[self.image_cols].values
		else:
			x = self.data[self.cols].values
		resp = self.compute_responsibilities(np.asarray(x, dtype=np.float64))
		self.__dict__["_responsibilities"] = resp
	return resp


def _set_responsibilities(self, value):
	self.__dict__["_responsibilities"] = None if value is None else np.asarray(value, dtype=np.float64)


TGMM.responsibilities = property(_get_responsibilities, _set_responsibilities,
								 doc="Posterior component probabilities [N, K] of `data` (computed lazily).")
