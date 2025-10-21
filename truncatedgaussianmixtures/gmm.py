from .julia_import import jl
from dataclasses import dataclass
from .conversions import jl_to_pandas, pandas_to_jl
from .julia_helpers import jl_array
import juliacall
import numpy as np
from typing import Any, List, Optional
from .save_load import save_dict_h5, load_dict_h5
from .transformations import Transformation

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
	sample_weights : Any = None

	def __post_init__(self):
		self._means = np.stack([jl_array(a.normal.μ) for a in self.gmm.components])
		self._covariances = np.stack([jl_array(a.normal.Σ) for a in self.gmm.components])
		jl.seval("using LinearAlgebra")
		jl.seval("using Distributions")
		jl.seval("import Distributions")
		jl.seval("get_znk(gmm,df) = [TruncatedGaussianMixtures.Zⁿ(gmm, collect(x)) for x in eachrow(df)]")
		jl.seval("get_comps(znk) = [rand(Distributions.Categorical(z)) for z in znk]")
		jl.seval("get_comps(gmm::Distributions.MixtureModel, df) = get_comps(get_znk(gmm,df))")
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
			self.image_cols = self.cols
			if self.data is not None:
				self.responsibilities = jl.get_znk(self.gmm, pandas_to_jl(self.data[self.cols]))
			else:
				print("Warning: No data output to the result object")


		if self.data is not None:
			if self.transformation is not None:
				self.transformed_data = jl_to_pandas(jl.forward(self.transformation.julia_object, jl.DataFrame(self.data)))
				self.responsibilities = jl.get_znk(self.gmm, pandas_to_jl(self.transformed_data[self.image_cols]))
			else:
				self.responsibilities = jl.get_znk(self.gmm, pandas_to_jl(self.data[self.cols]))
		else:
			print("Warning: No data output to the result object")


			

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
				'extra_funcs' : T.extra_funcs
				}
		else:
			transformation_save = None

		if 'components' in fit2.data.columns:
		    thedata = fit2.data.drop('components', axis=1);
		    component_assignment = fit2.data['components'].values
		else:
		    thedata = fit2.data

		tgmm_save = {
		    'cols' : fit2.cols,
		    'domain_cols' : fit2.domain_cols,
		    'image_cols' : fit2.image_cols,
		    'responsibilities' : np.array([np.array(x) for x in fit2.responsibilities]),
		    'block_structure' : fit2.block_structure,
		    'cov' : fit2.cov,
		    'data' : thedata,
		    'gmm' : {
		        'means' : fit2.means, 
		        'covariances' : fit2.covariances, 
		        'weights' : fit2.weights, 
		        'a' : np.array(fit2.gmm.components[0].a), 
		        'b' : np.array(fit2.gmm.components[0].b)
		    }
		    
		}

		tgmm_save['transformation'] = transformation_save
		if 'components' in fit2.data.columns:
			tgmm_save['components'] = fit2.data['components'].values

		save_dict_h5(filename, tgmm_save)

	@classmethod
	def load(cls, filename):
		result = load_dict_h5(filename)
		if (result.get('transformation', None) is not None):
		    T = Transformation(**result['transformation'])
		else:
		    T = None

		jl.seval("""
		using TruncatedGaussianMixtures, Distributions, DataFrames
		function create_gmm(means, covariances, weights, a, b)
		K = size(weights)[1];
		d = size(means)[2];
		gmm = MixtureModel([TruncatedMvNormal(MvNormal(means[k,:], covariances[k,:,:]), a, b) for k in 1:K], weights)
		gmm
		end
		""")

		gmm = jl.create_gmm(jl_array(result['gmm']['means']), 
		                    jl_array(result['gmm']['covariances']), 
		                    jl_array(result['gmm']['weights']), 
		                    jl_array(result['gmm']['a']), 
		                    jl_array(result['gmm']['b']));

		jl.seval("array_to_vec(v) = [v[i,:] for i in 1:size(v,1)]")

		if ('components' in result.keys()) and (result.get('data',None) is not None):
			result['data']['components'] = result['components']

		obj = cls(gmm = gmm,
		    cols = result['cols'],
		    domain_cols= result['domain_cols'],
		    image_cols= result['image_cols'],
		    transformation= T,
		    responsibilities= jl.array_to_vec(jl_array(result['responsibilities'])),
		    block_structure= result.get('block_structure',None),
		    cov = result.get('cov',None),
		    data = result.get('data',None),
		    sample_weights = result.get('sample_weights',None))

		return obj




	