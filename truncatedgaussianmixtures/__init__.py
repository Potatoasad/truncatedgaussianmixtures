# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
from .julia_import import jl, TruncatedGaussianMixtures, force_install_tgmm  # isort:skip
#force_install_tgmm(force=False)

jl.seval("using StatsBase: Weights")

def jlarray(x):
	return jl.pyconvert(jl.seval("Array"), x)

#fit_gmm  = TruncatedGaussianMixtures.fit_gmm
#transformation_jl  = TruncatedGaussianMixtures.Transformation



from .transformations import *
from .conversions import *
from .annealing_schedule import *
from .fit_gmm import *
from .fit_kde import *


#from .catalog


########
# weights = np
# dataframe = df OR X = np
# n_kernels
# a List or np.array
# b List or np.array
# cov "diag" or "full"
# weights None or ones_like(len(X))
# 
