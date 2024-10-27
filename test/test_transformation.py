from ..truncatedgaussianmixtures import *
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

def test_transformation():
	#force_install_tgmm()
	Transformation(['x'], "(x,) -> (x^2,)", ['y'], "(y,) -> (y^0.5,)")

def test_tgmm_run():
	#force_install_tgmm()

	T = Transformation(['x', 'y', 'z'],
						"(x,y,z) -> (x^2,y,cos(z))", 
						['x1','y', 'z1'], 
						"(x1,y,z1) -> (x^(0.5),y,acos(z))")


	df = pd.DataFrame({'x' : np.random.rand(1_000), 
					  'y' : np.random.rand(1_000), 
					  'z' : np.random.rand(1_000)})


	fit = fit_gmm(df, 3, [0,0,0], [1,1,1], cov="full", progress=True, 
				  boundary_unbiasing={'structure' : [1,1,1], 'bandwidth_scale' : 0.01},  # extra boundary correction will be applied to both columns
              	  block_structure=[0,0,0]);


def test_tgmm_run2():
	#force_install_tgmm()

	T = Transformation(['x', 'y', 'z'],
						"(x,y,z) -> (x^2,y,cos(z))", 
						['x1','y', 'z1'], 
						"(x1,y,z1) -> (x^(0.5),y,acos(z))")


	df = pd.DataFrame({'x' : np.random.rand(1_000), 
					  'y' : np.random.rand(1_000), 
					  'z' : np.random.rand(1_000)})

	fit2 = fit_gmm(df, 3, [0,0,0], [1,1,1], cov="diag", progress=True, transformation=T,
				  boundary_unbiasing={'structure' : [1,1,1], 'bandwidth_scale' : 0.01},  # extra boundary correction will be applied to both columns
              	  block_structure=[0,0,0]);









