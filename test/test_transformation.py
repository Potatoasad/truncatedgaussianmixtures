from ..truncatedgaussianmixtures import *

def test_transformation():
	force_install_tgmm()
	Transformation(['x'], "(x,) -> (x^2,)", ['y'], "(y,) -> (y^0.5,)")