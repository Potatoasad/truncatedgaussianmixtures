from .julia_import import jl, TruncatedGaussianMixtures
from .julia_helpers import jl_array
from juliacall import convert as jl_convert 
from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np

def AnnealingSchedule(beta_list : Union[np.array, List]):
	return TruncatedGaussianMixtures.AnnealingSchedule(jl_array(beta_list))