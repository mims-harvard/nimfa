
"""    
    This package contains implementations of initialization methods for matrix factorization.
"""

from .nndsvd import *
from .random import *
from .fixed import *
from .random_c import *
from .random_vcol import *

methods = {"random": Random,
           "fixed": Fixed,
           "nndsvd": Nndsvd,
           "random_c": Random_c,
           "random_vcol": Random_vcol,
           "none": None}
