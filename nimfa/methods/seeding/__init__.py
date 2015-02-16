
"""    
    This package contains implementations of initialization methods for matrix factorization.
"""

from . import nndsvd
from . import random
from . import fixed
from . import random_c
from . import random_vcol

methods = {"random": random.Random,
           "fixed": fixed.Fixed,
           "nndsvd": nndsvd.Nndsvd,
           "random_c": random_c.Random_c,
           "random_vcol": random_vcol.Random_vcol,
           "none": None}
