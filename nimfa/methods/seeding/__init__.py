
"""    
    This package contains implementations of initialization methods for matrix factorization.
    
"""

import nndsvd
import random
import fixed
import random_c
import random_vcol

methods = {"random": random.Random,
           "fixed": fixed.Fixed,
           "nndsvd": nndsvd.Nndsvd,
           "random_c": random_c.Random_c,
           "random_vcol": random_vcol.Random_vcol,
           "none": None }