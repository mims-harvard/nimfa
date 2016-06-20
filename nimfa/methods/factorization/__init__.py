
"""    
    This package contains implementations of matrix factorization algorithms. 
"""

from .bd import *
from .icm import *
from .lfnmf import *
from .lsnmf import *
from .nmf import *
from .nsnmf import *
from .pmf import *
from .psmf import *
from .snmf import *
from .bmf import *
from .snmnmf import *
from .pmfcc import *
from .sepnmf import *

methods = {"bd": Bd,
           "icm": Icm,
           "lfnmf": Lfnmf,
           "lsnmf": Lsnmf,
           "nmf": Nmf,
           "nsnmf": Nsnmf,
           "pmf": Pmf,
           "psmf": Psmf,
           "snmf": Snmf,
           "bmf": Bmf,
           "snmnmf": Snmnmf,
           "pmfcc": Pmfcc,
           "sepnmf": SepNmf,
           "none": None}
