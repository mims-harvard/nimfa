
"""    
    This package contains implementations of matrix factorization algorithms. 
"""

from . import bd
from . import icm
from . import lfnmf
from . import lsnmf
from . import nmf
from . import nsnmf
from . import pmf
from . import psmf
from . import snmf
from . import bmf
from . import snmnmf
from . import pmfcc

methods = {"bd": bd.Bd,
           "icm": icm.Icm,
           "lfnmf": lfnmf.Lfnmf,
           "lsnmf": lsnmf.Lsnmf,
           "nmf": nmf.Nmf,
           "nsnmf": nsnmf.Nsnmf,
           "pmf": pmf.Pmf,
           "psmf": psmf.Psmf,
           "snmf": snmf.Snmf,
           "bmf": bmf.Bmf,
           "snmnmf": snmnmf.Snmnmf,
           "pmfcc": pmfcc.Pmfcc,
           "none": None}
