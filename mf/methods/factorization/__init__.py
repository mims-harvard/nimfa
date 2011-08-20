
"""
    This package contains implementations of matrix factorization algorithms. 

"""

import bd
import icm
import lfnmf
import lsnmf
import nmf
import nsnmf
import pmf
import psmf
import snmf
import bmf
import snmnmf

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
           "none": None}