import bd, icm, lfnmf, lsnmf, nmf, nsnmf, pmf, psmf, snmf, bmf

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
           "none": None}