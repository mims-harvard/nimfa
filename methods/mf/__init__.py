from methods.mf import bd, icm, lnmf, lsnmf, nmf, nsnmf, pmf, psmf, snmf, bnmf

methods = {"bd": bd.Bd, 
           "icm": icm.Icm,
           "lnmf": lnmf.Lnmf,
           "lsnmf": lsnmf.Lsnmf,
           "nmf": nmf.Nmf,
           "nsnmf": nsnmf.Nsnmf,
           "pmf": pmf.Pmf,
           "psmf": psmf.Psmf,
           "snmf": snmf.Snmf,
           "bnmf": bnmf.Bnmf}