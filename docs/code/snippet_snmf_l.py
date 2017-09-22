import numpy as np

import nimfa

V = np.random.rand(40, 100)
snmf = nimfa.Snmf(V, seed="random_vcol", rank=10, max_iter=12, version='l',
                  eta=1., beta=1e-4, i_conv=10, w_min_change=0)
snmf_fit = snmf()
