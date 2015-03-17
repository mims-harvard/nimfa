import numpy as np

import nimfa

V = np.random.rand(40, 100)
snmf = nimfa.Snmf(V, seed="random_c", rank=10, max_iter=12, version='r', eta=1.,
                  beta=1e-4, i_conv=10, w_min_change=0)
snmf_fit = snmf()
