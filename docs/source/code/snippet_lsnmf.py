import numpy as np

import nimfa

V = np.random.rand(40, 100)
lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=10, max_iter=12, sub_iter=10,
                    inner_sub_iter=10, beta=0.1)
lsnmf_fit = lsnmf()
