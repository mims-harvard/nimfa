import numpy as np

import nimfa

V = np.random.rand(40, 100)
nmf = nimfa.Nmf(V, rank=10, seed="random_vcol", max_iter=200, update='euclidean',
                objective='conn', conn_change=40)
nmf_fit = nmf()
