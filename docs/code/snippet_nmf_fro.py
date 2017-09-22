import numpy as np

import nimfa

V = np.random.rand(40, 100)
nmf = nimfa.Nmf(V, seed="nndsvd", rank=10, max_iter=12, update='euclidean',
                objective='fro')
nmf_fit = nmf()
