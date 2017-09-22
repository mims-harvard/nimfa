import numpy as np

import nimfa

V = np.random.rand(40, 100)
psmf = nimfa.Psmf(V, seed=None, rank=10, max_iter=12, prior=np.random.rand(10))
psmf_fit = psmf()
