import numpy as np

import nimfa

V = np.random.rand(40, 100)
nsnmf = nimfa.Nsnmf(V, seed="random", rank=10, max_iter=12, theta=0.5)
nsnmf_fit = nsnmf()
