import numpy as np
import scipy.sparse as sp

import nimfa

V = np.random.rand(40, 100)
V1 = np.random.rand(40, 200)
snmnmf = nimfa.Snmnmf(V=V, V1=V1, seed="random_c", rank=10, max_iter=12,
                      A=sp.csr_matrix((V1.shape[1], V1.shape[1])),
                      B=sp.csr_matrix((V.shape[1], V1.shape[1])), gamma=0.01,
                      gamma_1=0.01, lamb=0.01, lamb_1=0.01)
snmnmf_fit = snmnmf()
