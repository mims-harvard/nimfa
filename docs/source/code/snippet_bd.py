import numpy as np

import nimfa

V = np.random.rand(40, 100)
bd = nimfa.Bd(V, seed="random_c", rank=10, max_iter=12, alpha=np.zeros((V.shape[0], 10)),
              beta=np.zeros((10, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((10, 1)), n_h=np.zeros((10, 1)), n_sigma=False)
bd_fit = bd()
