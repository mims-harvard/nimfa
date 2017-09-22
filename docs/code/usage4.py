import nimfa

import numpy as np

V = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print('Target:\n%s' % V)

# Initialization callback function
def init_info(model):
    print('Initialized basis matrix:\n%s' % model.basis())
    print('Initialized  mixture matrix:\n%s' % model.coef())

# Callback is called after initialization and prior to factorization in each run
icm = nimfa.Icm(V, seed='random_c', max_iter=10, rank=3, callback_init=init_info)
icm_fit = icm()

W = icm_fit.basis()
print('Basis matrix:\n%s' % W)
print(W)

H = icm_fit.coef()
print('Mixture matrix:\n%s' % H)

sm = icm_fit.summary()
print('Rss: %5.3f' % sm['rss'])
print('Evar: %5.3f' % sm['evar'])
print('Iterations: %d' % sm['n_iter'])
print('KL divergence: %5.3f' % sm['kl'])
print('Euclidean distance: %5.3f' % sm['euclidean'])
