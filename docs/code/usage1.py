import numpy as np
import scipy.sparse as spr

import nimfa

V = spr.csr_matrix([[1, 0, 2, 4], [0, 0, 6, 3], [4, 0, 5, 6]])
print('Target:\n%s' % V.todense())

nmf = nimfa.Nmf(V, max_iter=200, rank=2, update='euclidean', objective='fro')
nmf_fit = nmf()

W = nmf_fit.basis()
print('Basis matrix:\n%s' % W.todense())

H = nmf_fit.coef()
print('Mixture matrix:\n%s' % H.todense())

print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))

sm = nmf_fit.summary()
print('Sparseness Basis: %5.3f  Mixture: %5.3f' % (sm['sparseness'][0], sm['sparseness'][1]))
print('Iterations: %d' % sm['n_iter'])
print('Target estimate:\n%s' % np.dot(W.todense(), H.todense()))
