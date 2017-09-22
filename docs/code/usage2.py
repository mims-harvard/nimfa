import numpy as np

import nimfa

V = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print('Target:\n%s' % V)

lsnmf = nimfa.Lsnmf(V, max_iter=10, rank=3)
lsnmf_fit = lsnmf()

W = lsnmf_fit.basis()
print('Basis matrix:\n%s' % W)

H = lsnmf_fit.coef()
print('Mixture matrix:\n%s' % H)

print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))

print('Rss: %5.3f' % lsnmf_fit.fit.rss())
print('Evar: %5.3f' % lsnmf_fit.fit.evar())
print('Iterations: %d' % lsnmf_fit.n_iter)
print('Target estimate:\n%s' % np.dot(W, H))
