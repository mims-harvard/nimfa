import numpy as np

import nimfa

V = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print('Target:\n%s' % V)

lsnmf = nimfa.Lsnmf(V, seed='random_vcol', max_iter=10, rank=3, track_error=True)
lsnmf_fit = lsnmf()

W = lsnmf_fit.basis()
print('Basis matrix:\n%s' % W)

H = lsnmf_fit.coef()
print('Mixture matrix:\n%s' % H)

# Objective function value for each iteration
print('Error tracking:\n%s' % lsnmf_fit.fit.tracker.get_error())

sm = lsnmf_fit.summary()
print('Rss: %5.3f' % sm['rss'])
print('Evar: %5.3f' % sm['evar'])
print('Iterations: %d' % sm['n_iter'])
