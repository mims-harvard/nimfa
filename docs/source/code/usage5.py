import numpy as np

import nimfa

V = np.random.rand(23, 200)

# Factorization will be run 3 times (n_run) and factors will be tracked for computing
# cophenetic correlation. Note increased time and space complexity
bmf = nimfa.Bmf(V, max_iter=10, rank=30, n_run=3, track_factor=True)
bmf_fit = bmf()

print('K-L divergence: %5.3f' % bmf_fit.distance(metric='kl'))

sm = bmf_fit.summary()
print('Rss: %5.3f' % sm['rss'])
print('Evar: %5.3f' % sm['evar'])
print('Iterations: %d' % sm['n_iter'])
print('Cophenetic correlation: %5.3f' % sm['cophenetic'])
