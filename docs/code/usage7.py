import numpy as np

import nimfa

V = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print('Target:\n%s' % V)

lsnmf = nimfa.Lsnmf(V, seed='random_vcol', max_iter=10, rank=3, track_error=True)
lsnmf_fit = lsnmf()

W = lsnmf_fit.basis()
print('Basis matrix:\n%s' % W)

H = lsnmf_fit.coef()
print('Mixture matrix:\n%s' % H)

r = lsnmf.estimate_rank(rank_range=[2,3,4], what=['rss'])
pp_r = '\n'.join('%d: %5.3f' % (rank, vals['rss']) for rank, vals in r.items())
print('Rank estimate:\n%s' % pp_r)
