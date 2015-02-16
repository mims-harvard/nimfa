import nimfa

V = nimfa.examples.medulloblastoma.read(normalize=True)

fctr = nimfa.mf(V, seed='random_vcol', method='lsnmf', rank=40, max_iter=65)
fctr_res = nimfa.mf_run(fctr)

print('Rss: %5.4f' % fctr_res.fit.rss())
print('Evar: %5.4f' % fctr_res.fit.evar())
print('K-L divergence: %5.4f' % fctr_res.distance(metric='kl'))
print('Sparseness, W: %5.4f, H: %5.4f' % fctr_res.fit.sparseness())
