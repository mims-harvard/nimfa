import nimfa

import numpy as np

V = np.random.rand(23, 200)

# Run BMF.
# Factorization will be run 3 times (n_run) and factors will be tracked for computing
# cophenetic correlation. Note increased time and space complexity.
fctr = nimfa.mf(V, method="bmf", max_iter=10, rank=30, n_run=3, track_factor=True)
fctr_res = nimfa.mf_run(fctr)

print("Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric="kl"))

sm = fctr_res.summary()
print("Rss: %8.3f" % sm['rss'])
print("Evar: %8.3f" % sm['evar'])
print("Iterations: %d" % sm['n_iter'])
print("cophenetic: %8.3f" % sm['cophenetic'])
