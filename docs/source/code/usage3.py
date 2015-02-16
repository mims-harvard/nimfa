import nimfa
import numpy as np

V = np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print(V)

# LSNMF rank 3 algorithm
fctr = nimfa.mf(V, seed="random_vcol", method="lsnmf", max_iter=10, rank=3, track_error=True)
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print("Basis matrix")
print(W)

# Mixture matrix.
H = fctr_res.coef()
print("Coef")
print(H)

# Error tracking.
print("Error tracking")
# A list of objective function values for each iteration in factorization is printed.
# If error tracking is enabled and user specifies multiple runs of the factorization, get_error(run = n) return a list of objective values from n-th run.
# fctr_res.fit.tracker is an instance of Mf_track --
# isinstance(fctr_res.fit.tracker, nimfa.models.mf_track.Mf_track)
print(fctr_res.fit.tracker.get_error())

# Generic set of measures to evaluate factorization quality
sm = fctr_res.summary()
print("Rss: %8.3f" % sm['rss'])
print("Evar: %8.3f" % sm['evar'])
print("Iterations: %d" % sm['n_iter'])
