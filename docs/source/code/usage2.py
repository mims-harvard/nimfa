import nimfa

# Here we will work with numpy matrix
import numpy as np

V = np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print(V)

# LSNMF rank 3 algorithm
fctr = nimfa.mf(V, method="lsnmf", max_iter=10, rank=3)
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print("Basis matrix")
print(W)

# Mixture matrix.
H = fctr_res.coef()
print("Coef")
print(H)

# Kullback-Leibler divergence. By
print("Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric="kl"))

sm = fctr_res.summary()
# Residual sum of squares - can be used for factorization rank selection
print("Rss: %8.3f" % sm['rss'])
# Explained variance.
print("Evar: %8.3f" % sm['evar'])
# Actual number of iterations performed
print("Iterations: %d" % sm['n_iter'])

# Estimate of target matrix V
print("Estimate")
print(np.dot(W, H))
