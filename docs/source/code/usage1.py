import nimfa

# Construct sparse matrix in CSR format
from scipy.sparse import csr_matrix
from scipy import array
from numpy import dot

V = csr_matrix((array([1, 2, 3, 4, 5, 6]), array([0, 2, 2, 0, 1, 2]), array([0, 2, 3, 6])), shape=(3, 3))
print(V.todense())

# Standard NMF rank 4 algorithm
# Returned object is fitted factorization model
fctr = nimfa.mf(V, method="nmf", max_iter=30, rank=4, update='divergence', objective='div')
# The fctr_res's attribute `fit` contains factorization method attributes
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print("Basis matrix")
print(W.todense())

# Mixture matrix.
H = fctr_res.coef()
print("Coef")
print(H.todense())

# Kullback-Leibler divergence value
print("Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric="kl"))

# Generic measures of factorization quality
sm = fctr_res.summary()
# Sparseness (Hoyer, 2004) of basis and mixture matrix
print("Sparseness Basis: %5.3f  Mixture: %5.3f" % (sm['sparseness'][0], sm['sparseness'][1]))
# Actual number of iterations performed
print("Iterations: %d" % sm['n_iter'])

# Estimate of target matrix V
print("Estimate")
print(dot(W.todense(), H.todense()))
