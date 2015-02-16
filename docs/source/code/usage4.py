import nimfa

import numpy as np

V = np.matrix([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print(V)

# Initialization callback function
def init_info(model):
    print("Initialized basis matrix\n", model.basis())
    print("Initialized  mixture matrix\n", model.coef())

# ICM rank 3 algorithm
# We specify callback_init parameter by passing a init_info function
# Callback is called after initialization and prior to factorization in each run.
fctr = nimfa.mf(V, seed="random_c", method="icm", max_iter=10, rank=3, callback_init=init_info)
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print("Resulting basis matrix")
print(W)

# Mixture matrix.
H = fctr_res.coef()
print("Resulting mixture matrix")
print(H)

sm = fctr_res.summary()
print("Rss: %8.3e" % sm['rss'])
print("Evar: %8.3e" % sm['evar'])
print("Iterations: %d" % sm['n_iter'])
print("KL divergence: %5.3e" % sm['kl'])
print("Euclidean distance: %5.3e" % sm['euclidean'])
