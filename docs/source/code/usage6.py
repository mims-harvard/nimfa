import nimfa

import numpy as np

V = np.random.rand(30, 20)

# Generate random matrix factors which we will pass as fixed factors to Nimfa.
init_W = np.random.rand(30, 4)
init_H = np.random.rand(4, 20)

# Run NMF.
# We specify fixed initialization method and pass matrix factors.
fctr = nimfa.mf(V, method="nmf", seed="fixed", W=init_W, H=init_H, rank=4)
fctr_res = nimfa.mf_run(fctr)

print("Euclidean distance: %5.3e" % fctr_res.distance(metric="euclidean"))

# It should print 'fixed'.
print(fctr_res.seeding)

# By default, max 30 iterations are performed.
print(fctr_res.n_iter)
