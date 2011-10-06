
####
# EXAMPLE 1: 
####


# Import MF library entry point for factorization
import mf

# Construct sparse matrix in CSR format, which will be our input for factorization
from scipy.sparse import csr_matrix
from scipy import array
from numpy import dot
V = csr_matrix((array([1,2,3,4,5,6]), array([0,2,2,0,1,2]), array([0,2,3,6])), shape=(3,3))

# Print this tiny matrix in dense format
print V.todense()

# Run Standard NMF rank 4 algorithm
# Update equations and cost function are Standard NMF specific parameters (among others).
# If not specified the Euclidean update and Forbenius cost function would be used.
# We don't specify initialization method. Algorithm specific or random intialization will be used.
# In Standard NMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.
fit = mf.mf(V, method = "nmf", max_iter = 30, rank = 4, update = 'divergence', objective = 'div')

# Basis matrix. It is sparse, as input V was sparse as well.
W = fit.basis()
print "Basis matrix"
print W.todense()

# Mixture matrix. We print this tiny matrix in dense format.
H = fit.coef()
print "Coef"
print H.todense()

# Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler: %5.3e" % fit.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print sparseness (Hoyer, 2004) of basis and mixture matrix
print "Sparseness Basis: %5.3f  Mixture: %5.3f" % (sm['sparseness'][0], sm['sparseness'][1])
# Print actual number of iterations performed
print "Iterations: %d" % sm['n_iter']

# Print estimate of target matrix V
print "Estimate"
print dot(W.todense(), H.todense())


####
# EXAMPLE 2: 
####


# Import MF library entry point for factorization
import mf

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V

# Run LSNMF rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We don't specify initialization method. Algorithm specific or random intialization will be used. 
# In LSNMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.  
fit = mf.mf(V, method = "lsnmf", max_iter = 10, rank = 3)

# Basis matrix.
W = fit.basis()
print "Basis matrix"
print W

# Mixture matrix. 
H = fit.coef()
print "Coef"
print H

# Print the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler: %5.3e" % fit.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
print "Rss: %8.3f" % sm['rss']
# Print explained variance.
print "Evar: %8.3f" % sm['evar']
# Print actual number of iterations performed
print "Iterations: %d" % sm['n_iter']

# Print estimate of target matrix V 
print "Estimate"
print np.dot(W, H)


####
# EXAMPLE 3:
####


# Import MF library entry point for factorization
import mf

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V

# Run LSNMF rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We specify Random V Col initialization algorithm. 
# We enable tracking the error from each iteration of the factorization, by default only the final value of objective function is retained. 
# Perform initialization separately. 
model = mf.mf(V, seed = "random_vcol", method = "lsnmf", max_iter = 10, rank = 3, track_error = True, initialize_only = True)

# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.  
fit = mf.mf_run(model)

# Basis matrix.
W = fit.basis()
print "Basis matrix"
print W

# Mixture matrix. 
H = fit.coef()
print "Coef"
print H

# Error tracking. 
print "Error tracking"
# A list of objective function values for each iteration in factorization is printed.
# If error tracking is enabled and user specifies multiple runs of the factorization, get_error(run = n) return a list of objective values from n-th run. 
# fit.fit.tracker is an instance of Mf_track -- isinstance(fit.fit.tracker, mf.models.mf_track.Mf_track)
print fit.fit.tracker.get_error()

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
print "Rss: %8.3f" % sm['rss']
# Print explained variance.
print "Evar: %8.3f" % sm['evar']
# Print actual number of iterations performed
print "Iterations: %d" % sm['n_iter']


####
# Example 4:
####


# Import MF library entry point for factorization
import mf

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V


# This will be our callback_init function called prior to factorization.
# We will only print the initialized matrix factors.
def init_info(model):
    print "Initialized basis matrix\n", model.basis()
    print "Initialized  mixture matrix\n", model.coef() 

# Run ICM rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We specify Random C initialization algorithm.
# We specify callback_init parameter by passing a init_info function 
# This function is called after initialization and prior to factorization in each run.  
model = mf.mf(V, seed = "random_c", method = "icm", max_iter = 10, rank = 3, initialize_only = True, callback_init = init_info)

# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.  
fit = mf.mf_run(model)

# Basis matrix.
W = fit.basis()
print "Resulting basis matrix"
print W

# Mixture matrix. 
H = fit.coef()
print "Resulting mixture matrix"
print H

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
print "Rss: %8.3e" % sm['rss']
# Print explained variance.
print "Evar: %8.3e" % sm['evar']
# Print actual number of iterations performed
print "Iterations: %d" % sm['n_iter']
# Print distance according to Kullback-Leibler divergence
print "KL divergence: %5.3e" % sm['kl']
# Print distance according to Euclidean metric
print "Euclidean distance: %5.3e" % sm['euclidean'] 


