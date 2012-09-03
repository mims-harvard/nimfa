
####
# EXAMPLE 1: 
####


# Import nimfa library entry point for factorization
import nimfa

# Construct sparse matrix in CSR format, which will be our input for factorization
from scipy.sparse import csr_matrix
from scipy import array
from numpy import dot
V = csr_matrix((array([1,2,3,4,5,6]), array([0,2,2,0,1,2]), array([0,2,3,6])), shape=(3,3))

# Print this tiny matrix in dense format
print V.todense()

# Run Standard NMF rank 4 algorithm
# Update equations and cost function are Standard NMF specific parameters (among others).
# If not specified the Euclidean update and Frobenius cost function would be used.
# We don't specify initialization method. Algorithm specific or random initialization will be used.
# In Standard NMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fctr_res's attribute `fit` contains all the attributes of the factorization.
fctr = nimfa.mf(V, method = "nmf", max_iter = 30, rank = 4, update = 'divergence', objective = 'div')
fctr_res = nimfa.mf_run(fctr)

# Basis matrix. It is sparse, as input V was sparse as well.
W = fctr_res.basis()
print "Basis matrix"
print W.todense()

# Mixture matrix. We print this tiny matrix in dense format.
H = fctr_res.coef()
print "Coef"
print H.todense()

# Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fctr_res.summary()
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


# Import nimfa library entry point for factorization
import nimfa

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V

# Run LSNMF rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We don't specify initialization method. Algorithm specific or random initialization will be used. 
# In LSNMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fctr_res's attribute `fit` contains all the attributes of the factorization.  
fctr = nimfa.mf(V, method = "lsnmf", max_iter = 10, rank = 3)
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print "Basis matrix"
print W

# Mixture matrix. 
H = fctr_res.coef()
print "Coef"
print H

# Print the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fctr_res.summary()
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


# Import nimfa library entry point for factorization
import nimfa

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V

# Run LSNMF rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We specify Random V Col initialization algorithm. 
# We enable tracking the error from each iteration of the factorization, by default only the final value of objective function is retained. 
# Perform initialization. 
fctr = nimfa.mf(V, seed = "random_vcol", method = "lsnmf", max_iter = 10, rank = 3, track_error = True)

# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fctr_res's attribute `fit` contains all the attributes of the factorization.  
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print "Basis matrix"
print W

# Mixture matrix. 
H = fctr_res.coef()
print "Coef"
print H

# Error tracking. 
print "Error tracking"
# A list of objective function values for each iteration in factorization is printed.
# If error tracking is enabled and user specifies multiple runs of the factorization, get_error(run = n) return a list of objective values from n-th run. 
# fctr_res.fit.tracker is an instance of Mf_track -- isinstance(fctr_res.fit.tracker, nimfa.models.mf_track.Mf_track)
print fctr_res.fit.tracker.get_error()

# Compute generic set of measures to evaluate the quality of the factorization
sm = fctr_res.summary()
# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
print "Rss: %8.3f" % sm['rss']
# Print explained variance.
print "Evar: %8.3f" % sm['evar']
# Print actual number of iterations performed
print "Iterations: %d" % sm['n_iter']


####
# Example 4:
####


# Import nimfa library entry point for factorization
import nimfa

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
fctr = nimfa.mf(V, seed = "random_c", method = "icm", max_iter = 10, rank = 3, callback_init = init_info)

# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fctr_res's attribute `fit` contains all the attributes of the factorization.  
fctr_res = nimfa.mf_run(fctr)

# Basis matrix.
W = fctr_res.basis()
print "Resulting basis matrix"
print W

# Mixture matrix. 
H = fctr_res.coef()
print "Resulting mixture matrix"
print H

# Compute generic set of measures to evaluate the quality of the factorization
sm = fctr_res.summary()
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


####
# Example Script
####


import nimfa

V = nimfa.examples.medulloblastoma.read(normalize = True)

fctr = nimfa.mf(V, seed = 'random_vcol', method = 'lsnmf', rank = 40, max_iter = 65)
fctr_res = nimfa.mf_run(fctr)

print 'Rss: %5.4f' % fctr_res.fit.rss()
print 'Evar: %5.4f' % fctr_res.fit.evar()
print 'K-L divergence: %5.4f' % fctr_res.distance(metric = 'kl')
print 'Sparseness, W: %5.4f, H: %5.4f' % fctr_res.fit.sparseness()


####
# Example 5
####


# Import nimfa library entry point for factorization.
import nimfa

# Here we will work with numpy matrix.
import numpy as np
V = np.random.random((23, 200))

# Run BMF.
# We don't specify any algorithm parameters or initialization method. Defaults will be used.
# Factorization will be run 3 times (n_run) and factors will be tracked for computing 
# cophenetic correlation. Note increased time and space complexity.
fctr = nimfa.mf(V, method = "bmf", max_iter = 10, rank = 30, n_run = 3, track_factor = True)
fctr_res = nimfa.mf_run(fctr)

# Print the loss function according to Kullback-Leibler divergence. 
print "Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization.
sm = fctr_res.summary()
# Print residual sum of squares.
print "Rss: %8.3f" % sm['rss']
# Print explained variance.
print "Evar: %8.3f" % sm['evar']
# Print actual number of iterations performed.
print "Iterations: %d" % sm['n_iter']
# Print cophenetic correlation. Can be used for rank estimation.
print "cophenetic: %8.3f" % sm['cophenetic']


####
# Example 6
####


# Import nimfa library entry point for factorization.
import nimfa

# Here we will work with numpy matrix.
import numpy as np

# Generate random target matrix.
V = np.random.rand(30, 20)

# Generate random matrix factors which we will pass as fixed factors to Nimfa.
init_W = np.random.rand(30, 4)
init_H = np.random.rand(4, 20)
# Obviously by passing these factors we want to use rank = 4.

# Run NMF.
# We don't specify any algorithm parameters. Defaults will be used.
# We specify fixed initialization method and pass matrix factors.
fctr = nimfa.mf(V, method = "nmf", seed = "fixed", W = init_W, H = init_H, rank = 4)
fctr_res = nimfa.mf_run(fctr)

# Print the loss function (Euclidean distance between target matrix and its estimate). 
print "Euclidean distance: %5.3e" % fctr_res.distance(metric = "euclidean")

# It should print 'fixed'.
print fctr_res.seeding

# By default, max 30 iterations are performed.
print fctr_res.n_iter




