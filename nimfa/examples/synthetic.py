
"""
    ##################################
    Synthetic (``examples.synthetic``)
    ##################################
    
    This module contains examples of factorization runs. Since the data is artificially generated, 
    this is not a valid test of models applicability to real world situations. It can however
    be used for demonstration of the library. 
    
    Examples are performed on 20 x 30 dense matrix, whose values are drawn from normal 
    distribution with zero mean and variance of one (an absolute of values is taken because of 
    nonnegativity constraint).
    
    Only for the purpose of demonstration in all examples many optional (runtime or algorithm specific) 
    parameters are set. The user could as well run the factorization by providing only the target matrix.
    In that case the defaults would be used. General model parameters are explained in :mod:`nimfa.mf_run`, 
    algorithm specific parameters in Python module implementing the algorithm. Nevertheless for best results, 
    careful choice of parameters is recommended. No tracking is demonstrated here.
    
    .. note:: For most factorizations using artificially generated data is not the intended usage (e. g. SNMNMF is in [Zhang2011]_
              used for identification of the microRNA-gene regulatory networks). Consider this when discussing convergence
              and measurements output. 
        
    To run the examples simply type::
        
        python synthetic.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.synthetic.run()
"""

import nimfa
import numpy as np
import scipy.sparse as sp

def __fact_factor(X):
    """
    Return dense factorization factor, so that output is printed nice if factor is sparse.
    
    :param X: Factorization factor.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    """
    return X.todense() if sp.isspmatrix(X) else X

def print_info(fit, idx = None):
    """
    Print to stdout info about the factorization.
    
    :param fit: Fitted factorization model.
    :type fit: :class:`nimfa.models.mf_fit.Mf_fit`
    :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model. Therefore in factorizations 
                that follow standard or nonsmooth model, this parameter can be omitted. Currently, SNMNMF implements 
                multiple NMF model.
    :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
    """
    print "================================================================================================="
    print "Factorization method:", fit.fit
    print "Initialization method:", fit.fit.seed
    print "Basis matrix W: "
    print __fact_factor(fit.basis())
    print "Mixture (Coefficient) matrix H%d: " % (idx if idx != None else 0)
    print __fact_factor(fit.coef(idx))
    print "Distance (Euclidean): ", fit.distance(metric = 'euclidean', idx = idx)
    # We can access actual number of iteration directly through fitted model. 
    # fit.fit.n_iter
    print "Actual number of iterations: ", fit.summary(idx)['n_iter']
    # We can access sparseness measure directly through fitted model.
    # fit.fit.sparseness()
    print "Sparseness basis: %7.4f, Sparseness mixture: %7.4f" % (fit.summary(idx)['sparseness'][0], fit.summary(idx)['sparseness'][1])
    # We can access explained variance directly through fitted model.
    # fit.fit.evar()
    print "Explained variance: ", fit.summary(idx)['evar']
    # We can access residual sum of squares directly through fitted model.
    # fit.fit.rss()
    print "Residual sum of squares: ", fit.summary(idx)['rss']
    # There are many more ... but just cannot print out everything =] and some measures need additional data or more runs
    # e.g. entropy, predict, purity, coph_cor, consensus, select_features, score_features, connectivity  
    print "================================================================================================="

def run_snmnmf(V, V1):
    """
    Run sparse network-regularized multiple NMF. 
    
    :param V: First target matrix to estimate.
    :type V: :class:`numpy.matrix`
    :param V1: Second target matrix to estimate.
    :type V1: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(target = (V, V1), 
                  seed = "random_c", 
                  rank = rank, 
                  method = "snmnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  A = abs(sp.rand(V1.shape[1], V1.shape[1], density = 0.7, format = 'csr')),
                  B = abs(sp.rand(V.shape[1], V1.shape[1], density = 0.7, format = 'csr')), 
                  gamma = 0.01,
                  gamma_1 = 0.01,
                  lamb = 0.01,
                  lamb_1 = 0.01)
    fit = nimfa.mf_run(model)
    # print all quality measures concerning first target and mixture matrix in multiple NMF
    print_info(fit, idx = 0)
    # print all quality measures concerning second target and mixture matrix in multiple NMF
    print_info(fit, idx = 1)

def run_bd(V):
    """
    Run Bayesian decomposition.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_c", 
                  rank = rank, 
                  method = "bd", 
                  max_iter = 12, 
                  initialize_only = True,
                  alpha = np.mat(np.zeros((V.shape[0], rank))),
                  beta = np.mat(np.zeros((rank, V.shape[1]))),
                  theta = .0,
                  k = .0,
                  sigma = 1., 
                  skip = 100,
                  stride = 1,
                  n_w = np.mat(np.zeros((rank, 1))),
                  n_h = np.mat(np.zeros((rank, 1))),
                  n_sigma = False)
    fit = nimfa.mf_run(model)
    print_info(fit)

def run_bmf(V):
    """
    Run binary matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "bmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  lambda_w = 1.1,
                  lambda_h = 1.1)
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_icm(V):
    """
    Run iterated conditional modes.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    model = nimfa.mf(V, 
                  seed = "nndsvd", 
                  rank = rank, 
                  method = "icm", 
                  max_iter = 12, 
                  initialize_only = True,
                  iiter = 20,
                  alpha = pnrg.randn(V.shape[0], rank),
                  beta = pnrg.randn(rank, V.shape[1]), 
                  theta = 0.,
                  k = 0.,
                  sigma = 1.)
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_lfnmf(V):
    """
    Run local fisher nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    model = nimfa.mf(V, 
                  seed = None,
                  W = abs(pnrg.randn(V.shape[0], rank)), 
                  H = abs(pnrg.randn(rank, V.shape[1])),
                  rank = rank, 
                  method = "lfnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  alpha = 0.01)
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_lsnmf(V):
    """
    Run least squares nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "lsnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  sub_iter = 10,
                  inner_sub_iter = 10, 
                  beta = 0.1, 
                  min_residuals = 1e-5)
    fit = nimfa.mf_run(model)
    print_info(fit)

def run_nmf(V):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "nmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  update = 'euclidean',
                  objective = 'fro')
    fit = nimfa.mf_run(model)
    print_info(fit)
    # divergence
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "nmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  update = 'divergence',
                  objective = 'div')
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_nsnmf(V):
    """
    Run nonsmooth nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random", 
                  rank = rank, 
                  method = "nsnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  theta = 0.5)
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_pmf(V):
    """
    Run probabilistic matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "pmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  rel_error = 1e-5)
    fit = nimfa.mf_run(model)
    print_info(fit)
    
def run_psmf(V):
    """
    Run probabilistic sparse matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    prng = np.random.RandomState()
    model = nimfa.mf(V, 
                  seed = None,
                  rank = rank, 
                  method = "psmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  prior = prng.uniform(low = 0., high = 1., size = 10))
    fit = nimfa.mf_run(model)
    print_info(fit)

def run_snmf(V):
    """
    Run sparse nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # SNMF/R
    rank = 10
    model = nimfa.mf(V, 
                  seed = "random_c", 
                  rank = rank, 
                  method = "snmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  version = 'r',
                  eta = 1.,
                  beta = 1e-4, 
                  i_conv = 10,
                  w_min_change = 0)
    fit = nimfa.mf_run(model)
    print_info(fit)
    # SNMF/L
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "snmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  version = 'l',
                  eta = 1.,
                  beta = 1e-4, 
                  i_conv = 10,
                  w_min_change = 0)
    fit = nimfa.mf_run(model)
    print_info(fit)

def run(V = None, V1 = None):
    """
    Run examples.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    :param V1: (Second) Target matrix to estimate used in multiple NMF (e. g. SNMNMF).
    :type V1: :class:`numpy.matrix`
    """
    if V == None or V1 == None:
        prng = np.random.RandomState(42)
        # construct target matrix 
        V = abs(np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 30))))
        V1 = abs(np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 25))))
    run_snmnmf(V, V1)
    run_bd(V)
    run_bmf(V)
    run_icm(V)
    run_lfnmf(V)
    run_lsnmf(V)
    run_nmf(V)
    run_nsnmf(V)
    run_pmf(V)
    run_psmf(V)
    run_snmf(V)

if __name__ == "__main__":
    prng = np.random.RandomState(42)
    # construct target matrix 
    V = abs(np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 30))))
    V1 = abs(np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 25))))
    # run examples
    run(V, V1)
    
    