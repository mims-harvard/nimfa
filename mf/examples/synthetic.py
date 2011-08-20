
"""
    This module contains examples of factorization runs. Since the data is artificially generated, 
    this is not a valid test of models applicability to real world situations. It can however
    be used for demonstration of the library. 
    
    Examples are performed on 20 x 30 dense matrix, whose values are drawn from normal 
    distribution with zero mean one one variance (an absolute of values is taken because of 
    nonnegativity constraint).
    
    Only for the purpose of demonstration in all examples many optional (runtime or algorithm specific) 
    parameters are set. The user could as well run the factorization by providing only the target matrix.
    In that case the defaults would be used. General model parameters are explained in :mod:`mf.mf_run`, 
    algorithm specific parameters in Python module implementing the algorithm. Nevertheless for best results, 
    careful choice of parameters is recommended. No tracking is demonstrated here.
    
    .. seealso:: README.rst     
    
    To run the examples simply type::
        
        python synthetic.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.synthetic.run()
"""

import mf
import numpy as np
import scipy.sparse as sp

def __fact_factor(X):
    """
    Return dense factorization factor, so that output is printed nice if factor is sparse.
    
    :param X: Factorization factor.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    """
    return X.todense() if sp.isspmatrix(X) else X

def _print_info(fit):
    """
    Print to stdout info about the factorization.
    
    :param fit: Fitted factorization model.
    :type fit: :class:`mf.models.mf_fit.Mf_fit`
    """
    print "================================================================================================="
    print "Factorization method:", fit.fit
    print "Initialization method:", fit.fit.seed
    print "Basis matrix: "
    print __fact_factor(fit.basis())
    print "Mixture (Coefficient) matrix: "
    print __fact_factor(fit.coef())
    print "Distance (Euclidean): ", fit.distance(metric = 'euclidean')
    # We can access actual number of iteration directly through fitted model. 
    # fit.fit.n_iter
    print "Actual number of iterations: ", fit.summary()['n_iter']
    # We can access sparseness measure directly through fitted model.
    # fit.fit.sparseness()
    print "Sparseness basis: %7.4f, Sparseness mixture: %7.4f" % (fit.summary()['sparseness'][0], fit.summary()['sparseness'][1])
    # We can access explained variance directly through fitted model.
    # fit.fit.evar()
    print "Explained variance: ", fit.summary()['evar']
    # We can access residual sum of squares directly through fitted model.
    # fit.fit.rss()
    print "Residual sum of squares: ", fit.summary()['rss']
    # There are many more ... but just cannot print out everything =] and some measures need additional data or more runs
    # e.g. entropy, predict, purity, coph_cor, consensus, select_features, score_features, connectivity  
    print "================================================================================================="

def run_snmnmf(V):
    """
    Run sparse network-regularized multiple NMF. 
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
                  seed = "random_c", 
                  rank = rank, 
                  method = "snmnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  A = abs(sp.rand(self.V.shape[1], self.V1.shape[1], density = 0.7, format = 'csr', dtype = 'd')),
                  B = abs(sp.rand(self.V.shape[1], self.V1.shape[1], density = 0.7, format = 'csr', dtype = 'd')), 
                  gamma = 0.01,
                  gamma_1 = 0.01,
                  lamb = 0.01,
                  lamb_1 = 0.01)
    fit = mf.mf_run(model)
    _print_info(fit)

def run_bd(V):
    """
    Run Bayesian decomposition.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
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
    fit = mf.mf_run(model)
    _print_info(fit)

def run_bmf(V):
    """
    Run binary matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "bmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  lambda_w = 1.1,
                  lambda_h = 1.1)
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_icm(V):
    """
    Run iterated conditional modes.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    model = mf.mf(V, 
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
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_lfnmf(V):
    """
    Run local fisher nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    model = mf.mf(V, 
                  seed = None,
                  W = abs(pnrg.randn(V.shape[0], rank)), 
                  H = abs(pnrg.randn(rank, V.shape[1])),
                  rank = rank, 
                  method = "lfnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  alpha = 0.01)
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_lsnmf(V):
    """
    Run least squares nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "lsnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  sub_iter = 10,
                  inner_sub_iter = 10, 
                  beta = 0.1)
    fit = mf.mf_run(model)
    _print_info(fit)

def run_nmf(V):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    rank = 10
    model = mf.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "nmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  update = 'euclidean',
                  objective = 'fro')
    fit = mf.mf_run(model)
    _print_info(fit)
    # divergence
    model = mf.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "nmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  update = 'divergence',
                  objective = 'div')
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_nsnmf(V):
    """
    Run nonsmooth nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
                  seed = "random", 
                  rank = rank, 
                  method = "nsnmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  theta = 0.5)
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_pmf(V):
    """
    Run probabilistic matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    model = mf.mf(V, 
                  seed = "random_vcol", 
                  rank = rank, 
                  method = "pmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  rel_error = 1e-5)
    fit = mf.mf_run(model)
    _print_info(fit)
    
def run_psmf(V):
    """
    Run probabilistic sparse matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    prng = np.random.RandomState()
    model = mf.mf(V, 
                  seed = None,
                  rank = rank, 
                  method = "psmf", 
                  max_iter = 12, 
                  initialize_only = True,
                  prior = prng.uniform(low = 0., high = 1., size = 10))
    fit = mf.mf_run(model)
    _print_info(fit)

def run_snmf(V):
    """
    Run sparse nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # SNMF/R
    rank = 10
    model = mf.mf(V, 
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
    fit = mf.mf_run(model)
    _print_info(fit)
    # SNMF/L
    model = mf.mf(V, 
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
    fit = mf.mf_run(model)
    _print_info(fit)

def run(V = None):
    """
    Run examples.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    if V == None:
        prng = np.random.RandomState(42)
        # construct target matrix 
        V = abs(np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 30))))
    run_snmnmf(V)
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
    # run examples
    run(V)