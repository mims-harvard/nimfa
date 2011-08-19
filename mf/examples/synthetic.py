
"""
    This module contains examples of factorization runs. Since the data is artificially generated, this is not a valid test of
    models applicability to real world situations. It can however be used for demonstration of the library. 
    
    Examples are performed on 20 x 30 dense matrix, whose values are drawn from normal distribution wit zero mean one one variance.
    
    Only for the purpose of demonstration in all examples many optional (runtime or algorithm specific) parameters are set. The user could
    as well run the factorization by providing only the target matrix. In that case the defaults would be used. General model parameters
    are explained in :mod:`mf.mf_run`, algorithm specific parameters in Python module implementing the algorithm. Nevertheless for best results, 
    careful choice of parameters is recommended. 
    
    .. seealso:: README.rst     
    
    To run the examples simply type::
        
        python synthetic.py
        
    or call the module's function::
    
        synthetic.run()

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
    print "Factorization method:", fit.fit.name
    print "Initialization method:", fit.fit.seed.name
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
                  n_sigma = 0)
    fit = mf.mf_run(model)
    _print_info(fit)

def run_bmf(V):
    """
    Run binary matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_icm(V):
    """
    Run iterated conditional modes.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_lfnmf(V):
    """
    Run local fisher nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_lsnmf(V):
    """
    Run least squares nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """

def run_nmf(V):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_nsnmf(V):
    """
    Run nonsmooth nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_pmf(V):
    """
    Run probabilistic matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    
def run_psmf(V):
    """
    Run probabilistic sparse matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """

def run_snmf(V):
    """
    Run sparse nonnegative matrix factorizaiton.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """

def run(V):
    """
    Run examples.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
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
    V = np.mat(prng.normal(loc = 0.0, scale = 1.0, size = (20, 30)))
    # run examples
    run(V)