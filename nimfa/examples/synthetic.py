
"""
    ##########################################
    Simulated studies (``examples.synthetic``)
    ##########################################
    
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


def print_info(fit, idx=None):
    """
    Print to stdout info about the factorization.
    
    :param fit: Fitted factorization model.
    :type fit: :class:`nimfa.models.mf_fit.Mf_fit`
    :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model. Therefore in factorizations 
                that follow standard or nonsmooth model, this parameter can be omitted. Currently, SNMNMF implements 
                multiple NMF model.
    :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
    """
    print("=================================================================================================")
    print("Factorization method:", fit.fit)
    print("Initialization method:", fit.fit.seed)
    print("Basis matrix W: ")
    print(__fact_factor(fit.basis()))
    print("Mixture (Coefficient) matrix H%d: " % (idx if idx != None else 0))
    print(__fact_factor(fit.coef(idx)))
    print("Distance (Euclidean): ", fit.distance(metric='euclidean', idx=idx))
    # We can access actual number of iteration directly through fitted model.
    # fit.fit.n_iter
    print("Actual number of iterations: ", fit.summary(idx)['n_iter'])
    # We can access sparseness measure directly through fitted model.
    # fit.fit.sparseness()
    print("Sparseness basis: %7.4f, Sparseness mixture: %7.4f" % (fit.summary(idx)['sparseness'][0], fit.summary(idx)['sparseness'][1]))
    # We can access explained variance directly through fitted model.
    # fit.fit.evar()
    print("Explained variance: ", fit.summary(idx)['evar'])
    # We can access residual sum of squares directly through fitted model.
    # fit.fit.rss()
    print("Residual sum of squares: ", fit.summary(idx)['rss'])
    # There are many more ... but just cannot print out everything =] and some measures need additional data or more runs
    # e.g. entropy, predict, purity, coph_cor, consensus, select_features, score_features, connectivity
    print("=================================================================================================")


def run_snmnmf(V, V1):
    """
    Run sparse network-regularized multiple NMF. 
    
    :param V: First target matrix to estimate.
    :type V: :class:`numpy.matrix`
    :param V1: Second target matrix to estimate.
    :type V1: :class:`numpy.matrix`
    """
    rank = 10
    snmnmf = nimfa.Snmnmf(V, V1, seed="random_c", rank=rank, max_iter=12,
                          A=sp.csr_matrix((V1.shape[1], V1.shape[1])), B=sp.csr_matrix((V.shape[1], V1.shape[1])),
                          gamma = 0.01, gamma_1 = 0.01, lamb = 0.01, lamb_1 = 0.01)
    fit = snmnmf()
    # print all quality measures concerning first target and mixture matrix in
    # multiple NMF
    print_info(fit, idx=0)
    # print all quality measures concerning second target and mixture matrix
    # in multiple NMF
    print_info(fit, idx=1)


def run_bd(V):
    """
    Run Bayesian decomposition.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    bd = nimfa.Bd(V, seed="random_c", rank=rank, max_iter=12, alpha=np.zeros((V.shape[0], rank)),
                  beta=np.zeros((rank, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100,
                  stride=1, n_w=np.mat(np.zeros((rank, 1))), n_h=np.mat(np.zeros((rank, 1))),
                  n_sigma=False)
    fit = bd()
    print_info(fit)


def run_bmf(V):
    """
    Run binary matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    bmf = nimfa.Bmf(V, seed="random_vcol", rank=rank, max_iter=12, initialize_only=True,
                     lambda_w=1.1, lambda_h=1.1)
    fit = bmf()
    print_info(fit)


def run_icm(V):
    """
    Run iterated conditional modes.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    icm = nimfa.Icm(V, seed="nndsvd", rank=rank, max_iter=12, iiter=20,
                     alpha=pnrg.randn(V.shape[0], rank),
                     beta=pnrg.randn(rank, V.shape[1]), theta=0., k=0., sigma=1.)
    fit = icm()
    print_info(fit)


def run_lfnmf(V):
    """
    Run local fisher nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pnrg = np.random.RandomState()
    lfnmf = nimfa.Lfnmf(V, seed=None, W=pnrg.rand(V.shape[0], rank), H=pnrg.rand(rank, V.shape[1]),
                        rank=rank, max_iter=12, alpha=0.01)
    fit = lfnmf()
    print_info(fit)


def run_lsnmf(V):
    """
    Run least squares nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=rank, max_iter=12, sub_iter=10,
                        inner_sub_iter=10, beta=0.1, min_residuals=1e-5)
    fit = lsnmf()
    print_info(fit)


def run_nmf(V):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    rank = 10
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=12, update='euclidean',
                      objective='fro')
    fit = nmf()
    print_info(fit)
    # divergence
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=12, initialize_only=True,
                    update='divergence', objective='div')
    fit = nmf()
    print_info(fit)


def run_nsnmf(V):
    """
    Run nonsmooth nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    nsnmf = nimfa.Nsnmf(V, seed="random", rank=rank, max_iter=12, theta=0.5)
    fit = nsnmf()
    print_info(fit)


def run_pmf(V):
    """
    Run probabilistic matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    pmf = nimfa.Pmf(V, seed="random_vcol", rank=rank, max_iter=12, rel_error=1e-5)
    fit = pmf()
    print_info(fit)


def run_psmf(V):
    """
    Run probabilistic sparse matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    rank = 10
    prng = np.random.RandomState()
    psmf = nimfa.Psmf(V, seed=None, rank=rank, max_iter=12, prior=prng.rand(10))
    fit = psmf()
    print_info(fit)


def run_snmf(V):
    """
    Run sparse nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # SNMF/R
    rank = 10
    snmf = nimfa.Snmf(V, seed="random_c", rank=rank, max_iter=12, version='r', eta=1.,
                       beta=1e-4, i_conv=10, w_min_change=0)
    fit = snmf()
    print_info(fit)
    # SNMF/L
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank, max_iter=12, version='l', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    fit = snmf()
    print_info(fit)


def run_sepnmf(V):
    """
    Run standard nonnegative matrix factorization.

    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    rank = 10

    sepnmf = nimfa.SepNmf(V, rank=rank, selection='spa')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, selection='xray')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='qr', selection='xray')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='qr', selection='spa')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='structured',
                          selection='xray')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='structured',
                          selection='xray')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='count_gauss',
                          selection='xray')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='count_gauss',
                          selection='spa')
    fit = sepnmf()
    print_info(fit)

    sepnmf = nimfa.SepNmf(V, rank=rank, compression='count_gauss',
                          selection='none')
    fit = sepnmf()
    print_info(fit)


def run(V=None, V1=None):
    """
    Run examples.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    :param V1: (Second) Target matrix to estimate used in multiple NMF (e. g. SNMNMF).
    :type V1: :class:`numpy.matrix`
    """
    if V is None or V1 is None:
        prng = np.random.RandomState(42)
        # construct target matrices
        V = prng.rand(20, 30)
        V1 = prng.rand(20, 25)
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
    run_sepnmf(V)


if __name__ == "__main__":
    prng = np.random.RandomState(42)
    V = prng.rand(20, 30)
    V1 = prng.rand(20, 25)
    run(V, V1)
