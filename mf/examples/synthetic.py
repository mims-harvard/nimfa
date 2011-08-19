
"""
    This module contains examples of factorization runs. Since the data is artificially generated, this is not a valid test of
    models applicability to real world situations. It can however be used for demonstration of the library. 
    
    Examples are performed on 20 x 30 dense matrix, whose values are drawn from normal distribution wit zero mean one one variance.
    
    .. seealso:: README.rst     
    
    To run the examples simply type::
        
        python synthetic.py
        
    or call the module's function::
    
        synthetic.run()

"""

import mf
import numpy as np

def run_bd(V):
    """
    Run Bayesian decomposition.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    fit = mf.mf(V, )
    
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