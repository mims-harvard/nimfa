
"""
    ##############################################
    Recommendations (``examples.recommendations``)
    ##############################################
    
    In this examples of collaborative filtering we consider movie recommendation using common MovieLens data set. It 
    represents typical cold start problem. A recommender system compares the user's profile to reference
    characteristics from the user's social environment. In the collaborative filtering approach, the recommender
    system identify users who share the same preference with the active user and propose items which the like-minded
    users favoured (and the active user has not yet seen).     
    
    We used the MovieLens 100k data set in this example. This data set consists of 100 000 ratings (1-5) from 943
    users on 1682 movies. Each user has rated at least 20 movies. Simple demographic info for the users is included. 
    Factorization is performed on a split data set as provided by the collector of the data. The data is split into 
    two disjoint sets each consisting of training set and a test set with exactly 10 ratings per user. 
    
    It is common that matrices in the field of recommendation systems are very sparse (ordinary user rates only a small
    fraction of items from the large items' set), therefore ``scipy.sparse`` matrix formats are used in this example. 
    
    The configuration of this example is SNMF/R factorization method using Random Vcol algorithm for initialization. 
    
    .. note:: MovieLens movies' rating data set used in this example is not included in the `datasets` and need to be
      downloaded. Download links are listed in the ``datasets``. Download compressed version of the MovieLens 100k. 
      To run the example, the extracted data set must exist in the ``MovieLens`` directory under ``datasets``. 
      
    .. note:: No additional knowledge in terms of ratings' timestamps, information about items and their
       genres or demographic information about users is used in this example. 
      
    To run the example simply type::
        
        python recommendations.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.recommendations.run()
        
    .. note:: This example uses ``matplotlib`` library for producing visual interpretation of the RMSE error measure. 
    
"""

from os.path import dirname, abspath
from os.path import join
from warnings import warn

import numpy as np

import nimfa


try:
    import matplotlib.pylab as plb
except ImportError as exc:
    warn("Matplotlib must be installed to run Recommendations example.")


def run():
    """
    Run SNMF/R on the MovieLens data set.
    
    Factorization is run on `ua.base`, `ua.test` and `ub.base`, `ub.test` data set. This is MovieLens's data set split 
    of the data into training and test set. Both test data sets are disjoint and with exactly 10 ratings per user
    in the test set. 
    """
    for data_set in ['ua', 'ub']:
        V = read(data_set)
        W, H = factorize(V)
        rmse(W, H, data_set)


def factorize(V):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The MovieLens data matrix. 
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=30, max_iter=30, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                            fit.distance(metric='euclidean'),
                                                            sparse_w, sparse_h))
    return fit.basis(), fit.coef()


def read(data_set):
    """
    Read movies' ratings data from MovieLens data set. 
    
    :param data_set: Name of the split data set to be read.
    :type data_set: `str`
    """
    print("Read MovieLens data set")
    fname = join(dirname(dirname(abspath(__file__))), "datasets", "MovieLens", "%s.base" % data_set)
    V = np.ones((943, 1682)) * 2.5
    for line in open(fname):
        u, i, r, _ = list(map(int, line.split()))
        V[u - 1, i - 1] = r
    return V


def rmse(W, H, data_set):
    """
    Compute the RMSE error rate on MovieLens data set.
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
    :param H: Mixture matrix of the fitted factorization model.
    :type H: `numpy.matrix`
    :param data_set: Name of the split data set to be read. 
    :type data_set: `str`
    """
    fname = join(dirname(dirname(abspath(__file__))), "datasets", "MovieLens", "%s.test" % data_set)
    rmse = []
    for line in open(fname):
        u, i, r, _ = list(map(int, line.split()))
        sc = max(min((W[u - 1, :] * H[:, i - 1])[0, 0], 5), 1)
        rmse.append((sc - r) ** 2)
    print("RMSE: %5.3f" % np.sqrt(np.mean(rmse)))


if __name__ == "__main__":
    """Run the Recommendations example."""
    run()
