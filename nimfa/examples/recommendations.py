
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

import nimfa
import numpy as np
import scipy.sparse as sp
from os.path import dirname, abspath, sep
from warnings import warn

try:
    import matplotlib.pylab as plb
except ImportError, exc:
    warn("Matplotlib must be installed to run Recommendations example.")

def run():
    """
    Run SNMF/R on the MovieLens data set.
    
    Factorization is run on `ua.base`, `ua.test` and `ub.base`, `ub.test` data set. This is MovieLens's data set split 
    of the data into training and test set. Both test data sets are disjoint and with exactly 10 ratings per user
    in the test set. 
    """
    for data_set in ['ua', 'ub']:
        # read ratings from MovieLens data set 
        V = read(data_set)
        # preprocess MovieLens data matrix
        V, maxs = preprocess(V)
        # run factorization
        W, H = factorize(V.todense())
        # plot RMSE rate on MovieLens data set. 
        plot(W, H, data_set, maxs)
        
def factorize(V):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The MovieLens data matrix. 
    :type V: `scipy.sparse.csr_matrix`
    """
    model = nimfa.mf(V, 
                  seed = "random_vcol", 
                  rank = 12, 
                  method = "snmf", 
                  max_iter = 15, 
                  initialize_only = True,
                  version = 'r',
                  eta = 1.,
                  beta = 1e-4, 
                  i_conv = 10,
                  w_min_change = 0)
    print "Performing %s %s %d factorization ..." % (model, model.seed, model.rank) 
    fit = nimfa.mf_run(model)
    print "... Finished"
    sparse_w, sparse_h = fit.fit.sparseness()
    print """Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter, fit.distance(metric = 'euclidean'), sparse_w, sparse_h)
    return fit.basis(), fit.coef()
    
def read(data_set):
    """
    Read movies' ratings data from MovieLens data set. 
    
    Construct a user-by-item matrix. This matrix is sparse, therefore ``scipy.sparse`` format is used. For construction
    LIL sparse format is used, which is an efficient structure for constructing sparse matrices incrementally. 
    
    Return the MovieLens sparse data matrix in LIL format. 
    
    :param data_set: Name of the split data set to be read. 
    :type data_set: `str`
    """
    print "Reading MovieLens ratings data set ..."
    dir = dirname(dirname(abspath(__file__))) + sep + 'datasets' + sep + 'MovieLens' + sep + data_set + '.base'
    V = sp.lil_matrix((943, 1682))
    for line in open(dir): 
        u, i, r, _ = map(int, line.split())
        V[u - 1, i - 1] = r
    print "... Finished."
    return V 
            
def preprocess(V):
    """
    Preprocess MovieLens data matrix. Normalize data.
    
    Return preprocessed target sparse data matrix in CSR format and users' maximum ratings. Returned matrix's shape is 943 (users) x 1682 (movies). 
    The sparse data matrix is converted to CSR format for fast arithmetic and matrix vector operations. 
    
    :param V: The MovieLens data matrix. 
    :type V: `scipy.sparse.lil_matrix`
    """
    print "Preprocessing data matrix ..."
    V = V.tocsr()
    maxs = [np.max(V[i, :].todense()) for i in xrange(V.shape[0])]
    now = 0
    for row in xrange(V.shape[0]):
        upto = V.indptr[row+1]
        while now < upto:
            col = V.indices[now]
            V.data[now] /= maxs[row]
            now += 1
    print "... Finished." 
    return V, maxs
            
def plot(W, H, data_set, maxs):
    """
    Plot the RMSE error rate on MovieLens data set. 
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `scipy.sparse.csr_matrix`
    :param H: Mixture matrix of the fitted factorization model.
    :type H: `scipy.sparse.csr_matrix`
    :param data_set: Name of the split data set to be read. 
    :type data_set: `str`
    :param maxs: Users' maximum ratings (used in normalization). 
    :type maxs: `list`
    """
    print "Plotting RMSE rates ..."
    dir = dirname(dirname(abspath(__file__))) + sep + 'datasets' + sep + 'MovieLens' + sep + data_set + '.test'
    rmse = 0
    n = 0
    for line in open(dir): 
        u, i, r, _ = map(int, line.split())
        rmse += ((W[u - 1, :] * H[:, i - 1])[0, 0] + maxs[u - 1] - r)** 2
        n += 1
    rmse /= n
    print rmse
    print "... Finished."

if __name__ == "__main__":
    """Run the Recommendations example."""
    run()
