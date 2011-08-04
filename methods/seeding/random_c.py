import scipy.sparse as sp
import numpy as np
from operator import itemgetter

class Random_c(object):
    """
    Random C [17] is inexpensive initialization method for nonnegative matrix factorization. It is inspired by the C matrix in
    of the CUR decomposition. The Random C initialization is similar to the Random Vcol method (see mod:`methods.seeding.random_vcol`)
    except it chooses p columns at random from the longest (in 2-norm) columns in target matrix (V), which generally means the most
    dense columns of target matrix. 
    
    Initialization of each column of basis matrix is done by averaging p random columns of l longest columns of target matrix. Initialization 
    of mixture matrix is similar except for row operations.
    
    [17] Albright, R. et al., (2006). Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, (919), p.1-18. 
        Available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.2161&rep=rep1&type=pdf.
    """
    
    def __init__(self):
        self.name = "random_c"
       
    def initialize(self, V, rank, **options):
        """
        Return initialized basis and mixture matrix. Initialized matrices are of the same type as passed target matrix. 
        
        :param V: Target matrix, the matrix for MF method to estimate. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                        :param p_c: The number of columns of target matrix used to average the column of basis matrix.
                                    Default value for :param:`p_c` is 1/5 * (target.shape[1]).
                        :type p_c: `int`
                        :param p_r: The number of rows of target matrix used to average the row of basis matrix.
                                    Default value for :param:`p_r` is 1/5 * (target.shape[0]).
                        :type p_r: `int`
                        :param l_c: First l_c columns of target matrix sorted descending by length (2-norm). Default value for :param:`l_c` is 
                                    1/2 * (target.shape[1]).
                        :type l_c: `int`
                        :param l_r: First l_r rows of target matrix sorted descending by length (2-norm). Default value for :param:`l_r` is 
                                    1/2 * (target.shape[0]).
                        :type l_r: `int`
        """
        self.rank = rank
        self.p_c = options.get('p_c', 1 / 5 * V.shape[1])
        self.p_r = options.get('p_r', 1 / 5 * V.shape[0])
        self.l_c = options.get('l_c', 1 / 2 * V.shape[1])
        self.l_r = options.get('l_r', 1 / 2 * V.shape[0])
        if sp.isspmatrix(V):
            self.W = sp.lil_matrix((V.shape[0], self.rank))
            self.H = sp.lil_matrix((self.rank, V.shape[1]))
            top_c = sorted(enumerate([np.linalg.norm(V[:, i].todense()) for i in xrange(V.shape[1])]), key = itemgetter(1), reverse = True)[: self.l_c]
            top_r = sorted(enumerate([np.linalg.norm(V[i, :].todense()) for i in xrange(V.shape[0])]), key = itemgetter(1), reverse = True)[: self.l_r]
        else:
            self.W = np.matrix(np.zeros((V.shape[0], self.rank)))
            self.H = np.matrix(np.zeros((self.rank, V.shape[1])))
            top_c = sorted(enumerate([np.linalg.norm(V[:, i]) for i in xrange(V.shape[1])]), key = itemgetter(1), reverse = True)[: self.l_c]
            top_r = sorted(enumerate([np.linalg.norm(V[i, :]) for i in xrange(V.shape[0])]), key = itemgetter(1), reverse = True)[: self.l_r]
        top_c = np.matrix(zip(*top_c)[1])
        top_r = np.matrix(zip(*top_r)[1])
        for i in xrange(self.rank):
            self.W[:, i] = V[:, top_c[0, np.random.randint(self.l_c, size = self.p_c)]].mean(axis = 1)
            self.H[i, :] = V[top_r[0, np.random.randint(self.l_r, size = self.p_r)], :].mean(axis = 0)
        return self.W.asformat(V.getformat()), self.H.asformat(V.getformat()) if sp.isspmatrix(V) else self.W, self.H
    
    def __repr__(self):
        return "random_c.Random_c()"
    
    def __str__(self):
        return self.name
    