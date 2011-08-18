
from utils.linalg import *

class Random_vcol(object):
    """
    Random Vcol [17] is inexpensive initialization method for nonnegative matrix factorization. Random Vcol forms an initialization
    of each column of the basis matrix (W) by averaging p random columns of target matrix (V). Similarly, Random Vcol forms an initialization
    of each row of the mixture matrix (H) by averaging p random rows of target matrix (V). It makes more sense to build the 
    basis vectors from the given data than to form completely random basis vectors, as random initialization does. Sparse
    matrices are built from the original sparse data. 
    
    Method's performance lies between random initialization and centroid initialization, which is built from the centroid
    decomposition.  
    
    [17] Albright, R. et al., (2006). Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, 
        (919), p.1-18. Available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.2161&rep=rep1&type=pdf.
    """
    
    def __init__(self):
        self.name = "random_vcol"
       
    def initialize(self, V, rank, options):
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
        """
        self.rank = rank
        self.p_c = options.get('p_c', 1 / 5 * V.shape[1])
        self.p_r = options.get('p_r', 1 / 5 * V.shape[0])
        if sp.isspmatrix(V):
            self.W = sp.lil_matrix((V.shape[0], self.rank))
            self.H = sp.lil_matrix((self.rank, V.shape[1]))
        else:
            self.W = np.mat(np.zeros((V.shape[0], self.rank)))
            self.H = np.mat(np.zeros((self.rank, V.shape[1])))
        for i in xrange(self.rank):
            self.W[:, i] = V[:, np.random.randint(V.shape[1], size = self.p_c)].mean(axis = 1)
            self.H[i, :] = V[np.random.randint(V.shape[0], size = self.p_r), :].mean(axis = 0)
        return self.W.asformat(V.getformat()), self.H.asformat(V.getformat()) if sp.isspmatrix(V) else self.W, self.H
    
    def __repr__(self):
        return "random_vcol.Random_vcol()"
    
    def __str__(self):
        return self.name
    