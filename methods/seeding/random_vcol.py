import scipy.sparse as sp

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
       
    def initialize(self, V, rank, **options):
        """
        Return initialized basis and mixture matrix. 
        
        :param V: Target matrix, the matrix for MF method to estimate. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                        :param p_c: The number of columns of target matrix used to average the column of basis matrix.
                                    Default value for :param:`p_c` is 1/20 * (target.shape[1])
                        :type p_c: `int`
                        :param p_r: The number of rows of target matrix used to average the row of basis matrix.
                                    Default value for :param:`p_r` is 1/20 * (target.shape[0])
                        :type p_r: `int`
        """
        self.V = V
        self.rank = rank
        self.p_c = options.get('p_c', 1 / 20 * self.V.shape[1])
        self.p_r = options.get('p_r', 1 / 20 * self.V.shape[0])
    
    def __repr__(self):
        return "random_vcol.Random_vcol()"
    
    def __str__(self):
        return self.name