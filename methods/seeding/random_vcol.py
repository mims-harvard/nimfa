import scipy.sparse as sp

class Random_vcol(object):
    """
    Random Vcol [17] is inexpensive initialization method for nonnegative matrix factorization. Random Vcol forms an initialization
    of each column of the basis matrix (W) by averaging p random columns of target matrix (V). It makes more sense to build the 
    basis vectors from the given data than to form completely random basis vectors, as random initialization does. Sparse
    matrices are built from the original sparse data. 
    
    Method's performance lies between random initialization and centroid initialization, which is built from the centroid
    decomposition.  
    
    [17] Albright, R. et al., 2006. Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, (919), p.1-18. 
        Available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.2161&rep=rep1&type=pdf.
    """
    
    def __init__(self):
        self.name = "random_vcol"
       
    def initialize(self, V, rank, **options):
        """
        TODO
        """
        return None
    
    def __repr__(self):
        return "random_vcol.Random_vcol()"
    
    def __str__(self):
        return self.name