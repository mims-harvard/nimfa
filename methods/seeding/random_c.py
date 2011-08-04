import scipy.sparse as sp

class Random_c(object):
    """
    Random C [17] is inexpensive initialization method for nonnegative matrix factorization. It is inspired by the C matrix in
    of the CUR decomposition. The Random C initialization is similar to the Random Vcol method (see mod:`methods.seeding.random_vcol`)
    except it chooses p columns at random from the longest (in 2-norm) columns in target matrix (V), which generally means the most
    dense columns of target matrix.  
    
    [17] Albright, R. et al., (2006). Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, (919), p.1-18. 
        Available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.2161&rep=rep1&type=pdf.
    """
    
    def __init__(self):
        self.name = "random_c"
       
    def initialize(self, V, rank, **options):
        """
        TODO
        """
        return None
    
    def __repr__(self):
        return "random_c.Random_c()"
    
    def __str__(self):
        return self.name