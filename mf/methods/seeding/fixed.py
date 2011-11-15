
"""
#################################
Fixed (``methods.seeding.fixed``)
#################################

Fixed factorization. This is the option to completely specify the initial factorization by passing values for 
matrix factors.  
"""

from nimfa.utils.linalg import *

class Fixed(object):

    def __init__(self):
        self.name = "fixed"
        
    def _set_fixed(self, **factors):
        """Set initial factorization."""
        for k in factors.keys():
            if factors[k] != None: 
                factors[k] = np.matrix(factors[k]) if not sp.isspmatrix(factors[k]) else factors[k].copy()
            else:
                factors.pop(k)
        self.__dict__.update(factors) 
        
    def initialize(self, V, rank, options):
        """
        Return fixed initialized matrix factors.
        
        :param V: Target matrix, the matrix for MF method to estimate. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify:
                            #. algorithm;
                            #. model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                    
                        The following are Fixed options.
                        
                         :param idx: Name of the matrix (coefficient) matrix. Default is 0, corresponding to 
                                     factorization models with one mixture matrix (e.g. standard, nonsmooth model).
                         :type idx: `int`
        :type options: `dict`
        """
        self.idx = options.get('idx', 0)
        return (self.W, self.H) if self.idx == 0 else (self.W, getattr(self, 'H' + str(self.idx)))
    
    def __repr__(self):
        return "fixed.Fixed()"
    
    def __str__(self):
        return self.name