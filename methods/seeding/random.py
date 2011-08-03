import scipy.sparse as sp

from utils.linalg import argmax

class Random(object):
    """
    Random is the simplest MF initialization method.
    
    The entries of factors are drawn from a uniform distribution over [0, max(target matrix)]. Generated matrix factors are sparse
    matrices with the default density parameter of 0.01. 
    """
    
    def __init__(self):
        self.name = "random"
       
    def initialize(self, V, rank, **options):
        """
        Return initialized basis and mixture matrix (and additional factors if specified in :param:`options`). 
        
        :param V: Target matrix, the matrix for MF method to estimate.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.ndarray` or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                            #. :param Sn: n = 1, 2, 3, ..., n specify additional n matrix factors which need to be initialized.
                                          The value of each option Sn is a tuple, denoting matrix shape. Matrix factors are returned in the same
                                          order as their descriptions in input.
                               :type Sn: n tuples
                            #. :param density: Density of the generated matrices. Density of 1 means a full matrix, density of 0 means a 
                                               matrix with no nonzero items. Default value is 0.01. 
                               :type density: `float`
        :type options: `dict`
        """
        self.V = V
        self.rank = rank
        self.density = options.get('density', 0.01)
        max = argmax(self.V, axis = None)[0]
        self.W = max * sp.rand(self.V.shape[0], self.rank, density = self.density, format = 'csr', dtype = 'd')
        self.H = max * sp.rand(self.rank, self.V.shape[1], density = self.density, format = 'csr', dtype = 'd')
        mfs = [self.W, self.H]
        for sn in options:
            if sn[0] is 'S' and sn[1:].isdigit():
                mfs.append(max * sp.rand(options[sn][0], options[sn][1], density = self.density, format = 'csr', dtype = 'd'))
        return mfs
    
    def __repr__(self):
        return "random.Random()"
    
    def __str__(self):
        return self.name