import scipy.sparse as sp
import numpy as np

from utils.linalg import argmax

class Random(object):
    """
    Random is the simplest MF initialization method.
    
    The entries of factors are drawn from a uniform distribution over [0, max(target matrix)). Generated matrix factors are sparse
    matrices with the default density parameter of 0.01. 
    """
    
    def __init__(self):
        self.name = "random"
       
    def initialize(self, V, rank, **options):
        """
        Return initialized basis and mixture matrix (and additional factors if specified in :param:`Sn`, n = 1, 2, ..., k). 
        Initialized matrices are of the same type as passed target matrix. 
        
        :param V: Target matrix, the matrix for MF method to estimate.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                            #. :param Sn: n = 1, 2, 3, ..., k specify additional k matrix factors which need to be initialized.
                                          The value of each option Sn is a tuple, denoting matrix shape. Matrix factors are returned in the same
                                          order as their descriptions in input.
                               :type Sn: k tuples
                            #. :param density: Density of the generated matrices. Density of 1 means a full matrix, density of 0 means a 
                                               matrix with no nonzero items. Default value is 0.01. Density parameter is applied 
                                               only if passed target :param:`V` is an instance of one :class:`scipy.sparse` sparse
                                               types. 
                               :type density: `float`
        """
        self.rank = rank
        self.density = options.get('density', 0.01)
        self.max = argmax(V, axis = None)[0]
        self._format = V.getformat()
        gen = self._gen_sparse if sp.isspmatrix(V) else self._gen_dense
        self.W = gen(V.shape[0], self.rank)
        self.H = gen(self.rank, V.shape[1])
        mfs = [self.W, self.H]
        for sn in options:
            if sn[0] is 'S' and sn[1:].isdigit():
                mfs.append(gen(options[sn][0], options[sn][1]))
        return mfs
    
    def _gen_sparse(self, dim1, dim2):
        """
        Return randomly initialized sparse matrix of specified dimensions.
        
        :param dim1: Dimension along first axis.
        :type dim1: `int`
        :param dim2: Dimension along second axis.
        :type dim2: `int`
        """
        return self.max * sp.rand(dim1, dim2, density = self.density, format = self._format, dtype = 'd')
        
    def _gen_dense(self, dim1, dim2):
        """
        Return randomly initialized :class:`numpy.matrix` matrix of specified dimensions.
        
        :param dim1: Dimension along first axis.
        :type dim1: `int`
        :param dim2: Dimension along second axis.
        :type dim2: `int`
        """
        return self.max * np.matrix(np.random.rand(dim1, dim2))
    
    def __repr__(self):
        return "random.Random()"
    
    def __str__(self):
        return self.name