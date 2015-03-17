
"""
###################################
Random (``methods.seeding.random``)
###################################

Random is the simplest MF initialization method.

The entries of factors are drawn from a uniform distribution over
[0, max(target matrix)). Generated matrix factors are sparse matrices with the
default density parameter of 0.01.
"""

from nimfa.utils.linalg import *

__all__ = ['Random']

class Random(object):

    def __init__(self):
        self.name = "random"

    def initialize(self, V, rank, options):
        """
        Return initialized basis and mixture matrix (and additional factors if
        specified in :param:`Sn`, n = 1, 2, ..., k).
        Initialized matrices are of the same type as passed target matrix. 
        
        :param V: Target matrix, the matrix for MF method to estimate.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or
                :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify the algorithm and model specific options (e.g. initialization of
                extra matrix factor, seeding parameters).
                    
                Option ``Sn``, n = 1, 2, 3, ..., k specifies additional k matrix factors which
                need to be initialized. The value of each option Sn is a tuple denoting matrix
                shape. Matrix factors are returned in the same order as their descriptions in input.

                Option ``density`` represents density of generated matrices. Density of 1 means a
                full matrix, density of 0 means a matrix with no nonzero items. Default value is 0.7.
                Density parameter is applied only if passed target ``V`` is an instance of one :class:`scipy.sparse` sparse types.
        :type options: `dict`
        """
        self.rank = rank
        self.density = options.get('density', 0.7)
        if sp.isspmatrix(V):
            self.max = V.data.max()
            self._format = V.getformat()
            gen = self.gen_sparse
        else:
            self.max = V.max()
            self.prng = np.random.RandomState()
            gen = self.gen_dense
        self.W = gen(V.shape[0], self.rank)
        self.H = gen(self.rank, V.shape[1])
        mfs = [self.W, self.H]
        for sn in options:
            if sn[0] is 'S' and sn[1:].isdigit():
                mfs.append(gen(options[sn][0], options[sn][1]))
        return mfs

    def gen_sparse(self, dim1, dim2):
        """
        Return randomly initialized sparse matrix of specified dimensions.
        
        :param dim1: Dimension along first axis.
        :type dim1: `int`
        :param dim2: Dimension along second axis.
        :type dim2: `int`
        """
        rnd = sp.rand(dim1, dim2, density=self.density, format=self._format)
        return abs(self.max * rnd)

    def gen_dense(self, dim1, dim2):
        """
        Return randomly initialized :class:`numpy.matrix` matrix of specified
        dimensions.
        
        :param dim1: Dimension along first axis.
        :type dim1: `int`
        :param dim2: Dimension along second axis.
        :type dim2: `int`
        """
        return np.mat(self.prng.uniform(0, self.max, (dim1, dim2)))

    def __repr__(self):
        return "random.Random()"

    def __str__(self):
        return self.name
