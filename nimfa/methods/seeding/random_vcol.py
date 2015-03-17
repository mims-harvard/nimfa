
"""
#############################################
Random_vcol (``methods.seeding.random_vcol``)
#############################################

Random Vcol [Albright2006]_ is inexpensive initialization method for nonnegative
matrix factorization. Random Vcol forms an initialization of each column of the
basis matrix (W) by averaging p random columns of target matrix (V). Similarly,
Random Vcol forms an initialization of each row of the mixture matrix (H) by
averaging p random rows of target matrix (V). It makes more sense to build the
basis vectors from the given data than to form completely random basis vectors,
as random initialization does. Sparse matrices are built from the original
sparse data.

Method's performance lies between random initialization and centroid
initialization, which is built from the centroid decomposition.
"""

from nimfa.utils.linalg import *

__all__ = ['Random_vcol']


class Random_vcol(object):

    def __init__(self):
        self.name = "random_vcol"

    def initialize(self, V, rank, options):
        """
        Return initialized basis and mixture matrix. Initialized matrices are of
        the same type as passed target matrix.
        
        :param V: Target matrix, the matrix for MF method to estimate. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or
                :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify the algorithm and model specific options (e.g. initialization of
                extra matrix factor, seeding parameters).
                    
                Option ``p_c`` represents the number of columns of target matrix used
                to average the column of basis matrix. Default value for ``p_c`` is
                1/5 * (target.shape[1]).

                Option ``p_r`` represent the number of rows of target matrix used to
                average the row of basis matrix. Default value for ``p_r`` is 1/5 * (target.shape[0]).
        :type options: `dict`
        """
        self.rank = rank
        self.p_c = options.get('p_c', int(ceil(1. / 5 * V.shape[1])))
        self.p_r = options.get('p_r', int(ceil(1. / 5 * V.shape[0])))
        self.prng = np.random.RandomState()
        if sp.isspmatrix(V):
            self.W = sp.lil_matrix((V.shape[0], self.rank))
            self.H = sp.lil_matrix((self.rank, V.shape[1]))
        else:
            self.W = np.mat(np.zeros((V.shape[0], self.rank)))
            self.H = np.mat(np.zeros((self.rank, V.shape[1])))
        for i in range(self.rank):
            self.W[:, i] = V[:, self.prng.randint(
                low=0, high=V.shape[1], size=self.p_c)].mean(axis=1)
            self.H[i, :] = V[
                self.prng.randint(low=0, high=V.shape[0], size=self.p_r), :].mean(axis=0)
        # return sparse or dense initialization
        if sp.isspmatrix(V):
            return self.W.tocsr(), self.H.tocsr()
        else:
            return self.W, self.H

    def __repr__(self):
        return "random_vcol.Random_vcol()"

    def __str__(self):
        return self.name
