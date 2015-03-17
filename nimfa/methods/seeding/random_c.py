
"""
#######################################
Random_c (``methods.seeding.random_c``)
#######################################

Random C [Albright2006]_ is inexpensive initialization method for nonnegative
matrix factorization. It is inspired by the C matrix in of the CUR decomposition.
The Random C initialization is similar to the Random Vcol method (see
mod:`methods.seeding.random_vcol`) except it chooses p columns at random from
the longest (in 2-norm) columns in target matrix (V), which generally means the
most dense columns of target matrix.

Initialization of each column of basis matrix is done by averaging p random
columns of l longest columns of target matrix. Initialization
of mixture matrix is similar except for row operations.    
"""

from nimfa.utils.linalg import *

__all__ = ['Random_c']


class Random_c(object):

    def __init__(self):
        self.name = "random_c"

    def initialize(self, V, rank, options):
        """
        Return initialized basis and mixture matrix. Initialized matrices are
        of the same type as passed target matrix.
        
        :param V: Target matrix, the matrix for MF method to estimate. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or
                :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify the algorithm and model specific options (e.g. initialization of
                extra matrix factor, seeding parameters).

                Option ``p_c`` represents the number of columns of target matrix
                used to average the column of basis matrix. Default
                value for ``p_c`` is 1/5 * (target.shape[1]).

                Option ``p_r`` represents the number of rows of target matrix used
                to average the row of basis matrix. Default value for
                ``p_r`` is 1/5 * (target.shape[0]).

                Option ``l_c`` represents the first l_c columns of target matrix sorted
                descending by length (2-norm). Default value for ``l_c`` is  1/2 * (target.shape[1]).

                Option ``l_r`` represent first l_r rows of target matrix sorted
                descending by length (2-norm). Default value for
                ``l_r`` is 1/2 * (target.shape[0]).
        :type options: `dict`
        """
        self.rank = rank
        self.p_c = options.get('p_c', int(ceil(1. / 5 * V.shape[1])))
        self.p_r = options.get('p_r', int(ceil(1. / 5 * V.shape[0])))
        self.l_c = options.get('l_c', int(ceil(1. / 2 * V.shape[1])))
        self.l_r = options.get('l_r', int(ceil(1. / 2 * V.shape[0])))
        self.prng = np.random.RandomState()
        if sp.isspmatrix(V):
            self.W = sp.lil_matrix((V.shape[0], self.rank))
            self.H = sp.lil_matrix((self.rank, V.shape[1]))
            top_c = sorted(enumerate([norm(V[:, i], 2)
                           for i in range(V.shape[1])]), key=itemgetter(1), reverse=True)[:self.l_c]
            top_r = sorted(
                enumerate([norm(V[i, :], 2) for i in range(V.shape[0])]), key=itemgetter(1), reverse=True)[:self.l_r]
        else:
            self.W = np.mat(np.zeros((V.shape[0], self.rank)))
            self.H = np.mat(np.zeros((self.rank, V.shape[1])))
            top_c = sorted(enumerate([norm(V[:, i], 2)
                           for i in range(V.shape[1])]), key=itemgetter(1), reverse=True)[:self.l_c]
            top_r = sorted(
                enumerate([norm(V[i, :], 2) for i in range(V.shape[0])]), key=itemgetter(1), reverse=True)[:self.l_r]
        top_c = np.mat(list(zip(*top_c))[0])
        top_r = np.mat(list(zip(*top_r))[0])
        for i in range(self.rank):
            self.W[:, i] = V[
                :, top_c[0, self.prng.randint(low=0, high=self.l_c, size=self.p_c)].tolist()[0]].mean(axis=1)
            self.H[i, :] = V[
                top_r[0, self.prng.randint(low=0, high=self.l_r, size=self.p_r)].tolist()[0], :].mean(axis=0)
        # return sparse or dense initialization
        if sp.isspmatrix(V):
            return self.W.tocsr(), self.H.tocsr()
        else:
            return self.W, self.H

    def __repr__(self):
        return "random_c.Random_c()"

    def __str__(self):
        return self.name
