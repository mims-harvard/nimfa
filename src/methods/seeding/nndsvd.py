import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as sla
import numpy.linalg as nla
import random
from math import sqrt

import utils.utils as utils


class Nndsvd(object):
    """
    Nonnegative Double Singular Value Decomposition (NNDSVD) [1] is a new method designed to enhance the initialization
    stage of the nonnegative matrix factorization. The basic algorithm contains no randomization and is based on 
    two SVD processes, one approximating the data matrix, the other approximating positive sections of the 
    resulting partial SVD factors utilizing an algebraic property of unit rank matrices. 
    
    NNDSVD is well suited to initialize NMF algorithms with sparse factors. Numerical examples suggest that NNDSVD leads 
    to rapid reduction of the approximation error of many NMF algorithms. 
    
    [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head start for nonnegative matrix factorization, Pattern Recognition, 2007,
    doi:10.1016/j.patcog.2007.09.010 
    """


    def __init__(self):
        """
        :param V: Data instances to be clustered. If not None, clustering will be executed immediately after initialization unless initialize_only=True.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.ndarray` or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param flag: Indicates the variant of the NNDSVD algorithm. Possible values are 
                    0 -- NNDSVD
                    1 -- NNDSVDa (fill in the zero elements with the average)
                    2 -- NNDSVDar (fill in the zero elements with random values in the space [0:average/100])
        :type flag: `int`
        """
        self.name = "nndsvd"
        
    def initialize(self, V, rank, flag = 0):
        self.V = V
        self.rank = rank
        self.flag = 0
        if sp.isspmatrix(self.V):
            if any(self.V.data < 0):
                raise utils.MFError("The input matrix contains negative elements.")
            U, S, V = sla.svd(self.V, self.rank)
            avg = np.average(self.V.data)
        else:
            if any(self.V < 0):
                raise utils.MFError("The input matrix contains negative elements.")
            U, S, V = nla.svd(self.V)
            avg = np.average(self.V)
        self.W = np.matrix(np.zeros((self.V.shape[0], self.rank)))
        self.H = np.matrix(np.zeros((self.rank, self.V.shape[1])))
        
        # choose the first singular triplet to be nonnegative
        self.W[:,0] = sqrt(S[0]) * abs(U[:,0])
        self.H[0,:] = sqrt(S[0]) * abs(V[:,0])
        
        # second svd for the other factors
        for i in xrange(1, self.rank):
            uu = U[:,i]
            vv = V[:,i]
            uup = self.pos(uu); uun = self.neg(uu)
            vvp = self.pos(vv); vvn = self.neg(vv)
            n_uup = nla.norm(uup, 2); n_vvp = nla.norm(vvp, 2)
            n_uun = nla.norm(uun, 2); n_vvn = nla.norm(vvn, 2)
            termp = n_uup * n_vvp; termn = n_uun * n_vvn
            if (termp >= termn):
                self.W[:,i] = sqrt(S[i] * termp) * uup / n_uup 
                self.H[i,:] = sqrt(S[i] * termp) * vvp.T / n_vvp
            else:
                self.W[:,i] = sqrt(S[i] * termn) * uun / n_uun
                self.H[i,:] = sqrt(S[i] * termn) * vvn.T / n_vvn
        
        self.W[self.W < 1e-11] = 0
        self.H[self.H < 1e-11] = 0 
        
        # NNDSVDa
        if self.flag == 1:
            self.W[self.W == 0] = avg
            self.H[self.H == 0] = avg
        
        # NNDSVDar
        if self.flag == 2:
            n1 = len(self.W[self.W == 0])
            n2 = len(self.H[self.H == 0])
            self.W[self.W == 0] = avg * random.uniform(n1, 1) / 100
            self.H[self.H == 0] = avg * random.uniform(n2, 1) / 100
        return self.W, self.H
            
    def _pos(self, X):
        """Return positive section of matrix or vector."""
        return np.multiply(X >= 0, X)
    
    def _neg(self, X):
        """Return negative section of matrix or vector."""
        return np.multiply(X < 0, -X)
            
            
            
        