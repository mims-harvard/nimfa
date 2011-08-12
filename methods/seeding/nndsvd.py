import numpy as np
import scipy.sparse as sp
from math import sqrt

from utils.utils import *
from utils.linalg import *

class Nndsvd(object):
    """
    Nonnegative Double Singular Value Decomposition (NNDSVD) [1] is a new method designed to enhance the initialization
    stage of the nonnegative matrix factorization. The basic algorithm contains no randomization and is based on 
    two SVD processes, one approximating the data matrix, the other approximating positive sections of the 
    resulting partial SVD factors utilizing an algebraic property of unit rank matrices. 
    
    NNDSVD is well suited to initialize NMF algorithms with sparse factors. Numerical examples suggest that NNDSVD leads 
    to rapid reduction of the approximation error of many NMF algorithms. By setting algorithm options :param:`flag` dense factors can be
    generated. 
    
    [1] Boutsidis, C., Gallopoulos, E., (2007). SVD-based initialization: A head start for nonnegative matrix factorization, Pattern Recognition, 2007,
    doi:10.1016/j.patcog.2007.09.010.
    """

    def __init__(self):
        self.name = "nndsvd"
        
    def initialize(self, V, rank, options):
        """
        Return initialized basis and mixture matrix. 
        
        Initialized matrices are sparse :class:`scipy.sparse.csr_matrix` if NNDSVD variant is specified by the :param:`flag` option,
        else matrices are :class:`numpy.matrix`.
        
        :param V: Target matrix, the matrix for MF method to estimate. Data instances to be clustered. 
        :type V: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                        :param flag: Indicate the variant of the NNDSVD algorithm. Possible values are:
                                     #. 0 -- NNDSVD,
                                     #. 1 -- NNDSVDa (fill in the zero elements with the average),
                                     #. 2 -- NNDSVDar (fill in the zero elements with random values in the space [0:average/100]).
                                    Default is NNDSVD.
                        :type flag: `int`
        """
        self.rank = rank
        self.flag = options.get('flag', 0)
        if negative(V):
            raise MFError("The input matrix contains negative elements.")
        U, S, V = svd(V, self.rank)
        self.W = np.mat(np.zeros((V.shape[0], self.rank)))
        self.H = np.mat(np.zeros((self.rank, V.shape[1])))
        
        # choose the first singular triplet to be nonnegative
        self.W[:,0] = sqrt(S[0]) * abs(U[:,0])
        self.H[0,:] = sqrt(S[0]) * abs(V[:,0])
        
        # second svd for the other factors
        for i in xrange(1, self.rank):
            uu = U[:,i]
            vv = V[:,i]
            uup = self.pos(uu); uun = self.neg(uu)
            vvp = self.pos(vv); vvn = self.neg(vv)
            n_uup = norm(uup, 2); n_vvp = norm(vvp, 2)
            n_uun = norm(uun, 2); n_vvn = norm(vvn, 2)
            termp = n_uup * n_vvp; termn = n_uun * n_vvn
            if (termp >= termn):
                self.W[:,i] = sqrt(S[i] * termp) / n_uup * uup 
                self.H[i,:] = sqrt(S[i] * termp) / n_vvp * vvp.T 
            else:
                self.W[:,i] = sqrt(S[i] * termn) / n_uun * uun
                self.H[i,:] = sqrt(S[i] * termn) / n_vvn * vvn.T
        
        self.W[self.W < 1e-11] = 0
        self.H[self.H < 1e-11] = 0 
        if self.flag == 0:
            return sp.csr_matrix(self.W), sp.csr_matrix(self.H)
        
        # NNDSVDa
        if self.flag == 1:
            avg = V.mean()
            self.W[self.W == 0] = avg
            self.H[self.H == 0] = avg
        
        # NNDSVDar
        if self.flag == 2:
            avg = V.mean()
            n1 = len(self.W[self.W == 0])
            n2 = len(self.H[self.H == 0])
            self.W[self.W == 0] = avg * np.random.uniform(n1, 1) / 100
            self.H[self.H == 0] = avg * np.random.uniform(n2, 1) / 100
        return self.W, self.H
            
    def _pos(self, X):
        """Return positive section of matrix or vector."""
        return multiply(X >= 0, X)
    
    def _neg(self, X):
        """Return negative section of matrix or vector."""
        return multiply(X < 0, -X)
    
    def __repr__(self):
        return "nndsvd.Nndsvd()"
    
    def __str__(self):
        return self.name
            
            
            
        