from math import log
from operator import div, add

import models.nmf_ns as mns
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Nsnmf(mns.Nmf_ns):
    """
    Nonsmooth Nonnegative Matrix Factorization (NSNMF) [14]. 
    
    NSNMF aims at finding localized, part-based representations of nonnegative multivariate data items. Generally this method
    produces a set of basis and encoding vectors representing not only the original data but also extracting highly localized 
    patterns. Because of the multiplicative nature of the standard model, sparseness in one of the factors almost certainly
    forces nonsparseness (or smoothness) in the other in order to compensate for the final product to reproduce the data as best
    as possible. With the modified standard model in NSNMF global sparseness is achieved. 
    
    In the new model the target matrix is estimated as the product V = WSH, where V, W and H are the same as in the original NMF
    model. The positive symmetric square matrix S is a smoothing matrix defined as S = (1 - theta)I + (theta/rank)11', where
    I is an identity matrix, 1 is a vector of ones, rank is factorization rank and theta is a smoothing parameter (0<=theta<=1). 
    
    The interpretation of S as a smoothing matrix can be explained as follows: Let X be a positive, nonzero, vector.
    Consider the transformed vector Y = SX. As theta --> 1, the vector Y tends to the constant vector with all elements almost
    equal to the average of the elements of X. This is the smoothest possible vector in the sense of nonsparseness because 
    all entries are equal to the same nonzero value. The parameter theta controls the extent of smoothness of the matrix 
    operator S. Due to the multiplicative nature of the model, strong smoothing in S forces strong sparseness in
    both the basis and the encoding vectors. Therefore, the parameter theta controls the sparseness of the model.     
    
    [14] Pascual-Montano, A., Carazo, J. M., Kochi, K., Lehmann, D., and Pascual-Marqui, R. D., (2006). Nonsmooth nonnegative matrix 
        factorization (nsnmf). IEEE transactions on pattern analysis and machine intelligence, 28(3), 403-415.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        Algorithm specific model option is :param:`theta` which can be passed with value as keyword argument.
        Parameter :param:`theta` is the smoothing parameter. Its value should be 0<=theta<=1. If not specified, default value  
        theta = 0.5 is used.  
        """
        mns.Nmf_ns.__init__(self, params)
        self.name = "nsnmf"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
                
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.S = sop((1 - self.theta) * sp.spdiags([1 for _ in xrange(self.rank)], 0, self.rank, self.rank, 'csr'), 
                         self.theta / self.rank, add)
            pobj = cobj = self.objective()
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.tracker != None:
                self.tracker.append(mtrack.Mf_track(W = self.W.copy(), H = self.H.copy()))
        
        self.n_iter = iter
        self.final_obj = cobj
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iters and self.max_iters < iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def _set_params(self):
        self.theta = self.options.get('theta', .5)
        self.tracker = [] if self.options.get('track', 0) and self.n_run > 1 else None
            
    def update(self):
        """Update basis and mixture matrix based on modified divergence multiplicative update rules."""
        # update mixture matrix H
        W = dot(self.W, self.S)
        H1 = repmat(W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(self.H, elop(dot(W.T, elop(self.V, dot(W, self.H), div)), H1, div))
        # update basis matrix W
        H = dot(self.S, self.H)
        W1 = repmat(H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, H), div), H.T), W1, div))
        # normalize basis matrix W
        W2 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.W = elop(self.W, W2, div)
    
    def objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(dot(self.W, self.S), self.H)
        return (multiply(self.V, elop(self.V, Va, log)) - self.V + Va).sum()
    
        