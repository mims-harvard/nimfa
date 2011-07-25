from math import log

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
        
        Algorithm specific model option is 'theta' which can be passed with value as keyword argument.
        Parameter theta is the smoothing parameter. Its value should be 0<=theta<=1. If not specified, default value  
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
        self.theta = self.options['theta'] if self.options and 'theta' in self.options else .5
        self.tracker = [] if self.options and 'track' in self.options and self.options['track'] and self.n_run > 1 else None
        
    def update(self):
        """Update basis and mixture matrix."""
        pass
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
        