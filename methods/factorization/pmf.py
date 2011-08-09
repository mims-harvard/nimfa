from operator import div, add
from math import sqrt

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Pmf(mstd.Nmf_std):
    """
    Probabilistic Nonnegative Matrix Factorization (PMF) interpreting target matrix (V) as samples from a multinomial [9], [10], (Hansen, 2005)
    and using Euclidean distance for convergence test.
    
    [9] Laurberg, H.,et. al., (2008). Theorems on positive data: on the uniqueness of NMF. Computational intelligence and neuroscience.
    [10] Hansen, L. K., (2008). Generalization in high-dimensional factor models. Web: http://www.stanford.edu/group/mmds/slides2008/hansen.pdf.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        There are no algorithm specific model options for this method.
        """
        self.name = "pmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
                
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.W = elop(self.W, repmat(self.W.sum(axis = 0), self.V.shape[0], 1), div)
            self.H = elop(self.H, repmat(self.H.sum(axis = 1), 1, self.V.shape[1]), div)
            self.v_factor = self.V.sum()
            self.V_n = sop(self.V.copy(), self.v_factor, div)
            self.P = sp.spdiags([1. / self.rank for _ in xrange(self.rank)], 0, self.rank, self.rank, 'csr')
            self.sqrt_P = sop(self.P, s = None, op = sqrt) 
            pobj = cobj = self.objective() 
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                self._adjustment()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
                if self.track_error:
                    self.tracker._track_error(self.residuals())
            self.W = self.v_factor * dot(self.W, self.sqrt_P) 
            self.H = dot(self.sqrt_P, self.H)
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker._track_factor(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter - 1
        self.final_obj = cobj
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iter and self.max_iter < iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)
        
    def _set_params(self):
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mtrack.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
        
    def update(self):
        """Update basis and mixture matrix."""
        # E step
        Qnorm = dot(dot(self.W, self.P), self.H)
        for k in xrange(self.rank):
            # E-step
            Q = elop(self.P[k,k] * dot(self.W[:, k], self.H[k, :]), sop(Qnorm, np.finfo(Qnorm.dtype).eps, add), div)
            V_nQ = dot(self.V_n, Q)
            # M-step 
            dum = V_nQ.sum(axis = 1)
            s_dum = dum.sum()
            for i in xrange(self.W.shape[0]):
                self.W[i, k] = dum[i, 0] / s_dum
            dum = V_nQ.sum(axis = 0)
            s_dum = dum.sum()
            for i in xrange(self.H.shape[1]):
                self.H[k, i] = dum[0, i] / s_dum
    
    def objective(self):
        """Compute Euclidean distance cost function."""
        # relative error
        error_v_n =  abs(self.V_n - dot(self.W, self.H)).mean() / self.V_n.mean()
        # Euclidean distance
        return (elop(self.V - dot(dot(dot(self.W, self.sqrt_P) * self.v_factor, self.sqrt_P), self.H), 2, pow)).sum()
        
    def __str__(self):
        return self.name