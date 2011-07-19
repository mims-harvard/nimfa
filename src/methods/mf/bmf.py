from math import pow
from operator import div

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Bmf(mstd.Nmf_std):
    """
    Binary Matrix Factorization (BMF) [8].
    
    BMF extends standard NMF to binary matrices. Given a binary target matrix (V), we want to factorize it into binary 
    basis and mixture matrices, thus conserving the most important integer property of the target matrix. Common methodologies 
    include penalty function algorithm and thresholding algorithm. This class implements penalty function algorithm. 
    
    [8] Z. Zhang, T. Li, C. H. Q. Ding, X. Zhang: Binary Matrix Factorization with Applications. ICDM 2007
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        Algorithm specific model options are 'lambda_w' and 'lambda_h' parameters which controls how fast lambda 
        should increase. This influences convergence of basis (W) and mixture (H) matrices to binary values during the 
        update. 
            #. A value lambda < 1 will result in a nonbinary decompositions as the update rule effectively
              is a conventional NMF update rule. 
            #. A value lambda > 1 give more weight to make the factorization binary with increasing iterations.
        If parameters are not specified, default value of 1.1 is taken for both of them. 
        """
        mstd.Nmf_std.__init__(self, params)
        self.aname = "bnmf"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
        
        self._lambda_w = 1. / self.max_iters if self.max_iters else 1. / 10
        self._lambda_h = self._lambda_w         
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            pobj = cobj = self.objective()
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                self._adjustment()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.tracker != None:
                self.tracker.append(mtrack.Mf_track(W = self.W.copy(), H = self.H.copy()))
        
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
        self.lambda_w = self.options['lambda_w'] if self.options and 'lambda_w' in self.options else 1.1
        self.lambda_h = self.options['lambda_h'] if self.options and 'lambda_h' in self.options else 1.1
        self.tracker = [] if self.options and 'track' in self.options and self.options['track'] and self.n_run > 1 else None
    
    def update(self):
        """Update basis and mixture matrix."""
        H1 = dot(self.W.T, self.V) + 3. * self._lambda_h * multiply(self.H, self.H)
        H2 = dot(dot(self.W.T, self.W), self.H) + 2. * self._lambda_h * sop(self.H, 3, pow) + self._lambda_h * self.H
        self.H = multiply(self.H, elop(H1, H2, div))
        W1 = dot(self.V, self.H.T) + 3. * self._lambda_w * multiply(self.W, self.W)
        W2 = dot(self.W, dot(self.H, self.H.T)) + 2. * self._lambda_w * sop(self.W, 3, pow) + self._lambda_w * self.W
        self.W = multiply(self.W, elop(W1, W2, div))
        self._lambda_h = self.lambda_h * self._lambda_h
        self._lambda_w = self.lambda_w * self._lambda_w
        
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
    def _adjustment(self):
        """Adjust small value to factors to avoid numerical underflow."""
        self.H = sop(self.W, np.finfo(self.H.dtype).eps)
        self.W = sop(self.H, np.finfo(self.W.dtype).eps)
    
        