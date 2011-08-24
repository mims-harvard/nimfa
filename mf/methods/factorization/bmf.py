
"""
###################################
Bmf (``methods.factorization.bmf``)
###################################

**Binary Matrix Factorization (BMF)** [Zhang2007]_.

BMF extends standard NMF to binary matrices. Given a binary target matrix (V), we want to factorize it into binary 
basis and mixture matrices, thus conserving the most important integer property of the target matrix. Common methodologies 
include penalty function algorithm and thresholding algorithm. This class implements penalty function algorithm.

.. literalinclude:: /code/methods_snippets.py
    :lines: 38-47
         
"""

from mf.models import *
from mf.utils import *
from mf.utils.linalg import *

class Bmf(nmf_std.Nmf_std):
    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param lambda_w: It controls how fast lambda should increase and influences the convergence of the basis matrix (W)
                     to binary values during the update. 
                         #. :param:`lambda_w` < 1 will result in a nonbinary decompositions as the update rule effectively
                            is a conventional NMF update rule. 
                         #. :param:`lambda_w` > 1 give more weight to make the factorization binary with increasing iterations.
                     Default value is 1.1.
    :type lambda_w: `float`
    :param lambda_h: It controls how fast lambda should increase and influences the convergence of the mixture matrix (H)
                     to binary values during the update. 
                         #. :param:`lambda_h` < 1 will result in a nonbinary decompositions as the update rule effectively
                            is a conventional NMF update rule. 
                         #. :param:`lambda_h` > 1 give more weight to make the factorization binary with increasing iterations.
                     Default value is 1.1.
    :type lambda_h: `float`
    """

    def __init__(self, **params):
        self.name = "bmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self.set_params()
        
        self._lambda_w = 1. / self.max_iter if self.max_iter else 1. / 10
        self._lambda_h = self._lambda_w         
        for run in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            pobj = cobj = self.objective()
            iter = 0
            while self.is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                self._adjustment()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
                if self.track_error:
                    self.tracker._track_error(cobj, run)
            if self.callback:
                self.final_obj = cobj
                mffit = mf_fit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker._track_factor(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter 
        self.final_obj = cobj
        mffit = mf_fit.Mf_fit(self)
        return mffit
    
    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param p_obj: Objective function value from previous iteration. 
        :type p_obj: `float`
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True
    
    def set_params(self):
        """Set algorithm specific model options."""
        self.lambda_w = self.options.get('lambda_w', 1.1)
        self.lambda_h = self.options.get('lambda_h', 1.1)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
    
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
        return (sop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)

    def __str__(self):
        return self.name    
        