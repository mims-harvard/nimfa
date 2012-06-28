
"""
#######################################
Pmfcc (``methods.factorization.pmfcc``)
#######################################

**Penalized Matrix Factorization for Constrained Clustering (PMFCC)**. PMFCC implements factorization approach proposed in [FWang2008]_. 
Intra-type information is represented as constraints to guide the factorization process. The constraints are of two types: (i) must-link:
two data points belong to the same class, (ii) cannot-link: two data points cannot belong to the same class.

PMFCC solves the following problem. Given a target matrix V = [v_1, v_2, ..., v_n], it produces W = [f_1, f_2, ... f_rank], containing
cluster centers and matrix H of data point cluster membership values.    

Cost function includes centroid distortions and any associated constraint violations. Compared to the traditional NMF cost function, the only 
difference is the inclusion of the penalty term.  

.. literalinclude:: /code/methods_snippets.py
    :lines: 192-200
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

class Pmfcc(smf.Smf):
    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param theta: Constraint matrix (dimension: V.shape[1] x X.shape[1]). It contains known must-link (negative) and cannot-link 
                  (positive) constraints.
    :type theta: `numpy.matrix`
    """

    def __init__(self, **params):
        self.name = "pmfcc"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        smf.Smf.__init__(self, params)
        self.set_params()
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """     
        for run in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.W = elop(self.W, repmat(self.W.sum(axis = 0), self.V.shape[0], 1), div)
            self.H = elop(self.H, repmat(self.H.sum(axis = 1), 1, self.V.shape[1]), div)
            self.v_factor = self.V.sum()
            self.V_n = sop(self.V.copy(), self.v_factor, div)
            self.P = sp.spdiags([1. / self.rank for _ in xrange(self.rank)], 0, self.rank, self.rank, 'csr')
            self.sqrt_P = sop(self.P, s = None, op = sqrt) 
            p_obj = c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(p_obj, c_obj, iter):
                p_obj = c_obj if not self.test_conv or iter % self.test_conv == 0 else p_obj
                self.update()
                self._adjustment()
                iter += 1
                c_obj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else c_obj
                if self.track_error:
                    self.tracker.track_error(c_obj, run)
            self.W = self.v_factor * dot(self.W, self.sqrt_P) 
            self.H = dot(self.sqrt_P, self.H)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(W = self.W.copy(), H = self.H.copy(), final_obj = c_obj, n_iter = iter)
            # if multiple runs are performed, fitted factorization model with the lowest objective function value is retained 
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter 
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))
        
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
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        if self.rel_error and self.error_v_n < self.rel_error:
            return False
        return True
    
    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)
        
    def set_params(self):
        """Set algorithm specific model options."""
        self.rel_error = self.options.get('rel_error', False)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
        
    def update(self):
        """Update basis and mixture matrix. It is expectation maximization algorithm. """
        # E step
        Qnorm = dot(dot(self.W, self.P), self.H)
        for k in xrange(self.rank):
            # E-step
            Q = elop(self.P[k,k] * dot(self.W[:, k], self.H[k, :]), sop(Qnorm, np.finfo(Qnorm.dtype).eps, add), div)
            V_nQ = multiply(self.V_n, Q)
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
        self.error_v_n =  abs(self.V_n - dot(self.W, self.H)).mean() / self.V_n.mean()
        # Euclidean distance
        return (sop(self.V - dot(dot(dot(self.W, self.sqrt_P) * self.v_factor, self.sqrt_P), self.H), 2, pow)).sum()
        
    def __str__(self):
        return self.name 
    
    def __repr__(self):
        return self.name