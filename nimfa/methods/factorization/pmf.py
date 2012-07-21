
"""
###################################
Pmf (``methods.factorization.pmf``)
###################################

**Probabilistic Nonnegative Matrix Factorization (PMF).** 

PMF interprets target matrix (V) as samples from a multinomial [Laurberg2008]_, [Hansen2008]_ and uses Euclidean distance for 
convergence test. Factorization is guided by an expectation maximization algorithm. 

.. literalinclude:: /code/methods_snippets.py
    :lines: 141-149
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

class Pmf(nmf_std.Nmf_std):
    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param rel_error: In PMF only Euclidean distance cost function is used for convergence test by default. By specifying the value for 
                      minimum relative error, the relative error measure can be used as stopping criteria as well. In this case of 
                      multiple passed criteria, the satisfiability of one terminates the factorization run. Suggested value for
                      :param:`rel_error` is 1e-5.
    :type rel_error: `float`
    """

    def __init__(self, **params):
        self.name = "pmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
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
            self.error_v_n = c_obj
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
                    self.tracker.track_error(run, c_obj)
            self.W = self.v_factor * dot(self.W, self.sqrt_P) 
            self.H = dot(self.sqrt_P, self.H)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(run, W = self.W, H = self.H, final_obj = c_obj, n_iter = iter)
            # if multiple runs are performed, fitted factorization model with the lowest objective function value is retained 
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter 
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))
        
        mffit.fit.tracker = self.tracker
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
        return power(self.V - dot(dot(dot(self.W, self.sqrt_P) * self.v_factor, self.sqrt_P), self.H), 2).sum()
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name