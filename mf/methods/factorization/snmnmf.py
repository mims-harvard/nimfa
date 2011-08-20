
from mf.models import *
from mf.utils import *
from mf.utils.linalg import *

class Snmnmf(nmf_mm.Nmf_mm):
    """
    
    
    [18] Zhang, S. et. al., (2011). A novel computational framework for simultaneous integration of multiple types of 
         genomic data to identify microRNA-gene regulatory modules. Bioinformatics 2011, 27(13), i401-i409. 
         doi:10.1093/bioinformatics/btr206.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_run`.
        
        The following are algorithm specific model options which can be passed with values as keyword arguments.
        
          
        """
        self.name = "snmnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_mm.Nmf_mm.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
                
        for run in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            pobj = cobj = self.objective()
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update(iter)
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
                if self.track_error:
                    self.tracker._track_error(cobj, run)
            if self.callback:
                self.final_obj = cobj
                mffit = mf_fit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker._track_factor(W = self.W.copy(), H = self.H.copy(), sigma = self.sigma)
        
        self.n_iter = iter 
        self.final_obj = cobj
        mffit = mf_fit.Mf_fit(self)
        return mffit
        
    def _is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
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
        if self.min_residuals and iter > 0 and c_obj - p_obj <= self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True
    
    def _set_params(self):
        """Set algorithm specific model options."""
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
        
    def update(self, iter):
        """Update basis and mixture matrix."""
                    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
    
    def __str__(self):
        return self.name