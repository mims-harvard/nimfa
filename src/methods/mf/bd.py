
import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Bd(mstd.Nmf_std):
    """
    Bayesian Decomposition - Bayesian nonnegative matrix factorization Gibbs sampler [16].
    
    In the Bayesian framework knowledge of the distribution of the residuals is stated in terms of likelihood function and
    the parameters in terms of prior densities. In this method normal likelihood and exponential priors are chosen as these 
    are suitable for a wide range of problems and permit an efficient Gibbs sampling procedure. Using Bayes rule, the posterior
    can be maximized to yield an estimate of basis (W) and mixture (H) matrix. However, we are interested in estimating the 
    marginal density of the factors and because the marginals cannot be directly computed by integrating the posterior, an
    MCMC sampling method is used.    
    
    In Gibbs sampling a sequence of samples is drawn from the conditional posterior densities of the model parameters and this
    converges to a sample from the joint posterior. The conditional densities of basis and mixture matrices are proportional 
    to a normal multiplied by an exponential, i.e. rectified normal density. The conditional density of sigma**2 is an inverse 
    Gamma density. The posterior can be approximated by sequentially sampling from these conditional densities. 
    
    Bayesian NMF is concerned with the sampling from the posterior distribution of basis and mixture factors. Algorithm outline
    is: 
        #. Initialize basis and mixture matrix. 
        #. Sample from rectified Gaussian for each column in basis matrix.
        #. Sample from rectified Gaussian for each row in mixture matrix. 
        #. Sample from inverse Gamma for noise variance
        #. Repeat the previous three steps until some convergence criterion is met. 
    
    [16] Schmidt, M.N., Winther, O.,  and Hansen, L.K., (2009). Bayesian Non-negative Matrix Factorization. 
        In Proceedings of ICA. 2009, 540-547.
    """

    def __init__(self, **params):
        mstd.Nmf_std.__init__(self, params)
        self.name = "bd"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
        self.v = multiply(self.V, self.V).sum() / 2.
                
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
        self.tracker = [] if self.options and 'track' in self.options and self.options['track'] and self.n_run > 1 else None
        
    def update(self):
        """Update basis and mixture matrix."""
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()