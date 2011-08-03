from operator import div

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Icm(mstd.Nmf_std):
    """
    Iterated Conditional Modes nonnegative matrix factorization (ICM) [16]. 
    
    Iterated conditional modes algorithm is a deterministic algorithm for obtaining the configuration that maximizes the 
    joint probability of a Markov random field. This is done iteratively by maximizing the probability of each variable 
    conditioned on the rest.
    
    Most NMF algorithms can be seen as computing a maximum likelihood or maximum a posteriori (MAP) estimate of the 
    nonnegative factor matrices under some assumptions on the distribution of the data and factors. ICM algorithm computes
    the MAP estimate. In this approach, iterations over the parameters of the model set each parameter equal to the conditional
    mode and after a number of iterations the algorithm converges to a local maximum of the joint posterior density. This is a
    block coordinate ascent algorithm with the benefit that the optimum is computed for each block of parameters in each 
    iteration. 
    
    ICM has low computational cost per iteration as the modes of conditional densities have closed form expressions.   
    
    In [16] ICM is compared to the popular Lee and Seung's multiplicative update algorithm and fast Newton algorithm on image
    feature extraction test. ICM converges much faster than multiplicative update algorithm and with approximately the same
    rate per iteration as fast Newton algorithm. All three algorithms have approximately the same computational cost per
    iteration. 
    
    [16] Schmidt, M.N., Winther, O.,  and Hansen, L.K., (2009). Bayesian Non-negative Matrix Factorization. 
        In Proceedings of ICA. 2009, 540-547. 
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        The following are algorithm specific model options which can be passed with values as keyword arguments.
        
        :param iiter: Number of inner iterations. Default is 20. 
        :type iiter: `int`
        :param alpha: The prior for basis matrix (W) of proper dimensions. Default is zeros matrix prior.
        :type alpha: :class:`scipy.sparse.csr_matrix` or :class:`numpy.matrix`
        :param beta: The prior for mixture matrix (H) of proper dimensions. Default is zeros matrix prior.
        :type beta: :class:`scipy.sparse.csr_matrix` or :class:`numpy.matrix`
        :param theta: The prior for :param:`sigma`. Default is 0.
        :type theta: `float`
        :param k: The prior for :param:`sigma`. Default is 0. 
        :type k: `float`
        :param sigma: Initial value for noise variance (sigma**2). Default is 1. 
        :type sigma: `float`       
        """
        self.name = "icm"
        self.aseeds = ["random", "fixed", "nndsvd"]
        mstd.Nmf_std.__init__(self, params)
        
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
                self.tracker.add(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter
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
    
    def _set_params(self):
        self.iiter = self.options.get('iiter', 20)
        self.alpha = self.options.get('alpha', sp.csr_matrix((self.V.shape[0], self.rank)))
        self.beta = self.options.get('beta', sp.csr_matrix((self.rank, self.V.shape[1])))
        self.theta = self.options.get('theta', .0)
        self.k = self.options.get('k', .0)
        self.sigma = self.options.get('sigma', 1.) 
        self.tracker = mtrack.Mf_track() if self.options.get('track', 0) and self.n_run > 1 else None
        
    def update(self):
        """Update basis and mixture matrix."""
        # update basis matrix
        C = dot(self.H, self.H.T)
        D = dot(self.V, self.B.T)
        for _ in xrange(self.iiter):
            for n in xrange(self.rank):
                nn = list(xrange(n - 1)) + list(xrange(n, self.rank))
                temp = max(sop(D[:, n] - dot(self.W[:, nn], C[nn, n]) - self.sigma * self.alpha[:, n], C[n, n], div), 0)
                for i in xrange(self.W.shape[0]):
                    self.W[i, n] = temp[i, 0]
        # update sigma
        self.sigma = (self.theta + self.v + multiply(self.W, dot(self.W, C) - 2 * D).sum() / 2.) / (self.V.shape[0] * self.V.shape[1] / 2. + self.k + 1.)
        # update mixture matrix
        E = dot(self.W.T, self.W)
        F = dot(self.W.T, self.V)
        for _ in xrange(self.iiter):
            for n in xrange(self.rank):
                nn = list(xrange(n - 1)) + list(xrange(n, self.rank))
                temp = max(sop(F[n, :] - dot(E[n, nn], self.H[nn, :]) - self.sigma * self.beta[n, :], E[n, n], div), 0)
                for i in xrange(self.H.shape[1]):
                    self.H[n, i] = temp[0, i]
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
  
    def __str__(self):
        return self.name  
    