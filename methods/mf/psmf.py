from math import sqrt
from operator import div

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Psmf(mstd.Nmf_std):
    """
    Probabilistic Sparse Matrix Factorization (PSMF) [11], [12]. PSMF allows for varying levels of sensor noise in the
    data, uncertainty in the hidden prototypes used to explain the data and uncertainty as to the prototypes selected
    to explain each data vector stacked in target matrix (V). 
    
    This technique explicitly maximizes a lower bound on the log-likelihood of the data under a probability model. Found
    sparse encoding can be used for a variety of tasks, such as functional prediction, capturing functionally relevant
    hidden factors that explain gene expression data and visualization. As this algorithm computes probabilities 
    rather than making hard decisions, it can be shown that a higher data log-likelihood is obtained than from the 
    versions (iterated conditional modes) that make hard decisions [13].
    
    Given a target matrix (V [n, m]), containing n m-dimensional data points, basis matrix (factor loading matrix) (W) 
    and mixture matrix (matrix of hidden factors) (H) are found under a structural sparseness constraint that each row 
    of W contains at most N (of possible factorization rank number) non-zero entries. Intuitively, this corresponds to 
    explaining each row vector of V as a linear combination (weighted by the corresponding row in W) of a small subset 
    of factors given by rows of H. This framework includes simple clustering by setting N = 1 and ordinary low-rank 
    approximation N = factorization rank as special cases. 
    
    A probability model presuming Gaussian sensor noise in V (V = WH + noise) and uniformly distributed factor 
    assignments is constructed. Factorized variational inference method is used to perform tractable inference on the 
    latent variables and account for noise and uncertainty. The number of factors, r_g, contributing to each data point is
    multinomially distributed such that P(r_g = n) = v_n, where v is a user specified N-vector. PSMF model estimation using  
    factorized variational inference has greater computational complexity than basic NMF methods [12]. 
    
    Example of usage of PSMF for identifying gene transcriptional modules from gene expression data is described in [15]. 
    
    [11] ﻿Dueck, D., Morris, Q. D., Frey, B. J, (2005). Multi-way clustering of microarray data using probabilistic sparse matrix factorization.
         Bioinformatics 21. Suppl 1, i144-51.
    [12] ﻿Dueck, D., Frey, B. J., (2004). Probabilistic Sparse Matrix Factorization Probabilistic Sparse Matrix Factorization. University of
         Toronto Technical Report PSI-2004-23.
    [13] Srebro, N. and Jaakkola, T., (2001). Sparse Matrix Factorization of Gene Expression Data. Unpublished note, MIT Artificial 
         Intelligence Laboratory.
    [15] ﻿Li, H., Sun, Y., Zhan, M., (2007). The discovery of transcriptional modules by a two-stage matrix decomposition approach. Bioinformatics, 23(4), 473-479. 
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        Algorithm specific model option is 'prior' which can be passed with value as keyword argument.
        Parameter prior is the prior on the number of factors explaining each vector and should be a positive row vector. The 
        prior can be passed as a list, formatted as prior = [P(r_g = 1), P(r_g = 2), ... P(r_q = N)] or as a scalar N, in 
        which case uniform prior is taken, prior = 1. /(1:N), reflecting no knowledge about the distribution and giving equal 
        preference to all values of a particular r_g. Default value is prior = factorization rank, e. g. ordinary 
        low-rank approximations is performed. 
        """
        mstd.Nmf_std.__init__(self, params)
        self.name = "psmf"
        self.aseeds = ["none"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
        self.N = len(self.prior)
        sm = sum(self.prior)
        self.prior = [p / sm for p in self.prior]
        
        for _ in xrange(self.n_run):
            # initialize P and Q distributions
            # internal computation is done with numpy array as n-(n > 2)dimensionality is needed 
            self.W, self.H = sp.csr_matrix((self.V.shape[0], self.rank), dtype = 'd'), sp.csr_matrix((self.rank, self.V.shape[1]), dtype = 'd')
            self.s = np.zeros((self.V.shape[0], self.N), dtype = 'd')
            self.r = np.zeros((self.V.shape[0], 1), dtype = 'd')
            self.psi = np.array(std(self.V, axis = 1, ddof = 0)) 
            self.lamb = abs(np.tile(sqrt(self.psi), (1, self.rank)) * np.random.randn(self.V.shape[0], self.rank)) 
            self.zeta = np.random.rand(self.rank, self.V.shape[1])
            self.phi = np.random.rand(self.rank, 1)
            self.sigma = np.random.rand(self.V.shape[0], self.rank, self.N)
            self.sigma = self.sigma / np.tile(self.sigma.sum(axis = 1), (1, self.rank, 1))
            self.rho = np.tile(self.prior, (self.V.shape[0], 1))
            self._cross_terms()
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
    
    def _cross_terms(self):
        """Initialize the major cached parameter."""
        outer_zeta = dot(self.zeta, self.zeta.T)
        self.cross_terms = {}
        for n1 in xrange(self.N):
            for n2 in xrange(n1 + 1, self.N):
                for c in xrange(self.rank):
                    self.cross_terms[n1, n2] = np.zeros((self.V.shape[0], 1)) + sum(self.Q[:, :, 1] * self.lamb * 
                                            np.tile(self.sigma[:, c, n2] * self.lamb[:, c], (1, self.rank)) * 
                                            np.tile(outer_zeta[c, :], (self.V.shape[0], 1)), axis = 1)
    
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
        self.prior = self.options['prior'] if self.options and 'prior' in self.options else self.rank
        try:
            self.prior = [1. / self.prior for _ in xrange(int(round(self.prior)))]
        except TypeError:
            self.prior = self.optios['prior'] 
        self.tracker = [] if self.options and 'track' in self.options and self.options['track'] and self.n_run > 1 else None
        
    def update(self):
        """Update basis and mixture matrix."""
        pass
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
        