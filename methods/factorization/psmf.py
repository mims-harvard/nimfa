
import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
import utils.utils as utils
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
    
    [11] Dueck, D., Morris, Q. D., Frey, B. J, (2005). Multi-way clustering of microarray data using probabilistic sparse matrix factorization.
         Bioinformatics 21. Suppl 1, i144-51.
    [12] Dueck, D., Frey, B. J., (2004). Probabilistic Sparse Matrix Factorization Probabilistic Sparse Matrix Factorization. University of
         Toronto Technical Report PSI-2004-23.
    [13] Srebro, N. and Jaakkola, T., (2001). Sparse Matrix Factorization of Gene Expression Data. Unpublished note, MIT Artificial 
         Intelligence Laboratory.
    [15] Li, H., Sun, Y., Zhan, M., (2007). The discovery of transcriptional modules by a two-stage matrix decomposition approach. Bioinformatics, 23(4), 473-479.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        Algorithm specific model option is :param:`prior` which can be passed with value as keyword argument.
        Parameter :param:`prior` is the prior on the number of factors explaining each vector and should be a positive row vector. 
        The :param:`prior` can be passed as a list, formatted as prior = [P(r_g = 1), P(r_g = 2), ... P(r_q = N)] or as a scalar N, in 
        which case uniform prior is taken, prior = 1. /(1:N), reflecting no knowledge about the distribution and giving equal 
        preference to all values of a particular r_g. Default value is prior = factorization rank, e. g. ordinary 
        low-rank approximations is performed. 
        """
        self.name = "psmf"
        self.aseeds = ["none"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
        if not sp.isspmatrix(self.V):
            raise utils.MFError("Target matrix is not in sparse format.")
        self.N = len(self.prior)
        sm = sum(self.prior)
        self.prior = np.array([p / sm for p in self.prior])
        self.eps = 1e-5
        
        for _ in xrange(self.n_run):
            # initialize P and Q distributions
            # internal computation is done with numpy arrays as n-(n > 2)dimensionality is needed 
            self.W, self.H = sp.csr_matrix((self.V.shape[0], self.rank), dtype = 'd'), sp.csr_matrix((self.rank, self.V.shape[1]), dtype = 'd')
            self.s = np.zeros((self.V.shape[0], self.N), int)
            self.r = np.zeros((self.V.shape[0], 1), int)
            self.psi = np.array(std(self.V, axis = 1, ddof = 0))  
            self.lamb = abs(np.tile(np.sqrt(self.psi), (1, self.rank)) * np.random.randn(self.V.shape[0], self.rank)) 
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
                self.tracker.add(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter - 1
        self.final_obj = cobj 
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _cross_terms(self):
        """Initialize the major cached parameter."""
        outer_zeta = np.dot(self.zeta, self.zeta.T)
        self.cross_terms = {}
        for n1 in xrange(self.N):
            for n2 in xrange(n1 + 1, self.N):
                for c in xrange(self.rank):
                    self.cross_terms[n1, n2] = np.zeros((self.V.shape[0], 1)) + sum(self.sigma[:, :, n1] * self.lamb * 
                                            np.tile(self.sigma[:, c, n2] * self.lamb[:, c], (1, self.rank)) * 
                                            np.tile(outer_zeta[c, :], (self.V.shape[0], 1)), axis = 1)
    
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
        self.prior = self.options.get('prior', self.rank)
        try:
            self.prior = [1. / self.prior for _ in xrange(int(round(self.prior)))]
        except TypeError:
            self.prior = self.optios['prior'] 
        self.tracker = mtrack.Mf_track() if self.options.get('track', 0) and self.n_run > 1 else None
        
    def update(self):
        """Update basis and mixture matrix."""
        self._update_rho()
        self._update_phi()
        self._update_zeta()
        self._update_sigma()
        self._update_lamb()
        self._update_psi()
        
    def _update_psi(self):
        """Compute M-step and update psi."""
        self.psi = - sum(np.tile([i for i in xrange(1, self.N)], (self.V.shape[0], 1)) * 
                    self.rho[:, 1:self.N - 1], axis = 1) * sum(multiply(self.V, self.V), axis = 1)
        temp = np.zeros((self.V.shape[0], self.rank))
        for t in xrange(self.V.shape[1]):
            temp += (np.tile(self.V[:, t].toarray(), (1, self.rank)) -  self.lamb * np.tile(self.zeta[:,t].T, (self.V.shape[0], 1)))**2 + self.lamb**2 * np.tile(self.phi.T, (self.V.shape[0], 1))
        for n in xrange(self.N):
            self.psi += sum(self.rho[:, n:self.N - 1], axis = 1) * sum(self.sigma[:, :, n] * temp, axis = 1)
        for n in xrange(self.N):
            for nn in xrange(n + 1, self.N):
                self.psi += 2 * sum(self.rho[:, nn:self.N - 1], axis = 1) * self.cross_terms[n, nn]
        self.psi /= self.V.shape[1]
        # heuristic: variances cannot go lower than epsilon
        self.psi = np.maximum(self.psi, self.eps)
        
    def _update_lamb(self):
        """Compute M-step and update lambda."""
        D = np.zeros((self.rank, 1))
        V = np.zeros((self.V.shape[0], self.rank))
        for t in xrange(self.V.shape[1]):
            D += self.zeta[:, t]**2 + self.phi
            V += dot(self.V[:, t], self.zeta[:, t].T)
        temp = np.zeros((self.V.shape[0], self.rank))
        for n in xrange(self.N): 
            temp += np.tile(sum(self.rho[:, n:self.N - 1], axis = 1), (1, self.rank)) * self.sigma[:, :, n]
        V *= temp
        D = np.tile(D.T, (self.V.shape[0], 1)) * temp
        # heuristic: weak Gaussian prior on lambda for ill-conditioning prevention
        D += self.eps
        for g in xrange(self.V.shape[0]):
            M = np.zeros((self.rank, self.rank))
            for n in xrange(self.N):
                for nn in xrange(n + 1, self.N):
                    M += np.dot(sum(self.rho[g, nn:self.N - 1], axis = 0), np.dot(self.sigma[g, :, n].T, self.sigma[g, :, nn]))
            M = (M + M.T) * np.dot(self.zeta, self.zeta.T)
            self.lamb[g, :] = np.dot(V[g, :], np.linalg.inv(M + np.diag(D[g, :])))
        # heuristic:  negative mixing proportions not allowed
        self.lamb[self.lamb < 0] = 0 
        self.W = sp.csr_matrix((self.V.shape[0], self.rank))
        for n in xrange(self.N):
            locs = (self.r >= n).ravel().nonzero()
            if len(locs):
                locs = sub2ind((self.V.shape[0], self.rank), locs, self.s[locs, n])
                for l in locs:
                    self.W[l % self.V.shape[0], l / self.V.shape[0]] += self.lamb[l % self.V.shape[0], l / self.V.shape[0]]
        self.cross_terms = self._cross_terms()
        
    def _update_sigma(self):
        """Compute E-step and update sigma."""
        self.cross_terms = np.zeros((self.V.shape[0], self.rank, self.N))
        for cc in xrange(self.rank):
            self.cross_terms += np.tile(self.sigma[:, cc, :], (1, self.rank, 1)) * np.tile(self.lamb * np.tile(self.lamb[:, cc], (1, self.rank)) *
                                np.tile(np.dot(self.zeta[cc, :], self.zeta.T), (self.V.shape[0], 1) ), (1, 1, self.N))
        self.sigma = np.zeros(self.sigma.shape)
        for t in xrange(self.V.shape[1]):
            self.sigma -= 0.5 * np.tile(((np.tile(self.V[:, t].toarray(), (1, self.rank)) - self.lamb * 
                          np.tile(self.zeta[:, t].T, (self.V.shape[0], 1)))**2 + self.lamb**2 * np.tile(self.phi.T, (self.V.shape[1], 1))) 
                          / np.tile(self.psi, (1, self.rank)), (1, 1, self.N))
        for n in xrange(self.N):
            for nn in xrange(self.N):
                if nn != n:
                    self.sigma[:, :, n] -= np.tile(sum(1e-50 + self.rho[:, np.max(n, nn):self.N], axis = 1) / 
                                           sum(1e-50 + self.rho[:, n, self.N], axis = 1) / self.psi, (1, self.rank)) * self.cross_terms[:, :, nn]
        self.sigma = np.exp(self.sigma - np.tile(np.amax(self.sigma, 1), (1, self.rank)))
        self.sigma /= np.tile(self.sigma.sum(axis = 1), (1, self.rank))
        self.cross_terms = self._cross_terms()
        self.s = np.argmax(self.sigma, axis = 1)
        self.s = self.s.transpose([0, 2, 1])
        
    def _update_zeta(self):
        """Compute E-step and update zeta."""
        M = np.zeros((self.rank, self.rank))
        V = np.zeros((self.rank, self.V.shape[1]))
        for cc in xrange(self.rank): 
            for n in xrange(self.N):
                for nn in xrange(n + 1, self.N):
                    M[cc, :] += sum(np.tile(self.rho[:, nn:self.N - 1].sum(axis = 1) / self.psi, (1, self.rank)) * self.lamb * 
                                self.sigma[:, :, n] * np.tile(self.lamb[:, cc] * self.sigma[:, cc, nn], (1, self.rank)), axis = 0)
        M += M.T
        temp = np.zeros((self.V.shape[0], self.rank))
        for n in xrange(self.N):
            temp += np.tile(self.rho[:, n:self.N - 1].sum(axis = 1), (1, self.rank)) * self.sigma[:, :, n]
        M += np.diag(sum(self.lamb**2 / np.tile(self.psi, (1, self.rank)) * temp, axis = 0))
        for t in xrange(self.V.shape[1]):
            V[:, t] = sum(np.tile(self.V[:, t].toarray() / self.psi, (1, self.rank)) * self.lamb * self.temp, axis = 0).T
        self.zeta = np.linalg.solve(M + np.eye(self.rank), V)
        # heuristic: negative expression levels not allowed
        self.zeta[self.zeta < 0] = 0
        self.H = sp.csr_matrix(self.zeta)
        
    def _update_phi(self):
        """Compute E-step and update phi."""
        self.phi = np.ones(self.phi.shape)
        for n in xrange(self.N): 
            self.phi += self.lamb**2 / (np.tile(self.psi, (1, self.rank)) * self.sigma[:, :, n] * np.tile(self.rho[:, n:self.N - 1].sum(axis = 1), (1, self.C))).sum(axis = 0).T
        self.phi = 1. / self.phi
        # heuristic: variances cannot go lower than epsilon
        self.phi = np.maximum(self.phi, self.eps)
        
    def _update_rho(self):
        """Compute E-step and update rho."""
        self.rho = - (self.sigma * np.log(1e-50 + self.sigma)).sum(axis = 1).cumsum(axis = 2).transpose([0, 2, 1])
        temp = np.zeros((self.V.shape[0], self.rank))
        for t in xrange(self.V.shape[1]):
            temp -= 2 * self.lamb * np.dot(self.V[:, t].toarray(), self.zeta[:, t].T) + self.lamb**2 * np.tile(self.zeta[:, t].T**2 + self.phi.T, (self.V.shape[0], 1))
        for n in xrange(1, self.N):
            self.rho[:, n] -= 0.5 / self.psi * (self.sigma[:, :, 1:n].sum(axis = 2) * temp).sum(axis = 1)
            for n1 in xrange(n):
                for n2 in xrange(n1 + 1, n):
                    self.rho -= 1. / self.psi * self.cross_terms[n1, n2]
        self.rho = np.tile((self.prior / self.C)**np.array([i for i in xrange(1, self.N)]), (self.V.shape[0], 1)) * np.exp(self.rho - np.tile(np.amax(self.rho, 1), (1, self.N)))
        self.rho = self.rho / np.tile(self.rho.sum(axis = 1), (1, self.N))
        self.r = np.argmax(self.rho, axis = 1)
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (sop(self.V - dot(self.W, self.H), 2, pow)).sum()
       
    def __str__(self):
        return self.name 