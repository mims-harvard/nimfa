
"""
#################################
Bd (``methods.factorization.bd``)
#################################

**Bayesian Decomposition (BD) - Bayesian nonnegative matrix factorization Gibbs
sampler** [Schmidt2009]_.

In the Bayesian framework knowledge of the distribution of the residuals is
stated in terms of likelihood function and the parameters in terms of prior
densities. In this method normal likelihood and exponential priors are chosen as
these are suitable for a wide range of problems and permit an efficient Gibbs
sampling procedure. Using Bayes rule, the posterior can be maximized to yield
an estimate of basis (W) and mixture (H) matrix. However, we are interested in
estimating the marginal density of the factors and because the marginals cannot
be directly computed by integrating the posterior, an MCMC sampling method is
used.

In Gibbs sampling a sequence of samples is drawn from the conditional posterior
densities of the model parameters and this converges to a sample from the
joint posterior. The conditional densities of basis and mixture matrices are
proportional to a normal multiplied by an exponential, i. e. rectified normal
density. The conditional density of sigma**2 is an inverse Gamma density. The
posterior can be approximated by sequentially sampling from these conditional
densities.

Bayesian NMF is concerned with the sampling from the posterior distribution of
basis and mixture factors. Algorithm outline
is: 
    #. Initialize basis and mixture matrix. 
    #. Sample from rectified Gaussian for each column in basis matrix.
    #. Sample from rectified Gaussian for each row in mixture matrix. 
    #. Sample from inverse Gamma for noise variance
    #. Repeat the previous three steps until some convergence criterion is met. 
    
The sampling procedure could be used for estimating the marginal likelihood,
which is useful for model selection, i. e. choosing factorization rank.

.. literalinclude:: /code/methods_snippets.py
    :lines: 18-35
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *


class Bd(nmf_std.Nmf_std):

    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    If ``max_iter`` of the underlying model is not specified, default
    value of ``max_iter`` 30 is set. The meaning of ``max_iter`` for
    BD is the number of Gibbs samples to compute. Sequence of Gibbs samples
    converges to a sample from the joint posterior.
    
    The following are algorithm specific model options which can be passed with
    values as keyword arguments.
    
    :param alpha: The prior for basis matrix (W) of proper dimensions. Default
       is zeros matrix prior.
    :type alpha: :class:`scipy.sparse.csr_matrix` or :class:`numpy.matrix`

    :param beta: The prior for mixture matrix (H) of proper dimensions. Default
       is zeros matrix prior.
    :type beta: :class:`scipy.sparse.csr_matrix` or :class:`numpy.matrix`

    :param theta: The prior for ``sigma``. Default is 0.
    :type theta: `float`

    :param k: The prior for ``sigma``. Default is 0.
    :type k: `float`

    :param sigma: Initial value for noise variance (sigma**2). Default is 1.
    :type sigma: `float`  

    :param skip: Number of initial samples to skip. Default is 100.
    :type skip: `int`

    :param stride: Return every ``stride``'th sample. Default is 1.
    :type stride: `int`

    :param n_w: Method does not sample from these columns of basis matrix.
       Column i is not sampled if ``n_w``[i] is True. Default is sampling
       from all columns.
    :type n_w: :class:`numpy.ndarray` or list with shape (factorization rank,
       1) with logical values

    :param n_h: Method does not sample from these rows of mixture matrix. Row
       i is not sampled if ``n_h``[i] is True. Default is sampling from all
       rows.
    :type n_h: :class:`numpy.ndarray` or list with shape (factorization rank, 1)
       with logical values

    :param n_sigma: Method does not sample from ``sigma``. By default
       sampling is done.
    :type n_sigma: `bool`    
    """

    def __init__(self, **params):
        self.name = "bd"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        self.set_params()

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self.v = multiply(self.V, self.V).sum() / 2.

        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
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
                self.update(iter)
                iter += 1
                c_obj = self.objective(
                ) if not self.test_conv or iter % self.test_conv == 0 else c_obj
                if self.track_error:
                    self.tracker.track_error(run, c_obj)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    run, W=self.W, H=self.H, sigma=self.sigma, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on
        stopping parameters and objective function value.
        
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
        return True

    def set_params(self):
        """Set algorithm specific model options."""
        if not self.max_iter:
            self.max_iter = 30
        self.alpha = self.options.get(
            'alpha', sp.csr_matrix((self.V.shape[0], self.rank)))
        if sp.isspmatrix(self.alpha):
            self.alpha = self.alpha.tocsr()
        else:
            self.alpha = np.mat(self.alpha)
        self.beta = self.options.get(
            'beta', sp.csr_matrix((self.rank, self.V.shape[1])))
        if sp.isspmatrix(self.beta):
            self.beta = self.beta.tocsr()
        else:
            self.beta = np.mat(self.beta)
        self.theta = self.options.get('theta', .0)
        self.k = self.options.get('k', .0)
        self.sigma = self.options.get('sigma', 1.)
        self.skip = self.options.get('skip', 100)
        self.stride = self.options.get('stride', 1)
        self.n_w = self.options.get('n_w', np.zeros((self.rank, 1)))
        self.n_h = self.options.get('n_h', np.zeros((self.rank, 1)))
        self.n_sigma = self.options.get('n_sigma', False)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track(
        ) if self.track_factor and self.n_run > 1 or self.track_error else None

    def update(self, iter):
        """Update basis and mixture matrix."""
        for _ in range(self.skip * (iter == 0) + self.stride * (iter > 0)):
            # update basis matrix
            C = dot(self.H, self.H.T)
            D = dot(self.V, self.H.T)
            for n in range(self.rank):
                if not self.n_w[n]:
                    nn = list(range(n)) + list(range(n + 1, self.rank))
                    temp = self._randr(
                        sop(D[:, n] - dot(self.W[:, nn], C[nn, n]), C[
                            n, n] + np.finfo(C.dtype).eps, div),
                        self.sigma / (C[n, n] + np.finfo(C.dtype).eps), self.alpha[:, n])
                    if not sp.isspmatrix(self.W):
                        self.W[:, n] = temp
                    else:
                        for j in range(self.W.shape[0]):
                            self.W[j, n] = temp[j]
            # update sigma
            if self.n_sigma == False:
                scale = 1. / \
                    (self.theta + self.v + multiply(
                        self.W, dot(self.W, C) - 2 * D).sum() / 2.)
                self.sigma = 1. / \
                    np.random.gamma(
                        shape=(self.V.shape[0] * self.V.shape[1]) / 2. + 1. + self.k, scale = scale)
            # update mixture matrix
            E = dot(self.W.T, self.W)
            F = dot(self.W.T, self.V)
            for n in range(self.rank):
                if not self.n_h[n]:
                    nn = list(range(n)) + list(range(n + 1, self.rank))
                    temp = self._randr(
                        sop((F[n, :] - dot(E[n, nn], self.H[nn, :])).T, E[
                            n, n] + np.finfo(E.dtype).eps, div),
                        self.sigma / (E[n, n] + np.finfo(E.dtype).eps), self.beta[n, :].T)
                    if not sp.isspmatrix(self.H):
                        self.H[n, :] = temp.T
                    else:
                        for j in range(self.H.shape[1]):
                            self.H[n, j] = temp[j]

    def _randr(self, m, s, l):
        """Return random number from distribution with density
        p(x)=K*exp(-(x-m)^2/s-l'x), x>=0."""
        # m and l are vectors and s is scalar
        m = m.toarray() if sp.isspmatrix(m) else np.array(m)
        l = l.toarray() if sp.isspmatrix(l) else np.array(l)
        A = (l * s - m) / sqrt(2 * s)
        a = A > 26.
        x = np.zeros(m.shape)
        y = np.random.rand(m.shape[0], m.shape[1])
        x[a] = - np.log(y[a]) / ((l[a] * s - m[a]) / s)
        a = np.array(1 - a, dtype=bool)
        R = erfc(abs(A[a]))
        x[a] = erfcinv(y[a] * R - (A[a] < 0) * (2 * y[a] + R - 2)) * \
            sqrt(2 * s) + m[a] - l[a] * s
        x[np.isnan(x)] = 0
        x[x < 0] = 0
        x[np.isinf(x)] = 0
        return x.real

    def objective(self):
        """Compute squared Frobenius norm of a target matrix and
        its NMF estimate."""
        return power(self.V - dot(self.W, self.H), 2).sum()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
