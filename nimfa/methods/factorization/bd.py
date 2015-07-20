
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

.. literalinclude:: /code/snippet_bd.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Bd']


class Bd(nmf_std.Nmf_std):
    """
    :param V: The target matrix to estimate.
    :type V: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix` or tuple of instances of
       the latter classes.

    :param seed: Specify method to seed the computation of a factorization. If
       specified :param:`W` and :param:`H` seeding must be None. If neither seeding
       method or initial fixed factorization is specified, random initialization is
       used.
    :type seed: `str` naming the method or :class:`methods.seeding.nndsvd.Nndsvd`
       or None

    :param W: Specify initial factorization of basis matrix W. Default is None.
       When specified, :param:`seed` must be None.
    :type W: :class:`scipy.sparse` or :class:`numpy.ndarray` or
       :class:`numpy.matrix` or None

    :param H: Specify initial factorization of mixture matrix H. Default is None.
       When specified, :param:`seed` must be None.
    :type H: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix`, tuple of instances of the
       latter classes or None

    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`

    :param n_run: It specifies the number of runs of the algorithm. Default is
       1. If multiple runs are performed, fitted factorization model with the
       lowest objective function value is retained.
    :type n_run: `int`

    :param callback: Pass a callback function that is called after each run when
       performing multiple runs. This is useful if one wants to save summary
       measures or process the result before it gets discarded. The callback
       function is called with only one argument :class:`models.mf_fit.Mf_fit` that
       contains the fitted model. Default is None.
    :type callback: `function`

    :param callback_init: Pass a callback function that is called after each
       initialization of the matrix factors. In case of multiple runs the function
       is called before each run (more precisely after initialization and before
       the factorization of each run). In case of single run, the passed callback
       function is called after the only initialization of the matrix factors.
       This is useful if one wants to obtain the initialized matrix factors for
       further analysis or additional info about initialized factorization model.
       The callback function is called with only one argument
       :class:`models.mf_fit.Mf_fit` that (among others) contains also initialized
       matrix factors. Default is None.
    :type callback_init: `function`

    :param track_factor: When :param:`track_factor` is specified, the fitted
        factorization model is tracked during multiple runs of the algorithm. This
        option is taken into account only when multiple runs are executed
        (:param:`n_run` > 1). From each run of the factorization all matrix factors
        are retained, which can be very space consuming. If space is the problem
        setting the callback function with :param:`callback` is advised which is
        executed after each run. Tracking is useful for performing some quality or
        performance measures (e.g. cophenetic correlation, consensus matrix,
        dispersion). By default fitted model is not tracked.
    :type track_factor: `bool`

    :param track_error: Tracking the residuals error. Only the residuals from
        each iteration of the factorization are retained. Error tracking is not
        space consuming. By default residuals are not tracked and only the final
        residuals are saved. It can be used for plotting the trajectory of the
        residuals.
    :type track_error: `bool`

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

    **Stopping criterion**

    Factorization terminates if any of the specified criteria is satisfied.

    :param max_iter: Maximum number of factorization iterations. Note that the
       number of iterations depends on the speed of method convergence. Default
       is 30.
    :type max_iter: `int`

    :param min_residuals: Minimal required improvement of the residuals from the
       previous iteration. They are computed between the target matrix and its MF
       estimate using the objective function associated to the MF algorithm.
       Default is None.
    :type min_residuals: `float`

    :param test_conv: It indicates how often convergence test is done. By
       default convergence is tested each iteration.
    :type test_conv: `int`
    """
    def __init__(self, V, seed=None, W=None, H=None, rank=30, max_iter=30,
                 min_residuals=1e-5, test_conv=None, n_run=1, callback=None,
                 callback_init=None, track_factor=False, track_error=False,
                 alpha=None, beta=None, theta=0., k=0., sigma=1., skip=100,
                 stride=1, n_w=None, n_h=None, n_sigma=False,
                 **options):
        self.name = "bd"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        if self.alpha is None:
            self.alpha = sp.csr_matrix((self.V.shape[0], self.rank))
        self.alpha = self.alpha.tocsr() if sp.isspmatrix(self.alpha) else np.mat(self.alpha)
        if self.beta is None:
            self.beta = sp.csr_matrix((self.rank, self.V.shape[1]))
        self.beta = self.beta.tocsr() if sp.isspmatrix(self.beta) else np.mat(self.beta)
        if self.n_w is None:
            self.n_w = np.zeros((self.rank, 1))
        if self.n_h is None:
            self.n_h = np.zeros((self.rank, 1))
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

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
