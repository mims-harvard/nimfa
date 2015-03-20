
"""
#####################################
Psmf (``methods.factorization.psmf``)
#####################################

**Probabilistic Sparse Matrix Factorization (PSMF)** [Dueck2005]_, [Dueck2004]_. 

PSMF allows for varying levels of sensor noise in the
data, uncertainty in the hidden prototypes used to explain the data and
uncertainty as to the prototypes selected to explain each data vector stacked in
target matrix (V).

This technique explicitly maximizes a lower bound on the log-likelihood of the
data under a probability model. Found sparse encoding can be used for a variety
of tasks, such as functional prediction, capturing functionally relevant
hidden factors that explain gene expression data and visualization. As this
algorithm computes probabilities rather than making hard decisions, it can be
shown that a higher data log-likelihood is obtained than from the
versions (iterated conditional modes) that make hard decisions [Srebro2001]_.

Given a target matrix (V [n, m]), containing n m-dimensional data points, basis
matrix (factor loading matrix) (W) and mixture matrix (matrix of hidden factors)
(H) are found under a structural sparseness constraint that each row of W
contains at most N (of possible factorization rank number) non-zero entries.
Intuitively, this corresponds to explaining each row vector of V as a linear
combination (weighted by the corresponding row in W) of a small subset
of factors given by rows of H. This framework includes simple clustering by
setting N = 1 and ordinary low-rank approximation N = factorization rank as
special cases.

A probability model presuming Gaussian sensor noise in V (V = WH + noise) and
uniformly distributed factor assignments is constructed. Factorized variational
inference method is used to perform tractable inference on the latent variables
and account for noise and uncertainty. The number of factors, r_g, contributing
to each data point is multinomially distributed such that P(r_g = n) = v_n,
where v is a user specified N-vector. PSMF model estimation using factorized
variational inference has greater computational complexity than basic NMF
methods [Dueck2004]_.

Example of usage of PSMF for identifying gene transcriptional modules from gene
expression data is described in [Li2007]_.

.. literalinclude:: /code/snippet_psmf.py

"""

import numbers

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Psmf']


class Psmf(nmf_std.Nmf_std):
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

    :param prior: The prior on the number of factors explaining each vector. It
       should be a positive row vector. The ``prior`` can be passed as a
       list, formatted as prior = [P(r_g = 1), P(r_g = 2), ... P(r_q = N)] or as a
       scalar N, in which case uniform prior is taken, prior = 1. / (1:N),
       reflecting no knowledge about the distribution and giving equal preference to
       all values of a particular r_g. Default value for :param:`prior` is
       factorization rank.
    :type prior: `list` or `float`

    **Stopping criterion**

    Factorization terminates if any of specified criteria is satisfied.

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
    def __init__(self, V, seed=None, W=None, H=None, H1=None,
                 rank=30, max_iter=30, min_residuals=1e-5, test_conv=None,
                 n_run=1, callback=None, callback_init=None, track_factor=False,
                 track_error=False, prior=20, **options):
        self.name = "psmf"
        self.aseeds = ["none"]
        nmf_std.Nmf_std.__init__(self, vars())
        self.test_conv = 5 if not self.test_conv else self.test_conv
        if isinstance(self.prior, numbers.Real):
            self.prior = np.ones(int(self.prior)) / self.prior
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self.N = len(self.prior)
        sm = sum(self.prior)
        self.prior = np.array([p / sm for p in self.prior])
        self.eps = 1e-5
        if sp.isspmatrix(self.V):
            self.V = self.V.todense()

        for run in range(self.n_run):
            # initialize P and Q distributions
            # internal computation is done with numpy arrays as n-(n >
            # 2)dimensionality is needed
            if sp.isspmatrix(self.V):
                self.W = self.V.__class__(
                    (self.V.shape[0], self.rank), dtype='d')
                self.H = self.V.__class__(
                    (self.rank, self.V.shape[1]), dtype='d')
            else:
                self.W = np.mat(np.zeros((self.V.shape[0], self.rank)))
                self.H = np.mat(np.zeros((self.rank, self.V.shape[1])))
            self.s = np.zeros((self.V.shape[0], self.N), int)
            self.r = np.zeros((self.V.shape[0], 1), int)
            self.psi = np.array(std(self.V, axis=1, ddof=0))
            self.lamb = abs(np.tile(np.sqrt(self.psi), (1, self.rank))
                            * np.random.randn(self.V.shape[0], self.rank))
            self.zeta = np.random.rand(self.rank, self.V.shape[1])
            self.phi = np.random.rand(self.rank, 1)
            self.sigma = np.random.rand(self.V.shape[0], self.rank, self.N)
            self.sigma = self.sigma / np.tile(self.sigma.sum(axis=1).reshape(
                (self.sigma.shape[0], 1, self.sigma.shape[2])), (1, self.rank, 1))
            self.rho = np.tile(self.prior, (self.V.shape[0], 1))
            self._cross_terms()
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
                    run, W=self.W, H=self.H, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def _cross_terms(self):
        """Initialize the major cached parameter."""
        outer_zeta = np.dot(self.zeta, self.zeta.T)
        self.cross_terms = {}
        for n1 in range(self.N):
            for n2 in range(n1 + 1, self.N):
                self.cross_terms[n1, n2] = np.zeros((self.V.shape[0], 1))
                for c in range(self.rank):
                    sigmat = np.tile((self.sigma[:, c, n2] * self.lamb[:, c]).reshape((self.lamb.shape[0], 1)), (1, self.zeta.shape[0]))
                    outer_zetat = np.tile(outer_zeta[c, :], (self.rho.shape[0], 1))
                    self.cross_terms[n1, n2] += (self.sigma[:, :, n1] * self.lamb * 
                                                 sigmat * outer_zetat).sum(axis = 1).reshape(self.rho.shape[0], 1)

    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping
        parameters and objective function value.
        
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
        t_p1 = np.array(multiply(self.V, self.V).sum(axis=1))
        self.psi = - \
            (np.tile(list(range(1, self.N)), (self.V.shape[0], 1)) * self.rho[:, 1:self.N]).sum(
                axis=1) * t_p1[:, 0]
        self.psi = self.psi.reshape((self.V.shape[0], 1))
        temp = np.zeros((self.V.shape[0], self.rank))
        for t in range(self.V.shape[1]):
            temp += (np.tile(self.__arr(self.V[:, t]), (1, self.rank)) - self.lamb * np.tile(
                self.zeta[:, t].T, (self.V.shape[0], 1))) ** 2 + self.lamb ** 2 * np.tile(self.phi.T, (self.V.shape[0], 1))
        for n in range(self.N):
            self.psi += (self.rho[:, n:self.N].sum(axis = 1) * (self.sigma[:, :, n] * temp).sum(axis = 1)).reshape(self.psi.shape)
        for n in range(self.N):
            for nn in range(n + 1, self.N):
                self.psi += (2 * self.rho[:, nn:self.N].sum(
                    axis=1) * self.cross_terms[n, nn].T[0]).reshape((self.V.shape[0], 1))
        self.psi /= self.V.shape[1]
        # heuristic: variances cannot go lower than epsilon
        self.psi = np.maximum(self.psi, self.eps)

    def _update_lamb(self):
        """Compute M-step and update lambda."""
        D = np.zeros((self.rank, 1))
        V = np.zeros((self.V.shape[0], self.rank))
        for t in range(self.V.shape[1]):
            D += self.zeta[:, t].reshape(
                (self.zeta.shape[0], 1)) ** 2 + self.phi
            V += dot(self.V[:, t], self.zeta[:, t].T)
        temp = np.zeros((self.V.shape[0], self.rank))
        for n in range(self.N):
            temp += np.tile((self.rho[:, n:self.N]).sum(axis = 1).reshape((self.rho.shape[0], 1)), (1, self.rank)) * self.sigma[:, :, n]
        V *= temp
        D = np.tile(D.T, (self.V.shape[0], 1)) * temp
        # heuristic: weak Gaussian prior on lambda for ill-conditioning
        # prevention
        D += self.eps
        for g in range(self.V.shape[0]):
            M = np.zeros((self.rank, self.rank))
            for n in range(self.N):
                for nn in range(n + 1, self.N):
                    M += np.dot(self.rho[g, nn:self.N].sum(axis = 0), np.dot(self.sigma[g, :, n].T, self.sigma[g,:, nn]))
            M = (M + M.T) * np.dot(self.zeta, self.zeta.T)
            self.lamb[g, :] = np.dot(V[g,:], np.linalg.inv(M + np.diag(D[g,:])))
        # heuristic:  negative mixing proportions not allowed
        self.lamb[self.lamb < 0] = 0
        self.W = sp.lil_matrix((self.V.shape[0], self.rank))
        for n in range(self.N):
            locs = (self.r >= n).ravel().nonzero()[0]
            if len(locs):
                locs = sub2ind(
                    (self.V.shape[0], self.rank), locs, self.s[locs, n])
                for l in locs:
                    self.W[l % self.V.shape[0], l / self.V.shape[0]] = self.lamb[
                        l % self.V.shape[0], l / self.V.shape[0]]
        self.W = self.W.tocsr()
        self._cross_terms()

    def _update_sigma(self):
        """Compute E-step and update sigma."""
        self.cross_terms = np.zeros((self.V.shape[0], self.rank, self.N))
        for cc in range(self.rank):
            t_c1 = np.tile(self.sigma[:, cc, :].reshape((self.sigma.shape[0], 1, self.sigma.shape[2])), (1, self.rank, 1))
            t_c2 = np.tile(np.dot(self.zeta[cc, :], self.zeta.T), (self.V.shape[0], 1))
            t_c3 = np.tile((self.lamb * np.tile(self.lamb[:, cc].reshape((self.lamb.shape[0], 1)), (1, self.rank)) * t_c2).reshape(
                t_c2.shape[0], t_c2.shape[1], 1), (1, 1, self.N))
            self.cross_terms += t_c1 * t_c3
        self.sigma = np.zeros(self.sigma.shape)
        for t in range(self.V.shape[1]):
            t_s1 = np.tile(self.__arr(self.V[:, t]), (1, self.rank)) - self.lamb * np.tile(
                self.zeta[:, t].T, (self.V.shape[0], 1))
            t_s2 = t_s1 ** 2 + self.lamb ** 2 * \
                np.tile(self.phi.T, (self.V.shape[0], 1))
            self.sigma -= 0.5 * \
                np.tile((t_s2 / np.tile(self.psi, (1, self.rank))).reshape(
                    t_s2.shape[0], t_s2.shape[1], 1), (1, 1, self.N))
        for n in range(self.N):
            for nn in range(self.N):
                if nn != n:
                    t_s1 = (1e-50 + self.rho[:, max(n, nn):self.N]).sum(
                        axis=1) / (1e-50 + self.rho[:, n:self.N]).sum(axis=1)
                    self.sigma[:, :, n] -= np.tile(t_s1.reshape(self.psi.shape) / self.psi, (1, self.rank)) * self.cross_terms[:,:, nn]        
        self.sigma = np.exp(self.sigma - np.tile(np.amax(self.sigma, 1).reshape(
            (self.sigma.shape[0], 1, self.sigma.shape[2])), (1, self.rank, 1)))
        self.sigma /= np.tile(self.sigma.sum(axis=1).reshape(
            (self.sigma.shape[0], 1, self.sigma.shape[2])), (1, self.rank, 1))
        self.cross_terms = self._cross_terms()
        self.s = np.argmax(self.sigma, axis=1)
        self.s = self.s.transpose([0, 1])

    def _update_zeta(self):
        """Compute E-step and update zeta."""
        M = np.zeros((self.rank, self.rank))
        V = np.zeros((self.rank, self.V.shape[1]))
        for cc in range(self.rank):
            for n in range(self.N):
                for nn in range(n + 1, self.N):
                    t_m1 = np.tile(self.rho[:, nn:self.N].sum(axis=1).reshape(
                        (self.psi.shape[0], 1)) / self.psi, (1, self.rank))
                    t_m2 = np.tile((self.lamb[:, cc] * self.sigma[:, cc, nn]).reshape(
                        (self.lamb.shape[0], 1)), (1, self.rank))
                    t_m =  t_m1 * self.lamb * self.sigma[:, :, n] * t_m2
                    M[cc, :] += t_m.sum(axis = 0)
        M += M.T
        temp = np.zeros((self.V.shape[0], self.rank))
        for n in range(self.N):
            temp += np.tile(self.rho[:, n:self.N].sum(axis = 1).reshape((self.rho.shape[0], 1)), (1, self.rank)) * self.sigma[:, :, n]
        M += np.diag(
            (self.lamb ** 2 / np.tile(self.psi, (1, self.rank)) * temp).sum(axis=0))
        for t in range(self.V.shape[1]):
            t_v = np.tile(
                self.__arr(self.V[:, t]) / self.psi, (1, self.rank)) * self.lamb * temp
            V[:, t] = t_v.sum(axis=0)
        self.zeta = np.linalg.solve(M + np.eye(self.rank), V)
        # heuristic: negative expression levels not allowed
        self.zeta[self.zeta < 0] = 0.
        self.H = sp.csr_matrix(self.zeta)

    def _update_phi(self):
        """Compute E-step and update phi."""
        self.phi = np.ones(self.phi.shape)
        for n in range(self.N):
            rho_tmp = np.tile(self.rho[:, n:self.N].sum(axis = 1).reshape(self.rho.shape[0], 1), (1, self.rank))
            t_phi = np.tile(self.psi, (1, self.rank)) * self.sigma[:, :, n] * rho_tmp
            self.phi += (self.lamb ** 2 / (t_phi + np.finfo(t_phi.dtype).eps)).sum(
                axis=0).reshape((self.phi.shape[0], 1))
        self.phi = 1. / self.phi
        # heuristic: variances cannot go lower than epsilon
        self.phi = np.maximum(self.phi, self.eps)

    def _update_rho(self):
        """Compute E-step and update rho."""
        self.rho = - (self.sigma * np.log(1e-50 + self.sigma)).sum(axis=1).reshape(
            self.sigma.shape[0], 1, self.sigma.shape[2]).cumsum(axis=2).transpose([0, 2, 1])
        self.rho = self.rho.reshape((self.rho.shape[0], self.rho.shape[1]))
        temp = np.zeros((self.V.shape[0], self.rank))
        for t in range(self.V.shape[1]):
            t_dot = np.array(
                np.dot(self.__arr(self.V[:, t]).reshape((self.V.shape[0], 1)), self.zeta[:, t].T.reshape((1, self.zeta.shape[0]))))
            temp -= 2 * self.lamb * t_dot + self.lamb ** 2 * \
                np.tile(
                    self.zeta[:, t].T ** 2 + self.phi.T, (self.V.shape[0], 1))
        for n in range(1, self.N):
            self.rho[:, n] -= 0.5 / self.psi[:, 0] * (self.sigma[:, :, 1:n].sum(axis = 2) * temp).sum(axis = 1)
            for n1 in range(n + 1):
                for n2 in range(n1 + 1, n + 1):
                    self.rho[:, n] -= (
                        1. / self.psi * self.cross_terms[n1, n2])[:, 0]
        t_rho = np.exp(
            self.rho - np.tile(np.amax(self.rho, 1).reshape((self.rho.shape[0], 1)), (1, self.N)))
        self.rho = np.tile((self.prior / self.rank) ** list(
            range(1, self.N + 1)), (self.V.shape[0], 1)) * t_rho
        self.rho = self.rho / \
            np.tile(self.rho.sum(axis=1).reshape(
                (self.rho.shape[0], 1)), (1, self.N))
        self.r = np.argmax(self.rho, axis=1)

    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(self.W, self.H)
        return power(R, 2).sum()

    def __arr(self, X):
        """Return dense vector X."""
        if sp.isspmatrix(X):
            return X.toarray()
        else:
            return np.array(X)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
