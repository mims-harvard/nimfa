
"""
###################################
Pmf (``methods.factorization.pmf``)
###################################

**Probabilistic Nonnegative Matrix Factorization (PMF).** 

PMF interprets target matrix (V) as samples from a multinomial [Laurberg2008]_,
[Hansen2008]_ and uses Euclidean distance for convergence test. Factorization
is guided by an expectation maximization algorithm.

.. literalinclude:: /code/snippet_pmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Pmf']


class Pmf(nmf_std.Nmf_std):
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

    :param rel_error: In PMF only Euclidean distance cost function is used for
       convergence test by default. By specifying the value for minimum relative
       error, the relative error measure can be used as stopping criteria as well.
       In this case of multiple passed criteria, the satisfiability of one terminates
       the factorization run. Suggested value for ``rel_error`` is 1e-5.
    :type rel_error: `float`

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
    def __init__(self, V, seed=None, W=None, H=None, rank=30, max_iter=30,
                 min_residuals=1e-5, test_conv=None, n_run=1, callback=None,
                 callback_init=None, track_factor=False, track_error=False,
                 rel_error=False, **options):
        self.name = "pmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.W = elop(
                self.W, repmat(self.W.sum(axis=0), self.V.shape[0], 1), div)
            self.H = elop(
                self.H, repmat(self.H.sum(axis=1), 1, self.V.shape[1]), div)
            self.v_factor = self.V.sum()
            self.V_n = sop(self.V.copy(), self.v_factor, div)
            self.P = np.diag([1. / self.rank for _ in range(self.rank)])
            self.sqrt_P = sop(self.P, s=None, op=np.sqrt)
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
                c_obj = self.objective(
                ) if not self.test_conv or iter % self.test_conv == 0 else c_obj
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
        if self.rel_error and self.error_v_n < self.rel_error:
            return False
        return True

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)

    def update(self):
        """Update basis and mixture matrix. It is expectation maximization
        algorithm. """
        # E step
        Qnorm = dot(dot(self.W, self.P), self.H)
        for k in range(self.rank):
            # E-step
            Q = elop(self.P[k, k] * dot(self.W[:, k], self.H[k, :]), sop(
                Qnorm, np.finfo(Qnorm.dtype).eps, add), div)
            V_nQ = multiply(self.V_n, Q)
            # M-step
            dum = V_nQ.sum(axis=1)
            s_dum = dum.sum()
            for i in range(self.W.shape[0]):
                self.W[i, k] = dum[i, 0] / s_dum
            dum = V_nQ.sum(axis=0)
            s_dum = dum.sum()
            for i in range(self.H.shape[1]):
                self.H[k, i] = dum[0, i] / s_dum

    def objective(self):
        """Compute Euclidean distance cost function."""
        # relative error
        self.error_v_n = abs(
            self.V_n - dot(self.W, self.H)).mean() / self.V_n.mean()
        # Euclidean distance
        return power(self.V - dot(dot(dot(self.W, self.sqrt_P) * self.v_factor, self.sqrt_P), self.H), 2).sum()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
