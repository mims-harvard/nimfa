
"""
#######################################
Pmfcc (``methods.factorization.pmfcc``)
#######################################

**Penalized Matrix Factorization for Constrained Clustering
(PMFCC)** [FWang2008]_.

PMFCC is used for semi-supervised co-clustering. Intra-type information is
represented as constraints to guide the factorization process. The constraints
are of two types: (i) must-link: two data points belong to the same class,
(ii) cannot-link: two data points cannot belong to the same class.

PMFCC solves the following problem. Given a target matrix
V = [v_1, v_2, ..., v_n], it produces W = [f_1, f_2, ... f_rank], containing
cluster centers and matrix H of data point cluster membership values.    

Cost function includes centroid distortions and any associated constraint
violations. Compared to the traditional NMF cost function, the only difference
is the inclusion of the penalty term.

.. literalinclude:: /code/snippet_pmfcc.py

"""
import operator

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Pmfcc']


class Pmfcc(smf.Smf):
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

    :param Theta: Constraint matrix (dimension: V.shape[1] x X.shape[1]). It
       contains known must-link (negative) and cannot-link (positive) constraints.
    :type Theta: `numpy.matrix`

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
                 Theta=None, **options):
        self.name = "pmfcc"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        smf.Smf.__init__(self, vars())
        if not self.Theta:
            self.Theta = np.mat(np.zeros((self.V.shape[1], self.V.shape[1])))
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._Theta_p = multiply(self.Theta, sop(self.Theta, 0, operator.gt))
        self._Theta_n = multiply(self.Theta, sop(self.Theta, 0, operator.lt)*(-1))

        for run in range(self.n_run):
            # [FWang2008]_; H = G.T, W = F (Table 2)
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
        self.W = dot(self.V, dot(self.H.T, inv_svd(dot(self.H, self.H.T))))

        FtF = dot(self.W.T, self.W)
        XtF = dot(self.V.T, self.W)
        tmp1 = sop(FtF, 0, ge)
        tmp2 = tmp1.todense() - 1 if sp.isspmatrix(tmp1) else tmp1 - 1
        FtF_p = multiply(FtF, tmp1)
        FtF_n = multiply(FtF, tmp2)
        tmp1 = sop(XtF, 0, ge)
        tmp2 = tmp1.todense() - 1 if sp.isspmatrix(tmp1) else tmp1 - 1
        XtF_p = multiply(XtF, tmp1)
        XtF_n = multiply(XtF, tmp2)

        Theta_n_G = dot(self._Theta_n, self.H.T)
        Theta_p_G = dot(self._Theta_p, self.H.T)

        GFtF_p = dot(self.H.T, FtF_p)
        GFtF_n = dot(self.H.T, FtF_n)

        enum = XtF_p + GFtF_n + Theta_n_G
        denom = XtF_n + GFtF_p + Theta_p_G

        denom = denom.todense() + np.finfo(float).eps if sp.isspmatrix(
            denom) else denom + np.finfo(float).eps
        Ht = multiply(
            self.H.T, sop(elop(enum, denom, div), s=None, op=np.sqrt))
        self.H = Ht.T

    def objective(self):
        """Compute Frobenius distance cost function with penalization term."""
        return power(self.V - dot(self.W, self.H), 2).sum() + \
               trace(dot(self.H, dot(self.Theta, self.H.T)))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
