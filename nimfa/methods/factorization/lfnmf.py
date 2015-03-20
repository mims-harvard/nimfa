
"""
#######################################
Lfnmf (``methods.factorization.lfnmf``)
#######################################

**Fisher Nonnegative Matrix Factorization for learning Local features (LFNMF)**
[Wang2004]_.

LFNMF is based on nonnegative matrix factorization (NMF), which allows only
additive combinations of nonnegative basis components. The NMF bases are
spatially global, whereas local bases would be preferred. Li [Li2001]_ proposed
local nonnegative matrix factorization (LNFM) to achieve a localized NMF
representation by adding three constraints to enforce spatial locality:
minimize the number of basis components required to represent target matrix;
minimize redundancy between different bases by making different bases as
orthogonal as possible; maximize the total activity on each component, i. e. the
total squared projection coefficients summed over all training images.
However, LNMF does not encode discrimination information for a classification
problem.

LFNMF can produce both additive and spatially localized basis components as LNMF
and it also encodes characteristics of Fisher linear discriminant analysis (FLDA).
The main idea of LFNMF is to add Fisher constraint to the original NMF. Because
the columns of the mixture matrix (H) have a one-to-one correspondence with the
columns of the target matrix (V), between class scatter of H is maximized and
within class scatter of H is minimized.

Example usages are pattern recognition problems in classification, feature
generation and extraction for diagnostic classification purposes, face
recognition etc.

.. literalinclude:: /code/snippet_lfnmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Lfnmf']


class Lfnmf(nmf_std.Nmf_std):
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

    :param alpha: Weight used to minimize within class
       scatter and maximize between class scatter of the encoding mixture matrix.
       The objective function is the constrained divergence, which is the standard
       Lee's divergence rule with added terms ``alpha`` * S_w - ``alpha`` * S_h,
       where S_w and S_h are within class and between class
       scatter, respectively. It should be nonnegative. Default value is 0.01.
    :type alpha: `float`

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
                 callback_init=None, track_factor=False,
                 track_error=False, alpha=0.01, **options):
        self.name = "lfnmf"
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
            self.Sw, self.Sb = np.mat(
                np.zeros((1, 1))), np.mat(np.zeros((1, 1)))
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
        _, idxH = argmax(self.H, axis=0)
        c2m, avgs = self._encoding(idxH)
        C = len(c2m)
        ksi = 1.
        # update mixture matrix H
        for k in range(self.H.shape[0]):
            for l in range(self.H.shape[1]):
                n_r = len(c2m[idxH[0, l]])
                u_c = avgs[idxH[0, l]][k, 0]
                t_1 = (2 * u_c - 1.) / (4 * ksi)
                t_2 = (1. - 2 * u_c) ** 2 + 8 * ksi * self.H[k, l] * sum(self.W[i, k] * self.V[i, l] /
                      (dot(self.W[i, :], self.H[:, l])[0, 0] + 1e-5) for i in range(self.W.shape[0]))
                self.H[k, l] = t_1 + sqrt(t_2) / (4 * ksi)
        # update basis matrix W
        for i in range(self.W.shape[0]):
            for k in range(self.W.shape[1]):
                w_1 = sum(self.H[k, j] * self.V[i, j] / (dot(self.W[i, :], self.H[:, j])[0, 0] + 1e-5)
                          for j in range(self.V.shape[0]))
                self.W[i, k] = self.W[i, k] * w_1 / self.H[k, :].sum()
        W2 = repmat(self.W.sum(axis=0), self.V.shape[0], 1)
        self.W = elop(self.W, W2, div)
        # update within class scatter and between class
        self.Sw = sum(sum(dot(self.H[:, c2m[i][j]] - avgs[i], (self.H[:, c2m[i][j]] - avgs[i]).T)
                          for j in range(len(c2m[i]))) for i in c2m)
        avgs_t = np.mat(np.zeros((self.rank, 1)))
        for k in avgs:
            avgs_t += avgs[k]
        avgs_t /= len(avgs)
        self.Sb = sum(dot(avgs[i] - avgs_t, (avgs[i] - avgs_t).T) for i in c2m)

    def _encoding(self, idxH):
        """Compute class membership and mean class value of encoding (mixture) matrix H."""
        c2m = {}
        avgs = {}
        for i in range(idxH.shape[1]):
            # group columns of encoding matrix H by class membership
            c2m.setdefault(idxH[0, i], [])
            c2m[idxH[0, i]].append(i)
            # compute mean value of class idx in encoding matrix H
            avgs.setdefault(idxH[0, i], np.mat(np.zeros((self.rank, 1))))
            avgs[idxH[0, i]] += self.H[:, i]
        for k in avgs:
            avgs[k] /= len(c2m[k])
        return c2m, avgs

    def objective(self):
        """
        Compute constrained divergence of target matrix from its NMF estimate
        with additional factors of between class scatter and within class
        scatter of the mixture matrix (H).
        """
        Va = dot(self.W, self.H)
        return (multiply(self.V, elop(self.V, Va, np.log)) - self.V + Va).sum() + self.alpha * np.trace(self.Sw) - self.alpha * np.trace(self.Sb)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
