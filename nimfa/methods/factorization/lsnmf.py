
"""
#######################################
Lsnmf (``methods.factorization.lsnmf``)
#######################################

**Alternating Nonnegative Least Squares Matrix Factorization Using Projected
Gradient (bound constrained optimization) method for each subproblem
(LSNMF)** [Lin2007]_.

It converges faster than the popular multiplicative update approach. 

Algorithm relies on efficiently solving bound constrained subproblems. They are
solved using the projected gradient method. Each subproblem contains some (m)
independent nonnegative least squares problems. Not solving these separately but
treating them together is better because of: problems are closely related,
sharing the same constant matrices; all operations are matrix based, which
saves computational time.

The main task per iteration of the subproblem is to find a step size alpha such
that a sufficient decrease condition of bound constrained problem is satisfied.
In alternating least squares, each subproblem involves an optimization procedure
and requires a stopping condition. A common way to check whether current solution
is close to a stationary point is the form of the projected gradient [Lin2007]_.

.. literalinclude:: /code/snippet_lsnmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Lsnmf']


class Lsnmf(nmf_std.Nmf_std):
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

    :param sub_iter: Maximum number of subproblem iterations. Default value is 10.
    :type sub_iter: `int`

    :param inner_sub_iter: Number of inner iterations when solving subproblems.
       Default value is 10.
    :type inner_sub_iter: `int`

    :param beta: The rate of reducing the step size to satisfy the sufficient
       decrease condition when solving subproblems. Smaller beta more aggressively
       reduces the step size, but may cause the step size being too small. Default
       value is 0.1.
    :type beta: `float`

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
                 track_error=False, sub_iter=10, inner_sub_iter=10, beta=0.1, **options):
        self.name = "lsnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        self.min_residuals = 1e-5 if not self.min_residuals else self.min_residuals
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
            self.gW = dot(self.W, dot(self.H, self.H.T)) - dot(
                self.V, self.H.T)
            self.gH = dot(dot(self.W.T, self.W), self.H) - dot(
                self.W.T, self.V)
            self.init_grad = norm(vstack(self.gW, self.gH.T), p='fro')
            self.epsW = max(1e-3, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            # iterW and iterH are not parameters, as these values are used only
            # in first objective computation
            self.iterW = 10
            self.iterH = 10
            c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(c_obj, iter):
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

    def is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping
        parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param c_obj: Current objective function value. 
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if iter > 0 and c_obj < self.min_residuals * self.init_grad:
            return False
        if self.iterW == 0 and self.iterH == 0 and self.epsW + self.epsH < self.min_residuals * self.init_grad:
            # There was no move in this iteration
            return False
        return True

    def update(self):
        """Update basis and mixture matrix."""
        self.W, self.gW, self.iterW = self._subproblem(
            self.V.T, self.H.T, self.W.T, self.epsW)
        self.W = self.W.T
        self.gW = self.gW.T
        self.epsW = 0.1 * self.epsW if self.iterW == 0 else self.epsW
        self.H, self.gH, self.iterH = self._subproblem(
            self.V, self.W, self.H, self.epsH)
        self.epsH = 0.1 * self.epsH if self.iterH == 0 else self.epsH

    def _subproblem(self, V, W, Hinit, epsH):
        """
        Optimization procedure for solving subproblem (bound-constrained
        optimization).
        
        Return output solution, gradient and number of used iterations.
        
        :param V: Constant matrix.
        :type V: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
           dia or :class:`numpy.matrix`

        :param W: Constant matrix.
        :type W: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
           dia or :class:`numpy.matrix`

        :param Hinit: Initial solution to the subproblem.
        :type Hinit: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok,
           lil, dia or :class:`numpy.matrix`

        :param epsH: Tolerance for termination.
        :type epsH: `float`
        """
        H = Hinit
        WtV = dot(W.T, V)
        WtW = dot(W.T, W)
        # alpha is step size regulated by beta
        # beta is the rate of reducing the step size to satisfy the sufficient
        # decrease condition smaller beta more aggressively reduces the step
        # size, but may cause the step size alpha being too small
        alpha = 1.
        for iter in range(self.sub_iter):
            grad = dot(WtW, H) - WtV
            projgrad = norm(self.__extract(grad, H))
            if projgrad < epsH:
                break
            # search for step size alpha
            for n_iter in range(self.inner_sub_iter):
                Hn = max(H - alpha * grad, 0)
                d = Hn - H
                gradd = multiply(grad, d).sum()
                dQd = multiply(dot(WtW, d), d).sum()
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0
                if n_iter == 0:
                    decr_alpha = not suff_decr
                    Hp = H
                if decr_alpha:
                    if suff_decr:
                        H = Hn
                        break
                    else:
                        alpha *= self.beta
                else:
                    if not suff_decr or self.__alleq(Hp, Hn):
                        H = Hp
                        break
                    else:
                        alpha /= self.beta
                        Hp = Hn
        return H, grad, iter

    def objective(self):
        """Compute projected gradients norm."""
        return norm(vstack([self.__extract(self.gW, self.W), self.__extract(self.gH, self.H)]))

    def __alleq(self, X, Y):
        """
        Check element wise comparison for dense, sparse, mixed matrices.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
           dia or :class:`numpy.matrix`

        :param Y: Second input matrix.
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
           dia or :class:`numpy.matrix`
        """
        if sp.isspmatrix(X) and sp.isspmatrix(Y):
            X = X.tocsr()
            Y = Y.tocsr()
            if not np.all(X.data == Y.data):
                return False
            r1, c1 = X.nonzero()
            r2, c2 = Y.nonzero()
            if not np.all(r1 == r2) or not np.all(c1 == c2):
                return False
            else:
                return True
        else:
            return np.all(X == Y)

    def __extract(self, X, Y):
        """
        Extract elements for projected gradient norm.
        
        :param X: Gradient matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok,
           lil, dia or :class:`numpy.matrix`

        :param Y: Input matrix. 
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
           dia or :class:`numpy.matrix`
        """
        if sp.isspmatrix(X):
            X = X.tocsr()
            r1, c1 = X.nonzero()
            if r1.size != 0:
                xt = X[r1, c1] < 0
                xt = np.array(xt)
                xt = xt[0, :] if xt.shape[0] == 1 else xt[:, 0]
                r1 = r1[xt]
                c1 = c1[xt]

            Y = Y.tocsr()
            r2, c2 = Y.nonzero()
            if r2.size != 0:
                yt = Y[r2, c2] > 0
                yt = np.array(yt)
                yt = yt[0, :] if yt.shape[0] == 1 else yt[:, 0]
                r2 = r2[yt]
                c2 = c2[yt]

            idx1 = list(zip(r1, c1))
            idx2 = list(zip(r2, c2))

            idxf = set(idx1).union(set(idx2))
            rf, cf = list(zip(*idxf))
            return X[rf, cf].T
        else:
            return X[np.logical_or(X < 0, Y > 0)].flatten().T

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
