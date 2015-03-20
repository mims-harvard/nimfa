
"""
###################################
Bmf (``methods.factorization.bmf``)
###################################

**Binary Matrix Factorization (BMF)** [Zhang2007]_.

BMF extends standard NMF to binary matrices. Given a binary target matrix (V),
we want to factorize it into binary basis and mixture matrices, thus conserving
the most important integer property of the target matrix. Common methodologies
include penalty function algorithm and thresholding algorithm. 

BMF can be derived based on variant of Standard NMF, but some problems need to
be resolved:
    
    #. Uniqueness. Solution for basis and mixture matrix is not unique as it is
       always possible to find a diagonal matrix and incorporate it current solution
       to get a new.
    #. Scale. Scale problem arises when discretizing basis and mixture matrix into
       binary matrices. This problem can be resolved by using rescaling proposed in
       Boundedness Theorem in [Zhang2007]_. Therefore, discretization works properly
       because basis and mixture matrix are in the same scales. The factorization
       method is more robust in this way. It has been shown that the percentage of
       nonzero elements in normalized case is lower than in nonnormalized case.
       Without normalization the mixture matrix is often very sparse and the basis
       matrix very dense - much information, given via mixture matrix is lost and
       cannot be compensated with basis matrix.

This method implements penalty function algorithm. The problem of BMF can be
represented in terms of nonlinear programming and then solved by a penalty
function algorithm. The algorithm is described as follows:

    1. Initialize basis, mixture matrix and parameters. 
    2. Normalize basis and mixture using Boundedness Theorem in [Zhang2007]_.
    3. For basis and mixture, alternately solve nonlinear optimization problem
       with the objective function composed of three components: Euclidean distance
       of BMF estimate from target matrix; mixture penalty term and  basis penalty
       term.
    4. Update parameters based on the level of the binarization of the basis and
       mixture matrix.
    
In step 1, basis and mixture matrix can be initialized with common initialization
methods or with the result of the Standard NMF by passing fixed factors to the
factorization model. In step 3, the update rule is derived by taking the longest
step that can maintain the nonnegativity of the basis, mixture matrix during the
iterative process.

.. literalinclude:: /code/snippet_bmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Bmf']


class Bmf(nmf_std.Nmf_std):
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

    :param lambda_w: It controls how fast lambda should increase and influences
       the convergence of the basis matrix (W) to binary values during the update.

             #. ``lambda_w`` < 1 will result in a nonbinary
                decompositions as the update rule effectively is a
                conventional NMF update rule.
             #. ``lambda_w`` > 1 give more weight to make the
                factorization binary with increasing iterations.
       Default value is 1.1.
    :type lambda_w: `float`

    :param lambda_h: It controls how fast lambda should increase and influences
       the convergence of the mixture matrix (H) to binary values during the update.

             #. ``lambda_h`` < 1 will result in a nonbinary
                decompositions as the update rule effectively is a
                conventional NMF update rule.
             #. ``lambda_h`` > 1 give more weight to make the
                factorization binary with increasing iterations.
       Default value is 1.1.
    :type lambda_h: `float`

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
                 lambda_w=1.1, lambda_h=1.1, **options):
        self.name = "bmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._lambda_w = 1. / self.max_iter if self.max_iter else 1. / 10
        self._lambda_h = self._lambda_w
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.normalize()
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
                self._adjustment()
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
        # update mixture matrix
        H1 = dot(self.W.T, self.V) + 3. * \
            self._lambda_h * multiply(self.H, self.H)
        H2 = dot(dot(self.W.T, self.W), self.H) + 2. * \
            self._lambda_h * power(self.H, 3) + self._lambda_h * self.H
        self.H = multiply(self.H, elop(H1, H2, div))
        # update basis matrix,
        W1 = dot(self.V, self.H.T) + 3. * \
            self._lambda_w * multiply(self.W, self.W)
        W2 = dot(self.W, dot(self.H, self.H.T)) + 2. * \
            self._lambda_w * power(self.W, 3) + self._lambda_w * self.W
        self.W = multiply(self.W, elop(W1, W2, div))
        self._lambda_h = self.lambda_h * self._lambda_h
        self._lambda_w = self.lambda_w * self._lambda_w

    def normalize(self):
        """
        Normalize initialized basis and mixture matrix, using Boundedness
        Theorem in [Zhang2007]_. Normalization makes the BMF factorization more
        robust.
        
        Normalization produces basis and mixture matrix with values in [0, 1]. 
        """
        val_w, _ = argmax(self.W, axis=0)
        val_h, _ = argmax(self.H, axis=1)
        if sp.isspmatrix(self.W):
            D_w = sp.spdiags(val_w, 0, self.W.shape[1], self.W.shape[1])
        else:
            D_w = np.diag(val_w)
        if sp.isspmatrix(self.H):
            D_h = sp.spdiags(val_h, 0, self.H.shape[0], self.H.shape[0])
        else:
            D_h = np.diag(val_h)
        self.W = dot(dot(self.W, power(D_w, -0.5)), power(D_h, 0.5))
        self.H = dot(dot(power(D_h, -0.5), power(D_w, 0.5)), self.H)

    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(self.W, self.H)
        return (multiply(R, R)).sum()

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
