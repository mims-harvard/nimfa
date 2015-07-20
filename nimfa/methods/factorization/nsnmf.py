
"""
#######################################
Nsnmf (``methods.factorization.nsnmf``)
#######################################

**Nonsmooth Nonnegative Matrix Factorization (NSNMF)** [Montano2006]_. 

NSNMF aims at finding localized, part-based representations of nonnegative
multivariate data items. Generally this method produces a set of basis and
encoding vectors representing not only the original data but also extracting
highly localized patterns. Because of the multiplicative nature of the
standard model, sparseness in one of the factors almost certainly forces
nonsparseness (or smoothness) in the other in order to compensate for the final
product to reproduce the data as best as possible. With the modified standard
model in NSNMF global sparseness is achieved.

In the new model the target matrix is estimated as the product V = WSH, where
V, W and H are the same as in the original NMF model. The positive symmetric
square matrix S is a smoothing matrix defined as
S = (1 - theta)I + (theta/rank)11', where I is an identity matrix, 1 is a vector
of ones, rank is factorization rank and theta is a smoothing parameter
(0<=theta<=1).

The interpretation of S as a smoothing matrix can be explained as follows: Let X
be a positive, nonzero, vector. Consider the transformed vector Y = SX. As
theta --> 1, the vector Y tends to the constant vector with all elements almost
equal to the average of the elements of X. This is the smoothest possible vector
in the sense of nonsparseness because all entries are equal to the same nonzero
value. The parameter theta controls the extent of smoothness of the matrix
operator S. Due to the multiplicative nature of the model, strong smoothing in
S forces strong sparseness in both the basis and the encoding vectors. Therefore,
the parameter theta controls the sparseness of the model.

.. literalinclude:: /code/snippet_nmf_nsnmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Nsnmf']


class Nsnmf(nmf_ns.Nmf_ns):
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

    :param theta: The smoothing parameter. Its value should be 0<=``theta``<=1.
       With ``theta`` 0 the model corresponds to the basic divergence NMF.
       Strong smoothing forces strong sparseness in both the basis and the mixture
       matrices. If not specified, default value ``theta`` of 0.5 is used.
    :type theta: `float`

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
                 theta=0.5, **options):
        self.name = "nsnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_ns.Nmf_ns.__init__(self, vars())
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
            self.S = np.diag(np.ones(self.rank) - self.theta) + self.theta / self.rank
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
        """Update basis and mixture matrix based on modified divergence
        multiplicative update rules."""
        # update mixture matrix H
        W = dot(self.W, self.S)
        H1 = repmat(W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(
            self.H, elop(dot(W.T, elop(self.V, dot(W, self.H), div)), H1, div))
        H = dot(self.S, self.H)
        # update basis matrix W
        W1 = repmat(H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(
            self.W, elop(dot(elop(self.V, dot(self.W, H), div), H.T), W1, div))
        # normalize basis matrix W
        W2 = repmat(self.W.sum(0), self.V.shape[0], 1)
        self.W = elop(self.W, W2, div)

    def objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(dot(self.W, self.S), self.H)
        return (multiply(self.V, sop(elop(self.V, Va, div), op=np.log)) - self.V + Va).sum()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
