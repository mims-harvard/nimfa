
"""
###################################
Nmf (``methods.factorization.nmf``)
###################################

**Standard Nonnegative Matrix Factorization (NMF)** [Lee2001]_, [Lee1999]_.

Based on Kullback-Leibler divergence, it uses simple multiplicative updates
[Lee2001]_, [Lee1999]_, enhanced to avoid numerical underflow [Brunet2004]_.
Based on Euclidean distance, it uses simple multiplicative updates [Lee2001]_.
Different objective functions can be used, namely Euclidean distance, divergence
or connectivity matrix convergence.

Together with a novel model selection mechanism, NMF is an efficient method for
identification of distinct molecular patterns and provides a powerful method for
class discovery. It appears to have higher resolution such as HC or SOM and to
be less sensitive to a priori selection of genes. Rather than separating gene
clusters based on distance computation, NMF detects context-dependent patterns
of gene expression in complex biological systems.

Besides usages in bioinformatics NMF can be applied to text analysis,
image processing, multiway clustering, environmetrics etc.

.. literalinclude:: /code/snippet_nmf_fro.py

.. literalinclude:: /code/snippet_nmf_div.py

.. literalinclude:: /code/snippet_nmf_conn.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Nmf']


class Nmf(nmf_std.Nmf_std):
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

    :param update: Type of update equations used in factorization. When specifying
       model parameter ``update`` can be assigned to:

           #. 'euclidean' for classic Euclidean distance update
              equations,
           #. 'divergence' for divergence update equations.
       By default Euclidean update equations are used.
    :type update: `str`

    :param objective: Type of objective function used in factorization. When
       specifying model parameter :param:`objective` can be assigned to:

            #. 'fro' for standard Frobenius distance cost function,
            #. 'div' for divergence of target matrix from NMF
               estimate cost function (KL),
            #. 'conn' for measuring the number of consecutive
               iterations in which the connectivity matrix has not
               changed.
       By default the standard Frobenius distance cost function is used.
    :type objective: `str`

    :param conn_change: Stopping criteria used only if for :param:`objective`
       function connectivity matrix measure is selected. It specifies the minimum
       required of consecutive iterations in which the connectivity matrix has not
       changed. Default value is 30.
    :type conn_change: `int`

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
                 update='euclidean', objective='fro', conn_change=30, **options):
        self.name = "nmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, vars())
        if 'conn' not in self.objective:
            self.conn_change = None
        self._conn_change = 0
        self.update = getattr(self, self.update)
        self.objective = getattr(self, self.objective)
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
                if not self.test_conv or iter % self.test_conv == 0:
                    c_obj = self.objective()
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
        if self.conn_change is not None:
            return self.__is_satisfied(p_obj, c_obj, iter)
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True

    def __is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria if change of
        connectivity matrices is used for objective function.
        
        Return logical value denoting factorization continuation.   
        
        :param p_obj: Objective function value from previous iteration. 
        :type p_obj: `float`
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        self._conn_change = 0 if c_obj == 1 else self._conn_change + 1
        if self._conn_change >= self.conn_change:
            return False
        return True

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.H, np.finfo(self.H.dtype).eps)
        self.W = max(self.W, np.finfo(self.W.dtype).eps)

    def euclidean(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        self.H = multiply(
            self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        self.W = multiply(
            self.W, elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div))

    def divergence(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(
            self.H, elop(dot(self.W.T, elop(self.V, dot(self.W, self.H), div)), H1, div))
        W1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(
            self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), W1, div))

    def fro(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(self.W, self.H)
        return multiply(R, R).sum()

    def div(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(self.W, self.H)
        return (multiply(self.V, sop(elop(self.V, Va, div), op=np.log)) - self.V + Va).sum()

    def conn(self):
        """
        Compute connectivity matrix and compare it to connectivity matrix
        from previous iteration.

        Return logical value denoting whether connectivity matrix has changed
        from previous iteration.
        """
        _, idx = argmax(self.H, axis=0)
        mat1 = repmat(idx, self.V.shape[1], 1)
        mat2 = repmat(idx.T, 1, self.V.shape[1])
        cons = elop(mat1, mat2, eq)
        if not hasattr(self, 'consold'):
            self.cons = cons
            self.consold = np.mat(np.logical_not(cons))
        else:
            self.consold = self.cons
            self.cons = cons
        conn_change = elop(self.cons, self.consold, ne).sum()
        return conn_change > 0

    def __str__(self):
        return '%s - update: %s obj: %s' % (self.name, self.update.__name__, self.objective.__name__)

    def __repr__(self):
        return self.name
