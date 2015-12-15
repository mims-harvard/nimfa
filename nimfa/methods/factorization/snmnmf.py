
"""
#########################################
Snmnmf (``methods.factorization.snmnmf``)
#########################################

**Sparse Network-Regularized Multiple Nonnegative Matrix Factorization
(SNMNMF)** [Zhang2011]_.

It is semi-supervised learning method with constraints (e. g. in comodule
identification, any variables linked in A or B, are more likely placed in the
same comodule) to improve relevance and narrow down the search space.

The advantage of this method is the integration of multiple matrices for
multiple types of variables (standard NMF methods can be applied to a target
matrix containing just one type of variable) together with prior knowledge
(e. g. network representing relationship among variables). 

The objective function in [Zhang2011]_ has three components:
    #. first component models miRNA and gene expression profiles;
    #. second component models gene-gene network interactions;
    #. third component models predicted miRNA-gene interactions.
 
The inputs for the SNMNMF are:
    #. two sets of expression profiles (represented by the matrices V and V1 of
       shape s x m, s x n, respectively) for miRNA and genes measured on the same
       set of samples;
    #. (PRIOR KNOWLEDGE) a gene-gene interaction network (represented by the
       matrix A of shape n x n), including protein-protein interactions and
       DNA-protein interactions; the network is presented in the form of the
       adjacency matrix of gene network;
    #. (PRIOR KNOWLEDGE) a list of predicted miRNA-gene regulatory interactions
       (represented by the matrix B of shape m x n) based on sequence data; the
       network is presented in the form of the adjacency matrix of a bipartite
       miRNA-gene network. Network regularized constraints are used to enforce
       "must-link" constraints and to ensure that genes with known interactions
       have similar coefficient profiles.
       
Gene and miRNA expression matrices are simultaneously factored into a common
basis matrix (W) and two coefficients matrices (H and H1). Additional knowledge
is incorporated into this framework with network regularized constraints.
Because of the imposed sparsity constraints easily interpretable solution is
obtained. In [Zhang2011]_ decomposed matrix components are used to provide
information about miRNA-gene regulatory comodules. They identified the comodules
based on shared components (a column in basis matrix W) with significant
association values in the corresponding rows of coefficients matrices, H1 and H2.

In SNMNMF a strategy suggested by Kim and Park (2007) is adopted to make the
coefficient matrices sparse.

.. note:: In [Zhang2011]_ ``H1`` and ``H2`` notation corresponds to the ``H``
and ``H1`` here, respectively.

.. literalinclude:: /code/snippet_snmnmf.py

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

__all__ = ['Snmnmf']


class Snmnmf(nmf_mm.Nmf_mm):
    """
    :param V: The target matrix to estimate.
    :type V: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix` or tuple of instances of
       the latter classes.

    :param V1: The target matrix to estimate. Used by algorithms that consider
       more than one target matrix.
    :type V1: Instance of the :class:`scipy.sparse` sparse matrices types,
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

    :param A: Adjacency matrix of gene-gene interaction network (dimension:
       V1.shape[1] x V1.shape[1]).
    :type A: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
       or :class:`numpy.matrix`

    :param B: Adjacency matrix of a bipartite miRNA-gene network, predicted
       miRNA-target interactions (dimension: V.shape[1] x V1.shape[1]).
    :type B: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or
       :class:`numpy.matrix`

    :param gamma: Limit the growth of the basis matrix (W). Default is 0.01.
    :type gamma: `float`

    :param gamma_1: Encourage sparsity of the mixture (coefficient) matrices (H
       and H1). Default is 0.01.
    :type gamma_1: `float`

    :param lamb: Weight for the must-link constraints defined in ``A``.
       Default is 0.01.
    :type lamb: `float`

    :param lamb_1: Weight for the must-link constraints define in ``B``.
       Default is 0.01.
    :type lamb_1: `float`

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
    def __init__(self, V, V1, seed=None, W=None, H=None, H1=None,
                 rank=30, max_iter=30, min_residuals=1e-5, test_conv=None,
                 n_run=1, callback=None, callback_init=None, track_factor=False,
                 track_error=False, A=None, B=None, gamma=0.01, gamma_1=0.01,
                 lamb=0.01, lamb_1=0.01, **options):
        self.name = "snmnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_mm.Nmf_mm.__init__(self, vars())
        if self.A is None:
            self.A = sp.csr_matrix((self.V1.shape[1], self.V1.shape[1]))
        self.A = self.A.tocsr() if sp.isspmatrix(self.A) else np.mat(self.A)
        if self.B is None:
            self.B = sp.csr_matrix((self.V.shape[1], self.V1.shape[1]))
        self.B = self.B.tocsr() if sp.isspmatrix(self.B) else np.mat(self.B)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        if self.V.shape[0] != self.V1.shape[0]:
            raise utils.MFError(
                "Input matrices should have the same number of rows.")

        for run in range(self.n_run):
            self.options.update({'idx': 0})
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.options.update({'idx': 1})
            _, self.H1 = self.seed.initialize(self.V1, self.rank, self.options)
            self.options.pop('idx')
            p_obj = c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            self.err_avg = 1
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
                    run, W=self.W, H=self.H, H1=self.H1.copy(),
                    final_obj=c_obj, n_iter=iter)
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
        if self.err_avg < 1e-5:
            return False
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
        # update basis matrix
        temp_w1 = dot(self.V, self.H.T) + dot(self.V1, self.H1.T)
        temp_w2 = dot(self.W, dot(self.H, self.H.T) + dot(
            self.H1, self.H1.T)) + self.gamma / 2. * self.W
        self.W = multiply(self.W, elop(temp_w1, temp_w2, div))
        # update mixture matrices
        # update H1
        temp = sop(dot(self.W.T, self.W), s=self.gamma_1, op=add)
        temp_h1 = dot(self.W.T, self.V) + \
            self.lamb_1 / 2. * dot(self.H1, self.B.T)
        HH1 = multiply(self.H, elop(temp_h1, dot(temp, self.H), div))
        temp_h3 = dot(self.W.T, self.V1) + self.lamb * dot(
            self.H1, self.A) + self.lamb_1 / 2. * dot(self.H, self.B)
        temp_h4 = dot(temp, self.H1)
        self.H1 = multiply(self.H1, elop(temp_h3, temp_h4, div))
        # update H
        self.H = HH1

    def objective(self):
        """Compute three component objective function as defined in [Zhang2011]_."""
        err_avg1 = abs(self.V - dot(self.W, self.H)).mean() / self.V.mean()
        err_avg2 = abs(self.V1 - dot(self.W, self.H1)).mean() / self.V1.mean()
        self.err_avg = err_avg1 + err_avg2
        R1 = self.V - dot(self.W, self.H)
        eucl1 = (multiply(R1, R1)).sum()
        R2 = self.V1 - dot(self.W, self.H1)
        eucl2 = (multiply(R2, R2)).sum()
        tr1 = trace(dot(dot(self.H1, self.A), self.H1.T))
        tr2 = trace(dot(dot(self.H, self.B), self.H1.T))
        s1 = multiply(self.W, self.W).sum()
        s2 = multiply(self.H, self.H).sum() + multiply(self.H1, self.H1).sum()
        reg = - self.lamb * tr1 - self.lamb_1 * tr2 + self.gamma * s1 + self.gamma_1 * s2
        return eucl1 + eucl2  + reg

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
