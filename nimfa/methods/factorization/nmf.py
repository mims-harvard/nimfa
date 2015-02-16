
"""
###################################
Nmf (``methods.factorization.nmf``)
###################################

**Standard Nonnegative Matrix Factorization (NMF)** [Lee2001]_, [Lee1999]. 

Based on Kullback-Leibler divergence, it uses simple multiplicative updates
[Lee2001]_, [Lee1999], enhanced to avoid numerical underflow [Brunet2004]_.
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

.. literalinclude:: /code/methods_snippets.py
    :lines: 92-101
    
.. literalinclude:: /code/methods_snippets.py
    :lines: 104-113
    
.. literalinclude:: /code/methods_snippets.py
    :lines: 116-126
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *


class Nmf(nmf_std.Nmf_std):

    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with
    values as keyword arguments.
    
    :param update: Type of update equations used in factorization. When specifying
       model parameter ``update`` can be assigned to:

           #. 'Euclidean' for classic Euclidean distance update
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
    """

    def __init__(self, **params):
        self.name = "nmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        self.set_params()

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
        if self.conn_change != None:
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

    def set_params(self):
        """Set algorithm specific model options."""
        self.update = getattr(
            self, self.options.get('update', 'euclidean') + '_update')
        self.objective = getattr(
            self, self.options.get('objective', 'fro') + '_objective')
        self.conn_change = self.options.get(
            'conn_change', 30) if 'conn' in self.objective.__name__ else None
        self._conn_change = 0
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track(
        ) if self.track_factor and self.n_run > 1 or self.track_error else None

    def euclidean_update(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        self.H = multiply(
            self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        self.W = multiply(
            self.W, elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div))

    def divergence_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(
            self.H, elop(dot(self.W.T, elop(self.V, dot(self.W, self.H), div)), H1, div))
        W1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(
            self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), W1, div))

    def fro_objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(self.W, self.H)
        return multiply(R, R).sum()

    def div_objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(self.W, self.H)
        return (multiply(self.V, sop(elop(self.V, Va, div), op=np.log)) - self.V + Va).sum()

    def conn_objective(self):
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
        return self.name + " - update: " + self.options.get('update', 'euclidean') + " - obj: " + self.options.get('objective', 'fro')

    def __repr__(self):
        return self.name