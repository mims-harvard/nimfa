
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

.. literalinclude:: /code/methods_snippets.py
    :lines: 129-137
       
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *


class Nsnmf(nmf_ns.Nmf_ns):

    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with
    values as keyword arguments.
    
    :param theta: The smoothing parameter. Its value should be 0<=``theta``<=1.
       With ``theta`` 0 the model corresponds to the basic divergence NMF.
       Strong smoothing forces strong sparseness in both the basis and the mixture
       matrices. If not specified, default value ``theta`` of 0.5 is used.
    :type theta: `float`
    """

    def __init__(self, **params):
        self.name = "nsnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_ns.Nmf_ns.__init__(self, params)
        self.set_params()

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.S = sop(
                (1 - self.theta) * sp.spdiags(
                    [1 for _ in range(
                        self.rank)], 0, self.rank, self.rank, 'csr'),
                self.theta / self.rank, add)
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

    def set_params(self):
        """Set algorithm specific model options."""
        self.theta = self.options.get('theta', .5)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track(
        ) if self.track_factor and self.n_run > 1 or self.track_error else None

    def update(self):
        """Update basis and mixture matrix based on modified divergence
        multiplicative update rules."""
        # update mixture matrix H
        W = dot(self.W, self.S)
        H1 = repmat(W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(
            self.H, elop(dot(W.T, elop(self.V, dot(W, self.H), div)), H1, div))
        # update basis matrix W
        H = dot(self.S, self.H)
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