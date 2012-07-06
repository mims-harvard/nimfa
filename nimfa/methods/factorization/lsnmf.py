
"""
#######################################
Lsnmf (``methods.factorization.lsnmf``)
#######################################

**Alternating Nonnegative Least Squares Matrix Factorization Using Projected Gradient (bound constrained optimization)
method for each subproblem (LSNMF)** [Lin2007]_. 

It converges faster than the popular multiplicative update approach. 

Algorithm relies on efficiently solving bound constrained subproblems. They are solved using the projected gradient 
method. Each subproblem contains some (m) independent nonnegative least squares problems. Not solving these separately
but treating them together is better because of: problems are closely related, sharing the same constant matrices;
all operations are matrix based, which saves computational time. 

The main task per iteration of the subproblem is to find a step size alpha such that a sufficient decrease condition
of bound constrained problem is satisfied. In alternating least squares, each subproblem involves an optimization 
procedure and requires a stopping condition. A common way to check whether current solution is close to a 
stationary point is the form of the projected gradient [Lin2007]_.   

.. literalinclude:: /code/methods_snippets.py
    :lines: 79-89
      
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

class Lsnmf(nmf_std.Nmf_std):
    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    If :param:`min_residuals` of the underlying model is not specified, default value of :param:`min_residuals` 1e-5 is set.
    In LSNMF :param:`min_residuals` is used as an upper bound of quotient of projected gradients norm and initial gradient
    (initial gradient of basis and mixture matrix). It is a tolerance for a stopping condition. 
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param sub_iter: Maximum number of subproblem iterations. Default value is 10. 
    :type sub_iter: `int`
    :param inner_sub_iter: Number of inner iterations when solving subproblems. Default value is 10. 
    :type inner_sub_iter: `int`
    :param beta: The rate of reducing the step size to satisfy the sufficient decrease condition when solving subproblems.
                 Smaller beta more aggressively reduces the step size, but may cause the step size being too small. Default
                 value is 0.1.
    :type beta: `float`
    """

    def __init__(self, **params):
        self.name = "lsnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        self.set_params()
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        for run in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.gW = dot(self.W, dot(self.H, self.H.T)) - dot(self.V, self.H.T)
            self.gH = dot(dot(self.W.T, self.W), self.H) - dot(self.W.T, self.V)
            self.init_grad = norm(vstack(self.gW, self.gH.T), p = 'fro')
            self.epsW = max(1e-3, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            # iterW and iterH are not parameters, as these values are used only in first objective computation 
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
                c_obj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else c_obj
                if self.track_error:
                    self.tracker.track_error(run, c_obj)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(run, W = self.W, H = self.H, final_obj = c_obj, n_iter = iter)
            # if multiple runs are performed, fitted factorization model with the lowest objective function value is retained 
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter 
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))
        
        mffit.fit.tracker = self.tracker
        return mffit
    
    def is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
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
    
    def set_params(self):
        """Set algorithm specific model options."""
        if not self.min_residuals: self.min_residuals = 1e-5
        self.sub_iter = self.options.get('sub_iter', 10)
        self.inner_sub_iter = self.options.get('inner_sub_iter', 10)
        self.beta = self.options.get('beta', 0.1)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
            
    def update(self):
        """Update basis and mixture matrix."""
        self.W, self.gW, self.iterW = self._subproblem(self.V.T, self.H.T, self.W.T, self.epsW)
        self.W = self.W.T
        self.gW = self.gW.T
        self.epsW = 0.1 * self.epsW if self.iterW == 0 else self.epsW
        self.H, self.gH, self.iterH = self._subproblem(self.V, self.W, self.H, self.epsH)
        self.epsH = 0.1 * self.epsH if self.iterH == 0 else self.epsH
    
    def _subproblem(self, V, W, Hinit, epsH):
        """
        Optimization procedure for solving subproblem (bound-constrained optimization).
        
        Return output solution, gradient and number of used iterations.
        
        :param V: Constant matrix.
        :type V: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param W: Constant matrix.
        :type W: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Hinit: Initial solution to the subproblem.
        :type Hinit: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param epsH: Tolerance for termination.
        :type epsH: `float`
        """
        H = Hinit
        WtV = dot(W.T, V)
        WtW = dot(W.T, W)
        # alpha is step size regulated by beta
        # beta is the rate of reducing the step size to satisfy the sufficient decrease condition
        # smaller beta more aggressively reduces the step size, but may cause the step size alpha being too small
        alpha = 1.
        for iter in xrange(self.sub_iter):
            grad = dot(WtW, H) - WtV
            projgrad = norm(self.__extract(grad, H))
            if projgrad < epsH: 
                break
            # search for step size alpha
            for n_iter in xrange(self.inner_sub_iter):
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
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Second input matrix.
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
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
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Input matrix. 
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
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
            
            idx1 = zip(r1,c1)
            idx2 = zip(r2,c2)
             
            idxf = set(idx1).union(set(idx2))
            rf, cf = zip(*idxf)
            return X[rf,cf].T
        else:
            return X[np.logical_or(X<0, Y>0)].flatten().T
        
    def __str__(self):
        return self.name     
    
    def __repr__(self):
        return self.name 