from operator import ne

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Lsnmf(mstd.Nmf_std):
    """
    Alternating Nonnegative Least Squares Matrix Factorization Using Projected Gradient (bound constrained optimization)
    method for each subproblem (LSNMF) [4]. It converges faster than the popular multiplicative update approach. 
    
    Algorithm relies on efficiently solving bound constrained subproblems. They are solved using the projected gradient 
    method. Each subproblem contains some (m) independent nonnegative least squares problems. Not solving these separately
    but treating them together is better because of: problems are closely related, sharing the same constant matrices;
    all operations are matrix based, which saves computational time. 
    
    The main task per iteration of the subproblem is to find a step size alpha such that a sufficient decrease condition
    of bound constrained problem is satisfied. In alternating least squares, each subproblem involves an optimization 
    procedure and requires a stopping condition. A common way to check whether current solution is close to a 
    stationary point is the form of the projected gradient [4]. 
    
    [4] Lin, C.-J., (2007). Projected gradient methods for nonnegative matrix factorization. Neural computation, 19(10), 2756-79. doi: 10.1162/neco.2007.19.10.2756. 
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        If :param:`min_residuals` of the underlying model is not specified, default value of :param:`min_residuals` 1e-5 is set.
        In LSNMF :param:`min_residuals` is used as an upper bound of quotient of projected gradients norm and initial gradient
        (initial gradient of basis and mixture matrix).    
        
        The following are algorithm specific model options which can be passed with values as keyword arguments.
        
        :param sub_iter: Maximum number of subproblem iterations. Default value is 1000. 
        :type sub_iter: `int`
        :param inner_sub_iter: Number of inner iterations when solving subproblems. Default value is 20. 
        :type inner_sub_iter: `int`
        :param beta: The rate of reducing the step size to satisfy the sufficient decrease condition when solving subproblems.
                     Smaller beta more aggressively reduces the step size, but may cause the step size being too small. Default
                     value is 0.1.
        :type beta: `float`
        """
        self.name = "lsnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.gW = dot(self.W, dot(self.H, self.H.T)) - dot(self.V, self.H.T)
            self.gH = dot(dot(self.W.T, self.W), self.H) - dot(self.W.T, self.V)
            self.init_grad = norm(vstack(self.gW, self.gH.T), p = 'fro')
            self.epsW = max(1e-3, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            cobj = self.objective() 
            iter = 0
            while self._is_satisfied(cobj, iter):
                self.update()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
                if self.track_error:
                    self.tracker._track_error(self.residuals())
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker._track_factor(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter - 1
        self.final_obj = cobj
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter < iter:
            return False
        if iter > 0 and c_obj < self.min_residuals * self.init_grad:
            return False
        return True
    
    def _set_params(self):
        """Set algorithm specific model options."""
        if not self.min_residuals: self.min_residuals = 1e-5
        self.sub_iter = self.options.get('sub_iter', 1000)
        self.inner_sub_iter = self.options.get('inner_sub_iter', 20)
        self.beta = self.options.get('beta', 0.1)
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mtrack.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
            
    def update(self):
        """Update basis and mixture matrix."""
        self.W, self.gW, iter = self._subproblem(self.V.T, self.H.T, self.W.T, self.epsW)
        self.W = self.W.T
        self.gW = self.gW.T
        self.epsW = 0.1 * self.epsW if iter == 0 else self.epsW
        self.H, self.gH, iter = self._subproblem(self.V, self.W, self.H, self.epsH)
        self.epsH = 0.1 * self.epsH if iter == 0 else self.epsH
    
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
        if sp.isspmatrix(X) or sp.isspmatrix(Y):
            X, Y = Y, X if not sp.isspmatrix(X) and sp.isspmatrix(Y) else X, Y
            now = 0
            for row in range(X.shape[0]):
                upto = X.indptr[row+1]
                while now < upto:
                    col = X.indices[now]
                    if  X[row, col] != Y[row, col]:
                        return False
                    now += 1
            return True
        else:
            return (X == Y).all()
    
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
            R = []
            now = 0
            for row in range(X.shape[0]):
                upto = X.indptr[row+1]
                while now < upto:
                    col = X.indices[now]
                    if  X[row, col] < 0 or Y[row, col] > 0: 
                        R.append(X[row, col])
                    now += 1
            return np.mat(R).T
        else:
            return X[np.logical_or(X<0, Y>0)].flatten().T
        
    def __str__(self):
        return self.name      