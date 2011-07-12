from operator import ne

import models.mf_fit as mfit
from utils.linalg import *

class Lsnmf(object):
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
    
    [4] ﻿Lin, C.-J. (2007). Projected gradient methods for nonnegative matrix factorization. Neural computation, 19(10), 2756-79. doi: 10.1162/neco.2007.19.10.2756. 
    """

    def __init__(self):
        self.aname = "lsnmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self, model):
        """
        Return fitted factorization model.
        
        :param model: The underlying model of matrix factorization.
                      If min_residuals of the underlying model is not specified, default value of min_residuals 0.001 is set.  
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
        if not self.min_residuals: self.min_residuals = 0.001
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.gW = dot(self.W, dot(self.H, self.H.T)) - dot(self.V, self.H.T)
            self.gH = dot(dot(self.W.T, self.W), self.H) - dot(self.W.T, self.V)
            self.init_grad = norm(vstack(self.gW, self.gH.T))
            self.epsW = max(0.001, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            cobj = self.objective() 
            iter = 0
            while self._is_satisfied(cobj, iter):
                self.update()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            self.final_obj = cobj
            mffit = mfit.Mf_fit(self)
            if self.callback: self.callback(mffit)
        return mffit
    
    def _is_satisfied(self, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iters and self.max_iters > iter:
            return False
        if iter > 0 and cobj < self.min_residuals * self.init_grad:
            return False
        return True
            
    def update(self):
        """Update basis and mixture matrix."""
        self.W, self.gW, iter = self.subproblem(self.V.T, self.H.T, self.W.T, self.epsW, 1000)
        self.W = self.W.T
        self.gW = self.gW.T
        self.epsW = 0.1 * self.epsW if iter == 1 else self.epsW
        self.H, self.gH, iter = self.subproblem(self.V, self.W, self.H, self.epsH, 1000)
        self.epsH = 0.1 * self.epsH if iter == 1 else self.epsH
    
    def _subproblem(self, V, W, Hinit, epsH, max_iter):
        """
        Optimization procedure for solving subproblem (bound-constrained optimization).
        
        Return output solution, gradient and number of used iterations.
        
        :param V: Constant matrix.
        :type V: sparse or dense matrix
        :param W: Constant matrix.
        :type W: sparse or dense matrix
        :param Hinit: Initial solution to subproblem.
        :type Hinit: sparse or dense matrix
        :param epsH: Tolerance for termination.
        :type epsH: `float`
        :param max_iter: Maximum number of subproblem iterations.
        :type max_iter: `int`
        """
        H = Hinit
        WtV = dot(W.T, V)
        WtW = dot(W.T, W.T)
        # step size
        alpha = 1.
        # the rate of reducing the step size to satisfy the sufficient decrease condition
        # smaller beta more aggressively reduces the step size, but may cause the step size being too small
        beta = 0.1
        
        for iter in xrange(max_iter):
            grad = dot(WtW, H) - WtV
            projgrad = norm(self.__extract(grad, H))
            if projgrad < epsH: 
                break
            # search for step size alpha
            for n_iter in xrange(20):
                Hn = max(H - alpha * grad, 0)
                d = Hn - H
                gradd = multiply(grad, d).sum()
                dQd = multiply(dot(WtW, d), d).sum()
                suff_decr = 0.99 * gradd + sop(0.5 * dQd, 0, ne)
                if n_iter == 1:
                    decr_alpha = not suff_decr
                    Hp = H
                if decr_alpha:
                    if suff_decr:
                        H = Hn
                        break
                    else:
                        alpha *= beta
                else:
                    if not suff_decr or self.__alleq(Hp, Hn):
                        H = Hp
                        break
                    else:
                        alpha /= beta
                        Hp = Hn
        return H, grad, iter
        
    def objective(self):
        """Compute projected gradients norm.""" 
        return norm(vstack([self.__extract(self.gW, self.W), self.__extract(self.gH, self.H)]))
    
    def __alleq(self, X, Y):
        """Check element wise comparison for dense, sparse, mixed matrices."""
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
        """Extract elements for projected gradient norm."""
        if sp.isspmatrix(X):
            R = sp.lil_matrix(X.shape, format = X.format)
            now = 0
            for row in range(X.shape[0]):
                upto = X.indptr[row+1]
                while now < upto:
                    col = X.indices[now]
                    if  X[row, col] < 0 or Y[row, col] > 0: 
                        R[row, col] = X[row, col]
                    now += 1
            return R.tocsr()
        else:
            return X[np.logical_or(X<0, Y>0)]
        
        