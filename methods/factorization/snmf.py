from operator import div, ne, ge, le
from math import sqrt

import models.nmf_std as mstd
import utils.utils as utils
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Snmf(mstd.Nmf_std):
    """
    Sparse Nonnegative Matrix Factorization (SNMF) based on alternating nonnegativity constrained least squares [5].
    
    In order to enforce sparseness on basis or mixture matrix, SNMF can be used, namely two formulations: SNMF/L for 
    sparse W (sparseness is imposed on the left factor) and SNMF/R for sparse H (sparseness imposed on the right factor).
    These formulations utilize L1-norm minimization. Each subproblem is solved by a fast nonnegativity constrained
    least squares (FCNNLS) algorithm (van Benthem and Keenan, 2004) that is improved upon the active set based NLS method. 
    
    SNMF/R contains two subproblems for two-block minimization scheme. The objective function is coercive on the 
    feasible set. It can be shown (Grippo and Sciandrome, 2000) that two-block minimization process is convergent, 
    every accumulation point is a critical point of the corresponding problem. Similarly, the algorithm SNMF/L converges
    to a stationary point. 
   
    [5] Kim H., Park H., (2007). Sparse Non-negative Matrix Factorizations via Alternating Non-negativity-constrained Least Squares 
        for Microarray Data Analysis, Bioinformatics.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        The parameter :param:`min_residuals` of the underlying model is used as KKT convergence test and should have 
        positive value. If not specified, value 1e-4 is used. 
        
        The following are algorithm specific model options which can be passed with values as keyword arguments.
        
        :param version: Specifiy version of the SNMF algorithm. it has two accepting values, 'r' and 'l' for SNMF/R and 
                        SNMF/L, respectively. Default choice is SNMF/R.
        :type version: `str`
        :param eta: Used for suppressing Frobenius norm on the basis matrix (W). Default value is maximum value of the target 
                    matrix (V). If :param:`eta` is negative, maximum value of target matrix is used for it. 
        :type eta: `float`
        :param beta: It controls sparseness. Larger :param:`beta` generates higher sparseness on H. Too large :param:`beta` 
                     is not recommended. It should have positive value. Default value is 0.01.
        :type beta: `float`
        :param i_conv: Part of the biclustering convergence test. It decides convergence if row clusters and column clusters have 
                       not changed for :param:`i_conv` convergence tests. It should have nonnegative value.
                       Default value is 10.
        :type i_conv: `int`
        :param w_min_change: Part of the biclustering convergence test. It specifies the minimal allowance of the change of 
                             row clusters. It should have nonnegative value. Default value is 0.
        :type w_min_change: `int`
        """
        self.name = "snmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization. 
                
        Return fitted factorization model.
        """
        self._set_params()
        # in version SNMF/L, V is transposed while W and H are swapped and transposed.
        if self.version == 'l':
            self.V = self.V.T
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            iter = 0
            self.idxWold = np.mat(np.zeros((self.V.shape[0], 1)))
            self.idxHold = np.mat(np.zeros((1, self.V.shape[1])))
            cobj = self.objective() 
            # count the number of convergence checks that column clusters and row clusters have not changed.
            self.inc = 0
            # normalize W
            self.W = elop(self.W, repmat(sop(multiply(self.W, self.W).sum(axis = 0), op = sqrt), self.V.shape[0], 1), div)
            self.I_k = self.eta * np.mat(np.eye(self.rank))
            self.betavec = sqrt(self.beta) * np.ones((1, self.rank))
            self.nrestart = 0
            while self._is_satisfied(cobj, iter):
                self.update()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
                if self.track_error:
                    self.tracker._track_error(self.residuals())
            # transpose and swap the roles back if SNMF/L
            if self.version == 'l':
                self.V = self.V.T
                self.W, self.H = self.H.T, self.W.T
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
    
    def _set_params(self): 
        """Set algorithm specific model options."""   
        self.version = self.options.get('version', 'r')
        self.eta = self.options.get('eta', np.max(self.V))
        if self.eta < 0: self.eta = np.max(self.V)
        self.beta = self.options.get('beta', 0.01)
        self.i_conv = self.options.get('i_conv', 10)
        self.w_min_change = self.options.get('w_min_change', 0)
        self.min_residuals = self.min_residuals if self.min_residuals else 1e-4
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mtrack.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
    
    def _is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if iter == 0:
            self.init_erravg = c_obj
        if self.max_iter and self.max_iter < iter:
            return False
        if self.inc >= self.i_conv and c_obj <= self.min_residuals * self.init_erravg:
            return False
        return True
            
    def update(self):
        """Update basis and mixture matrix."""
        # min_h ||[[W; 1 ... 1]*H  - [A; 0 ... 0]||, s.t. H>=0, for given A and W.
        self.H, _ = self._fcnnls(vstack((self.W, self.betavec)), vstack((self.V, np.zeros((1, self.V.shape[1])))))
        if any(self.H.sum(1) == 0):
            self.nrestart += 1
            if self.nrestart >= 10:
                raise utils.MFError("Too many restarts due to too large beta parameter.")
            self.idxWold = np.mat(np.zeros((self.V.shape[0], 1)))
            self.idxHold = np.mat(np.zeros((1, self.V.shape[1])))
            self.inc = 0
            self.W, _ = self.seed.initialize(self.V, self.rank, self.options)
            # normalize W
            self.W = elop(self.W, repmat(sop(multiply(self.W, self.W).sum(axis = 0), op = sqrt), self.V.shape[0], 1), div)
            return 
        # min_w ||[H'; I_k]*W' - [A'; 0]||, s.t. W>=0, for given A and H.
        Wt, _ = self._fcnnls(vstack((self.H.T, self.I_k)), vstack((self.V.T,  np.zeros((self.rank, self.V.shape[0])))))
        self.W = Wt.T   
    
    def fro_error(self):
        """Compute NMF objective value with additional sparsity constraints."""
        return 0.5 * norm(self.V - dot(self.W, self.H), "fro")**2 + self.eta * norm(self.W, "fro")**2 + self.beta * sum(norm(self.H[:,j], 1)**2 for j in self.H.shape[1])
        
    def objective(self):
        """Compute convergence test."""
        _, idxW = argmax(self.W, axis = 1)
        _, idxH = argmax(self.H, axis = 0) 
        changedW = count(elop(idxW, self.idxWold, ne), 1) 
        changedH = count(elop(idxH, self.idxHold, ne), 1)
        if changedW <= self.w_min_change and changedH == 0:
            self.inc += 1
        else:
            self.inc = 0
        resmat = elop(self.H, dot(dot(self.W.T, self.W), self.H) - dot(self.W.T, self.V) + dot(self.beta * np.ones((self.rank, self.rank)), self.H), min)
        resmat1 = elop(self.W, dot(self.W, dot(self.H, self.H.T)) - dot(self.V, self.H.T) + self.eta**2 * self.W, min)
        resvec = nz_data(resmat) + nz_data(resmat1)
        # L1 norm
        self.conv = norm(np.mat(resvec), 1)
        erravg = self.conv / len(resvec)
        self.idxWold = idxW
        self.idxHold = idxH
        return erravg
        
    def _fcnnls(self, C, A):
        """
        Nonnegative least squares solver (NNLS) using normal equations and fast combinatorial strategy (van Benthem and Keenan, 2004). 
        
        Given A and C this algorithm solves for the optimal K in a least squares sense, using that A = C*K in the problem
        ||A - C*K||, s.t. K>=0 for given A and C. 
        
        C is the nObs x lVar coefficient matrix
        A is the nObs x pRHS matrix of observations
        K is the lVar x pRHS solution matrix
        
        Pset is set of passive sets, one for each column. 
        Fset is set of column indices for solutions that have not yet converged. 
        Hset is set of column indices for currently infeasible solutions. 
        Jset is working set of column indices for currently optimal solutions. 
        """
        C = C.todense() if sp.isspmatrix(C) else C
        A = A.todense() if sp.isspmatrix(A) else A
        _, lVar = C.shape
        pRHS = A.shape[1]
        W = np.mat(np.zeros((lVar, pRHS)))
        iter = 0
        maxiter = 2 * lVar
        # precompute parts of pseudoinverse
        CtC = dot(C.T, C)
        CtA = dot(C.T, A)
        # obtain the initial feasible solution and corresponding passive set
        # K is not sparse
        K = self.__cssls(CtC, CtA)
        Pset = K > 0
        K[np.logical_not(Pset)] = 0
        D = K.copy() 
        Fset = np.array(find(np.logical_not(all(Pset, axis = 0))))
        # active set algorithm for NNLS main loop
        oitr = 0
        while len(Fset) > 0:
            oitr += 1    
            # solve for the passive variables
            K[:, Fset] = self.__cssls(CtC, CtA[:, Fset], Pset[:, Fset])
            # find any infeasible solutions
            idx = find(any(K[:, Fset] < 0, axis = 0))
            if idx != []:
                Hset = Fset[idx]
            else:
                Hset = []
            # make infeasible solutions feasible (standard NNLS inner loop)
            if len(Hset) > 0:
                nHset = len(Hset)
                alpha = np.mat(np.zeros((lVar, nHset)))
                while len(Hset) > 0 and iter < maxiter:
                    iter += 1
                    alpha[:,:nHset] = np.Inf
                    # find indices of negative variables in passive set
                    idx_f = find(np.logical_and(Pset[:, Hset], K[:, Hset] < 0))
                    i = [l % Pset.shape[0] for l in idx_f]
                    j = [l / Pset.shape[0] for l in idx_f]
                    if len(i) == 0:
                        break
                    hIdx = sub2ind((lVar, nHset), i, j)
                    l_1h = [l % lVar for l in hIdx]
                    l_2h = [l / lVar for l in hIdx]
                    if nHset == 1:
                        h_n = Hset * np.ones((len(j), 1))
                        negIdx = sub2ind(K.shape, i, h_n)
                    else:
                        negIdx = sub2ind(K.shape, i, Hset[j].T)
                    l_1n = [l % K.shape[0] for l in negIdx]
                    l_2n = [l / K.shape[0] for l in negIdx]
                    t_d = D[l_1n, l_2n] / (D[l_1n, l_2n] - K[l_1n, l_2n])
                    for i in xrange(len(l_1h)):
                        alpha[l_1h[i], l_2h[i]] = t_d[0, i]
                    alphaMin, minIdx =  argmin(alpha[:, :nHset], axis = 0)
                    minIdx = minIdx.tolist()[0]
                    alpha[:, :nHset] = repmat(alphaMin, lVar, 1)
                    D[:,Hset] = D[:,Hset] - multiply(alpha[:, :nHset], (D[:,Hset] - K[:,Hset]))
                    idx2zero = sub2ind(D.shape, minIdx, Hset)
                    l_1z = [l % D.shape[0] for l in idx2zero]
                    l_2z = [l / D.shape[0] for l in idx2zero]
                    D[l_1z, l_2z] = 0
                    Pset[l_1z, l_2z] = 0
                    K[:, Hset] = self.__cssls(CtC, CtA[:,Hset], Pset[:,Hset])
                    Hset = find(any(K < 0, axis = 0))
                    nHset = len(Hset)
            # make sure the solution has converged and check solution for optimality
            W[:,Fset] = CtA[:,Fset] - dot(CtC, K[:,Fset])
            Jset = find(all(multiply(np.logical_not(Pset[:,Fset]), W[:,Fset]) <= 0, axis = 0))
            if Jset != []:
                f_j = Fset[Jset]
            else:
                f_j = []
            Fset = np.setdiff1d(np.asarray(Fset), np.asarray(f_j))
            # for non-optimal solutions, add the appropriate variable to Pset
            if len(Fset) > 0:
                _, mxidx = argmax(multiply(np.logical_not(Pset[:,Fset]), W[:,Fset]), axis = 0)
                mxidx = mxidx.tolist()[0]
                idxs = sub2ind((lVar, pRHS), mxidx, Fset)
                l_1 = [l % lVar for l in idxs]
                l_2 = [l / lVar for l in idxs]
                Pset[l_1, l_2] = 1
                D[:,Fset] = K[:,Fset]
        return K, Pset
    
    def __cssls(self, CtC, CtA, Pset = None):
        """
        Solve the set of equations CtA = CtC * K for variables defined in set Pset
        using the fast combinatorial approach (van Benthem and Keenan, 2004).
        """
        K = np.mat(np.zeros(CtA.shape))
        if Pset == None or Pset.size == 0 or all(Pset):
            # equivalent if CtC is square matrix
            K = np.linalg.lstsq(CtC, CtA)[0]
            # K = pinv(CtC) * CtA
        else:
            lVar, pRHS = Pset.shape
            codedPset = dot(np.mat(2**np.array(range(lVar - 1, -1, -1))), Pset)
            sortedPset, sortedEset = sort(codedPset)
            breaks = diff(np.mat(sortedPset))
            breakIdx = [-1] + find(np.mat(breaks)) + [pRHS]
            for k in xrange(len(breakIdx) - 1):
                cols2solve = sortedEset[breakIdx[k] + 1 : breakIdx[k + 1] + 1]
                vars = Pset[:, sortedEset[breakIdx[k] + 1]]
                vars = [i for i in xrange(vars.shape[0]) if vars[i, 0]]
                K[:,cols2solve][vars, :] = np.linalg.lstsq(CtC[:,vars][vars, :], CtA[:,cols2solve][vars, :])[0]
                # K(vars,cols2solve) = pinv(CtC(vars,vars)) * CtA(vars,cols2solve);
        return K
    
    def __str__(self):
        return self.name   
        
        