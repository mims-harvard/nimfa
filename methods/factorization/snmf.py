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
        
        Algorithm specific model options are :param:`version`, :param:`eta`, :param:`beta`, :param:`i_conv` and 
        param:`w_min_change` which can be passed with values as keyword arguments.
        The parameter :param:`min_residuals` of the underlying model is used as KKT convergence test and should have 
        positive value. If not specified, value 1e-4 is used. 
            #. Parameter param:`version` specifies version of the SNMF algorithm. it has two accepting values,
               'r' and 'l' for SNMF/R and SNMF/L, respectively. Default choice is SNMF/R.
            #. Parameter :param:`eta` is used for suppressing Frobenius norm on basis matrix (W). Default value
               is maximum value in target matrix (V). If :param:`eta` is negative, maximum value in target matrix is
               used for it. 
            #. Parameter :param:`beta` controls sparseness. Larger :param:`beta` generates higher sparseness on H. Too large
               :param:`beta` is not recommended. It should have positive value. Default value is 0.01
            #. Parameter :param:`i_conv` is part of biclustering convergence test. It decides convergence if row clusters
               and column clusters have not changed for :param:`i_conv` convergence checks. It should have nonnegative value.
               Default value is 10.
            #. Parameter :param:`w_min_change` is part of biclustering convergence test. It specifies the minimal allowance
               of the change of row clusters. It should have nonnegative value. Default value is 0.
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
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            iter = 0
            self.idxWold = np.matrix(np.zeros((self.V.shape[0], 1)))
            self.idxHold = np.matrix(np.zeros((1, self.V.shape[1])))
            cobj = self.objective() 
            # count the number of convergence checks that column clusters and row clusters have not changed.
            self.inc = 0
            # normalize W
            self.W = elop(self.W, repmat(sop(sum(multiply(self.W, self.W), 0), sqrt), self.V[0], 1), div)
            self.I_k = self.eta * sp.eye(self.rank)
            self.betavec = sqrt(self.beta) * np.ones((1, self.rank))
            self.nrestart = 0
            while self._is_satisfied(cobj, iter):
                self.update()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            # transpose and swap the roles back
            if self.version == 'l':
                self.V = self.V.T
                self.W, self.H = self.H.T, self.W.T
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.tracker != None:
                self.tracker.add(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter - 1
        self.final_obj = cobj
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _set_params(self):    
        # in version l, V is transposed while W and H are swapped and transposed.
        self.version = self.options.get('version', 'r')
        self.V = self.V.T if self.version == 'l' else self.V
        self.eta = self.options.get('eta', max(self.V))
        if self.eta < 0: self.eta = max(self.V)
        self.beta = self.options.get('beta', 0.01)
        self.i_conv = self.options.egt('i_conv', 10)
        self.w_min_change = self.options.get('w_min_change', 0)
        self.min_residuals = self.min_residuals if self.min_residuals else 1e-4
        self.tracker = mtrack.Mf_track() if self.options.get('track', 0) and self.n_run > 1 else None
    
    def _is_satisfied(self, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if iter == 0:
            self.init_erravg = cobj
        if self.max_iter and self.max_iter < iter:
            return False
        if self.inc >= self.i_conv and cobj <= self.min_residuals * self.init_erravg:
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
            self.idxWold = np.matrix(np.zeros((self.V.shape[0], 1)))
            self.idxHold = np.matrix(np.zeros((1, self.V.shape[1])))
            self.inc = 0
            self.W, _ = self.seed.initialize(self.V, self.rank, self.options)
            # normalize W
            self.W = elop(self.W, repmat(sop(sum(multiply(self.W, self.W), 0), sqrt), self.V[0], 1), div)
            return 
        # min_w ||[H'; I_k]*W' - [A'; 0]||, s.t. W>=0, for given A and H.
        self.W, _ = self._fcnnls(vstack((self.H.T, self.I_k)),vstack((self.V.T,  np.zeros((self.rank, self.V.shape[0]))))).T  
    
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
        self.conv = norm(resvec, 1)
        erravg = self.conv / len(resvec)
        self.idxWold = idxW
        self.idxHold = idxH
        return erravg
        
    def _fcnnls(self, C, A):
        """
        NNLS using normal equations and fast combinatorial strategy (van Benthem and Keenan, 2004). 
        
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
        W = np.matrix(np.zeros((lVar, pRHS)))
        iter = 0
        maxiter = 2 * lVar
        # precompute parts of pseudoinverse
        CtC = dot(C.T, C)
        CtA = dot(C.T, A)
        # obtain the initial feasible solution and corresponding passive set
        # K is not sparse
        K = self.__cssls(CtC, CtA)
        Pset = sop(K, 0, ge)
        K = sop(K, 0, le)
        D = K 
        Fset  = find(any(Pset, 0))
        # active set algorithm for NNLS main loop
        oitr = 0
        while Fset.size > 0:
            oitr += 1    
            # solve for the passive variables
            K[:, Fset] = self.__cssls(CtC, CtA[:, Fset], Pset[:, Fset])
            # find any infeasible solutions
            Hset = Fset[find(any(K[:, Fset] < 0))]
            # make infeasible solutions feasible (standard NNLS inner loop)
            if Hset.size > 0:
                nHset = len(Hset)
                alpha = np.matrix(np.zeros((lVar, nHset)))
                while Hset and iter < maxiter:
                    iter += 1
                    alpha[:,:nHset] = np.Inf
                    # find indices of negative variables in passive set
                    i,j = find(elop(Pset[:, Hset], (K[:, Hset] < 0), np.logical_and))
                    if i.size == 0:
                        break
                    hIdx = sub2ind((lVar, nHset), i, j)
                    if nHset == 1:
                        negIdx = sub2ind(K.shape, i, Hset * np.matrix(np.ones((len(j),1))))
                    else:
                        negIdx = sub2ind(K.shape, i, Hset[j].T)
                    alpha[hIdx] = D[negIdx] / (D[negIdx] - K[negIdx])
                    alphaMin, minIdx =  argmin(alpha[:, :nHset])
                    alpha[:, :nHset] = repmat(alphaMin, lVar, 1)
                    D[:,Hset] = D[:,Hset] - multiply(alpha[:, :nHset], (D[:,Hset] - K[:,Hset]))
                    idx2zero = sub2ind(D.shape, minIdx, Hset)
                    D[idx2zero] = 0
                    Pset[idx2zero] = 0
                    K[:, Hset] = self.__cssls(CtC, CtA[:,Hset], Pset[:,Hset])
                    Hset = find(any(K < 0))
                    nHset = len(Hset)
            # make sure the solution has converged and check solution for optimality
            W[:,Fset] = CtA[:,Fset] - dot(CtC, K[:,Fset])
            Jset = find(all(multiply(np.logical_not(Pset[:,Fset]), W[:,Fset] <= 0)))
            Fset = np.setdiff1d(np.asarray(Fset), np.asarray(Fset[Jset]))
            # for non-optimal solutions, add the appropriate variable to Pset
            if Fset.size > 0:
                _, mxidx = argmax(multiply(np.logical_not(Pset[:,Fset]), W[:,Fset]))
                Pset[sub2ind((lVar, pRHS), mxidx, Fset)] = 1
                D[:,Fset] = K[:,Fset]
        return K, Pset
    
    def __cssls(self, CtC, CtA, Pset):
        """Solve the set of equations CtA = CtC * K for variables defined in set Pset
        using the fast combinatorial approach (van Benthem and Keenan, 2004)."""
        K = np.matrix(np.zeros(CtA.shape))
        if Pset.size == 0 or all(Pset):
            # equivalent if CtC is square matrix
            K = dot(np.linalg.inv(CtC), CtA)
            # K = pinv(CtC) * CtA
        else:
            lVar, pRHS = Pset.shape
            codedPset = dot(sop(np.matrix(range(lVar - 1, 0, -1)).T, 2, pow), Pset)
            sortedPset, sortedEset = sort(codedPset)
            breaks = diff(sortedPset)
            breakIdx = [0] + find(breaks) + [pRHS]
            for k in xrange(len(breakIdx) - 1):
                cols2solve = sortedEset[breakIdx[k] + 1 : breakIdx[k + 1]]
                vars = Pset[:, sortedEset[breakIdx[k] + 1]]
                K[vars,cols2solve] = dot(np.linalg.inv(CtC[vars,vars]), CtA[vars,cols2solve])
                # K(vars,cols2solve) = pinv(CtC(vars,vars)) * CtA(vars,cols2solve);
        return K
    
    def __str__(self):
        return self.name   
        
        