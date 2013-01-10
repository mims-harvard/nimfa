
"""
#####################################
Snmf (``methods.factorization.snmf``)
#####################################

**Sparse Nonnegative Matrix Factorization (SNMF)** based on alternating nonnegativity constrained least squares [Park2007]_.

In order to enforce sparseness on basis or mixture matrix, SNMF can be used, namely two formulations: SNMF/L for 
sparse W (sparseness is imposed on the left factor) and SNMF/R for sparse H (sparseness imposed on the right factor).
These formulations utilize L1-norm minimization. Each subproblem is solved by a fast nonnegativity constrained
least squares (FCNNLS) algorithm (van Benthem and Keenan, 2004) that is improved upon the active set based NLS method. 

SNMF/R contains two subproblems for two-block minimization scheme. The objective function is coercive on the 
feasible set. It can be shown (Grippo and Sciandrome, 2000) that two-block minimization process is convergent, 
every accumulation point is a critical point of the corresponding problem. Similarly, the algorithm SNMF/L converges
to a stationary point. 

.. literalinclude:: /code/methods_snippets.py
    :lines: 163-175
    
.. literalinclude:: /code/methods_snippets.py
    :lines: 178-190
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

class Snmf(nmf_std.Nmf_std):
    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
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
                 is not recommended. It should have positive value. Default value is 1e-4.
    :type beta: `float`
    :param i_conv: Part of the biclustering convergence test. It decides convergence if row clusters and column clusters have 
                   not changed for :param:`i_conv` convergence tests. It should have nonnegative value.
                   Default value is 10.
    :type i_conv: `int`
    :param w_min_change: Part of the biclustering convergence test. It specifies the minimal allowance of the change of 
                         row clusters. It should have nonnegative value. Default value is 0.
    :type w_min_change: `int`
    """

    def __init__(self, **params):
        self.name = "snmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        self.set_params()
        
    def factorize(self):
        """
        Compute matrix factorization. 
                
        Return fitted factorization model.
        """
        # in version SNMF/L, V is transposed while W and H are swapped and transposed.
        if self.version == 'l':
            self.V = self.V.T
        
        for run in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            if sp.isspmatrix(self.W):
                self.W = self.W.tolil()
            if sp.isspmatrix(self.H):
                self.H = self.H.tolil()
            iter = 0
            self.idx_w_old = np.mat(np.zeros((self.V.shape[0], 1)))
            self.idx_h_old = np.mat(np.zeros((1, self.V.shape[1])))
            c_obj = sys.float_info.max 
            best_obj = c_obj if run == 0 else best_obj
            # count the number of convergence checks that column clusters and row clusters have not changed.
            self.inc = 0
            # normalize W
            self.W = elop(self.W, repmat(sop(multiply(self.W, self.W).sum(axis = 0), op = np.sqrt), self.V.shape[0], 1), div)
            if sp.isspmatrix(self.V):
                self.beta_vec = sqrt(self.beta) * sp.lil_matrix(np.ones((1, self.rank)) , dtype = self.V.dtype)
                self.I_k = self.eta * sp.eye(self.rank, self.rank, format = 'lil')  
            else:
                self.beta_vec = sqrt(self.beta) * np.ones((1, self.rank))
                self.I_k = self.eta * np.mat(np.eye(self.rank))
            self.n_restart = 0
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
            # basis and mixture matrix are now constructed and are now converted to CSR for fast LA operations
            if sp.isspmatrix(self.W):
                self.W = self.W.tocsr()
            if sp.isspmatrix(self.H): 
                self.H = self.H.tocsr()
            # transpose and swap the roles back if SNMF/L
            if self.version == 'l':
                self.V = self.V.T
                self.W, self.H = self.H.T, self.W.T
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
    
    def set_params(self): 
        """Set algorithm specific model options."""   
        self.version = self.options.get('version', 'r')
        self.eta = self.options.get('eta', np.max(self.V) if not sp.isspmatrix(self.V) else np.max(self.V.data))
        if self.eta < 0: self.eta = np.max(self.V) if not sp.isspmatrix(self.V) else 0.
        self.beta = self.options.get('beta', 1e-4)
        self.i_conv = self.options.get('i_conv', 10)
        self.w_min_change = self.options.get('w_min_change', 0)
        self.min_residuals = self.min_residuals if self.min_residuals else 1e-4
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
    
    def is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if iter == 0:
            self.init_erravg = c_obj
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if self.inc >= self.i_conv and c_obj < self.min_residuals * self.init_erravg:
            return False
        return True
            
    def update(self):
        """Update basis and mixture matrix."""
        if sp.isspmatrix(self.V):
            v1 = self.V.__class__((1, self.V.shape[1]), dtype = self.V.dtype)
            v1t = self.V.__class__((self.rank, self.V.shape[0]), dtype = self.V.dtype)
        else:
            v1 = np.zeros((1, self.V.shape[1]))
            v1t = np.zeros((self.rank, self.V.shape[0]))
        # min_h ||[[W; 1 ... 1]*H  - [A; 0 ... 0]||, s.t. H>=0, for given A and W
        if sp.isspmatrix(self.V):
            self.H = self._spfcnnls(vstack((self.W, self.beta_vec)), vstack((self.V, v1)))
        else:
            self.H = self._fcnnls(vstack((self.W, self.beta_vec)), vstack((self.V, v1)))
        if any(self.H.sum(axis = 1) == 0):
            self.n_restart += 1
            if self.n_restart >= 100:
                raise utils.MFError("Too many restarts due to too large beta parameter.")
            self.idx_w_old = np.mat(np.zeros((self.V.shape[0], 1)))
            self.idx_h_old = np.mat(np.zeros((1, self.V.shape[1])))
            self.inc = 0
            self.W, _ = self.seed.initialize(self.V, self.rank, self.options)
            # normalize W and convert to lil
            if (sp.issparse(self.W)):
                self.W = elop(self.W, repmat(sop(multiply(self.W, self.W).sum(axis = 0), op = sqrt), self.V.shape[0], 1), div).tolil()
            else:
                self.W = elop(self.W, repmat(sop(multiply(self.W, self.W).sum(axis = 0), op = sqrt), self.V.shape[0], 1), div)
            return 
        # min_w ||[H'; I_k]*W' - [A'; 0]||, s.t. W>=0, for given A and H.
        if sp.isspmatrix(self.V):
            Wt = self._spfcnnls(vstack((self.H.T, self.I_k)), vstack((self.V.T, v1t)))
        else:
            Wt = self._fcnnls(vstack((self.H.T, self.I_k)), vstack((self.V.T, v1t)))
        self.W = Wt.T   
    
    def fro_error(self):
        """Compute NMF objective value with additional sparsity constraints."""
        return 0.5 * norm(self.V - dot(self.W, self.H), "fro")**2 + self.eta * norm(self.W, "fro")**2 + self.beta * sum(norm(self.H[:,j], 1)**2 for j in self.H.shape[1])
        
    def objective(self):
        """Compute convergence test."""
        _, idx_w = argmax(self.W, axis = 1)
        _, idx_h = argmax(self.H, axis = 0) 
        changed_w = count(elop(idx_w, self.idx_w_old, ne), 1) 
        changed_h = count(elop(idx_h, self.idx_h_old, ne), 1)
        if changed_w <= self.w_min_change and changed_h == 0:
            self.inc += 1
        else:
            self.inc = 0
        resmat = elop(self.H, dot(dot(self.W.T, self.W), self.H) - dot(self.W.T, self.V) + dot(self.beta * np.ones((self.rank, self.rank)), self.H), min)
        resmat1 = elop(self.W, dot(self.W, dot(self.H, self.H.T)) - dot(self.V, self.H.T) + self.eta**2 * self.W, min)
        res_vec = nz_data(resmat) + nz_data(resmat1)
        # L1 norm
        self.conv = norm(np.mat(res_vec), 1)
        err_avg = self.conv / len(res_vec)
        self.idx_w_old = idx_w
        self.idx_h_old = idx_h
        return err_avg
    
    def _spfcnnls(self, C, A):
        """
        NNLS for sparse matrices.
        
        Nonnegative least squares solver (NNLS) using normal equations and fast combinatorial strategy (van Benthem and Keenan, 2004). 
        
        Given A and C this algorithm solves for the optimal K in a least squares sense, using that A = C*K in the problem
        ||A - C*K||, s.t. K>=0 for given A and C. 
        
        C is the n_obs x l_var coefficient matrix
        A is the n_obs x p_rhs matrix of observations
        K is the l_var x p_rhs solution matrix
        
        p_set is set of passive sets, one for each column. 
        f_set is set of column indices for solutions that have not yet converged. 
        h_set is set of column indices for currently infeasible solutions. 
        j_set is working set of column indices for currently optimal solutions. 
        """
        C = C.tolil()
        A = A.tolil()
        _, l_var = C.shape
        p_rhs = A.shape[1]
        W = sp.lil_matrix((l_var, p_rhs))
        iter = 0
        max_iter = 3 * l_var
        # precompute parts of pseudoinverse
        CtC = dot(C.T, C)
        CtA = dot(C.T, A)
        # obtain the initial feasible solution and corresponding passive set
        K = self.__spcssls(CtC, CtA)
        p_set = sop(K, 0, ge).tolil()
        for i in xrange(K.shape[0]):
            for j in xrange(K.shape[1]):
                if not p_set[i, j]: K[i, j] = 0.
        D = K.copy() 
        f_set = np.array(find(np.logical_not(all(p_set, axis = 0))))
        # active set algorithm for NNLS main loop
        while len(f_set) > 0:
            # solve for the passive variables
            K[:, f_set] = self.__spcssls(CtC, CtA[:, f_set], p_set[:, f_set])
            # find any infeasible solutions
            idx = find(any(sop(K[:, f_set], 0, le), axis = 0))
            h_set = f_set[idx] if idx != [] else []
            # make infeasible solutions feasible (standard NNLS inner loop)
            if len(h_set) > 0:
                n_h_set = len(h_set)
                alpha = np.mat(np.zeros((l_var, n_h_set)))
                while len(h_set) > 0 and iter < max_iter:
                    iter += 1
                    alpha[:,:n_h_set] = np.Inf
                    # find indices of negative variables in passive set
                    tmp = sop(K[:, h_set], 0, le).tolil()
                    tmp_f = sp.lil_matrix(K.shape, dtype = 'bool')
                    for i in xrange(K.shape[0]):
                        for j in xrange(len(h_set)):
                            if p_set[i, h_set[j]] and tmp[i, h_set[j]]: tmp_f[i, h_set[j]] = True
                    idx_f = find(tmp_f[:, h_set])
                    i_f = [l % p_set.shape[0] for l in idx_f]
                    j_f = [l / p_set.shape[0] for l in idx_f]
                    if len(i_f) == 0:
                        break
                    if n_h_set == 1:
                        h_n = h_set * np.ones((1, len(j_f)))
                        l_1n = i_f
                        l_2n = h_n.tolist()[0]
                    else:
                        l_1n = i_f
                        l_2n = [h_set[e] for e in j_f]
                    t_d = D[l_1n, l_2n] / (D[l_1n, l_2n] - K[l_1n, l_2n])
                    for i in xrange(len(i_f)):
                        alpha[i_f[i], j_f[i]] = t_d.todense().flatten()[0, i]
                    alpha_min, min_idx = argmin(alpha[:, :n_h_set], axis = 0)
                    min_idx = min_idx.tolist()[0]
                    alpha[:, :n_h_set] = repmat(alpha_min, l_var, 1)
                    D[:, h_set] = D[:, h_set] - multiply(alpha[:, :n_h_set], D[:, h_set] - K[:, h_set])
                    D[min_idx, h_set] = 0
                    p_set[min_idx, h_set] = 0
                    K[:, h_set] = self.__spcssls(CtC, CtA[:, h_set], p_set[:, h_set])
                    h_set = find(any(sop(K, 0, le), axis = 0))
                    n_h_set = len(h_set)
            # make sure the solution has converged and check solution for optimality
            W[:, f_set] = CtA[:, f_set] - dot(CtC, K[:, f_set])
            tmp = sp.lil_matrix(p_set.shape, dtype = 'bool')
            for i in xrange(p_set.shape[0]):
                for j in f_set:
                    if not p_set[i, j]: tmp[i, j] = True
            j_set = find(all(sop(multiply(tmp[:, f_set], W[:, f_set]), 0, le), axis = 0))
            f_j = f_set[j_set] if j_set != [] else []
            f_set = np.setdiff1d(np.asarray(f_set), np.asarray(f_j))
            # for non-optimal solutions, add the appropriate variable to Pset
            if len(f_set) > 0:
                tmp = sp.lil_matrix(p_set.shape, dtype = 'bool')
                for i in xrange(p_set.shape[0]):
                    for j in f_set:
                        if not p_set[i, j]: tmp[i, j] = True
                _, mxidx = argmax(multiply(tmp[:, f_set], W[:, f_set]), axis = 0)
                mxidx = mxidx.tolist()[0]
                p_set[mxidx, f_set] = 1
                D[:, f_set] = K[:, f_set]
        return K.tolil()
    
    def __spcssls(self, CtC, CtA, p_set = None):
        """
        Solver for sparse matrices.
        
        Solve the set of equations CtA = CtC * K for variables defined in set p_set
        using the fast combinatorial approach (van Benthem and Keenan, 2004).
        
        It returns matrix in LIL sparse format.
        """    
        K = sp.lil_matrix(CtA.shape)
        if p_set == None or p_set.size == 0 or all(p_set):
            # equivalent if CtC is square matrix
            for k in xrange(CtA.shape[1]):
                ls = sp.linalg.gmres(CtC, CtA[:, k].toarray())[0]
                K[:, k] = sp.lil_matrix(np.mat(ls).T)
            # K = dot(np.linalg.pinv(CtC), CtA)
        else:
            l_var, p_rhs = p_set.shape
            coded_p_set = dot(sp.lil_matrix(np.mat(2**np.array(range(l_var - 1, -1, -1)))), p_set)
            sorted_p_set, sorted_idx_set = sort(coded_p_set.todense())
            breaks = diff(np.mat(sorted_p_set))
            break_idx = [-1] + find(np.mat(breaks)) + [p_rhs]
            for k in xrange(len(break_idx) - 1):
                cols2solve = sorted_idx_set[break_idx[k] + 1 : break_idx[k + 1] + 1]
                vars = p_set[:, sorted_idx_set[break_idx[k] + 1]]
                vars = [i for i in xrange(vars.shape[0]) if vars[i, 0]]
                tmp_ls = CtA[:, cols2solve][vars, :]
                sol = sp.lil_matrix(K.shape)
                for k in xrange(tmp_ls.shape[1]):
                    ls = sp.linalg.gmres(CtC[:, vars][vars, :], tmp_ls[:, k].toarray())[0]
                    sol[:, k] = sp.lil_matrix(np.mat(ls).T)
                i = 0
                for c in cols2solve:
                    j = 0
                    for v in vars:
                        K[v, c] = sol[j,i]
                        j += 1
                    i += 1
                # K[vars, cols2solve] = dot(np.linalg.pinv(CtC[vars, vars]), CtA[vars, cols2solve])
        return K.tolil()
        
    def _fcnnls(self, C, A):
        """
        NNLS for dense matrices.
        
        Nonnegative least squares solver (NNLS) using normal equations and fast combinatorial strategy (van Benthem and Keenan, 2004). 
        
        Given A and C this algorithm solves for the optimal K in a least squares sense, using that A = C*K in the problem
        ||A - C*K||, s.t. K>=0 for given A and C. 
        
        C is the n_obs x l_var coefficient matrix
        A is the n_obs x p_rhs matrix of observations
        K is the l_var x p_rhs solution matrix
        
        p_set is set of passive sets, one for each column. 
        f_set is set of column indices for solutions that have not yet converged. 
        h_set is set of column indices for currently infeasible solutions. 
        j_set is working set of column indices for currently optimal solutions. 
        """
        C = C.todense() if sp.isspmatrix(C) else C
        A = A.todense() if sp.isspmatrix(A) else A
        _, l_var = C.shape
        p_rhs = A.shape[1]
        W = np.mat(np.zeros((l_var, p_rhs)))
        iter = 0
        max_iter = 3 * l_var
        # precompute parts of pseudoinverse
        CtC = dot(C.T, C)
        CtA = dot(C.T, A)
        # obtain the initial feasible solution and corresponding passive set
        # K is not sparse
        K = self.__cssls(CtC, CtA)
        p_set = K > 0
        K[np.logical_not(p_set)] = 0
        D = K.copy() 
        f_set = np.array(find(np.logical_not(all(p_set, axis = 0))))
        # active set algorithm for NNLS main loop
        while len(f_set) > 0:
            # solve for the passive variables
            K[:, f_set] = self.__cssls(CtC, CtA[:, f_set], p_set[:, f_set])
            # find any infeasible solutions
            idx = find(any(K[:, f_set] < 0, axis = 0))
            h_set = f_set[idx] if idx != [] else []
            # make infeasible solutions feasible (standard NNLS inner loop)
            if len(h_set) > 0:
                n_h_set = len(h_set)
                alpha = np.mat(np.zeros((l_var, n_h_set)))
                while len(h_set) > 0 and iter < max_iter:
                    iter += 1
                    alpha[:,:n_h_set] = np.Inf
                    # find indices of negative variables in passive set
                    idx_f = find(np.logical_and(p_set[:, h_set], K[:, h_set] < 0))
                    i_f = [l % p_set.shape[0] for l in idx_f]
                    j_f = [l / p_set.shape[0] for l in idx_f]
                    if len(i_f) == 0:
                        break
                    if n_h_set == 1:
                        h_n = h_set * np.ones((1, len(j_f)))
                        l_1n = i_f
                        l_2n = h_n.tolist()[0]
                    else:
                        l_1n = i_f
                        l_2n = [h_set[e] for e in j_f]
                    t_d = D[l_1n, l_2n] / (D[l_1n, l_2n] - K[l_1n, l_2n])
                    for i in xrange(len(i_f)):
                        alpha[i_f[i], j_f[i]] = t_d.flatten()[0, i]
                    alpha_min, min_idx = argmin(alpha[:, :n_h_set], axis = 0)
                    min_idx = min_idx.tolist()[0]
                    alpha[:, :n_h_set] = repmat(alpha_min, l_var, 1)
                    D[:, h_set] = D[:, h_set] - multiply(alpha[:, :n_h_set], D[:, h_set] - K[:, h_set])
                    D[min_idx, h_set] = 0
                    p_set[min_idx, h_set] = 0
                    K[:, h_set] = self.__cssls(CtC, CtA[:, h_set], p_set[:, h_set])
                    h_set = find(any(K < 0, axis = 0))
                    n_h_set = len(h_set)
            # make sure the solution has converged and check solution for optimality
            W[:, f_set] = CtA[:, f_set] - dot(CtC, K[:, f_set])
            j_set = find(all(multiply(np.logical_not(p_set[:, f_set]), W[:, f_set]) <= 0, axis = 0))
            f_j = f_set[j_set] if j_set != [] else []
            f_set = np.setdiff1d(np.asarray(f_set), np.asarray(f_j))
            # for non-optimal solutions, add the appropriate variable to Pset
            if len(f_set) > 0:
                _, mxidx = argmax(multiply(np.logical_not(p_set[:, f_set]), W[:, f_set]), axis = 0)
                mxidx = mxidx.tolist()[0]
                p_set[mxidx, f_set] = 1
                D[:, f_set] = K[:, f_set]
        return K
    
    def __cssls(self, CtC, CtA, p_set = None):
        """
        Solver for dense matrices. 
        
        Solve the set of equations CtA = CtC * K for variables defined in set p_set
        using the fast combinatorial approach (van Benthem and Keenan, 2004).
        """
        K = np.mat(np.zeros(CtA.shape))
        if p_set == None or p_set.size == 0 or all(p_set):
            # equivalent if CtC is square matrix
            K = np.linalg.lstsq(CtC, CtA)[0]
            # K = dot(np.linalg.pinv(CtC), CtA)
        else:
            l_var, p_rhs = p_set.shape
            coded_p_set = dot(np.mat(2**np.array(range(l_var - 1, -1, -1))), p_set)
            sorted_p_set, sorted_idx_set = sort(coded_p_set)
            breaks = diff(np.mat(sorted_p_set))
            break_idx = [-1] + find(np.mat(breaks)) + [p_rhs]
            for k in xrange(len(break_idx) - 1):
                cols2solve = sorted_idx_set[break_idx[k] + 1 : break_idx[k + 1] + 1]
                vars = p_set[:, sorted_idx_set[break_idx[k] + 1]]
                vars = [i for i in xrange(vars.shape[0]) if vars[i, 0]]
                if vars != [] and cols2solve != []: 
                    sol = np.linalg.lstsq(CtC[:, vars][vars, :], CtA[:, cols2solve][vars, :])[0]
                    i = 0
                    for c in cols2solve:
                        j = 0
                        for v in vars:
                            K[v, c] = sol[j,i]
                            j += 1
                        i += 1
                    # K[vars, cols2solve] = dot(np.linalg.pinv(CtC[vars, vars]), CtA[vars, cols2solve])
        return K
    
    def __str__(self):
        return self.name + " - " + self.version
    
    def __repr__(self):
        return self.name 
        
        