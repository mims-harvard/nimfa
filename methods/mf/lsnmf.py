import numpy as np
from operator import div, pow, eq, ne
from math import log

import models.mf_fit as mfit
from utils.linalg import *

class Lsnmf(object):
    """
    Alternating nonnegative least squares MF using the projected gradient (bound constrained optimization) method for 
    each subproblem [4]. It converges faster than the popular multiplicative update approach.
    
    Algorithm relies on efficiently solving bound constrained subproblems. They are solved using the projected gradient 
    method. Each subproblem contains some (m) independent nonnegative least squares problems. Not solving these separately
    but treating them together is better because of: problems are closely related, sharing the same constant matrices;
    all operations are matrix based, which saves computational time. 
    
    The main task per iteration of the subproblem is to find a step size alpha such that a sufficient decrease condition
    of bound constrained problem is satisfied. In alternating least squares, each subproblem involves an optimization 
    procedure and requires a stopping condition. A common way to check whether current solution is close to a 
    stationary point is the form of the projected gradient [4].
    
    [4] ﻿Lin, C.J. (2007). Projected gradient methods for nonnegative matrix factorization. Neural computation, 19(10), 2756-79. doi: 10.1162/neco.2007.19.10.2756. 
    """

    def __init__(self):
        self.aname = "lsnmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self, model):
        """
        :param model: The underlying model of matrix factorization.
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            pobj = cobj = self.objective()
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                cobj = self.objective()
                iter += 1
            mffit = mfit.Mf_fit(self)
            if self.callback: self.callback(mffit)
        return mffit
    
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        return True
    
    def update(self):
        """Update basis and mixture matrix."""
        pass
    
    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        pass
    
        