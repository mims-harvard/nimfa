

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


    def __init__(self, params):
        self.aname = "lsnmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["nndsvd"]
        
    def factorize(self, model):
        self.__dict__.update(model.__dict__)
        