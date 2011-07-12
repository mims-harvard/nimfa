

class Snmf(object):
    """
    Sparse nonnegative matrix factorization (SNMF) based on alternating nonnegativity constrained least squares. [5]
    
    In order to enforce sparseness on basis or mixture matrix, SNMF can be used, namely two formulations: SNMF/L for 
    sparse W (sparseness is imposed on the left factor) and SNMF/R for sparse H (sparseness imposed on the right factor).
    These formulations utilize L1-norm minimization. Each subproblem is solved by a fast nonnegativity constrained
    least squares (NLS) algorithm (van Benthem ane Keenan, 2004) that is improved upon the active set based NLS method. 
    
    SNMF/R contains two subproblems for two-block minimization scheme. The objective function is coercive on the 
    feasible set. It can be shown (Grippo and Sciandrome, 2000) that two-block minimization process is convergent, 
    every accumulation point is a critical point of the corresponding problem. Similarly, the algorithm SNMF/L converges
    to a stationary point. 
   
    [5] Sparse Non-negative Matrix Factorizations via Alternating Non-negativity-constrained Least Squares for Microarray Data Analysis
        Hyunsoo Kim and Haesun Park, Bioinformatics, 2007.
    """

    def __init__(self, params):
        self.aname = "snmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self, model):
        """
        :param model: The underlying model of matrix factorization. Algorithm specific model options are 
                      'version', 'eta', 'beta', 'i_conv', 'w_min_change' which can be passed with values as 
                      keyword arguments to the underlying model. The min_residuals of the underlying model is used
                      as KKT convergence test. 
                      #. Parameter version specifies version of the SNMF algorithm. it has two accepting values,
                         'r' and 'l' for SNMF/R and SNMF/L, respectively. 
                      #. Parameter eta is used for suppressing Frobenius norm on basis matrix (W).
                      #. Parameter beta controls sparseness. Larger beta generates higher sparseness on H. Too large
                         beta is not recommended. 
                      #. Parameter i_conv is part of biclustering convergence test. Decide convergence if row clusters
                         and column clusters have not changed for i_conv convergence checks. Default value is 10
                      #. Parameter w_min_change is part of biclustering convergence test. It specifies the minimal allowance
                         of the change of row clusters. Default value is 0.
                      
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
        