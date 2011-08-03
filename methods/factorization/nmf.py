from operator import div, pow, eq, ne
from math import log

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Nmf(mstd.Nmf_std):
    """
    Standard Nonnegative Matrix Factorization (NMF). Based on Kullbach-Leibler divergence, it uses simple multiplicative
    updates [2], enhanced to avoid numerical underflow [3]. Based on Euclidean distance, it uses simple multiplicative
    updates [2]. Different objective functions can be used, namely Euclidean distance, divergence or connectivity 
    matrix convergence. 
    
    Together with a novel model selection mechanism, NMF is an efficient method for identification of distinct molecular
    patterns and provides a powerful method for class discovery. It appears to have higher resolution such as HC or 
    SOM and to be less sensitive to a priori selection of genes. Rather than separating gene clusters based on distance
    computation, NMF detects context-dependent patterns of gene expression in complex biological systems. 
    
    Besides usages in bioinformatics NMF can be applied to text analysis, image processing, multiway clustering,
    environmetrics etc. 
    
    [2] Lee, D..D., and Seung, H.S., (2001). Algorithms for Non-negative Matrix Factorization, Adv. Neural Info. Proc. Syst. 13, 556-562.
    [3] Brunet, J.-P., Tamayo, P., Golub, T. R., Mesirov, J. P., (2004). Metagenes and molecular pattern discovery using matrix factorization. Proceedings of the National Academy of Sciences of the United States of America, 101(12), 4164-9. doi: 10.1073/pnas.0308531101.
    """


    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        Algorithm specific model options are type of update equations and type of objective function. 
        When specifying model, user can pass :param:`update` parameter with one of possible values: 
            #. 'Euclidean' for classic Euclidean distance update equations, 
            #. 'divergence' for divergence update equations.
        When specifying model, user can pass :param:`objective` parameter with one of possible values:
            #. 'fro' for standard Frobenius distance cost function,
            #. 'div' for divergence of target matrix from NMF estimate cost function (KL),
            #. 'conn' for connectivity matrix changed elements cost function. 
        Default are 'Euclidean' for :param:`update` equations and 'Euclidean' for :param:`objective` function. 
        """
        self.name = "nmf"
        self.aseeds = ["random", "fixed", "nndsvd"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._set_params()
                
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            pobj = cobj = self.objective()
            iter = 0
            while self._is_satisfied(pobj, cobj, iter):
                pobj = cobj
                self.update()
                self._adjustment()
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.tracker != None:
                self.tracker.add(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter
        self.final_obj = cobj
        mffit = mfit.Mf_fit(self)
        return mffit
    
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iter and self.max_iter < iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = max(self.W, np.finfo(self.H.dtype).eps)
        self.W = max(self.H, np.finfo(self.W.dtype).eps)
        
    def _set_params(self):
        self.update = getattr(self, self.options.get('update', 'euclidean') + '_update') 
        self.objective = getattr(self, self.options.get('objective', 'fro') + '_objective')
        self.tracker = mtrack.Mf_track() if self.options.get('track', 0) and self.n_run > 1 else None
        
    def euclidean_update(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        self.H = multiply(self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        self.W = multiply(self.W , elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div)) 
        
    def divergence_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(self.H, elop(dot(self.W.T, elop(self.V, dot(self.W, self.H), div)), H1, div))
        W1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), W1, div))
        
    def fro_objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
    def div_objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(self.W, self.H)
        return (multiply(self.V, elop(self.V, Va, log)) - self.V + Va).sum()
    
    def conn_objective(self):
        """
        Compute connectivity matrix changes -- number of changing elements.
        if the number of instances changing the cluster is lower or equal to min_residuals, terminate factorization run.
        """
        _, idx = argmax(self.H, axis = 0)
        mat1 = repmat(idx, self.V.shape[1], 1)
        mat2 = repmat(idx.T, 1, self.V.shape[1])
        cons = elop(mat1, mat2, eq)
        if not hasattr(self, 'consold'):
            self.consold = np.ones_like(self.cons) - cons
            self.cons = cons
        else:
            self.consold = self.cons
            self.cons = cons
        return elop(self.cons, self.consold, ne).sum()
        
    def __str__(self):
        return self.name