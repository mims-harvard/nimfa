from operator import div, pow, eq, ne
from math import log

import models.mf_fit as mfit
from utils.linalg import *

class Nmf(object):
    """
    Standard Nonnegative Matrix Factorization (NMF). Based on Kullbach-Leibler divergence, it uses simple multiplicative
    updates [2], enhanced to avoid numerical underflow [3]. Based on euclidean distance, it uses simple multiplicative
    updates [2]. Different objective functions can be used, namely euclidean distance, divergence or connectivity 
    matrix convergence. 
    
    Together with a novel model selection mechanism, NMF is an efficient method for identification of distinct molecular
    patterns and provides a powerful method for class discovery. It appears to have higher resolution such as HC or 
    SOM and to be less sensitive to a priori selection of genes. Rather than separating gene clusters based on distance
    computation, NMF detects context-dependent patterns of gene expression in complex biological systems. 
    
    Besides usages in bioinformatics NMF can be applied to text analysis, image processing, multiway clustering,
    environmetrics etc. 
    
    [2] Lee, D..D., and Seung, H.S., (2001), Algorithms for Non-negative Matrix Factorization, Adv. Neural Info. Proc. Syst. 13, 556-562.
    [3] ﻿Brunet, J.-P., Tamayo, P., Golub, T. R., Mesirov, J. P. (2004). Metagenes and molecular pattern discovery using matrix factorization. Proceedings of the National Academy of Sciences of the United States of America, 101(12), 4164-9. doi: 10.1073/pnas.0308531101.
    """

    def __init__(self):
        self.name = "nmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self, model):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        
        :param model: The underlying model of matrix factorization. Algorithm specific model options are type of 
                      update equations and type of objective function. 
                      When specifying model, user can pass 'update' keyword argument with one of
                      possible values: 
                          #. 'euclidean' for classic Euclidean distance update equations, 
                          #. 'divergence' for divergence update equations.
                      When specifying model, user can pass 'objective' keyword argument with one of
                      possible values:
                          #. 'fro' for standard Frobenius distance cost function,
                          #. 'div' for divergence of target matrix from NMF estimate cost function (KL),
                          #. 'conn' for connectivity matrix changed elements cost function. 
                      Default are 'euclidean' update equations and 'euclidean' cost function. 
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
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
            self.final_obj = cobj
            mffit = mfit.Mf_fit(self)
            if self.callback: self.callback(mffit)
        return mffit
    
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iters and self.max_iters < iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def _adjustment(self):
        """Adjust small value to factors to avoid numerical underflow."""
        self.H = sop(self.W, np.finfo(self.H.dtype).eps)
        self.W = sop(self.H, np.finfo(self.W.dtype).eps)
        
    def _set_params(self):
        self.update = getattr(self, self.options['update'] + '_update') if self.options and 'update' in self.options else self.euclidean_update()
        self.objective = getattr(self, self.options['objective'] + '_objective') if self.options and 'objective' in self.options else self.fro_error()
        
    def euclidean_update(self):
        """Update basis and mixture matrix based on euclidean distance multiplicative update rules."""
        self.H = multiply(self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        self.W = multiply(self.W , elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div)) 
        
    def divergence_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        h1 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(self.H, elop(dot(self.W.T, elop(self.V, dot(self.W, self.H), div)), h1, div))
        w1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), w1, div))
        
    def fro_error(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
    def div_error(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = dot(self.W, self.H)
        return (multiply(self.V, elop(self.V, Va, log)) - self.V + Va).sum()
    
    def conn_error(self):
        """Compute connectivity matrix changes -- number of changing elements.
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
        