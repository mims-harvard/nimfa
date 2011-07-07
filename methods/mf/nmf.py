import numpy as np
from operator import div, pow
from math import log

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
    
    [2] Lee, D..D., and Seung, H.S., (2001), 'Algorithms for Non-negative Matrix Factorization', Adv. Neural Info. Proc. Syst. 13, 556-562.
    [3] ﻿Brunet, J.-P., Tamayo, P., Golub, T. R., Mesirov, J. P. (2004). Metagenes and molecular pattern discovery using matrix factorization. Proceedings of the National Academy of Sciences of the United States of America, 101(12), 4164-9. doi: 10.1073/pnas.0308531101.
    """

    def __init__(self):
        self.name = "nmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["nndsvd"]
        
    def factorize(self, model):
        """
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
        self.W, self.H = self.seed.initialize(self.V, self.rank)
        
    def euclidean_update(self):
        """Update basis and mixture matrix based on euclidean distance multiplicative update rules."""
        self.H = multiply(self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        self.W = multiply(self.W , elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div)) 
        
    def divergence_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        x1 = repmat(self.W.sum(0).T, 1, self.V.shape[1])
        self.H = multiply(self.H, elop(dot(self.W.T, elop(self.V, dot(self.W, self.H), div)), x1, div))
        x2 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), x2, div))
        
    def fro_error(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate.""" 
        return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
    
    def div_error(self):
        """Compute divergence of traget matrix from its NMF estimate."""
        return (multiply(self.V, elop(self.V, dot(self.W, self.H), log)) - self.V + dot(self.W, self.H)).sum()
    
    def conn_error(self):
        pass    
        