import numpy as np

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
                          #. 'euclidean' for classic euclidean distance update equations, 
                          #. 'divergence' for divergence update equations.
                      When specifying model, user can pass 'objective' keyword argument with one of
                      possible values:
                          #. 'euclidean' for standard euclidean distance cost function,
                          #. 'divergence' for divergence of target matrix from NMF estimate cost function (KL),
                          #. 'connectivity' for connectivity matrix changed elements cost function. 
                        Default are 'euclidean' update equations and 'euclidean' cost function. 
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
        self.W, self.H = self.seed.initialize(self.V, self.rank)
        
    def euclidean_update(self):
        """Update basis and mixture matrix based on euclidean distance multiplicative update rules."""
        self.H = np.multiply(self.H, (self.W.T * self.V) / (self.W.T * self.W * self.H))
        self.W = np.multiply(self.W , (self.V * self.H.T) / (self.W * self.H * self.H.T))
        
    def divergence_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        # self.H = np.multiply(self.H / , self.W.T * (self.V / (self.W * self.H)))
        