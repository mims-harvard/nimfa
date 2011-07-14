
import models.mf_fit as mfit
from utils.linalg import *

class Lfnmf(object):
    """
    Fisher Nonnegative Matrix Factorization for learning Local features (LFNMF) [6].
    
    LFNMF is based on nonnegative matrix factorization (NMF), which allows only additive combinations of nonnegative 
    basis components. The NMF bases are spatially global, whereas local bases would be preferred. Li [7] proposed 
    local nonnegative matrix factorization (LNFM) to achieve a localized NMF representation by adding three constraints
    to enforce spatial locality: minimize the number of basis components required to represent target matrix; minimize
    redundancy between different bases by making different bases as orthogonal as possible; maximize the total activity
    on each component, i. e. the total squared projection coefficients summed over all training images. 
    However, LNMF does not encode discrimination information for a classification problem. 
    
    LFNMF can produce both additive and spatially localized basis components as LNMF and it also encodes characteristics of
    Fisher linear discriminant analysis (FLDA). The main idea of LFNMF is to add Fisher constraint to the original NMF. 
    Because the columns of the mixture matrix (H) have a one-to-one correspondence with the columns of the target matrix
    (V), between class scatter of H is maximized and within class scatter of H is minimized. 
    
    Example usages are pattern recognition problems in classification, feature generation and extraction for diagnostic 
    classification purposes, face recognition etc.  
    
    [6] ﻿Wang, Y., et. al. Fisher non-negative matrix factorization for learning local features. Proc. Asian Conf. on Comp. Vision. 2004.                                                                                                                
    [7] ﻿Li, S. Z., et. al. Learning spatially localized, parts-based representation. Proc. of the 2001 IEEE Comp. Soc.
        Conf. on Comp. Vision and Pattern Recognition. CVPR 2001, I-207-I-212. IEEE Comp. Soc. doi: 10.1109/CVPR.2001.990477.
    """

    def __init__(self, params):
        self.aname = "lnmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self, model):
        """
        Compute matrix factorization. 
        
        Return fitted factorization model.
        
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
                cobj = self.objective() if not self.test_conv or iter % self.test_conv == 0 else cobj
                iter += 1
            self.final_obj = cobj
            mffit = mfit.Mf_fit(self)
            if self.callback: self.callback(mffit)
        return mffit
     
    def _is_satisfied(self, pobj, cobj, iter):
        """Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value."""
        if self.max_iters and self.max_iters > iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def update(self):
        """Update basis and mixture matrix."""
        pass
    
    def objective(self):
        """Compute constrained divergence of target matrix from its NMF estimate with additional factors of between
        class scatter and within class scatter of the mixture matrix (H).
        """ 
        pass
    
        