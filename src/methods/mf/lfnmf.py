from math import log, sqrt
from operator import div

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
        
        :param model: The underlying model of matrix factorization. Algorithm specific model option is 'alpha'
                      which can be passed with value as keyword argument to the underlying model.
                      Parameter alpha is weight within class scatter and between class scatter of encoding mixture
                      matrix. It should be nonnegative.
        :type model: :class:`models.nmf_std.Nmf_std`
        """
        self.__dict__.update(model.__dict__)
        self._set_params()
        
        for _ in xrange(self.n_run):
            self.W, self.H = self.seed.initialize(self.V, self.rank, self.options)
            self.Sw, self.Sb = 0, 0
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
        if self.max_iters and self.max_iters < iter:
            return False
        if self.min_residuals and iter > 0 and cobj - pobj <= self.min_residuals:
            return False
        if iter > 0 and cobj >= pobj:
            return False
        return True
    
    def _set_params(self):
        self.alpha = self.options['alpha'] if self.options and 'alpha' in self.options else 0.01
    
    def update(self):
        """Update basis and mixture matrix."""
        _, idxH = argmax(self.H, axis = 0)
        cls, avgs = self._encoding(idxH)
        C = len(cls)
        # update mixture matrix H
        for k in xrange(self.H.shape[0]):
            for l in xrange(self.H.shape[1]):
                b = 4 / (C * (C - 1) * len(cls[idxH[0, l]])) * sum(avgs[j][k, 0] - avgs[idxH[0,l]][k,0] + self.H[k,l] / len(cls[idxH[0, l]]) for j in xrange(C)) - 2 * avgs[idxH[0,l]][k, 0] / (len(cls[idxH[0, l]]) * C) + 1    
                self.H[k, l] = -b + sqrt(b**2 + 4 * self.H[k, l] * sum(self.V[i, l] * self.W[i, k] / dot(self.W[i, :], self.H[:, l]) for i in xrange(self.V.shape[0])) 
                                         * (2 / (len(cls[idxH[0, l]]) * C) - 4 / (len(cls[idxH[0, l]])**2 * (C - 1))))
        # update basis matrix W
        w1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), w1, div))
        w2 = repmat(self.W.sum(0), self.V.shape[0], 1)
        self.W = elop(self.W, w2, div)
        self.Sw = 1/C * sum(1/len(cls[i]) * sum(dot((self.H[:, cls[i][j]] - avgs[i]).T, self.H[:, cls[i][j]] - avgs[i]) for j in xrange(len(cls[i]))) for i in cls)
        self.Sb = 1/(C * (C - 1)) * sum( dot((avgs[i] - avgs[j]).T, avgs[i] - avgs[j]) for i in cls for j in cls)
         
    def _encoding(self, idxH):
        """Compute class membership and mean class value of encoding (mixture) matrix H."""
        cls = {}
        avgs = {}
        for i in xrange(idxH.shape[1]):
            # group columns of encoding matrix H by class membership 
            cls[idxH[0, i]] = cls.get(idxH[0, i], []).append(i)
            # compute mean value of class idx in encoding matrix H
            avgs[idxH[0, i]] = avgs.get(idxH[0, i], np.matrix(np.zeros((self.rank, 1)))) + self.H[:, i]
        for k in avgs:
            avgs[k] /= len(cls[k])
        return cls, avgs
    
    def objective(self):
        """Compute constrained divergence of target matrix from its NMF estimate with additional factors of between
        class scatter and within class scatter of the mixture matrix (H).
        """ 
        Va = dot(self.W, self.H)
        return (multiply(self.V, elop(self.V, Va, log)) - self.V + Va).sum() + self.alpha * self.Sw - self.alpha * self.Sb
    
        