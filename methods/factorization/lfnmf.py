from math import log, sqrt
from operator import div

import models.nmf_std as mstd
import models.mf_fit as mfit
import models.mf_track as mtrack
from utils.linalg import *

class Lfnmf(mstd.Nmf_std):
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
    
    [6] Wang, Y., et. al., (2004). Fisher non-negative matrix factorization for learning local features. Proc. Asian Conf. on Comp. Vision. 2004.    
    [7] Li, S. Z., et. al., (2001). Learning spatially localized, parts-based representation. Proc. of the 2001 IEEE Comp. Soc.
        Conf. on Comp. Vision and Pattern Recognition. CVPR 2001, I-207-I-212. IEEE Comp. Soc. doi: 10.1109/CVPR.2001.990477.
    """

    def __init__(self, **params):
        """
        For detailed explanation of the general model parameters see :mod:`mf_methods`.
        
        The following are algorithm specific model options which can be passed with values as keyword arguments.
        
        :param alpha: Parameter :param:`alpha` is weight used to minimize within class scatter and maximize between class scatter of the 
                      encoding mixture matrix. The objective function is the constrained divergence, which is the standard Lee's divergence
                      rule with added terms :param:`alpha` * S_w - :param:`alpha` * S_h, where S_w and S_h are within class and between class
                      scatter, respectively. It should be nonnegative. Default value is 0.01.
        :type alpha: `float`
        """
        self.name = "lfnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        mstd.Nmf_std.__init__(self, params)
        
    def factorize(self):
        """
        Compute matrix factorization. 
        
        Return fitted factorization model.
        """
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
                if self.track_error:
                    self.tracker._track_error(self.residuals())
            if self.callback:
                self.final_obj = cobj
                mffit = mfit.Mf_fit(self) 
                self.callback(mffit)
            if self.track_factor:
                self.tracker._track_factor(W = self.W.copy(), H = self.H.copy())
        
        self.n_iter = iter - 1
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
    
    def _set_params(self):
        self.alpha = self.options.get('alpha', 0.01) 
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mtrack.Mf_track() if self.track_factor and self.n_run > 1 or self.track_error else None
    
    def update(self):
        """Update basis and mixture matrix."""
        _, idxH = argmax(self.H, axis = 0)
        c2m, avgs = self._encoding(idxH)
        C = len(c2m)
        # update mixture matrix H
        for k in xrange(self.H.shape[0]):
            for l in xrange(self.H.shape[1]):
                cl = idxH[0, l]
                ni = len(c2m[cl])
                #b = 4. / (C * C * ni) * sum(avgs[k][j, 0] - avgs[k][cl, 0] + self.H[k,l] / ni for j in xrange(self.H.shape[0])) - 2 * avgs[k][cl, 0] / (ni * C) + 1.
                b = 4. / (C * C * ni) * sum(avgs[j][k, 0] - avgs[cl][k, 0] + self.H[k,l] / ni for j in c2m) - 2 * avgs[cl][k, 0] / (ni * C) + 1.
                self.H[k, l] = -b + sqrt(b**2 + 4. * self.H[k, l] * sum(self.V[i, l] * self.W[i, k] / dot(self.W[i, :], self.H[:, l])[0, 0] 
                              for i in xrange(self.V.shape[0])) * (2. / ((ni + 2) * C) - 4. / ((ni + 2)**2 * C)))
        # update basis matrix W
        W1 = repmat(self.H.sum(1).T, self.V.shape[0], 1)
        self.W = multiply(self.W, elop(dot(elop(self.V, dot(self.W, self.H), div), self.H.T), W1, div))
        W2 = repmat(self.W.sum(0), self.V.shape[0], 1)
        self.W = elop(self.W, W2, div)
        # update between class and within class scatter
        self.Sw = 1. / C * sum(1. / len(c2m[i]) * sum(dot((self.H[:, c2m[i][j]] - avgs[i]).T, self.H[:, c2m[i][j]] - avgs[i]) 
                  for j in xrange(len(c2m[i]))) for i in c2m)[0, 0]
        self.Sb = 1. / (C * C) * sum(dot((avgs[i] - avgs[j]).T, avgs[i] - avgs[j]) for i in c2m for j in c2m)[0, 0]
         
    def _encoding(self, idxH):
        """Compute class membership and mean class value of encoding (mixture) matrix H."""
        c2m = {}
        avgs = {}
        for i in xrange(idxH.shape[1]):
            # group columns of encoding matrix H by class membership
            c2m.setdefault(idxH[0, i], [])
            c2m[idxH[0, i]].append(i)
            # compute mean value of class idx in encoding matrix H
            avgs.setdefault(idxH[0, i], np.matrix(np.zeros((self.rank, 1))))
            avgs[idxH[0, i]] += self.H[:, i]
        for k in avgs:
            avgs[k] /= len(c2m[k])
        return c2m, avgs
    
    def objective(self):
        """
        Compute constrained divergence of target matrix from its NMF estimate with additional factors of between
        class scatter and within class scatter of the mixture matrix (H).
        """ 
        Va = dot(self.W, self.H)
        return (multiply(self.V, elop(self.V, Va, np.log)) - self.V + Va).sum() + self.alpha * self.Sw - self.alpha * self.Sb

    def __str__(self):
        return self.name
        