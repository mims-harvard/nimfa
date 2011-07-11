import scipy.sparse as sp
import numpy as np

import methods.mf as mf
import methods.seeding as seed
import utils.utils as utils

class Nmf(object):
    '''
    This class defines a common interface / model to handle NMF models in generic way.
    
    It contains definitions of the minimum set of generic methods that are used in 
    common computations and NM factorizations. 
    
    .. attribute:: rank
    
        Factorization rank
        
    .. attribute:: V
        
        Target matrix, the matrix for MF method to estimate. The columns of target matrix V are called samples, the rows of target
        matrix V are called features. 
        
    .. attribute:: seed
    
        Method to seed the computation of a factorization
        
    .. attribute:: method
    
        The algorithm to use to perform MF on target matrix
        
    .. attribute:: n_run 
    
        The number of runs of the algorithm
        
    .. attribute:: callback
    
        A callback function that is called after each run if performing multiple runs 
        
    .. attribute:: options
    
        Runtime / algorithm specific options
        
    .. attribute:: max_iters
    
        Maximum number of factorization iterations
        
    .. attribute:: min_residuals
    
        Minimal required improvement of the residuals from the previous iteration
        
    .. attribute:: test_conv
        
        Indication how often convergence test is done.
    '''

    def __init__(self, **params):
        '''
        Constructor
        '''
        self.__dict__.update(params)
        
    def _is_smdefined(self):
        """Check if MF and seeding methods are well defined."""
        if type(self.method) is str:
            if self.method in mf.methods:
                self.method = mf.methods[self.method]()
            else: raise utils.MFError("Unrecognized MF method.")
        else:
            if not self.method.name in mf.methods:
                raise utils.MFError("Unrecognized MF method.")
        if type(self.seed) is str:
            if self.seed in seed.methods:
                self.seed = seed.methods[self.seed]()
            else: raise utils.MFError("Unrecognized seeding method.")
        else:
            if not self.seed.name in seed.methods:
                raise utils.MFError("Unrecognized seeding method.")
        self._compatibility()
        
    def _compatibility(self):
        """Check if MF model is compatible with MF method and seeding method."""
        if not self.name in self.method.amodels and self.seed.name in self.method.aseeds:
            raise utils.MFError("MF model is incompatible with chosen MF method and seeding method.")   
    
    def run(self):
        """Run the specified MF algorithm."""
        return self.method.factorize(self)
        
    def basis(self):
        """Return the matrix of basis vectors."""
        pass
        
    def coef(self):
        """Return the matrix of mixture coefficients."""
        pass
    
    def fitted(self):
        """Compute the estimated target matrix according to the NMF model."""
        pass
    
    def distance(self):
        """Return the loss function value."""
        pass
        
    def connectivity(self):
        """Compute the connectivity matrix associated to the clusters based on NM factorization."""
        pass
    
    def consensus(self):
        """Compute consensus matrix as the mean connectivity matrix across the runs."""
        pass
        
    def dim(self):
        """Return triple containing the dimension of the target matrix and NM factorization rank."""
        return (self.V.shape, self.rank)
    
    def entropy(self):
        """Compute the entropy of the NMF model given a priori known groups of samples."""
        pass
    
    def evar(self):
        """Compute the explained variance of the NMF estimate of the target matrix."""
        pass
    
    def feature_score(self):
        """Compute the score for each feature representing specificity to basis vectors.
            (Kim, Park, 2007)"""
        pass
    
    def extract_feature(self):
        """Compute most specific features for basis vectors. (Kim, Park, 2007)"""
        pass
    
    def purity(self):
        """Compute the purity given a priori known groups of samples. (Kim, Park, 2007)"""
        pass
    
    def rss(self):
        """Compute Residual Sum of Squares (RSS) between NMF estimate and target (Hutchins, 2008)."""
        pass
    
    def sparseness(self, X):
        """Compute average sparseness of matrix (mixture coefficients, basis vectors matrix). (Hoyer, 2004)"""
        pass
    
    def residuals(self):
        """Compute residuals between the target matrix and its NMF estimate."""
        pass
    
    def cophcor(self):
        """Compute cophenetic correlation coefficient of consensus matrix, generally obtained from multiple
        NMF runs. 
        The cophenetic correlation coefficient is based on the average of connectivity matrices. (Brunet, 2004) It
        measures the stability of the clusters obtained from NMF. 
        """
        pass
    
    def dispersion(self):
        """Compute the dispersion coefficient of consensus matrix, generally obtained from multiple
        NMF runs.
        The dispersion coefficient is based on the average of connectivity matrices. (Kim, Park, 2007) It 
        measures the reproducibility of the clusters obtained from NMF. 
        """
        pass
    
    