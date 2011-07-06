
import methods.mf as mf
import methods.seeding as seed
import utils.utils as utils

class Nmf(object):
    '''
    This class defines a common interface to handle NMF models in generic way.
    
    It contains definitions of the minimum set of generic methods that are used in 
    common computations and NM factorizations. 
    
    .. attribute:: rank
    
        Factorization rank
        
    .. attribute:: V
        
        Target matrix. The columns of target matrix V are called samples, the rows of target
        matrix V are called features. 
    '''

    def __init__(self, **params):
        '''
        Constructor
        '''
        self.__dict__.update(params)
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
        self.compatibility()
        
    def compatibility(self):
        """Check if MF model is compatible with MF method and seeding method."""
        if not self.name in self.method.amodels and self.seed.name in self.method.aseeds:
            raise utils.MFError("MF model is incompatible with chosen MF method and seeding method.")   
    
    def run(self):
        """Run the specified MF algorithm."""
        self.method.factorize(self)
        
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
    
    def sparseness(self):
        """Compute average sparseness of mixture coefficients and basis vectors. (Hoyer, 2004)"""
        pass
    
    