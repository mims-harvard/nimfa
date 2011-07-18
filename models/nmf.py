from math import sqrt, log

import methods.mf as mf
import methods.seeding as seed
import utils.utils as utils
from utils.linalg import *

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
        
    def coef(self):
        """Return the matrix of mixture coefficients."""
    
    def fitted(self):
        """Compute the estimated target matrix according to the NMF model."""
    
    def distance(self, metric):
        """Return the loss function value."""
        
    def residuals(self):
        """Compute residuals between the target matrix and its NMF estimate."""
        
    def connectivity(self):
        """Compute the connectivity matrix associated to the clusters based on NM factorization."""
        pass
    
    def consensus(self):
        """Compute consensus matrix as the mean connectivity matrix across the runs."""
        pass
        
    def dim(self):
        """Return triple containing the dimension of the target matrix and NM factorization rank."""
        return (self.V.shape, self.rank)
    
    def entropy(self, membership = None):
        """
        Compute the entropy of the NMF model given a priori known groups of samples (Kim, Park, 2007).
        
        The entropy is a measure of performance of a clustering method in recovering classes defined by a list a priori known (true class
        labels). 
        
        Return the real number. The smaller the entropy, the better the clustering performance.
        
        :param membership: Specify known class membership for each sample. 
        :type membership: `list`
        """
        if not membership:
            raise utils.MFError("Known class membership for each sample is not specified.")
        n = self.V.shape[1]
        mbs = self.predict(what = "samples", prob = False)
        dmbs, dmembership = {}, {}
        [dmbs.setdefault(mbs[i], set()).add(i) for i in xrange(len(mbs))]
        [dmembership.setdefault(membership[i], set()).add(i) for i in xrange(len(membership))]
        return -1. / (n * log(len(dmembership), 2)) * sum(sum( len(dmbs[k].intersection(dmembership[j])) * 
               log(len(dmbs[k].intersection(dmembership[j])) / float(len(dmbs[k])), 2) for j in dmembership) for k in dmbs)
        
    def predict(self, what = 'samples', prob = False):
        """
        Compute the dominant basis components. The dominant basis component is computed as the row index for which
        the entry is the maximum within the column. 
        
        If :param:`prob` is not specified, list is returned which contains computed index for each sample (feature). Otherwise
        tuple is returned where first element is a list as specified before and second element is a list of associated
        probabilities, relative contribution of the maximum entry within each column. 
        
        :param what: Specify target for dominant basis components computation. Two values are possible, 'samples' or
                     'features'. When what='samples' is specified, dominant basis component for each sample is determined based
                     on its associated entries in the mixture coefficient matrix (H). When what='features' computation is performed
                     on the transposed basis matrix (W.T). 
        :type what: `str`
        :param prob: Specify dominant basis components probability inclusion. 
        :type prob: `bool` equivalent
        """
        X = self.H if what == "samples" else self.W.T if what == "features" else None
        if not X:
            raise utils.MFError("Dominant basis components can be computed for samples or features.")
        eX, idxX = argmax(X, axis = 0)
        if not prob:
            return idxX
        sums = X.sum(axis = 0)
        prob = [e / sums[0, s] for e, s in zip(eX, X.shape[1])]
        return idxX, prob
    
    def evar(self):
        """
        Compute the explained variance of the NMF estimate of the target matrix.
        
        This measure can be used for comparing the ability of models for accurately reproducing the original target matrix. 
        Some methods specifically aim at minimizing the RSS and maximizing the explained variance while others not, which 
        one should note when using this measure. 
        """
        return 1. - self.rss() / multiply(self.V, self.V).sum()
        
    def feature_score(self):
        """Compute the score for each feature representing specificity to basis vectors (Kim, Park, 2007)."""
        pass
    
    def extract_feature(self):
        """Compute most specific features for basis vectors (Kim, Park, 2007)."""
        pass
    
    def purity(self, membership = None):
        """
        Compute the purity given a priori known groups of samples (Kim, Park, 2007).
        
        The purity is a measure of performance of a clustering method in recovering classes defined by a list a priori known (true class
        labels). 
        
        Return the real number in [0,1]. The larger the purity, the better the clustering performance. 
        
        :param membership: Specify known class membership for each sample. 
        :type membership: `list`
        """
        if not membership:
            raise utils.MFError("Known class membership for each sample is not specified.")
        n = self.V.shape[1]
        mbs = self.predict(what = "samples", prob = False)
        dmbs, dmembership = {}, {}
        [dmbs.setdefault(mbs[i], set()).add(i) for i in xrange(len(mbs))]
        [dmembership.setdefault(membership[i], set()).add(i) for i in xrange(len(membership))]
        return 1. / n * sum(max( len(dmbs[k].intersection(dmembership[j])) for j in dmembership) for k in dmbs)
    
    def rss(self):
        """
        Compute Residual Sum of Squares (RSS) between NMF estimate and target matrix (Hutchins, 2008).
        
        This measure can be used to estimate optimal factorization rank. (Hutchins et. al., 2008) suggested to choose
        the first value where the RSS curve presents an inflection point. (Frigyesi, 2008) suggested to use the 
        smallest value at which the decrease in the RSS is lower than decrease of the RSS obtained from random data. 
        """
        X = self.residuals()
        xX = self.V - X 
        return multiply(xX, xX).sum()
    
    def sparseness(self):
        """
        Compute sparseness of matrix (mixture coefficients, basis vectors matrix) (Hoyer, 2004). This sparseness 
        measure quantifies how much energy of a vector is packed into only few components. The sparseness of a vector
        is a real number in [0, 1]. Sparser vector has value closer to 1. The measure is 1 iff vector contains single
        nonzero component and the measure is equal to 0 iff all components are equal. 
        
        Sparseness of a matrix is the mean sparseness of its column vectors. 
        
        Return tuple that contains sparseness of the basis and mixture coefficients matrices. 
        """
        def sparseness(x):
            x1 = sqrt(x.shape[0]) - abs(x).sum() / sqrt(multiply(x, x).sum())
            x2 = sqrt(x.shape[0]) - 1
            return x1 / x2 
        W = self.basis()
        H = self.coef()
        return np.mean([sparseness(W[:, i]) for i in xrange(W.shape[1])]), np.mean([sparseness(H[:, i]) for i in xrange(H.shape[1])])
        
    def cophcor(self):
        """
        Compute cophenetic correlation coefficient of consensus matrix, generally obtained from multiple NMF runs. 
        
        The cophenetic correlation coefficient is based on the average of connectivity matrices (Brunet, 2004). It
        measures the stability of the clusters obtained from NMF. 
        """
        pass
    
    def dispersion(self):
        """
        Compute the dispersion coefficient of consensus matrix, generally obtained from multiple
        NMF runs.
        
        The dispersion coefficient is based on the average of connectivity matrices (Kim, Park, 2007). It 
        measures the reproducibility of the clusters obtained from NMF. 
        """
        pass
    
    