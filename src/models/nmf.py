from math import sqrt, log
from operator import eq, div
from scipy.cluster.hierarchy import linkage, cophenet
import warnings

import methods.seeding as seed
import utils.utils as utils
from utils.linalg import *

class Nmf(object):
    """
    This class defines a common interface / model to handle NMF models in generic way.
    
    It contains definitions of the minimum set of generic methods that are used in 
    common computations and matrix factorizations. Besides it contains some quality and performance measures 
    about factorizations. 
    
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
        
    .. attribute:: max_iter
    
        Maximum number of factorization iterations
        
    .. attribute:: min_residuals
    
        Minimal required improvement of the residuals from the previous iteration
        
    .. attribute:: test_conv
        
        Indication how often convergence test is done.
    """

    def __init__(self, params):
        """
        Construct generic factorization model.
        
        :param params: MF runtime and algorithm parameters and options. For detailed explanation of the general model 
                       parameters see :mod:`mf_methods`. For algorithm specific model options see documentation of chosen
                       factorization method. 
        :type params: `dict`
        """
        self.__dict__.update(params)
        
    def _is_smdefined(self):
        """Check if MF and seeding methods are well defined."""
        if isinstance(self.seed, str):
            if self.seed in seed.methods:
                self.seed = seed.methods[self.seed]()
            else: raise utils.MFError("Unrecognized seeding method.")
        else:
            if not str(self.seed).lower() in seed.methods:
                raise utils.MFError("Unrecognized seeding method.")
        self._compatibility()
        
    def _compatibility(self):
        """Check if MF model is compatible with the seeding method."""
        if not str(self.seed).lower() in self.aseeds:
            raise utils.MFError("MF model is incompatible with chosen seeding method.")   
    
    def run(self):
        """Run the specified MF algorithm."""
        return self.method.factorize()
        
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
        
    def connectivity(self, H = None):
        """
        Compute the connectivity matrix for the samples based on their mixture coefficients. 
        
        The connectivity matrix C is a symmetric matrix which shows the shared membership of the samples: entry C_ij is 1 iff sample i and 
        sample j belong to the same cluster, 0 otherwise. Sample assignment is determined by its largest metagene expression value. 
        
        Return connectivity matrix.
        """
        H = self.coef() if not H else H
        _, idx = argmax(H, axis = 0)
        mat1 = repmat(idx, self.V.shape[1], 1)
        mat2 = repmat(idx.T, 1, self.V.shape[1])
        return elop(mat1, mat2, eq)
    
    def consensus(self):
        """
        Compute consensus matrix as the mean connectivity matrix across multiple runs of the factorization. It has been
        proposed by Brunet et. all (2004) to help visualize and measure the stability of the clusters obtained by NMF.
        
        Tracking of matrix factors across multiple runs must be enabled for computing consensus matrix. For results
        of a single NMF run, the consensus matrix reduces to the connectivity matrix.
        """
        if not self.tracker:
            warnings.warn("Tracking matrix factors across runs is not enabled. Connectivity matrix will be computed.", UserWarning, 2)
        cons = np.matrix(np.zeros((self.V.shape[1], self.V.shape[1])))
        for i in xrange(self.n_run):
            cons += self.connectivity(self.tracker.get(i).H)
        return sop(cons, self.n_run, div)
        
    def dim(self):
        """Return triple containing the dimension of the target matrix and matrix factorization rank."""
        return (self.V.shape[0], self.V.shape[1], self.rank)
    
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
        X = self.coef() if what == "samples" else self.basis().T if what == "features" else None
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
        
    def score_features(self):
        """
        Compute the score for each feature that represents its specificity to one of the basis vector (Kim, Park, 2007).
        
        A row vector of the basis matrix (W) indicates the contributions of a gene to the r (i.e. columns of W) biological pathways or
        processes. As genes can participate in more than one biological process, it is beneficial to investigate genes that have relatively 
        large coefficient in each biological process. 
        
        Return the list containing score for each feature. The feature scores are real values in [0,1]. The higher the feature score the more 
        basis-specific the corresponding feature.  
        """
        W = self.basis()
        def prob(i, q):
            """Return probability that the i-th feature contributes to the basis q."""
            return W[i, q] / sum(W[i, :])
        res = []
        for f in xrange(W.shape[0]):
            res.append(1 + 1 / log(W.shape[1], 2) * sum( prob(f, q) * log(prob(f,q), 2) for q in W.shape[1]))
        return res
    
    def select_features(self):
        """
        Compute the most basis-specific features for each basis vector (Kim, Park, 2007).
        
        (Kim, Park, 2007) scoring schema and feature selection method is used. The features are first scored using the :func:`score_features`.
        Then only the features that fulfill both the following criteria are retained:
        #. score greater than u + 3s, where u and s are the median and the median absolute deviation (MAD) of the scores, resp.,
        #. the maximum contribution to a basis component (i.e the maximal value in the corresponding row of the basis matrix (W)) is larger 
           than the median of all contributions (i.e. of all elements of basis matrix (W)).
        
        Return list of retained features' indices.  
        """
        scores = self.score_features()
        u = np.median(scores)
        s = np.median(abs(scores - u))
        res = [i for i in xrange(len(scores)) if scores[i] > u + 3. * s]
        W = self.basis()
        m = np.median(W.toarray() if sp.isspmatrix(W) else W.tolist())
        return [i for i in res if np.max(W[:, i].toarray() if sp.isspmatrix(W) else W[:, i]) > m]
    
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
        the first value where the RSS curve presents an inflection point. (Frigyesi and Hoglund, 2008) suggested to use the 
        smallest value at which the decrease in the RSS is lower than the decrease of the RSS obtained from random data. 
        
        Return real value
        """
        X = self.residuals()
        xX = self.V - X 
        return multiply(xX, xX).sum()
    
    def sparseness(self):
        """
        Compute sparseness of matrix (basis vectors matrix, mixture coefficients) (Hoyer, 2004). This sparseness 
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
        
        The cophenetic correlation coefficient is measure which indicates the dispersion of the consensus matrix and is based 
        on the average of connectivity matrices. It measures the stability of the clusters obtained from NMF. 
        It is computed as the Pearson correlation of two distance matrices: the first is the distance between samples induced by the 
        consensus matrix; the second is the distance between samples induced by the linkage used in the reordering of the consensus 
        matrix (Brunet, 2004).
        
        Return real number. In a perfect consensus matrix, cophenetic correlation equals 1. When the entries in consensus matrix are
        scattered between 0 and 1, the cophenetic correlation is < 1. We observe how this coefficient changes as factorization rank 
        increases. We select the first rank, where the magnitude of the cophenetic correlation coefficient begins to fall (Brunet, 2004).  
        """
        A = self.consensus()
        # upper diagonal elements of consensus
        avec = np.array([A[i,j] for i in xrange(A.shape[0] - 1) for j in xrange(i + 1, A.shape[1])])
        # consensus entries are similarities, conversion to distances
        Y = 1 - avec
        Z = linkage(Y, method = 'average')
        # cophenetic correlation coefficient of a hierarchical clustering defined by the linkage matrix Z and matrix Y from which Z was generated
        return cophenet(Z, Y)[0]
    
    def dispersion(self):
        """
        Compute the dispersion coefficient of consensus matrix, generally obtained from multiple
        NMF runs.
        
        The dispersion coefficient is based on the average of connectivity matrices (Kim, Park, 2007). It 
        measures the reproducibility of the clusters obtained from multiple NMF runs.
        
        Return the real value in [0,1]. Dispersion is 1 iff for a perfect consensus matrix, where all entries are 0 or 1.
        A perfect consensus matrix is obtained only when all the connectivity matrices are the same, meaning that
        the algorithm gave the same clusters at each run.  
        """
        C = self.consensus()
        return sum(sum(4 * (C[i,j] - 0.5)**2 for j in xrange(C.shape[1])) for i in xrange(C.shape[0]))
    