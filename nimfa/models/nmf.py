
"""
    #####################
    Nmf (``models.nmf``)
    #####################
"""

import nimfa.utils.utils as utils
from nimfa.utils.linalg import *
from nimfa.models import mf_track
from nimfa.methods import seeding


class Nmf(object):
    """
    This class defines a common interface / model to handle NMF models in a generic way.
    
    It contains definitions of the minimum set of generic methods that are used in 
    common computations and matrix factorizations. Besides it contains some quality and performance measures 
    about factorizations. 
    
    .. attribute:: rank
    
        Factorization rank
        
    .. attribute:: V
        
        Target matrix, the matrix for the MF method to estimate. The columns of target matrix V are called samples, the rows of target
        matrix V are called features. Some algorithms (e. g. multiple NMF) specify more than one target matrix. In that case
        target matrices are passed as tuples. Internally, additional attributes with names following Vn pattern are created, 
        where n is the consecutive index of target matrix. Zero index is omitted (there are V, V1, V2, V3, etc. matrices and
        then H, H1, H2, etc. and W, W1, W2, etc. respectively - depends on the algorithm). 
        
    .. attribute:: seed
    
        Method to seed the computation of a factorization
        
    .. attribute:: method
    
        The algorithm to use to perform MF on target matrix
        
    .. attribute:: n_run 
    
        The number of runs of the algorithm
        
    .. attribute:: n_iter
    
        The number of iterations performed
        
    .. attribute:: final_obj
    
        Final value (of the last performed iteration) of the objective function
        
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
        
        :param params: MF runtime and algorithm parameters and options. For detailed explanation
           of the general model parameters see :mod:`mf_run`. For algorithm specific
           model options see documentation of chosen factorization method.
        :type params: `dict`
        """
        self.__dict__.update(params)
        # do not copy target and factor matrices into the program
        if sp.isspmatrix(self.V):
            self.V = self.V.tocsr().astype('d')
        else:
            self.V = np.asmatrix(self.V) if self.V.dtype == np.dtype(
                float) else np.asmatrix(self.V, dtype='d')
        if self.V1 is not None:
            if sp.isspmatrix(self.V1):
                self.V1 = self.V1.tocsr().astype('d')
            else:
                self.V1 = np.asmatrix(self.V1) if self.V1.dtype == np.dtype(
                    float) else np.asmatrix(self.V1, dtype='d')
        if self.W is not None:
            if sp.isspmatrix(self.W):
                self.W = self.W.tocsr().astype('d')
            else:
                self.W = np.asmatrix(self.W) if self.W.dtype == np.dtype(
                    float) else np.asmatrix(self.W, dtype='d')
        if self.H is not None:
            if sp.isspmatrix(self.H):
                self.H = self.H.tocsr().astype('d')
            else:
                self.H = np.asmatrix(self.H) if self.H.dtype == np.dtype(
                    float) else np.asmatrix(self.H, dtype='d')
        if self.H1 is not None:
            if sp.isspmatrix(self.H1):
                self.H1 = self.H1.tocsr().astype('d')
            else:
                self.H1 = np.asmatrix(self.H1) if self.H1.dtype == np.dtype(
                    float) else np.asmatrix(self.H1, dtype='d')
        self._compatibility()

    def __call__(self):
        """Run the specified MF algorithm."""
        return self.factorize()

    def basis(self):
        """Return the matrix of basis vectors. See NMF specific model."""

    def target(self, idx=None):
        """Return the target matrix. See NMF specific model."""

    def coef(self, idx=None):
        """
        Return the matrix of mixture coefficients. See NMF specific model.
        
        :param idx: Used in the multiple NMF model. In factorizations following standard
            NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """

    def fitted(self, idx=None):
        """
        Compute the estimated target matrix according to the NMF model. See NMF specific model.

        :param idx: Used in the multiple NMF model. In factorizations following standard
            NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """

    def distance(self, metric='euclidean', idx=None):
        """
        Return the loss function value. See NMF specific model.
        
        :param distance: Specify distance metric to be used. Possible are Euclidean
            and Kullback-Leibler (KL) divergence. Strictly, KL is not a metric.
        :type distance: `str` with values 'euclidean' or 'kl'

        :param idx: Used in the multiple NMF model. In factorizations following
            standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """

    def residuals(self, idx=None):
        """
        Compute residuals between the target matrix and its NMF estimate. See NMF specific model.
        
        :param idx: Used in the multiple NMF model. In factorizations following standard
            NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """

    def connectivity(self, H=None, idx=None):
        """
        Compute the connectivity matrix for the samples based on their mixture coefficients. 
        
        The connectivity matrix C is a symmetric matrix which shows the shared membership of the samples: entry C_ij is 1 iff sample i and 
        sample j belong to the same cluster, 0 otherwise. Sample assignment is determined by its largest metagene expression value. 
        
        Return connectivity matrix.
        
        :param idx: Used in the multiple NMF model. In factorizations following
            standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        H = self.coef(idx) if H is None else H
        _, idx = argmax(H, axis=0)
        mat1 = repmat(idx, V.shape[1], 1)
        mat2 = repmat(idx.T, 1, V.shape[1])
        conn = elop(mat1, mat2, eq)
        if sp.isspmatrix(conn):
            return conn.__class__(conn, dtype='d')
        else:
            return np.mat(conn, dtype='d')

    def consensus(self, idx=None):
        """
        Compute consensus matrix as the mean connectivity matrix across multiple runs of the factorization. It has been
        proposed by [Brunet2004]_ to help visualize and measure the stability of the clusters obtained by NMF.
        
        Tracking of matrix factors across multiple runs must be enabled for computing consensus matrix. For results
        of a single NMF run, the consensus matrix reduces to the connectivity matrix.
        
        :param idx: Used in the multiple NMF model. In factorizations following
            standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        if self.track_factor:
            if sp.isspmatrix(V):
                cons = V.__class__((V.shape[1], V.shape[1]), dtype=V.dtype)
            else:
                cons = np.mat(np.zeros((V.shape[1], V.shape[1])))
            for i in range(self.n_run):
                cons += self.connectivity(
                    H=self.tracker.get_factor(i).H, idx=idx)
            return sop(cons, self.n_run, div)
        else:
            return self.connectivity(H=self.coef(idx), idx=idx)

    def dim(self, idx=None):
        """
        Return triple containing the dimension of the target matrix and matrix factorization rank.
        
        :param idx: Used in the multiple NMF model. In factorizations following
            standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        return (V.shape[0], V.shape[1], self.rank)

    def entropy(self, membership=None, idx=None):
        """
        Compute the entropy of the NMF model given a priori known groups of
        samples [Park2007]_.
        
        The entropy is a measure of performance of a clustering method in
        recovering classes defined by a list a priori known (true class labels).
        
        Return the real number. The smaller the entropy, the better the
        clustering performance.
        
        :param membership: Specify known class membership for each sample. 
        :type membership: `list`

        :param idx: Used in the multiple NMF model. In factorizations following
           standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        if not membership:
            raise utils.MFError(
                "Known class membership for each sample is not specified.")
        n = V.shape[1]
        mbs = self.predict(what="samples", prob=False, idx=idx)
        dmbs, dmembership = {}, {}
        [dmbs.setdefault(mbs[i], set()).add(i) for i in range(len(mbs))]
        [dmembership.setdefault(membership[i], set()).add(i)
         for i in range(len(membership))]
        return -1. / (n * log(len(dmembership), 2)) * sum(sum(len(dmbs[k].intersection(dmembership[j])) *
                                                              log(len(dmbs[k].intersection(dmembership[j])) / float(len(dmbs[k])), 2) for j in dmembership) for k in dmbs)

    def predict(self, what='samples', prob=False, idx=None):
        """
        Compute the dominant basis components. The dominant basis component is
        computed as the row index for which the entry is the maximum within the column.
        
        If ``prob`` is not specified, list is returned which contains computed index
        for each sample (feature). Otherwise tuple is returned where first element
        is a list as specified before and second element is a list of associated
        probabilities, relative contribution of the maximum entry within each column. 
        
        :param what: Specify target for dominant basis components computation. Two values
           are possible, 'samples' or 'features'. When what='samples' is specified,
           dominant basis component for each sample is determined based on its associated
           entries in the mixture coefficient matrix (H). When what='features' computation
           is performed on the transposed basis matrix (W.T).
        :type what: `str`

        :param prob: Specify dominant basis components probability inclusion.
        :type prob: `bool` equivalent

        :param idx: Used in the multiple NMF model. In factorizations following
           standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        X = self.coef(idx) if what == "samples" else self.basis(
        ).T if what == "features" else None
        if X is None:
            raise utils.MFError(
                "Dominant basis components can be computed for samples or features.")
        eX, idxX = argmax(X, axis=0)
        if not prob:
            return idxX
        sums = X.sum(axis=0)
        prob = [e / (sums[0, s] + 1e-5) for e, s in zip(eX, list(range(X.shape[1])))]
        return idxX, prob

    def evar(self, idx=None):
        """
        Compute the explained variance of the NMF estimate of the target matrix.
        
        This measure can be used for comparing the ability of models for accurately
        reproducing the original target matrix. Some methods specifically aim at
        minimizing the RSS and maximizing the explained variance while others not, which
        one should note when using this measure. 
        
        :param idx: Used in the multiple NMF model. In factorizations following
           standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        return 1. - self.rss(idx=idx) / multiply(V, V).sum()

    def score_features(self, idx=None):
        """
        Score features in terms of their specificity to the basis vectors [Park2007]_.
        
        A row vector of the basis matrix (W) indicates contributions of a feature
        to the r (i.e. columns of W) latent components. It might be informative to
        investigate features that have strong component-specific membership values
        to the latent components.
        
        Return array with feature scores. Feature scores are real-valued from interval [0,1].
        Higher value indicates greater feature specificity.

        :param idx: Used in the multiple NMF model. In standard NMF model or nonsmooth NMF model
           ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        W = self.basis()
        scores = np.zeros(W.shape[0])
        for f in range(W.shape[0]):
            # probability that the i-th feature contributes to q-th basis vector.
            prob = W[f, :] / (W[f, :].sum() + np.finfo(W.dtype).eps)
            prob = prob.todense() if sp.isspmatrix(prob) else prob
            scores[f] = np.dot(prob, np.log2(prob + np.finfo(W.dtype).eps).T)
        scores = 1. + 1. / np.log2(W.shape[1]) * scores
        return scores

    def select_features(self, idx=None):
        """
        Compute the most basis-specific features for each basis vector [Park2007]_.
        
        [Park2007]_ scoring schema and feature selection method is used. The features are
        first scored using the :func:`score_features`. Then only the features that fulfill
        both the following criteria are retained:

        #. score greater than u + 3s, where u and s are the median and the median
           absolute deviation (MAD) of the scores, resp.,
        #. the maximum contribution to a basis component (i.e the maximal value in
           the corresponding row of the basis matrix (W)) is larger
           than the median of all contributions (i.e. of all elements of basis matrix (W)).
        
        Return a boolean array indicating whether features were selected.
        
        :param idx: Used in the multiple NMF model. In standard NMF model or nonsmooth NMF
           model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        scores = self.score_features(idx=idx)
        th = np.median(scores) + 3 * np.median(abs(scores - np.median(scores)))
        sel = scores > th
        W = self.basis()
        m = np.median(W.toarray() if sp.isspmatrix(W) else W.tolist())
        sel = np.array([sel[i] and np.max(W[i, :].toarray() if sp.isspmatrix(W) else W[i,:]) > m
               for i in range(W.shape[0])])
        return sel

    def purity(self, membership=None, idx=None):
        """
        Compute the purity given a priori known groups of samples [Park2007]_.
        
        The purity is a measure of performance of a clustering method in recovering
        classes defined by a list a priori known (true class labels).
        
        Return the real number in [0,1]. The larger the purity, the better the
        clustering performance.
        
        :param membership: Specify known class membership for each sample. 
        :type membership: `list`

        :param idx: Used in the multiple NMF model. In factorizations following
           standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        V = self.target(idx)
        if not membership:
            raise utils.MFError(
                "Known class membership for each sample is not specified.")
        n = V.shape[1]
        mbs = self.predict(what="samples", prob=False, idx=idx)
        dmbs, dmembership = {}, {}
        [dmbs.setdefault(mbs[i], set()).add(i) for i in range(len(mbs))]
        [dmembership.setdefault(membership[i], set()).add(i)
         for i in range(len(membership))]
        return 1. / n * sum(max(len(dmbs[k].intersection(dmembership[j])) for j in dmembership) for k in dmbs)

    def rss(self, idx=None):
        """
        Compute Residual Sum of Squares (RSS) between NMF estimate and
        target matrix [Hutchins2008]_.
        
        This measure can be used to estimate optimal factorization rank.
        [Hutchins2008]_ suggested to choose the first value where the RSS curve
        presents an inflection point. [Frigyesi2008]_ suggested to use the
        smallest value at which the decrease in the RSS is lower than the
        decrease of the RSS obtained from random data.
        
        RSS tells us how much of the variation in the dependent variables our
        model did not explain.
        
        Return real value.
        
        :param idx: Used in the multiple NMF model. In factorizations following
           standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        X = self.residuals(idx=idx)
        return multiply(X, X).sum()

    def sparseness(self, idx=None):
        """
        Compute sparseness of matrix (basis vectors matrix, mixture coefficients) [Hoyer2004]_.

        Sparseness of a vector quantifies how much energy is packed into its components.
        The sparseness of a vector is a real number in [0, 1], where sparser vector
        has value closer to 1. Sparseness is 1 iff the vector contains a single
        nonzero component and is equal to 0 iff all components of the vector are equal.
        
        Sparseness of a matrix is mean sparseness of its column vectors.
        
        Return tuple that contains sparseness of the basis and mixture coefficients matrices. 
        
        :param idx: Used in the multiple NMF model. In standard NMF model or nonsmooth NMF
           model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        def sparseness(x):
            eps = np.finfo(x.dtype).eps if 'int' not in str(x.dtype) else 1e-9
            x1 = sqrt(x.shape[0]) - (abs(x).sum() + eps) / \
                (sqrt(multiply(x, x).sum()) + eps)
            x2 = sqrt(x.shape[0]) - 1
            return x1 / x2
        W = self.basis()
        H = self.coef(idx)
        spars_W = np.mean([sparseness(W[:, i]) for i in range(W.shape[1])])
        spars_H = np.mean([sparseness(H[:, i]) for i in range(H.shape[1])])
        return spars_W, spars_H

    def coph_cor(self, idx=None):
        """
        Compute cophenetic correlation coefficient of consensus matrix, generally obtained from multiple NMF runs. 
        
        The cophenetic correlation coefficient is measure which indicates the dispersion of the consensus matrix and is based 
        on the average of connectivity matrices. It measures the stability of the clusters obtained from NMF. 
        It is computed as the Pearson correlation of two distance matrices: the first is the distance between samples induced by the 
        consensus matrix; the second is the distance between samples induced by the linkage used in the reordering of the consensus 
        matrix [Brunet2004]_.
        
        Return real number. In a perfect consensus matrix, cophenetic correlation equals 1. When the entries in consensus matrix are
        scattered between 0 and 1, the cophenetic correlation is < 1. We observe how this coefficient changes as factorization rank 
        increases. We select the first rank, where the magnitude of the cophenetic correlation coefficient begins to fall [Brunet2004]_.
        
        :param idx: Used in the multiple NMF model. In factorizations following standard NMF model or nonsmooth NMF model
                    :param:`idx` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        A = self.consensus(idx=idx)
        # upper diagonal elements of consensus
        avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                        for j in range(i + 1, A.shape[1])])
        # consensus entries are similarities, conversion to distances
        Y = 1 - avec
        Z = linkage(Y, method='average')
        # cophenetic correlation coefficient of a hierarchical clustering
        # defined by the linkage matrix Z and matrix Y from which Z was
        # generated
        return cophenet(Z, Y)[0]

    def dispersion(self, idx=None):
        """
        Compute dispersion coefficient of consensus matrix
        
        Dispersion coefficient [Park2007]_ measures the reproducibility of clusters obtained
        from multiple NMF runs.
        
        Return the real value in [0,1]. Dispersion is 1 for a perfect consensus matrix and
        has value in [0,0] for a scattered consensus matrix.

        :param idx: Used in the multiple NMF model. In standard NMF model or nonsmooth NMF
           model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        C = self.consensus(idx=idx)
        dispersion = np.sum(4 * np.multiply(C - 0.5, C - 0.5)) / C.size
        return dispersion

    def estimate_rank(self, rank_range=range(30, 51), n_run=10, idx=0, what='all'):
        """
        Choosing factorization parameters carefully is vital for success of a factorization.
        However, the most critical parameter is factorization rank. This method tries
        different values for ranks, performs factorizations, computes some quality
        measures of the results and chooses the best value according to [Brunet2004]_
        and [Hutchins2008]_.
        
        .. note:: The process of rank estimation can be lengthy.   
        
        .. note:: Matrix factors are tracked during rank estimation. This is needed
           for computing cophenetic correlation coefficient.
        
        Return a `dict` (keys are values of rank from range, values are `dict`s of measures)
        of quality measures for each value in rank's range. This can be passed to the
        visualization model, from which estimated rank can be established.
        
        :param rank_range: Range of factorization ranks to try. Default is ``range(30, 51)``.
        :type rank_range: list or tuple like range of `int`

        :param n_run: The number of runs to be performed for each value in range. Default is 10.  
        :type n_run: `int`

        :param what: Specify quality measures of the results computed for each rank.
           By default, summary of the fitted factorization model is computed. Instead,
           user can supply list of strings that matches some of the following quality measures:
                     
             * `sparseness`
             * `rss`
             * `evar`
             * `residuals`
             * `connectivity`
             * `dispersion`
             * `cophenetic`
             * `consensus`
             * `euclidean`
             * `kl`
        :type what: list or tuple like of `str`

        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple
           NMF model. Default is 0 (first coefficient matrix).
        :type idx: `str` or `int`
        """
        self.n_run = n_run
        self.track_factor = True
        self.tracker = mf_track.Mf_track()

        def _measures(measure):
            return {
                'sparseness': fctr.fit.sparseness,
                'rss': fctr.fit.rss,
                'evar': fctr.fit.evar,
                'residuals': fctr.fit.residuals,
                'connectivity': fctr.fit.connectivity,
                'dispersion': fctr.fit.dispersion,
                'cophenetic': fctr.fit.coph_cor,
                'consensus': fctr.fit.consensus}[measure]
        summaries = {}
        for rank in rank_range:
            self.rank = rank
            fctr = self()
            if what == 'all':
                summaries[rank] = fctr.summary(idx)
            else:
                summaries[rank] = {
                    'rank': fctr.fit.rank,
                    'n_iter': fctr.fit.n_iter,
                    'n_run': fctr.fit.n_run}
                for measure in what:
                    if measure == 'euclidean':
                        summaries[rank][measure] = fctr.distance(
                            metric='euclidean', idx=idx)
                    elif measure == 'kl':
                        summaries[rank][measure] = fctr.distance(
                            metric='kl', idx=idx)
                    else:
                        summaries[rank][measure] = _measures(
                            measure)(idx=idx)
        return summaries

    def _compatibility(self):
        """
        Check if chosen seeding method is compatible with chosen factorization
        method or fixed initialization is passed.

        :param mf_model: The underlying initialized model of matrix factorization.
        :type mf_model: Class inheriting :class:`models.nmf.Nmf`
        """
        W = self.basis()
        H = self.coef(0)
        H1 = self.coef(1) if self.model_name == 'mm' else None
        if self.seed is None and W is None and H is None and H1 is None:
            self.seed = None if "none" in self.aseeds else "random"
        if W is not None and H is not None:
            if self.seed is not None and self.seed is not "fixed":
                raise utils.MFError("Initial factorization is fixed.")
            else:
                self.seed = seeding.fixed.Fixed()
                self.seed._set_fixed(W=W, H=H, H1=H1)
        self.__is_smdefined()
        self.__compatibility()

    def __is_smdefined(self):
        """Check if MF and seeding methods are well defined."""
        if isinstance(self.seed, str):
            if self.seed in seeding.methods:
                self.seed = seeding.methods[self.seed]()
            else:
                raise utils.MFError("Unrecognized seeding method.")
        else:
            if not str(self.seed).lower() in seeding.methods:
                raise utils.MFError("Unrecognized seeding method.")

    def __compatibility(self):
        """Check if MF model is compatible with the seeding method."""
        if not str(self.seed).lower() in self.aseeds:
            raise utils.MFError("MF model is incompatible with the seeding method.")
