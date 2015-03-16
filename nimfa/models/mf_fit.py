
"""
    ##########################
    Mf_fit (``models.mf_fit``)
    ##########################
"""


class Mf_fit():
    """
    Base class for storing MF results.
    
    It contains generic functions and structure for handling the results of MF algorithms. 
    It contains a slot with the fitted MF model and data about parameters and methods used for
    factorization.  
    
    The purpose of this class is to handle in a generic way the results of MF algorithms and acts as a wrapper for the 
    fitted model. Its attribute attribute:: fit contains the fitted model and its configuration 
    can therefore be used directly in following calls to factorization.    
    
    .. attribute:: fit
        
        The fitted NMF model
    
    .. attribute:: algorithm 

        NMF method of factorization.

    .. attribute:: n_iter

        The number of iterations performed.

    .. attribute:: n_run

        The number of NMF runs performed.

    .. attribute:: seeding

        The seeding method used to seed the algorithm that fitted NMF model. 

    .. attribute:: options

        Extra parameters specific to the algorithm used to fit the model.
    """
    def __init__(self, fit):
        """
        Construct fitted factorization model. 
        
        :param fit: Matrix factorization algorithm model. 
        :type fit: class from methods.mf package
        """
        self.fit = fit
        self.algorithm = str(self.fit)
        self.n_iter = self.fit.n_iter
        self.n_run = self.fit.n_run
        self.seeding = str(self.fit.seed)
        self.options = self.fit.options

    def basis(self):
        """Return the matrix of basis vectors."""
        return self.fit.basis()

    def coef(self, idx=None):
        """
        Return the matrix of mixture coefficients.
        
        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        return self.fit.coef(idx)

    def distance(self, metric=None, idx=None):
        """
        Return the loss function value. If metric is not supplied, final objective function value associated to the MF algorithm is returned.
        
        :param metric: Measure of distance between a target matrix and a MF estimate. Metric 'kl' and 'euclidean' 
           are defined.
        :type metric: 'str'

        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        if metric == None:
            return self.fit.final_obj
        else:
            return self.fit.distance(metric, idx)

    def fitted(self, idx=None):
        """
        Compute the estimated target matrix according to the MF algorithm model.
        
        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        return self.fit.fitted(idx)

    def fit(self):
        """Return the MF algorithm model."""
        return self.fit

    def summary(self, idx=None):
        """
        Return generic set of measures to evaluate the quality of the factorization.
        
        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        if idx == 'coef':
            idx = 0
        if idx == 'coef1':
            idx = 1
        if hasattr(self, 'summary_data'):
            if idx not in self.summary_data:
                self.summary_data[idx] = self._compute_summary(idx)
            return self.summary_data[idx]
        else:
            self.summary_data = {}
            self.summary_data[idx] = self._compute_summary(idx)
            return self.summary_data[idx]

    def _compute_summary(self, idx=None):
        """
        Compute generic set of measures to evaluate the quality of the factorization.
        
        :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
        """
        return {
            'rank': self.fit.rank,
            'sparseness': self.fit.sparseness(idx=idx),
            'rss': self.fit.rss(idx=idx),
            'evar': self.fit.evar(idx=idx),
            'residuals': self.fit.residuals(idx=idx),
            'connectivity': self.fit.connectivity(idx=idx),
            'predict_samples': self.fit.predict(what='samples', prob=True, idx=idx),
            'predict_features': self.fit.predict(what='features', prob=True, idx=idx),
            'score_features': self.fit.score_features(idx=idx),
            'select_features': self.fit.select_features(idx=idx),
            'dispersion': self.fit.dispersion(idx=idx),
            'cophenetic': self.fit.coph_cor(idx=idx),
            'consensus': self.fit.consensus(idx=idx),
            'euclidean': self.fit.distance(metric='euclidean', idx=idx),
            'kl': self.fit.distance(metric='kl', idx=idx),
            'n_iter': self.fit.n_iter,
            'n_run': self.fit.n_run
        }
