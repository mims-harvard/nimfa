
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

    .. attribute:: niter

        The number of iterations performed.

    .. attribute:: nrun

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
        
    def basis(self):
        """Return the matrix of basis vectors."""
        return self.fit.basis()
    
    def coef(self):
        """Return the matrix of mixture coefficients."""
        return self.fit.coef()  
    
    def distance(self, metric = None):
        """
        Return the loss function value. If metric is not supplied, final objective function value associated to the MF algorithm is returned.
        
        :param metric: Measure of distance between a target matrix and a MF estimate. Metric 'kl' and 'euclidean' 
                       are defined.  
        :type metric: 'str'
        """
        if not metric:
            return self.fit.final_obj
        else:
            self.fit.distance(metric)
            
    def fitted(self):
        """Compute the estimated target matrix according to the MF algorithm model."""
        return self.fit.fitted()
    
    def fit(self):
        """Return the MF algorithm model."""
        return self.fit 
    
    def summary(self):
        """Compute generic set of measures to evaluate the quality of the factorization."""
        return {
                'rank': self.fit.rank,
                'sparseness basis': self.fit.sparseness()[0],
                'sparseness coef': self.fit.sparseness()[1],
                'rss': self.fit.rss(),
                'evar': self.fit.evar(),
                'residuals': self.fit.residuals(),
                'connectivity': self.fit.connectivity(),
                'predict_samples': self.fit.predict(what = 'samples', prob = True),
                'predict_features': self.fit.predict(what = 'features', prob = True),
                'score_features': self.fit.score_features(),
                'select_features': self.fit.select_features(), 
                'niter': self.fit.niter,
                'nrun': self.fit.nrun
                }
    
    
        