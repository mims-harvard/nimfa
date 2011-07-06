

class Nmf_fit():
    '''
    Base class for storing NMF results.
    
    It contains generic functions and structure for handling the results of NMF algorithms. 
    It contains a slot with the fitted NMF model and data about parameters and methods used for
    factorization. 
    
    .. attribute:: fit
        
        The fitted NMF model
    
    .. attribute:: algorithm 

        NMF method of factorization.

    .. attribute:: niter

        The number of iterations performed.

    .. attribute:: nrun

        The number of NMF runs performed.

    .. attribute:: residuals

        Residuals track between the target matrix and its NMF estimate. 

    .. attribute:: seeding

        The seeding method used to seed the algorithm that fitted NMF model. 

    .. attribute:: parameters

        Extra parameters specific to the algorithm used to fit the model.
    
    '''


    def __init__(self, fit, model_data):
        '''
        Constructor
        '''
        self.fit = fit
        self.__dict__.update(model_data.__dict__.update())
        
    def basis(self):
        return self.fit.W.todense()
    
    def coef(self):
        return self.fit.H.todense()
    
    def distance(self, metric = "KL"):
        """Return the loss function value."""
        if self.distance_method:
            self.fit.distance(metric)
        else:
            print "Distance function not defined."
            
    def fitted(self):
        """Compute the estimated target matrix according to the NMF model."""
        return (self.fit.W * self.fit.H).todense()
    
    
        