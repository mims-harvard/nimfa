
"""
    ##############################
    Mf_track (``models.mf_track``)
    ##############################
"""

class Mf_track():
    """
    Base class for tracking MF fitted model across multiple runs of the factorizations or tracking
    the residuals error across iterations of single/multiple runs. 
    
    The purpose of this class is to store matrix factors from multiple runs of the algorithm which can then be used
    for performing quality and performance measures. Because of additional space consumption for storing multiple 
    matrix factors, tracking is used only if explicitly specified by user through runtime option. 
    
    The purpose of this class is to store residuals across iterations which can then be used for plotting the trajectory 
    of the residuals track or estimating proper number of iterations. 
    """

    def __init__(self):
        """
        Construct model for tracking fitted factorization model across multiple runs or tracking the residuals error across iterations. 
        """
        self._factors = []
        self._residuals = {}
        
    def _track_error(self, residuals, run):
        """
        Add residuals error after one iteration. 
        
        :param residuals: Residuals between the target matrix and its MF estimate.
        :type residuals: `float`
        :param run: Specify the run to which :param:`residuals` belongs. Error tracking can be also used if multiple runs are enabled. 
        :type run: `int`
        """
        self._residuals.setdefault(run, [])
        self._residuals[run].append(residuals)
        
    def _track_factor(self, **track_model):
        """
        Add matrix factorization factors (and method specific model data) after one factorization run.
        
        :param track_model: Matrix factorization factors.
        :type track_model:  algorithm specific
        """
        
    def get_factor(self, run):
        """
        Return matrix factorization factors from run :param:`run`.
        
        :param run: Saved factorization factors (and method specific model data) of :param:`run`'th run are returned. 
        :type run: `int`
        """
        return self._factors[run - 1]
    
    def get_error(self, run = 1):
        """
        Return residuals track from one run of the factorization.
        
        :param run: Specify the run of which error track is desired. By default :param:`run` is 1. 
        :type run: `int`
        """
        return self._residuals[run - 1]
    