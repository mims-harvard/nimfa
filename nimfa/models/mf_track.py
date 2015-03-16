
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
    matrix factors, tracking is used only if explicitly specified by user through runtime option. In summary, when
    tracking factors, the following is retained from each run:
        
        #. fitted factorization model depending on the factorization method; 
        #. final value of objective function; 
        #. performed number of iterations. 
        
    Instead of tracking fitted factorization model, callback function can be set, which will be called after each 
    factorization run. For more details see :mod:`mf_run`.
    
    The purpose of this class is to store residuals across iterations which can then be used for plotting the trajectory 
    of the residuals track or estimating proper number of iterations. 
    """
    def __init__(self):
        """
        Construct model for tracking fitted factorization model across multiple runs or tracking the residuals error across iterations. 
        """
        self._factors = {}
        self._residuals = {}

    def track_error(self, run, residuals):
        """
        Add residuals error after one iteration. 
        
        :param run: Specify the run to which ``residuals`` belongs. Error tracking can be
           also used if multiple runs are enabled.
        :type run: `int`

        :param residuals: Residuals between the target matrix and its MF estimate.
        :type residuals: `float`
        """
        self._residuals.setdefault(run, [])
        self._residuals[run].append(residuals)

    def track_factor(self, run, **track_model):
        """
        Add matrix factorization factors (and method specific model data) after one factorization run.
        
        :param run: Specify the run to which ``track_model`` belongs.
        :type run: 'int'
        :param track_model: Matrix factorization factors.
        :type track_model:  algorithm specific
        """
        self._factors[run] = t_model(track_model)

    def get_factor(self, run=0):
        """
        Return matrix factorization factors from run :param:`run`.
        
        :param run: Saved factorization factors (and method specific model data) of
           ``run``'th run are returned.
        :type run: `int`
        """
        return self._factors[run]

    def get_error(self, run=0):
        """
        Return residuals track from one run of the factorization.
        
        :param run: Specify the run of which error track is desired. By default ``run`` is 1.
        :type run: `int`
        """
        return self._residuals[run]


class t_model:

    """
    Tracking factors model.
    """

    def __init__(self, td):
        self.__dict__.update(td)
