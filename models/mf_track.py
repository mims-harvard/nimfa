
class Mf_track():
    """
    Base class for tracking MF results across multiple runs of the factorizations. 
    
    The purpose of this class is to store matrix factors from multiple runs of the algorithm which can then be used
    for performing quality and performance measures. Because of additional space consumption for storing multiple 
    matrix factors, tracking is used only if explicitly specified by user through runtime option. 
    """

    def __init__(self):
        """
        Construct model for tracking results across multiple runs. 
        """
        self._runs = []
        
    def add(self, **track_model):
        """
        Add matrix factorization factors (and method specific model data) after one factorization run.
        
        :param track_model: Matrix factorization factors.
        :type track_model:  algorithm specific
        """
        
    def get(self, run):
        """
        Return matrix factorization factors from run :param:`run`.
        
        :param run: Saved factorization factors (and method specific model data) of :param:`run`'th run are returned. 
        :type run: `int`
        """
        return self._runs[run - 1]
    