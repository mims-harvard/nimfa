
class Mf_track():
    """
    Base class for tracking MF results across multiple runs of the factorizations. 
    
    The purpose of this class is to store matrix factors from multiple runs of the algorithm which can then be used
    for performing quality and performance measures. Because of additional space consumption for storing multiple 
    matrix factors, tracking is used only if explicitly specified by user through runtime option. 
    """


    def __init__(self, **track_model):
        """
        Construct model for tracking results across multiple runs. 
        
        :param track_model: Matrix factorization factors.
        :type track_model:  algorithm specific
        """
        self.__dict__.update(track_model)