

class Fixed(object):
    """
    Fixed factorization. This is the option to completely specify the initial factorization by passing values for 
    matrix factors.  
    """


    def __init__(self):
        self.name = "fixed"
        
    def _set_fixed(self, *ff):
        """Set initial factorization."""
        self.ff = ff
        
    def initialize(self, *args, **kwargs):
        """
        Return fixed matrix factors.
        """
        return self.ff
    
    def __repr__(self):
        return "fixed.Fixed()"
    
    def __str__(self):
        return self.name