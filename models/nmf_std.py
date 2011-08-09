from math import log

import nmf

class Nmf_std(nmf.Nmf):
    """
    Implementation of the standard model to manage factorizations that follow standard NMF model.
     
    It is the underlying model of matrix factorization and provides a general structure of standard NMF model.
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in standard factorization
        
    .. attribute:: H
    
        Mixture matrix -- the second matrix factor in standard factorization
    """
    
    def __init__(self, params):
        """
        Construct factorization model that manages standard NMF models.
        
        :param params: MF runtime and algorithm parameters and options. For detailed explanation of the general model 
                       parameters see :mod:`mf_methods`. For algorithm specific model options see documentation of chosen
                       factorization method. 
        :type params: `dict`
        """
        nmf.Nmf.__init__(self, params)
        if not self.seed and not self.W and not self.H: self.seed = None if "none" in self.aseeds else "random"
        if self.W and self.H:
            if self.seed:
                raise nmf.utils.MFError("Initial factorization is fixed. Seeding method cannot be used.")
            else:
                self.seed = nmf.seed.fixed.Fixed()
                self.seed._set_fixed(self.W, self.H)
        self._is_smdefined()
        if nmf.sp.isspmatrix(self.V) and (self.V.data < 0).any() or (self.V.data < 0).any():
            raise nmf.utils.MFError("The input matrix contains negative elements.")    
            
    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W
    
    def coef(self):
        """Return the matrix of mixture coefficients."""
        return self.H
    
    def fitted(self):
        """Compute the estimated target matrix according to the NMF algorithm model."""
        return nmf.dot(self.W, self.H)
    
    def distance(self, metric):
        """Return the loss function value."""
        if metric == 'euclidean':
            return (nmf.elop(self.V - nmf.dot(self.W, self.H), 2, pow)).sum()
        elif metric == 'kl':
            Va = nmf.dot(self.W, self.H)
            return (nmf.multiply(self.V, nmf.elop(self.V, Va, log)) - self.V + Va).sum()
        else:
            raise nmf.utils.MFError("Unknown distance metric.")
    
    def residuals(self):
        """Return residuals between the target matrix and its NMF estimate."""
        return self.V - nmf.dot(self.W, self.H)
        