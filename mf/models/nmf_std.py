
from nmf import *

class Nmf_std(Nmf):
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
                       parameters see :mod:`mf`. For algorithm specific model options see documentation of chosen
                       factorization method. 
        :type params: `dict`
        """
        Nmf.__init__(self, params)
        if sp.isspmatrix(self.V) and (self.V.data < 0).any() or not sp.isspmatrix(self.V) and (self.V < 0).any():
            raise utils.MFError("The input matrix contains negative elements.")    
            
    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W
    
    def coef(self):
        """Return the matrix of mixture coefficients."""
        return self.H
    
    def fitted(self):
        """Compute the estimated target matrix according to the NMF algorithm model."""
        return dot(self.W, self.H)
    
    def distance(self, metric = 'euclidean'):
        """Return the loss function value."""
        if metric == 'euclidean':
            return (sop(self.V - dot(self.W, self.H), 2, pow)).sum()
        elif metric == 'kl':
            Va = dot(self.W, self.H)
            return (multiply(self.V, sop(elop(self.V, Va, div), op = log)) - self.V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")
    
    def residuals(self):
        """Return residuals between the target matrix and its NMF estimate."""
        return self.V - dot(self.W, self.H)
        