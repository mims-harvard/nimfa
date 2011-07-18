from math import log

import utils.utils as utils
import nmf
from utils.linalg import *
from methods.seeding.fixed import *

class Nmf_std(nmf.Nmf):
    '''
    Implementation of the standard model to manage factorizations that follow NMF standard model. 
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in standard factorization
        
    .. attribute:: H
    
        Mixture matrix -- the second matrix factor in standard factorization
        
    '''
    
    def __init__(self, **params):
        '''
        Constructor
        '''
        nmf.Nmf.__init__(self, params)
        if not self.seed and not self.W and not self.H: self.seed = "random"
        if self.W and self.H:
            if self.seed:
                raise utils.MFError("Initial factorization is fixed. Seeding method cannot be used.")
            else:
                self.seed = Fixed()
                self.seed._set_fixed(self.W, self.H)
        self._is_smdefined()
        if any(self.V.data < 0):
            raise utils.MFError("The input matrix contains negative elements.")    
            
    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W
    
    def coef(self):
        """Return the matrix of mixture coefficients."""
        return self.H
    
    def fitted(self):
        """Compute the estimated target matrix according to the NMF model."""
        return dot(self.W, self.H)
    
    def distance(self, metric):
        """Return the loss function value."""
        if metric == 'euclidean':
            return (elop(self.V - dot(self.W, self.H), 2, pow)).sum()
        elif metric == 'kl':
            Va = dot(self.W, self.H)
            return (multiply(self.V, elop(self.V, Va, log)) - self.V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")
    
    def residuals(self):
        """Return residuals between the target matrix and its NMF estimate."""
        return self.V - dot(self.W, self.H)
        