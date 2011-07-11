import scipy.sparse as sp
import numpy as np

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
        return self.W
    
    def coef(self):
        return self.H
    
    def fitted(self):
        return dot(self.W, self.H)
    
    def residuals(self):
        return 