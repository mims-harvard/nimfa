import scipy.sparse as sp
import numpy as np

import utils.utils as utils
import nmf
from utils.linalg import *

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