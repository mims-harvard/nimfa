
import utils.utils as utils
import methods.seeding.fixed as fixed
import nmf

class Nmf_ns(nmf.Nmf):
    """
    Implementation of the alternative model to manage factorizations that follow NMF model. This modification is 
    required by the Nonsmooth NMF algorithm. The Nonsmooth NMF algorithm [14] is a modification of the standard divergence
    based NMF methods. By introducing a smoothing matrix it is aimed to achieve global sparseness. 
     
    It is the underlying model of matrix factorization and provides structure of modified standard NMF model. 
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in standard factorization
        
    .. attribute:: H
    
        Mixture matrix -- the second matrix factor in standard factorization
        
    .. attribute:: S
    
        Smoothing matrix -- positive symmetric matrix modifying standard model (V = WSH).
        
    [14] Pascual-Montano, A., Carazo, J. M., Kochi, K., Lehmann, D., and Pascual-Marqui, R. D., (2006). Nonsmooth nonnegative matrix 
        factorization (nsnmf). IEEE transactions on pattern analysis and machine intelligence, 28(3), 403-415.
    """


    def __init__(self, params):
        """
        Constructor
        """
        nmf.Nmf.__init__(self, params)
        if not self.seed and not self.W and not self.H: self.seed = "random"
        if self.W and self.H:
            if self.seed:
                raise utils.MFError("Initial factorization is fixed. Seeding method cannot be used.")
            else:
                self.seed = fixed.Fixed() 
                self.seed._set_fixed(self.W, self.H)
        self._is_smdefined()
        if any(self.V.data < 0):
            raise utils.MFError("The input matrix contains negative elements.") 
        