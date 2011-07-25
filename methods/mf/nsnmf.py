
import models.nmf_ns as mns

class Nsnmf(mns.Nmf_ns):
    """
    Nonsmooth Nonnegative Matrix Factorization (NSNMF) [14].
    
    [14] Pascual-Montano, A., Carazo, J. M., Kochi, K., Lehmann, D., and Pascual-Marqui, R. D., (2006). Nonsmooth nonnegative matrix 
        factorization (nsnmf). IEEE transactions on pattern analysis and machine intelligence, 28(3), 403-415.
    """

    def __init__(self, **params):
        mns.Nmf_ns.__init__(self, params)
        self.name = "nsnmf"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        