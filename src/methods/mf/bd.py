
import models.nmf_std as mstd

class Bd(mstd.Nmf_std):
    '''
    classdocs
    '''


    def __init__(self, **params):
        mstd.Nmf_std.__init__(self, params)
        self.name = "bd"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        