
import models.nmf_std as mstd

class Nsnmf(mstd.Nmf_std):
    '''
    classdocs
    '''


    def __init__(self, **params):
        mstd.Nmf_std.__init__(self, params)
        self.aname = "nsnmf"
        self.aseeds = ["random", "fixed", "nndsvd"]
        
    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        