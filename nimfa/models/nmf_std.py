
"""
    ############################
    Nmf_std (``models.nmf_std``)
    ############################
"""

from .nmf import *


class Nmf_std(Nmf):
    """
    Implementation of the standard model to manage factorizations that follow standard NMF model.
     
    It is the underlying model of matrix factorization and provides a general structure
    of standard NMF model.
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in standard factorization
        
    .. attribute:: H
    
        Mixture matrix -- the second matrix factor in standard factorization
    """
    def __init__(self, params):
        """
        Construct factorization model that manages standard NMF models.
        
        :param params: MF runtime and algorithm parameters and options. For detailed
           explanation of the general model parameters see :mod:`mf_run`. For
           algorithm specific model options see documentation of chosen factorization method.
        :type params: `dict`
        """
        self.model_name = "std"
        self.V1 = None
        self.H1 = None
        Nmf.__init__(self, params)
        if sp.isspmatrix(self.V) and (self.V.data < 0).any() or not sp.isspmatrix(self.V) and (self.V < 0).any():
            raise utils.MFError("The input matrix contains negative elements.")

    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W

    def target(self, idx=None):
        """
        Return the target matrix to estimate.
        
        :param idx: Used in the multiple NMF model. In standard NMF ``idx`` is always None.
        :type idx: None
        """
        return self.V

    def coef(self, idx=None):
        """
        Return the matrix of mixture coefficients.
        
        :param idx: Used in the multiple NMF model. In standard NMF ``idx`` is always None.
        :type idx: None
        """
        return self.H

    def fitted(self, idx=None):
        """
        Compute the estimated target matrix according to the NMF algorithm model.
        
        :param idx: Used in the multiple NMF model. In standard NMF ``idx`` is always None.
        :type idx: None
        """
        return dot(self.W, self.H)

    def distance(self, metric='euclidean', idx=None):
        """
        Return the loss function value.
        
        :param distance: Specify distance metric to be used. Possible are Euclidean and
           Kullback-Leibler (KL) divergence. Strictly, KL is not a metric.
        :type distance: `str` with values 'euclidean' or 'kl'

        :param idx: Used in the multiple NMF model. In standard NMF ``idx`` is always None.
        :type idx: None
        """
        if metric.lower() == 'euclidean':
            R = self.V - dot(self.W, self.H)
            return (power(R, 2)).sum()
        elif metric.lower() == 'kl':
            Va = dot(self.W, self.H)
            return (multiply(self.V, sop(elop(self.V, Va, div), op=np.log)) - self.V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")

    def residuals(self, idx=None):
        """
        Return residuals matrix between the target matrix and its NMF estimate.
        
        :param idx: Used in the multiple NMF model. In standard NMF ``idx`` is always None.
        :type idx: None
        """
        return self.V - dot(self.W, self.H)
