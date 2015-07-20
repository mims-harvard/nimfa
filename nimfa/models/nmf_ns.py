
"""
    ##########################
    Nmf_ns (``models.nmf_ns``)
    ##########################
"""

from .nmf import *


class Nmf_ns(Nmf):
    """
    Implementation of the alternative model to manage factorizations that follow
    nonstandard NMF model. This modification is required by the Nonsmooth NMF
    algorithm (NSNMF) [Montano2006]_. The Nonsmooth NMF algorithm is a modification
    of the standard divergence based NMF methods. By introducing a smoothing matrix
    it is aimed to achieve global sparseness.
     
    It is the underlying model of matrix factorization and provides structure of
    modified standard NMF model.
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in the nonsmooth NMF model
        
    .. attribute:: H
    
        Mixture matrix -- the third matrix factor in the nonsmooth NMF model
        
    .. attribute:: S
    
        Smoothing matrix -- the middle matrix factor (V = WSH) in the nonsmooth NMF model
        
    The interpretation of the basis and mixture matrix is such as in the standard NMF model.
    The smoothing matrix is an extra square matrix whose entries depends on smoothing
    parameter theta which can be specified as algorithm specific model option. For detailed
    explanation of the NSNMF algorithm see :mod:`methods.factorization.nsnmf`.
    """
    def __init__(self, params):
        """
        Construct factorization model that manages nonsmooth NMF models.
        
        :param params: MF runtime and algorithm parameters and options. For detailed
           explanation of the general model parameters see :mod:`mf_run`. For
           algorithm specific model options see documentation of chosen
           factorization method.
        :type params: `dict`
        """
        self.model_name = "ns"
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
        
        :param idx: Used in the multiple NMF model. In nonsmooth NMF ``idx`` is always None.
        :type idx: None
        """
        return self.V

    def coef(self, idx=None):
        """
        Return the matrix of mixture coefficients.
        
        :param idx: Used in the multiple NMF model. In nonsmooth NMF ``idx`` is always None.
        :type idx: None
        """
        return self.H

    def smoothing(self):
        """Return the smoothing matrix."""
        return self.S

    def fitted(self, idx=None):
        """
        Compute the estimated target matrix according to the nonsmooth NMF algorithm model.
        
        :param idx: Used in the multiple NMF model. In nonsmooth NMF ``idx`` is always None.
        :type idx: None
        """
        return dot(dot(self.W, self.S), self.H)

    def distance(self, metric='euclidean', idx=None):
        """
        Return the loss function value.
        
        :param distance: Specify distance metric to be used. Possible are Euclidean and
           Kullback-Leibler (KL) divergence. Strictly, KL is not a metric.
        :type distance: `str` with values 'euclidean' or 'kl'

        :param idx: Used in the multiple NMF model. In nonsmooth NMF ``idx`` is always None.
        :type idx: None
        """
        if metric.lower() == 'euclidean':
            R = self.V - dot(dot(self.W, self.S), self.H)
            return power(R, 2).sum()
        elif metric.lower() == 'kl':
            Va = dot(dot(self.W, self.S), self.H)
            return (multiply(self.V, sop(elop(self.V, Va, div), op=np.log)) - self.V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")

    def residuals(self, idx=None):
        """
        Return residuals matrix between the target matrix and its nonsmooth NMF estimate.
        
        :param idx: Used in the multiple NMF model. In nonsmooth NMF ``idx`` is always None.
        :type idx: None
        """
        return self.V - dot(dot(self.W, self.S), self.H)
