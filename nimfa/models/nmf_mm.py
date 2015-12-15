
"""
    ##########################
    Nmf_mm (``models.nmf_mm``)
    ##########################
"""

from .nmf import *


class Nmf_mm(Nmf):
    """
    Implementation of the alternative model to manage factorizations that follow
    NMF nonstandard model. This modification is required by the Multiple NMF
    algorithms (e. g. SNMNMF [Zhang2011]_). The Multiple NMF algorithms modify the
    standard divergence or Euclidean based NMF methods by introducing multiple
    mixture (coefficients) matrices and target matrices.
     
    It is the underlying model of matrix factorization and provides structure of
    modified standard NMF model.
    
    .. attribute:: W
        
        Basis matrix -- the first matrix factor in the multiple NMF model
        
    .. attribute:: H
    
        Mixture matrix -- the second matrix factor in the multiple NMF model (coef0)
    
    .. attribute:: H1
    
        Mixture matrix -- the second matrix factor in the multiple NMF model (coef1)
        
    .. attribute:: V1
    
        Target matrix, the matrix for the MF method to estimate.
        
    The interpretation of the basis and mixture matrix is such as in the standard NMF model. 
    
    Multiple NMF specify more than one target matrix. In that case target matrices are
    passed as tuples. Internally, additional attributes with names following Vn pattern
    are created, where n is the consecutive index of target matrix. Zero index is omitted
    (there are V, V1, V2, V3, etc. matrices and then H, H1, H2, etc. and W, W1, W2, etc.
    respectively).
    
    Currently, in implemented multiple NMF method V, V1 and H, H1 are needed. There is only
    one basis matrix (W).
    """
    def __init__(self, params):
        """
        Construct factorization model that manages multiple NMF models.
        
        :param params: MF runtime and algorithm parameters and options. For detailed
           explanation of the general model parameters see :mod:`mf_run`. For algorithm
           specific model options see documentation of chosen factorization method.
        :type params: `dict`
        """
        self.model_name = "mm"
        Nmf.__init__(self, params)
        if sp.isspmatrix(self.V) and (self.V.data < 0).any() or not sp.isspmatrix(self.V) and (self.V < 0).any():
            raise utils.MFError("The input matrix contains negative elements.")
        if sp.isspmatrix(self.V1) and (self.V1.data < 0).any() or not sp.isspmatrix(self.V1) and (self.V1 < 0).any():
            raise utils.MFError("The input matrix contains negative elements.")

    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W

    def target(self, idx):
        """
        Return the target matrix to estimate.
        
        :param idx: Name of the matrix (coefficient) matrix.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1 respectively) 
        """
        if idx == 'coef' or idx == 0:
            return self.V
        elif idx == 'coef1' or idx == 1:
            return self.V1
        raise utils.MFError("Unknown specifier for the target matrix.")

    def coef(self, idx):
        """
        Return the matrix of mixture coefficients.
        
        :param idx: Name of the matrix (coefficient) matrix.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1 respectively) 
        """
        if idx == 'coef' or idx == 0:
            return self.H
        elif idx == 'coef1' or idx == 1:
            return self.H1
        raise utils.MFError("Unknown specifier for the mixture matrix.")

    def fitted(self, idx):
        """
        Compute the estimated target matrix according to the nonsmooth NMF algorithm model.
        
        :param idx: Name of the matrix (coefficient) matrix.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1 respectively) 
        """
        if idx == 'coef' or idx == 0:
            return dot(self.W, self.H)
        elif idx == 'coef1' or idx == 1:
            return dot(self.W, self.H1)
        raise utils.MFError("Unknown specifier for the mixture matrix.")

    def distance(self, metric='euclidean', idx=None):
        """
        Return the loss function value.

        :param distance: Specify distance metric to be used. Possible are Euclidean
           and Kullback-Leibler (KL) divergence. Strictly, KL is not a metric.
        :type distance: `str` with values 'euclidean' or 'kl'

        :param idx: Name of the matrix (coefficient) matrix.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1 respectively) 
        """
        if idx == 'coef' or idx == 0:
            H = self.H
            V = self.V
        elif idx == 'coef1' or idx == 1:
            H = self.H1
            V = self.V1
        else:
            raise utils.MFError("Unknown specifier for the mixture matrix.")
        if metric.lower() == 'euclidean':
            return power(V - dot(self.W, H), 2).sum()
        elif metric.lower() == 'kl':
            Va = dot(self.W, H)
            return (multiply(V, sop(elop(V, Va, div), op=np.log)) - V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")

    def residuals(self, idx):
        """
        Return residuals matrix between the target matrix and its multiple NMF estimate.
        
        :param idx: Name of the matrix (coefficient) matrix.
        :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1 respectively) 
        """
        if idx == 'coef' or idx == 0:
            H = self.H
            V = self.V
        elif idx == 'coef1' or idx == 1:
            H = self.H1
            V = self.V1
        else:
            raise utils.MFError("Unknown specifier for the mixture matrix.")
        return V - dot(self.W, H)
