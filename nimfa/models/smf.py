"""
    #####################
    Smf (``models.smf``)
    #####################
"""

import nimfa.utils.utils as utils
from nimfa.utils.linalg import *
from nimfa.methods import seeding


class Smf(object):
    """
    This class defines a common interface / model to handle standard MF models in
    a generic way.
    
    It contains definitions of the minimum set of generic methods that are used in 
    common computations and matrix factorizations. Besides it contains some quality
    and performance measures about factorizations.
    """
    def __init__(self, params):
        self.model_name = "smf"
        self.__dict__.update(params)
        self.V1 = None
        self.H1 = None
        # do not copy target and factor matrices into the program
        if sp.isspmatrix(self.V):
            self.V = self.V.tocsr().astype('d')
        else:
            self.V = np.asmatrix(self.V) if self.V.dtype == np.dtype(
                float) else np.asmatrix(self.V, dtype='d')
        if self.W is not None or self.H is not None or self.H1 is not None:
            raise utils.MFError("Fixed initialized is not supported by SMF model.")
        self._compatibility()

    def __call__(self):
        """Run the specified MF algorithm."""
        return self.factorize()

    def basis(self):
        """Return the matrix of basis vectors (factor 1 matrix)."""
        return self.W

    def target(self, idx=None):
        """
        Return the target matrix to estimate.
        
        :param idx: Used in the multiple MF model. In standard MF ``idx`` is always None.
        :type idx: None
        """
        return self.V

    def coef(self, idx=None):
        """
        Return the matrix of mixture coefficients (factor 2 matrix).
        
        :param idx: Used in the multiple MF model. In standard MF ``idx`` is always None.
        :type idx: None
        """
        return self.H

    def fitted(self, idx=None):
        """
        Compute the estimated target matrix according to the MF algorithm model.
        
        :param idx: Used in the multiple MF model. In standard MF ``idx`` is always None.
        :type idx: None
        """
        return dot(self.W, self.H)

    def distance(self, metric='euclidean', idx=None):
        """
        Return the loss function value.
        
        :param distance: Specify distance metric to be used. Possible are Euclidean and
           Kullback-Leibler (KL) divergence. Strictly, KL is not a metric.
        :type distance: `str` with values 'euclidean' or 'kl'

        :param idx: Used in the multiple MF model. In standard MF ``idx`` is always None.
        :type idx: None
        """
        if metric.lower() == 'euclidean':
            R = self.V - dot(self.W, self.H)
            return power(R, 2).sum()
        elif metric.lower() == 'kl':
            Va = dot(self.W, self.H)
            return (multiply(self.V, sop(elop(self.V, Va, div), op=log)) - self.V + Va).sum()
        else:
            raise utils.MFError("Unknown distance metric.")

    def residuals(self, idx=None):
        """
        Return residuals matrix between the target matrix and its MF estimate.
        
        :param idx: Used in the multiple MF model. In standard MF ``idx`` is always None.
        :type idx: None
        """
        return self.V - dot(self.W, self.H)

    def _compatibility(self):
        """
        Check if chosen seeding method is compatible with chosen factorization
        method or fixed initialization is passed.

        :param mf_model: The underlying initialized model of matrix factorization.
        :type mf_model: Class inheriting :class:`models.nmf.Nmf`
        """
        W = self.basis()
        H = self.coef(0)
        H1 = self.coef(1) if self.model_name == 'mm' else None
        if self.seed is None and W is None and H is None and H1 is None:
            self.seed = None if "none" in self.aseeds else "random"
        if W is not None and H is not None:
            if self.seed is not None and self.seed is not "fixed":
                raise utils.MFError("Initial factorization is fixed.")
            else:
                self.seed = seeding.fixed.Fixed()
                self.seed._set_fixed(W=W, H=H, H1=H1)
        self.__is_smdefined()
        self.__compatibility()

    def __is_smdefined(self):
        """Check if MF and seeding methods are well defined."""
        if isinstance(self.seed, str):
            if self.seed in seeding.methods:
                self.seed = seeding.methods[self.seed]()
            else:
                raise utils.MFError("Unrecognized seeding method.")
        else:
            if not str(self.seed).lower() in seeding.methods:
                raise utils.MFError("Unrecognized seeding method.")

    def __compatibility(self):
        """Check if MF model is compatible with the seeding method."""
        if not str(self.seed).lower() in self.aseeds:
            raise utils.MFError("MF model is incompatible with the seeding method.")
