
"""    
    This package contains factorization models used in the library MF. Specifically, it contains the following:
    
        #. Generic factorization model for handling common computations and assessing quality and performance 
           measures. 
        #. Implementation of the standard model to manage factorizations that follow standard NMF model.
        #. Implementation of the alternative nonsmooth model to manage factorizations that follow nonstandard NMF model 
           (e.g. nsnmf)
        #. Implementation of the alternative multiple model to manage factorizations that follow NMF nonstandard model 
           (e.g. snmnmf)        
        #. Tracker of MF fitted model across multiple runs of the factorizations and tracker of
           the residuals error across iterations of single/multiple runs.  
        #. Implementation for storing MF fitted model and MF results. 
    
"""

import nmf
import nmf_std
import nmf_ns
import nmf_mm
import mf_track
import mf_fit