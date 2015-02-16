
"""
    This package contains nimfa library examples of usage. It demonstrates the following:
        
        #. Single run of the specified factorization algorithm.
        #. Multiple runs of the specified factorization algorithm.
        #. Tracking fitted results across multiple runs and tracking residuals error across single / multiple runs.
        #. Quality and performance measures of executed factorizations.
        #. Running factorization algorithms with algorithm specific settings and initializations. 
        
    Applications of factorization methods on both synthetic and real world data sets are provided. 
    
    Example using synthetic data set is intended as demonstration of the nimfa library since all currently implemented 
    factorization algorithms with different initialization methods and specific settings are ran. Others include 
    applications on real world data sets in:
    
        * bioinformatics,
        * functional genomics, 
        * text analysis,
        * image processing,
        * recommendation systems.
"""

from . import synthetic
from . import all_aml
from . import medulloblastoma
from . import cbcl_images
from . import documents
from . import orl_images
from . import recommendations
from . import gene_func_prediction
