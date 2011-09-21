
"""
    ########################################################
    Gene_func_prediction (``examples.gene_func_prediction``)
    ########################################################
    
    .. note:: This example is in progress.
    
    As a background reading before this example, we suggest reading [Schietgat2010]_ and [Schachtner2008]_
        
    This example from functional genomics deals with gene function prediction. Two main characteristics of function 
    prediction task are:
    
        #. single gene can have multiple functions, 
        #. the functions are organized in a hierarchy, in particular in a hierarchy structered as a rooted tree -- MIPS's
           FunCat. In example is used dataset that originates from S. cerevisiae and has annotations from the MIPS Functional
           Catalogue. A gene related to some function is automatically related to all its ancestor functions.
    
    These characteristics describe hierarchical multi-label classification setting. 
    
    Here is the outline of this gene function prediction task. 
    
        #. Dataset Preprocessing.
        #. Gene selection
        #. Feature generation. 
        #. Feature selection
        #. Classification of the mixture matrix and comply with the hierarchy constraint. 
    
    To run the example simply type::
        
        python gene_func_prediction.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.gene_func_prediction.run()
        
    .. note:: This example uses matplotlib library for producing visual interpretation.
"""

import mf
import numpy as np
import scipy.sparse as sp
from os.path import dirname, abspath, sep

try:
    import matplotlib.pylab as plb
except ImportError, exc:
    raise SystemExit("Matplotlib must be installed to run this example.")
    

def run():
    """Run the NMF - Divergence on the S. cerevisiae dataset."""
    pass

if __name__ == "__main__": 
    """Run the gene function prediction example."""
    run()



    