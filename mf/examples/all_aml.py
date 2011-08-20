
"""
    This module demonstrates the ability of NMF to recover meaningful biological information from 
    cancer related microarray data. NMF appers to have advantages over other methods such as HC or SOM. 
    
    Leukemia data set is used in this example. This data set is a benchmark in the cancer classification
    community. This example is inspired by [3]. In [3] authors applied NMF to the data set. With rank, rank = 2, 
    NMF recovered the AML-ALL biological distinction with high accuracy and robustness. Higher ranks revealed further
    partitioning of the samples. 
    
    .. figure:: all_aml_consensus2.png
       :scale: 80 %
       :alt: Consensus matrix generated for rank, rank = 2. 

       Consensus matrix generated for rank, rank = 2. Reordered consensus matrices averaging 50 connectivity 
       matrices computed at rank = 2-5 for the leukemia data set with the 5000 most highly varying genes 
       according to their coefficient of variation. Samples are hierarchically clustered by using 
       distances derived from consensus clustering matrix entries, coloured from 0 (deep blue, samples
       are never in the same cluster) to 1 (dark red, samples are always in the same cluster).   
       
       
    .. figure:: all_aml_consensus3.png
       :scale: 80 %
       :alt: Consensus matrix generated for rank, rank = 3. 

       Consensus matrix generated for rank, rank = 3.
    
     
    [3] Brunet, J.-P., Tamayo, P., Golub, T. R., Mesirov, J. P., (2004). Metagenes and molecular pattern discovery using 
        matrix factorization. Proceedings of the National Academy of Sciences of the United States of America, 
        101(12), 4164-9. doi: 10.1073/pnas.0308531101.
    
    .. seealso:: README.rst     
    
    To run the examples simply type::
        
        python all_aml.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.all_aml.run()
"""

import mf
import numpy as np
import scipy.sparse as sp

def run():
    """Run Standard NMF on Leukemia data set."""


if __name__ == "__main__":
    run()