
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
        
    .. note:: This example uses matplotlib library for producing a heatmap of a consensus matrix.
"""

import mf
import numpy as np
import os.path
import matplotlib.pyplot as plt

def run():
    """Run Standard NMF on Leukemia data set."""
    V = read()
    for rank in xrange(2, 5):
         _run(V, rank)
         
def read():
    """
    Read ALL AML gene expression data. The matrix's shape is 5000 (genes) x 38 (samples). 
    It contains only positive data.
    """
    V = np.matrix(np.zeros((5000, 38)))
    i = 0
    for line in open(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ '/datasets/ALL_AML_data.txt'):
        V[i, :] =  map(float, line.split('\t'))
        i += 1
    return V

def reorder(consensus):
    """
    Reorder consensus matrix.
    
    :param consensus: Consensus matrix.
    :type consensus: `numpy.matrix`
    """    
    return 0

def _run(V, rank):
    """
    Run standard NMF on Leukemia data set.
    
    :param V: Target matrix with gene expression data.
    :type V: `numpy.matrix` (of course it could be any format of scipy.sparse, but we will use numpy here) 
    :param rank: Factorization rank.
    :type rank: `int`
    """
    # read gene expression data
    V = read()
    consensus = np.mat(np.zeros((V.shape[1], V.shape[1])))
    for _ in xrange(50):
        # Standard NMF with euclidean update equations is used. For initialization random Vcol method is used. 
        # Objective function is connectivity matrix changes - if the number of instances changing the cluster is lower
        # or equal to min_residuals parameter, factorization is terminated. 
        model = mf.mf(V, 
                    method = "nmf", 
                    rank = rank, 
                    seed = "random_vcol", 
                    max_iter = 12, 
                    update = 'euclidean', 
                    objective = 'conn',
                    min_residuals = 1,
                    initialize_only = True)
        print "Factorization %s running with rank %d ..." % (model.name, model.rank)
        fit = mf.mf_run(model)
        print "... %d iterations performed" % fit.fit.n_iter
        # Compute connectivity matrix of factorization.
        # Again, we could use multiple runs support of the MF library, track factorization model across 50 runs and then
        # just call fit.consensus()
        consensus += fit.fit.connectivity()
    # averaging connectivity matrices
    consensus /= 50.
    # reorder consensus matrix
    perm = reorder(consensus)
    # display heatmap
    plt.set_cmap("RdBu")
    plt.imshow(perm) 
    plt.savefig("all_aml_consensus" + rank + ".png")

if __name__ == "__main__":
    """Run the ALL AML example."""
    run()
    