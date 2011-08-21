
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
       matrices computed at rank = 2, 3 for the leukemia data set with the 5000 most highly varying genes 
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
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from os.path import dirname, abspath

def run():
    """Run Standard NMF on Leukemia data set."""    
    # read gene expression data
    V = __read()
    for rank in xrange(2, 4):
         _run(V, rank)

def _run(V, rank):
    """
    Run standard NMF on Leukemia data set.
    
    :param V: Target matrix with gene expression data.
    :type V: `numpy.matrix` (of course it could be any format of scipy.sparse, but we will use numpy here) 
    :param rank: Factorization rank.
    :type rank: `int`
    """
    print "================= Rank = %d =================" % rank
    consensus = np.mat(np.zeros((V.shape[1], V.shape[1])))
    for i in xrange(50):
        # Standard NMF with euclidean update equations is used. For initialization random Vcol method is used. 
        # Objective function is the number of consecutive iterations in which the connectivity matrix has not changed. 
        # We demand that factorization does not terminate before 30 consecutive iterations in which connectivity matrix
        # does not change. For a backup we also specify the maximum number of iterations. Note that the satisfiability
        # of one stopping criteria terminates the run (there is no chance for divergence). 
        model = mf.mf(V, 
                    method = "nmf", 
                    rank = rank, 
                    seed = "random_vcol", 
                    max_iter = 200, 
                    update = 'euclidean', 
                    objective = 'conn',
                    conn_change = 40,
                    initialize_only = True)
        fit = mf.mf_run(model)
        print "%d / 50 :: %s running with  ... %d iterations ..." % (i + 1, fit.fit, fit.fit.n_iter)
        # Compute connectivity matrix of factorization.
        # Again, we could use multiple runs support of the MF library, track factorization model across 50 runs and then
        # just call fit.consensus()
        consensus += fit.fit.connectivity()
    # averaging connectivity matrices
    consensus /= 50.
    # reorder consensus matrix
    p_consensus, ivl = __reorder(consensus)
    # display heatmap
    plt.set_cmap("RdBu_r")
    plt.imshow(np.array(p_consensus)) 
    plt.savefig("all_aml_consensus" + str(rank) + ".png")
    
def __reorder(C):
    """
    Reorder consensus matrix.
    
    :param C: Consensus matrix.
    :type C: `numpy.matrix`
    """    
    c_vec = np.array([C[i,j] for i in xrange(C.shape[0] - 1) for j in xrange(i + 1, C.shape[1])])
    # convert similarities to distances
    Y = 1 - c_vec
    Z = linkage(Y, method = 'average')
    # get node ids as they appear in the tree from left to right(corresponding to observation vector idx)
    ivl = leaves_list(Z)
    return C[ivl], ivl
    
def __read():
    """
    Read ALL AML gene expression data. The matrix's shape is 5000 (genes) x 38 (samples). 
    It contains only positive data.
    """
    V = np.matrix(np.zeros((5000, 38)))
    i = 0
    for line in open(dirname(dirname(abspath(__file__)))+ '/datasets/ALL_AML_data.txt'):
        V[i, :] =  map(float, line.split('\t'))
        i += 1
    return V

if __name__ == "__main__":
    """Run the ALL AML example."""
    run()
    