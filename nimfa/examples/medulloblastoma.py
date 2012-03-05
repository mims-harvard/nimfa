
"""
    ##############################################
    Medulloblastoma (``examples.medulloblastoma``)
    ##############################################

    This module demonstrates the ability of NMF to recover meaningful biological information from childhood 
    brain tumors microarray data. 
    
    Medulloblastoma data set is used in this example. The pathogenesis of these childhood brain tumors is not well 
    understood but is accepted that there are two known histological subclasses; classic (C) and desmoplastic (D). 
    These subclasses can be clearly seen under microscope.   
    
    .. note:: Medulloblastoma data set used in this example is included in the `datasets` and does not need to be
          downloaded. However, download links are listed in the ``datasets``. To run the example, the data set
          must exist in the ``Medulloblastoma`` directory under `datasets`. 
    
    This example is inspired by [Brunet2004]_. In [Brunet2004]_ authors applied NMF to the medulloblastoma data set and managed to expose a
    separate desmoplastic (D) class. In [Brunet2004]_ authors also applied SOM and HC to these data but were unable to find a distinct
    desmoplastic class. Using HC desmoplastic samples were scattered among leaves and there was no level of the tree
    where they could split the branches to expose a clear desmoplastic cluster. They applied SOM by using two to eight 
    centroids but did not recover distinct desmoplastic class as well. 
    
    .. figure:: /images/medulloblastoma_consensus2.png
       :scale: 60 %
       :alt: Consensus matrix generated for rank, rank = 2.
       :align: center 
       
       Reordered consensus matrix generated for rank, rank = 2. Reordered consensus matrix averages 50 connectivity 
       matrices computed at rank = 2, 3 for the medulloblastoma data set consisting of 25 classic and 9 desmoplastic
       medulloblastoma tumors. Consensus matrix is reordered with HC by using distances derived from consensus clustering 
       matrix entries, coloured from 0 (deep blue, samples are never in the same cluster) to 1 (dark red, samples are 
       always in the same cluster).   
       
    .. figure:: /images/medulloblastoma_consensus3.png
       :scale: 60 %
       :alt: Consensus matrix generated for rank, rank = 3. 
       :align: center
       
       Reordered consensus matrix generated for rank, rank = 3.
       
       
    .. table:: Standard NMF Class assignments results obtained with this example for rank = 2, rank = 3 and rank = 5.  

       ====================  ========== ========== ========== ==========
              Sample           Class     rank = 2   rank = 3   rank = 5 
       ====================  ========== ========== ========== ==========
        Brain_MD_7                C        0            1        3
        Brain_MD_59               C        1            0        2
        Brain_MD_20               C        1            1        3
        Brain_MD_21               C        1            1        3
        Brain_MD_50               C        1            1        4
        Brain_MD_49               C        0            2        3
        Brain_MD_45               C        1            1        3
        Brain_MD_43               C        1            1        3
        Brain_MD_8                C        1            1        3
        Brain_MD_42               C        0            2        4
        Brain_MD_1                C        0            2        3
        Brain_MD_4                C        0            2        3 
        Brain_MD_55               C        0            2        3
        Brain_MD_41               C        1            1        2
        Brain_MD_37               C        1            0        3
        Brain_MD_3                C        1            2        3
        Brain_MD_34               C        1            2        4
        Brain_MD_29               C        1            1        2
        Brain_MD_13               C        0            1        2
        Brain_MD_24               C        0            1        3
        Brain_MD_65               C        1            0        2
        Brain_MD_5                C        1            0        1
        Brain_MD_66               C        1            0        1
        Brain_MD_67               C        1            0        3
        Brain_MD_58               C        0            2        3
        Brain_MD_53               D        0            2        4
        Brain_MD_56               D        0            2        4
        Brain_MD_16               D        0            2        4
        Brain_MD_40               D        0            1        0
        Brain_MD_35               D        0            2        4
        Brain_MD_30               D        0            2        4
        Brain_MD_23               D        0            2        4
        Brain_MD_28               D        1            2        1
        Brain_MD_60               D        1            0        0
       ====================  ========== ========== ========== ==========   
    
    To run the example simply type::
        
        python medulloblastoma.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.medulloblastoma.run()
        
    .. note:: This example uses ``matplotlib`` library for producing a heatmap of a consensus matrix.
"""

import nimfa
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from os.path import dirname, abspath, sep
from warnings import warn

try:
    from matplotlib.pyplot import savefig, imshow, set_cmap
except ImportError, exc:
    warn("Matplotlib must be installed to run Medulloblastoma example.")

def run():
    """Run Standard NMF on medulloblastoma data set. For each rank 50 Standard NMF runs are performed. """    
    # read gene expression data
    V = read()
    for rank in xrange(2, 4):
        run_one(V, rank)

def run_one(V, rank):
    """
    Run standard NMF on medulloblastoma data set. 50 runs of Standard NMF are performed and obtained consensus matrix
    averages all 50 connectivity matrices.  
    
    :param V: Target matrix with gene expression data.
    :type V: `numpy.matrix` (of course it could be any format of scipy.sparse, but we will use numpy here) 
    :param rank: Factorization rank.
    :type rank: `int`
    """
    print "================= Rank = %d =================" % rank
    consensus = np.mat(np.zeros((V.shape[1], V.shape[1])))
    for i in xrange(50):
        # Standard NMF with Euclidean update equations is used. For initialization random Vcol method is used. 
        # Objective function is the number of consecutive iterations in which the connectivity matrix has not changed. 
        # We demand that factorization does not terminate before 30 consecutive iterations in which connectivity matrix
        # does not change. For a backup we also specify the maximum number of iterations. Note that the satisfiability
        # of one stopping criteria terminates the run (there is no chance for divergence). 
        model = nimfa.mf(V, 
                    method = "nmf", 
                    rank = rank, 
                    seed = "random_vcol", 
                    max_iter = 200, 
                    update = 'euclidean', 
                    objective = 'conn',
                    conn_change = 40,
                    initialize_only = True)
        fit = nimfa.mf_run(model)
        print "%2d / 50 :: %s - init: %s ran with  ... %3d / 200 iters ..." % (i + 1, fit.fit, fit.fit.seed, fit.fit.n_iter)
        # Compute connectivity matrix of factorization.
        # Again, we could use multiple runs support of the nimfa library, track factorization model across 50 runs and then
        # just call fit.consensus()
        consensus += fit.fit.connectivity()
    # averaging connectivity matrices
    consensus /= 50.
    # reorder consensus matrix
    p_consensus = reorder(consensus)
    # plot reordered consensus matrix 
    plot(p_consensus, rank)
    
def plot(C, rank):
    """
    Plot reordered consensus matrix.
    
    :param C: Reordered consensus matrix.
    :type C: `numpy.matrix`
    :param rank: Factorization rank.
    :type rank: `int`
    """
    set_cmap("RdBu_r")
    imshow(np.array(C)) 
    savefig("medulloblastoma_consensus" + str(rank) + ".png")
    
def reorder(C):
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
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]
    
def read(normalize = False):
    """
    Read the medulloblastoma gene expression data. The matrix's shape is 5893 (genes) x 34 (samples). 
    It contains only positive data.
    
    Return the gene expression data matrix. 
    """
    V = np.matrix(np.zeros((5893, 34)))
    i = 0
    for line in open(dirname(dirname(abspath(__file__)))+ sep + 'datasets' + sep + 'Medulloblastoma' + sep + 'Medulloblastoma_data.txt'):
        V[i, :] =  map(float, line.split('\t'))
        i += 1
    if normalize:
        V -= V.min()
        V /= V.max()
    return V

if __name__ == "__main__":
    """Run the medulloblastoma example."""
    run()
    