
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

from os.path import dirname, abspath
from os.path import join
from warnings import warn

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np

import nimfa

try:
    from matplotlib.pyplot import savefig, imshow, set_cmap
except ImportError as exc:
    warn("Matplotlib must be installed to run Medulloblastoma example.")


def run():
    """Run Standard NMF on medulloblastoma data set. """
    V = read()
    for rank in range(2, 4):
        run_one(V, rank)


def run_one(V, rank):
    """
    Run standard NMF on medulloblastoma data set. 50 runs of Standard NMF are performed and obtained consensus matrix
    averages all 50 connectivity matrices.  
    
    :param V: Target matrix with gene expression data.
    :type V: `numpy.ndarray`
    :param rank: Factorization rank.
    :type rank: `int`
    """
    print("================= Rank = %d =================" % rank)
    consensus = np.zeros((V.shape[1], V.shape[1]))
    for i in range(50):
        nmf = nimfa.Nmf(V, rank=rank, seed="random_vcol", max_iter=200, update='euclidean',
                         objective='conn', conn_change=40)
        fit = nmf()
        print("Algorithm: %s\nInitialization: %s\nRank: %d" % (nmf, nmf.seed, nmf.rank))
        consensus += fit.fit.connectivity()
    consensus /= 50.
    p_consensus = reorder(consensus)
    plot(p_consensus, rank)


def plot(C, rank):
    """
    Plot reordered consensus matrix.
    
    :param C: Reordered consensus matrix.
    :type C: `numpy.ndarray`
    :param rank: Factorization rank.
    :type rank: `int`
    """
    set_cmap("RdBu_r")
    imshow(C)
    savefig("medulloblastoma_consensus_%s.png" % rank)


def reorder(C):
    """
    Reorder consensus matrix.
    
    :param C: Consensus matrix.
    :type C: `numpy.ndarray`
    """
    Y = 1 - C
    Z = linkage(squareform(Y), method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]


def read(normalize=False):
    """
    Read the medulloblastoma gene expression data. The matrix's shape is 5893 (genes) x 34 (samples). 
    It contains only positive data.
    
    Return the gene expression data matrix. 
    """
    fname = join(dirname(dirname(abspath(__file__))), 'datasets', 'Medulloblastoma',  'Medulloblastoma_data.txt')
    V = np.loadtxt(fname)
    if normalize:
        V = (V - V.min()) / (V.max() - V.min())
    return V


if __name__ == "__main__":
    """Run the medulloblastoma example."""
    run()
