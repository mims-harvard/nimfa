
"""
    #######################################
    ALL AML Leukemia (``examples.aml_all``)
    #######################################
    
    This module demonstrates the ability of NMF to recover meaningful biological information from 
    cancer related microarray data. NMF appears to have advantages over other methods such as HC or SOM. 
    Instead of separating gene clusters based on distance computation, NMF detects context-dependent patterns 
    of gene expression in complex biological systems.
    
    `Leukemia`_ data set is used in this example. This data set is a benchmark in the cancer classification
    community. It contains two ALL samples that are consistently misclassified or classified with low 
    confidence by most methods. There are a number of possible explanations for this, 
    including incorrect diagnosis of the samples. They are included them in example.The distinction between AML 
    and ALL, as well as the division of ALL into T and B cell subtypes is well known. 
    
    .. note:: Leukemia data set used in this example is included in the `datasets` and does not need to be
              downloaded. However, download links are listed in the ``datasets``. To run the example, the data set
              must exist in the ``ALL_AML`` directory under `data sets`. 
    
    .. _Leukemia: http://orange.biolab.si/data sets/leukemia.htm 
    
    This example is inspired by [Brunet2004]_. In [Brunet2004]_ authors applied NMF to the leukemia data set. With rank, rank = 2, 
    NMF recovered the AML-ALL biological distinction with high accuracy and robustness. Higher ranks revealed further
    partitioning of the samples. Clear block diagonal patterns in reordered consensus matrices attest to the 
    robustness of models with 2, 3 and 4 classes. 
    
    .. figure:: /images/all_aml_consensus2.png
       :scale: 60 %
       :alt: Consensus matrix generated for rank, rank = 2. 
       :align: center

       Reordered consensus matrix generated for rank, rank = 2. Reordered consensus matrix averages 50 connectivity 
       matrices computed at rank = 2, 3 for the leukemia data set with the 5000 most highly varying genes 
       according to their coefficient of variation. Samples are hierarchically clustered by using 
       distances derived from consensus clustering matrix entries, coloured from 0 (deep blue, samples
       are never in the same cluster) to 1 (dark red, samples are always in the same cluster).   
       
       
    .. figure:: /images/all_aml_consensus3.png
       :scale: 60 %
       :alt: Consensus matrix generated for rank, rank = 3.
       :align: center 

       Reordered consensus matrix generated for rank, rank = 3.
    
    
    .. table:: Standard NMF Class assignments obtained with this example for rank = 2 and rank = 3. 

       ====================  ========== ==========
              Sample          rank = 2   rank = 3
       ====================  ========== ==========
        ALL_19769_B-cell        0            2
        ALL_23953_B-cell        0            2
        ALL_28373_B-cell        0            2
        ALL_9335_B-cell         0            2
        ALL_9692_B-cell         0            2
        ALL_14749_B-cell        0            2
        ALL_17281_B-cell        0            2
        ALL_19183_B-cell        0            2
        ALL_20414_B-cell        0            2
        ALL_21302_B-cell        0            1
        ALL_549_B-cell          0            2
        ALL_17929_B-cell        0            2
        ALL_20185_B-cell        0            2
        ALL_11103_B-cell        0            2
        ALL_18239_B-cell        0            2
        ALL_5982_B-cell         0            2
        ALL_7092_B-cell         0            2
        ALL_R11_B-cell          0            2
        ALL_R23_B-cell          0            2
        ALL_16415_T-cell        0            1
        ALL_19881_T-cell        0            1
        ALL_9186_T-cell         0            1
        ALL_9723_T-cell         0            1
        ALL_17269_T-cell        0            1
        ALL_14402_T-cell        0            1
        ALL_17638_T-cell        0            1
        ALL_22474_T-cell        0            1       
        AML_12                  1            0
        AML_13                  0            0
        AML_14                  1            1
        AML_16                  1            0
        AML_20                  1            0
        AML_1                   1            0
        AML_2                   1            0
        AML_3                   1            0
        AML_5                   1            0 
        AML_6                   1            0
        AML_7                   1            0
       ====================  ========== ========== 
    
    To run the example simply type::
        
        python all_aml.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.all_aml.run()
        
    .. note:: This example uses ``matplotlib`` library for producing a heatmap of a consensus matrix.
"""

from os.path import dirname, abspath, join
from warnings import warn

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np

import nimfa

try:
    from matplotlib.pyplot import savefig, imshow, set_cmap
except ImportError as exc:
    warn("Matplotlib must be installed to run ALL AML example.")


def run():
    """Run Standard NMF on leukemia data set. """
    V = read()
    for rank in range(2, 4):
        run_one(V, rank)


def run_one(V, rank):
    """
    Run standard NMF on leukemia data set. 50 runs of Standard NMF are performed and obtained consensus matrix
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
        print("%2d/50 : %s - init: %s (%3d/200 iterations)" % (i + 1, fit.fit,
                                                               fit.fit.seed, fit.fit.n_iter))
        consensus += fit.fit.connectivity()
    consensus /= 50.
    p_consensus = reorder(consensus)
    plot(p_consensus, rank)


def plot(C, rank):
    """
    Plot reordered consensus matrix.

    :param C: Reordered consensus matrix.
    :type C: numpy.ndarray`
    :param rank: Factorization rank.
    :type rank: `int`
    """
    imshow(C)
    set_cmap("RdBu_r")
    savefig("all_aml_consensus_%d.png" % rank)


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


def read():
    """
    Read ALL AML gene expression data. The matrix's shape is 5000 (genes) x 38 (samples).
    It contains only positive data.

    Return the gene expression data matrix.
    """
    fname = join(dirname(dirname(abspath(__file__))), 'datasets', 'ALL_AML', 'ALL_AML_data.txt')
    V = np.loadtxt(fname)
    return V


if __name__ == "__main__":
    """Run the ALL AML example."""
    run()
