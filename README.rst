
MF is a Python scripting library which includes a number of published matrix factorization algorithms, initialization methods, quality and performance measures and facilitates the combination of these to produce new strategies. The library represents a unified and efficient interface to matrix factorization algorithms and methods.

The library has support for multiple runs of the algorithms which can be used for some quality measures. By setting runtime specific options tracking the residuals error within one (or more) run or tracking 
fitted factorization model is possible. Extensive documentation with working examples which demonstrate real applications, commonly used benchmark data and visualization methods are provided to help with the interpretation and comprehension of the results.

Project wiki is at http://orange.biolab.si/trac/wiki/MatrixFactorization. MF is a result of the Google Summer of Code 2011 program by the `Orange`_ organization. 

.. _Orange: http://orange.biolab.si

Content
=======

**Matrix Factorization Methods**

    **BD - Bayesian nonnegative matrix factorization Gibbs sampler**

    Schmidt, M.N., Winther, O.,  and Hansen, L.K., (2009). Bayesian Non-negative Matrix Factorization. In Proceedings of ICA. 2009, 540-547.    

    :BMF:

    :ICM:

    :LFNMF:

    :LSNMF:

    :NMF:

    :NSNMF:

    :PMF:

    :PSMF:

    :SNMF:

    :SNMNMF:

**Initialization Methods**

    - Random
    - Fixed
    - NNDSVD
    - Random C
    - Random VCol

**Quality and Performance Measures**

    - Distance
    - Residuals
    - Connectivity matrix
    - Consensus matrix
    - Entropy of the NMF model
    - Dominant basis components computation
    - Explained variance
    - Feature score computation representing its specificity to basis vectors
    - Computation of most basis specific features for basis vectors
    - Purity
    - Residual sum of squares
    - Sparseness
    - Cophenetic correlation coefficient of consensus matrix
    - Dispersion
    - Selected matrix factorization method specific

Install
=======



Configuration
=============



Usage
====




