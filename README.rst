
MF is a Python scripting library which includes a number of published matrix factorization algorithms, initialization methods, quality and performance measures and facilitates the combination of these to produce new strategies. The library represents a unified and efficient interface to matrix factorization algorithms and methods.

The MF works with numpy dense matrices and scipy sparse matrices (where this is possible to save on space). The library has support for multiple runs of the algorithms which can be used for some quality measures. By setting runtime specific options tracking the residuals error within one (or more) run or tracking fitted factorization model is possible. Extensive documentation with working examples which demonstrate real applications, commonly used benchmark data and visualization methods are provided to help with the interpretation and comprehension of the results.

Project wiki is at http://orange.biolab.si/trac/wiki/MatrixFactorization. MF is a result of the Google Summer of Code 2011 program by the `Orange`_ organization. 

.. _Orange: http://orange.biolab.si

.. image:: http://orange.biolab.si/small-logo.png
	:target: http://orange.biolab.si
	

.. image:: http://code.google.com/p/google-summer-of-code/logo?cct=1280260724
	:target: http://code.google.com/soc/

Content
=======

**Matrix Factorization Methods**

    **BD - Bayesian nonnegative matrix factorization Gibbs sampler**

        Schmidt, M.N., Winther, O.,  and Hansen, L.K., (2009). Bayesian Non-negative Matrix Factorization. In Proceedings of ICA. 2009, 540-547.    

    **BMF - Binary matrix factorization**

        Zhang Z., Li T., Ding C. H. Q., Zhang X., (2007). Binary Matrix Factorization with Applications. ICDM 2007.

    **ICM - Iterated conditional modes nonnegative matrix factorization**

        Schmidt, M.N., Winther, O.,  and Hansen, L.K., (2009). Bayesian Non-negative Matrix Factorization. In Proceedings of ICA. 2009, 540-547. 

    **LFNMF - Fisher nonnegative matrix factorization for learning Local features**

        Wang, Y., et. al., (2004). Fisher non-negative matrix factorization for learning local features. Proc. Asian Conf. on Comp. Vision. 2004.    

        Li, S. Z., et. al., (2001). Learning spatially localized, parts-based representation. Proc. of the 2001 IEEE Comp. Soc. Conf. on Comp. Vision and Pattern Recognition. CVPR 2001, I-207-I-212. IEEE Comp. Soc. doi: 10.1109/CVPR.2001.990477.

    **LSNMF - Alternative nonnegative least squares matrix factorization using projected gradient method for subproblems**

        Lin, C.-J., (2007). Projected gradient methods for nonnegative matrix factorization. Neural computation, 19(10), 2756-79. doi: 10.1162/neco.2007.19.10.2756.

    **NMF - Standard nonnegative matrix factorization with Euclidean/Kullback-Leibler update equations and Frobenius/divergence/connectivity cost functions**

        Lee, D..D., and Seung, H.S., (2001). Algorithms for Non-negative Matrix Factorization, Adv. Neural Info. Proc. Syst. 13, 556-562.

        Brunet, J.-P., Tamayo, P., Golub, T. R., Mesirov, J. P., (2004). Metagenes and molecular pattern discovery using matrix factorization. Proceedings of the National Academy of Sciences of the United States of America, 101(12), 4164-9. doi: 10.1073/pnas.0308531101.

    **NSNMF - Nonsmooth nonnegative matrix factorization**

        Pascual-Montano, A., Carazo, J. M., Kochi, K., Lehmann, D., and Pascual-Marqui, R. D., (2006). Nonsmooth nonnegative matrix factorization (nsnmf). IEEE transactions on pattern analysis and machine intelligence, 28(3), 403-415.

    **PMF - Probabilistic nonnegative matrix factorization**

        Laurberg, H.,et. al., (2008). Theorems on positive data: on the uniqueness of NMF. Computational intelligence and neuroscience.

        Hansen, L. K., (2008). Generalization in high-dimensional factor models. Web: http://www.stanford.edu/group/mmds/slides2008/hansen.pdf.

    **PSMF - Probabilistic sparse matrix factorization**

        Dueck, D., Morris, Q. D., Frey, B. J, (2005). Multi-way clustering of microarray data using probabilistic sparse matrix factorization. Bioinformatics 21. Suppl 1, i144-51.

        Deck, D., Frey, B. J., (2004). Probabilistic Sparse Matrix Factorization Probabilistic Sparse Matrix Factorization. University of Toronto Technical Report PSI-2004-23.

        Srebro, N. and Jaakkola, T., (2001). Sparse Matrix Factorization of Gene Expression Data. Unpublished note, MIT Artificial Intelligence Laboratory.

        Li, H., Sun, Y., Zhan, M., (2007). The discovery of transcriptional modules by a two-stage matrix decomposition approach. Bioinformatics, 23(4), 473-479.

    **SNMF - Sparse nonnegative matrix factorization based on alternating nonnegativity constrained least squares**
    
        Kim H., Park H., (2007). Sparse Non-negative Matrix Factorizations via Alternating Non-negativity-constrained Least Squares for Microarray Data Analysis, Bioinformatics.

    **SNMNMF - Sparse network regularized multiple nonnegative matrix factorization**

        ﻿Zhang, S. et. al., (2011). A novel computational framework for simultaneous integration of multiple types of genomic data to identify microRNA-gene regulatory modules. Bioinformatics 2011, 27(13), i401-i409. doi:10.1093/bioinformatics/btr206.

**Initialization Methods**

    - Random
    - Fixed
    - NNDSVD 
    	Boutsidis, C., Gallopoulos, E., (2007). SVD-based initialization: A head start for nonnegative matrix factorization, Pattern Recognition, 2007, doi:10.1016/j.patcog.2007.09.010.
    - Random C 
    	Albright, R. et al., (2006). Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, (919), p.1-18.
    - Random VCol 
		Albright, R. et al., (2006). Algorithms, initializations, and convergence for the nonnegative matrix factorization. Matrix, (919), p.1-18.

**Quality and Performance Measures**

    - Distance
    - Residuals
    - Connectivity matrix
    - Consensus matrix
    - Entropy of the fitted NMF model (Kim, Park, 2007)
    - Dominant basis components computation
    - Explained variance
    - Feature score computation representing its specificity to basis vectors (Kim, Park, 2007)
    - Computation of most basis specific features for basis vectors (Kim, Park, 2007)
    - Purity (Kim, Park, 2007)
    - Residual sum of squares - can be used for rank estimate (Hutchins, 2008) (Frigyesi, Hoglund, 2008)
    - Sparseness (Hoyer, 2004)
    - Cophenetic correlation coefficient of consensus matrix - can be used for rank estimate (Brunet, 2004)
    - Dispersion (Kim, Park, 2007)
    - Selected matrix factorization method specific

Install
=======

No special installation procedure is specified. However, the MF library makes extensive use of `SciPy`_ and `NumPy`_ libraries for fast and convenient deanse and sparse matrix manipulation and some linear
algebra operations. There are not any additional prerequisites. 

.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/

To build and install run::
	
	python setup.py install

Configuration
=============

Methods configuration goes through runtime specific options (e. g. tracking fitted model across multiple runs, tracking residuals across iterations, etc.) or algorithm specific options (e. g. prior 
information with PSMF, type of update equations with NMF, initial value for noise variance with ICM, etc.). 

For details and descriptions on algorithm specific options see specific algorithm documentation. For deatils on runtime specific options and explanation of the general model parameters see :mod:`mf_run`.

Usage
====

Following are two basic usage examples that employ Standard NMF algorithm and LSNMF algorithm. For more see examples and 
methods' documentation.

Example No. 1::

    # Import MF library entry point for factorization
    import mf

    # Construct sparse matrix in CSR format, which will be our input for factorization
    from scipy.sparse import csr_matrix
    from scipy import array
    from numpy import dot
    V = csr_matrix((array([1,2,3,4,5,6]), array([0,2,2,0,1,2]), array([0,2,3,6])), shape=(3,3))

    # Print this tiny matrix in dense format
    print V.todense()

    # Run Standard NMF rank 3 algorithm
    # Update equations and cost function are Standard NMF specific parameters (among others). 
    # If not specified the Euclidean update and Forbenius cost function would be used.
    # We don't specify initialization method. Algorithm specific or random intialization will be used. 
    # In Standard NMF case, by default random is used.
    # Returned object is fitted factorization model. Through it user can access quality and performance measures.
    # The fit's attribute `fit` contains all the attributes of the factorization.
    fit = mf.mf(V, method = "nmf", max_iter = 30, rank = 3, update = 'divergence', objective = 'fro')

    # Basis matrix. It is sparse, as input V was sparse as well. 
    W = fit.basis()
    print "Basis matrix"
    print W.todense()

    # Mixture matrix. We print this tiny matrix in dense format.
    H = fit.coef()
    print "Coef"
    print H.todense()

    # Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
    print "Distance Kullback-Leibler", fit.distance(metric = "kl")

    # Compute generic set of measures to evaluate the quality of the factorization
    sm = fit.summary()
    # Print sparseness (Hoyer, 2004) of basis and mixture matrix
    print "Sparseness Basis: %5.3f  Mixture: %5.3f" % (sm['sparseness'][0], sm['sparseness'][1])
    # Print actual number of iterations performed
    print "Iterations", sm['n_iter']

    # Print estimate of target matrix V 
    print "Estimate"
    print dot(W.todense(), H.todense())

Example No. 2::

	# Import MF library entry point for factorization
	import mf
	
	# Here we will work with numpy matrix
	import numpy as np
	V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])
	
	# Print this tiny matrix 
	print V
	
	# Run LSNMF rank 3 algorithm
	# We don't specify any algorithm specific parameters. Defaults will be used.
	# We don't specify initialization method. Algorithm specific or random intialization will be used. 
	# In LSNMF case, by default random is used.
	# Returned object is fitted factorization model. Through it user can access quality and performance measures.
	# The fit's attribute `fit` contains all the attributes of the factorization.  
	fit = mf.mf(V, method = "lsnmf", max_iter = 10, rank = 3)
	
	# Basis matrix.
	W = fit.basis()
	print "Basis matrix"
	print W
	
	# Mixture matrix. 
	H = fit.coef()
	print "Coef"
	print H
	
	# Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
	print "Distance Kullback-Leibler", fit.distance(metric = "kl")
	
	# Compute generic set of measures to evaluate the quality of the factorization
	sm = fit.summary()
	# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
	print "Rss: %8.3f" % sm['rss']
	# Print explained variance.
	print "Evar: %8.3f" % sm['evar']
	# Print actual number of iterations performed
	print "Iterations", sm['n_iter']
	
	# Print estimate of target matrix V 
	print "Estimate"
	print np.dot(W, H)
	


