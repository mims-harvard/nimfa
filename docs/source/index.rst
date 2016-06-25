
Welcome to Nimfa
================

`Nimfa`_ is a Python library for nonnegative matrix factorization. It includes implementations of several factorization methods, initialization approaches, and quality scoring.
Both dense and sparse matrix representation are supported.

Nimfa is distributed under the BSD license.

****

The sample script using Nimfa on medulloblastoma gene expression data is given below. It uses alternating least squares nonnegative matrix 
factorization with projected gradient method for subproblems [Lin2007]_ and Random Vcol [Albright2006]_ initialization algorithm. An object returned by ``nimfa.mf_run`` is 
fitted factorization model through which user can access matrix factors and estimate quality measures.
    
.. literalinclude:: /code/usage.py

Running this script produces the following output, where slight differences in reported scores across different 
runs can be attributed to randomness of the Random Vcol initialization method::
	
	Rss: 0.2668
	Evar: 0.9997
	K-L divergence: 38.8744
	Sparseness, W: 0.7297, H: 0.8796

****

An IPython notebook shows how to `use Nimfa to analyze breast cancer transcriptome
data sets`_ from The International Cancer Genome Consortium (`ICGC`_). The analysis of cancer data was highlighted in the `ACM XRDS magazine`_.

.. _use Nimfa to analyze breast cancer transcriptome data sets: http://nbviewer.ipython.org/github/marinkaz/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb
.. _ICGC: https://dcc.icgc.org
.. _ACM XRDS magazine: http://dl.acm.org/citation.cfm?id=2809623.2788526&coll=portal&dl=ACM
.. _Nimfa: http://jmlr.csail.mit.edu/papers/v13/zitnik12a.html


###################
Scripting Reference
###################

.. toctree::
   :maxdepth: 3

   nimfa.mf_run

   nimfa.models
   
   nimfa.methods
   
   nimfa.utils
   
   nimfa.examples
   
   nimfa.datasets
	


#######
Content
#######

************************
Factorization Algorithms
************************

.. include:: /content-factorization.rst

*************************
Initialization Algorithms
*************************

.. include:: /content-initialization.rst

****************
Quality Measures
****************

.. include:: /content-quality-performance.rst

*****
Utils
*****

.. include:: /content-utils.rst


	
############
Installation
############

Nimfa is compatible with Python 2 and Python 3 versions.
The recommended way to install Nimfa is by issuing::

	pip install nimfa

from the command line.

Nimfa makes extensive use of `SciPy`_ and `NumPy`_
libraries for fast and convenient dense and sparse matrix manipulation and some linear
algebra operations. There are not any additional prerequisites.

.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/

Alternatively, you can download source code from `Github`_.

.. _Github: http://github.com/marinkaz/mf

Unzip the archive. To build and install run::
	
	python setup.py install



#############
Configuration
#############

Methods configuration goes through:

	#. runtime specific options (e. g. tracking fitted model across multiple runs, tracking residuals across iterations, etc.);
	#. algorithm specific options (e. g. prior information with PSMF, type of update equations with NMF, initial value for noise variance with ICM, etc.). 

For details and descriptions on algorithm specific options see specific algorithm documentation. For details on runtime specific options and explanation of the general model parameters see :mod:`mf_run`.



#####
Usage
#####

Following are basic usage examples that employ different implemented factorization algorithms.

Standard NMF - Divergence on ``scipy.sparse`` matrix with matrix factors estimation. 

.. literalinclude:: /code/usage1.py


LSNMF on ``numpy`` dense matrix with quality and performance measures.

.. literalinclude:: /code/usage2.py


LSNMF with Random VCol initialization and error tracking.

.. literalinclude:: /code/usage3.py


LSNMF with Random VCol initialization and rank estimation.

.. literalinclude:: /code/usage7.py

   		
ICM with Random C initialization and passed callback initialization function.
   
.. literalinclude:: /code/usage4.py

   		
BMF with default parameters, multiple runs and factor tracking for consensus matrix computation.
   
.. literalinclude:: /code/usage5.py


Standard NMF - Euclidean update equations and fixed initialization (passed matrix factors).

.. literalinclude:: /code/usage6.py



##########
References
##########

.. [Schmidt2009] Mikkel N. Schmidt, Ole Winther, and Lars K. Hansen. Bayesian non-negative matrix factorization. In Proceedings of the 9th International Conference on Independent Component Analysis and Signal Separation, pages 540-547, Paraty, Brazil, 2009.

.. [Zhang2007] Zhongyuan Zhang, Tao Li, Chris H. Q. Ding and Xiangsun Zhang. Binary Matrix Factorization with applications. In Proceedings of 7th IEEE International Conference on Data Mining, pages 391-400, Omaha, USA, 2007.

.. [Wang2004] Yuan Wang, Yunde Jia, Changbo Hu and Matthew Turk. Fisher non-negative matrix factorization for learning local features. In Proceedings of the 6th Asian Conference on Computer Vision, pages 27-30, Jeju, Korea, 2004.

.. [Li2001] Stan Z. Li, Xinwen Huo, Hongjiang Zhang and Qian S. Cheng. Learning spatially localized, parts-based representation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 207-212, Kauai, USA, 2001.

.. [Lin2007] Chin J. Lin. Projected gradient methods for nonnegative matrix factorization. Neural Computation, 19(10): 2756-2779, 2007.

.. [Lee2001] Daniel D. Lee and H. Sebastian Seung. Algorithms for non-negative matrix factorization. In Proceedings of the Neural Information Processing Systems, pages 556-562, Vancouver, Canada, 2001.

.. [Lee1999] Daniel D. Lee and H. Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755): 788-791, 1999.

.. [Brunet2004] Jean-P. Brunet, Pablo Tamayo, Todd R. Golub and Jill P. Mesirov. Metagenes and molecular pattern discovery using matrix factorization. In Proceedings of the National Academy of Sciences of the USA, 101(12): 4164-4169, 2004.
  
.. [Montano2006] Alberto Pascual-Montano, J. M. Carazo, Kieko Kochi, Dietrich Lehmann and Roberto D. Pascual-Marqui. Nonsmooth nonnegative matrix factorization (nsnmf). In IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(3): 403-415, 2006.

.. [Laurberg2008] Hans Laurberg, Mads G. Christensen, Mark D. Plumbley, Lars K. Hansen and Soren H. Jensen. Theorems on positive data: on the uniqueness of NMF. Computational Intelligence and Neuroscience, doi: 10.1155/2008/764206, 2008.

.. [Hansen2008] Lars K. Hansen. Generalization in high-dimensional factor models. Web: http://www.stanford.edu/group/mmds/slides2008/hansen.pdf, 2008.

.. [Dueck2005] Delbert Dueck, Quaid D. Morris and Brendan J. Frey. Multi-way clustering of microarray data using probabilistic sparse matrix factorization. Bioinformatics, 21(Suppl 1): 144-151, 2005.

.. [Dueck2004] Delbert Dueck and Brendan J. Frey. Probabilistic sparse matrix factorization. University of Toronto Technical Report PSI-2004-23, Probabilistic and Statistical Inference Group, University of Toronto, 2004.

.. [Srebro2001] Nathan Srebro and Tommi Jaakkola. Sparse matrix Factorization of gene expression data. Artificial Intelligence Laboratory, Massachusetts Institute of Technology, 2001.

.. [Li2007] Huan Li, Yu Sun and Ming Zhan. The discovery of transcriptional modules by a two-stage matrix decomposition approach. Bioinformatics, 23(4): 473-479, 2007.

.. [Park2007] Hyuonsoo Kim and Haesun Park. Sparse non-negative matrix factorizations via alternating non-negativity-constrained least squares for microarray data analysis. Bioinformatics, 23(12): 1495-1502, 2007.

.. [Zhang2011] Shihua Zhang, Qingjiao Li and Xianghong J. Zhou. A novel computational framework for simultaneous integration of multiple types of genomic data to identify microRNA-gene regulatory modules. Bioinformatics, 27(13): 401-409, 2011.

.. [Boutsidis2007] Christos Boutsidis and Efstratios Gallopoulos. SVD-based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 41(4): 1350-1362, 2008.

.. [Albright2006] Russell Albright, Carl D. Meyer and Amy N. Langville. Algorithms, initializations, and convergence for the nonnegative matrix factorization. NCSU Technical Report Math 81706, NC State University, Releigh, USA, 2006.

.. [Hoyer2004] Patrik O. Hoyer. Non-negative matrix factorization with sparseness constraints. Journal of Machine Learning Research, 5: 1457-1469, 2004.

.. [Hutchins2008] Lucie N. Hutchins, Sean P. Murphy, Priyam Singh and Joel H. Graber. Position-dependent motif characterization using non-negative matrix factorization. Bioinformatics, 24(23): 2684-2690, 2008.

.. [Frigyesi2008] Attila Frigyesi and Mattias Hoglund. Non-negative matrix factorization for the analysis of complex gene expression data: identification of clinically relevant tumor subtypes. Cancer Informatics, 6: 275-292, 2008.

.. [FWang2008] Fei Wang, Tao Li, Changshui Zhang. Semi-Supervised Clustering via Matrix Factorization. SDM 2008, 1-12, 2008.

.. [Schietgat2010] Leander Schietgat, Celine Vens, Jan Struyf, Hendrik Blockeel, Dragi Kocev and Saso Dzeroski. Predicting gene function using hierarchical multi-label decision tree ensembles. BMC Bioinformatics, 11(2), 2010.
 
.. [Schachtner2008] R. Schachtner, D. Lutter, P. Knollmueller, A. M. Tome, F. J. Theis, G. Schmitz, M. Stetter, P. Gomez Vilda and E. W. Lang. Knowledge-based gene expression classification via matrix factorization. Bioinformatics, 24(15): 1688-1697, 2008.

.. [Damle2014] Anil Damle, Sun Yuekai . A geometric approach to archetypal analysis and non-negative matrix factorization. arXiv preprint arXiv:1405.4275, 2014.

.. [Benson2014] Austin R. Benson, Jason D. Lee, Bartek Rajwa, David F. Gleich. Scalable methods for nonnegative matrix factorizations of near-separable tall-and-skinny matrices. NIPS, 945-953, 2014.

.. [Kumar2013] Abhishek Kumar, Vikas Sindhwani, Prabhanjan Kambadur. Fast conical hull algorithms for near-separable non-negative matrix factorization. ICML, 231-239, 2013.

.. [Gillis2014] Gillis, Nicolas, and Stephen A. Vavasis. Fast and robust recursive algorithms for separable nonnegative matrix factorization. IEEE TPAMI, 36(4): 698-714, 2014.

.. [Tepper2015] Mariano Tepper, Guillermo Sapiro. Compressed nonnegative matrix factorization is fast and accurate. IEEE TSP, 64(9): 2269-2283, 2016

.. [Kapralov2016] Michael, Kapralov, Vamsi Potluru, David Woodruff. How to fake multiply by a Gaussian Matrix. ICML, 2101-2110. 2016.


########
Citation
########

.. code-block:: none

    @article{Zitnik2012,
     title     = {Nimfa: A Python Library for Nonnegative Matrix Factorization},
     author    = {Zitnik, Marinka and Zupan, Blaz},
     journal   = {Journal of Machine Learning Research},
     volume    = {13},
     pages     = {849-853},
     year      = {2012}
   }

	
##########
Disclaimer	
##########	
	
This software and data is provided as-is, and there are no guarantees
that it fits your purposes or that it is bug-free. Use it at your own 
risk!
