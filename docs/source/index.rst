.. MF - Matrix Factorization Techniques for Data Mining documentation master file, created by
   sphinx-quickstart on Tue Aug 23 13:23:27 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
 

MF - Matrix Factorization Techniques for Data Mining
================================================================================

#####
About
#####

**MF - Matrix Factorization Techniques for Data Mining**  is a Python scripting library which includes a number of published matrix 
factorization algorithms, initialization methods, quality and performance measures and facilitates the combination of these to 
produce new strategies. The library represents a unified and efficient interface to matrix factorization algorithms and methods.

The MF works with numpy dense matrices and scipy sparse matrices (where this is possible to save on space). The library has support for 
multiple runs of the algorithms which can be used for some quality measures. By setting runtime specific options tracking the 
residuals error within one (or more) run or tracking fitted factorization model is possible. Extensive documentation with working 
examples which demonstrate real applications, commonly used benchmark data and visualization methods are provided to help with the 
interpretation and comprehension of the results.

Matrix factorization methods have been shown to be a useful decomposition for multivariate data as low dimensional data representations are crucial 
to numerous applications in statistics, signal processing and machine learning.

An incomplete list of applications of matrix factorization methods includes:

	* bioinformatics,
	* environmetrics and chemometrics,
	* image processing and computer graphics,
	* text analysis,
	* miscelllaneous, such as extracting speech features, transcription of polyphonic music passages, object characterization, spectral data 
	  analysis, multiway clustering, learning sound dictionaries, etc. 

With the library of factorization methods in Orange an easy-to-use interface to established algorithms is provided. 
The MF library expands the already vast usage of Orange methods in numerous applications without the need to use external
programming packages for factorizations.

.. note:: `Project wiki`_ is available at `Orange`_ site. 

.. note:: `Short presentation`_ about MF - Matrix Factorization Techniques for Data Mining library is available.

.. note:: `Document describing possible visualizations`_ is available. These visualizations will be included in near-future 
		  release of the library and possibly some supported as widgets when integration with Orange will be done.  

.. _Document describing possible visualizations: http://helikoid.si/mf/visualizations_MF.pdf

.. _Orange: http://orange.biolab.si

.. _Project wiki: http://orange.biolab.si/trac/wiki/MatrixFactorization

.. _Short presentation: http://helikoid.si/mf/GSoC_MF.pdf

###################
Scripting Reference
###################

.. toctree::
   :maxdepth: 3

   mf.mf_run

   mf.models
   
   mf.methods
   
   mf.utils
   
   mf.examples
   
   mf.datasets
	


#######
Content
#######

****************************
Matrix Factorization Methods
****************************

.. include:: /content-factorization.rst

**********************
Initialization Methods
**********************

.. include:: /content-initialization.rst

********************************
Quality and Performance Measures
********************************

.. include:: /content-quality-performance.rst

*****
Utils
*****

.. include:: /content-utils.rst


	
#######
Install
#######

************************************************************
MF - Matrix Factorization Techniques for Data Mining Library
************************************************************

No special installation procedure is specified. However, the MF library makes extensive use of `SciPy`_ and `NumPy`_ 
libraries for fast and convenient dense and sparse matrix manipulation and some linear
algebra operations. There are not any additional prerequisites. 

.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/

Download source code from `Github repository`_.

.. _Github repository: http://github.com/marinkaz/mf

To build and install run::
	
	python setup.py install

******************************************************************
MF - Matrix Factorization Techniques for Data Mining Documentation
******************************************************************

For building the documentation use Sphinx 1.0 or newer. Sphinx is available at `Sphinx home page`_ and
MF library documentation sources are available at `Github repository`_. Before building documentation, 
please install MF library.

Documentation can be built by issuing::

    make html

Resulting documentation is saved to `html` directory. Without make 
utility, execute::

	cd docs/source

    sphinx-build -b html <source dir [.]>  <build dir [html]>
    
from the MF library root directory. 

.. note:: The MF library documentation is contained in docs/source directory and in scripts in mf directory.  

.. _Sphinx home page: http://sphinx.pocoo.org




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

Following are three basic usage examples that employ Standard NMF algorithm and LSNMF algorithm. For examples with real world
applications see Examples section and methods' documentation. 

.. note:: Consider these examples as Hello World examples for the MF library.

.. note:: These examples demonstrate factorization of ``scipy.sparse`` matrix, ``numpy`` dense matrix, computation 
		  of quality and performance measures, error tracking, computation of MF estimate and matrix factors. 

Example No. 1

	.. literalinclude:: /code/usage.py
   		:lines: 7-50


Example No. 2

	.. literalinclude:: /code/usage.py
   		:lines: 58-100

Example No. 3

	.. literalinclude:: /code/usage.py
   		:lines: 108-153

##########
References
##########

.. include:: /content-references.rst



###############
Acknowledgement
###############

The MF - Matrix Factorization Techniques for Data Mining library is part of the Google Summer of Code 2011 program 
under supervision of the `Orange`_ organization and mentor `Prof. Blaz Zupan, PhD`_. 

.. _Prof. Blaz Zupan, PhD: http://www.biolab.si/en/blaz/

.. image:: http://orange.biolab.si/small-logo.png
	:target: http://orange.biolab.si
	

.. image:: http://code.google.com/p/google-summer-of-code/logo?cct=1280260724
	:target: http://code.google.com/soc/
	
	
##########
Disclaimer	
##########	
	
This software and data is provided as-is, and there are no guarantees
that it fits your purposes or that it is bug-free. Use it at your own 
risk! 

		

################
Index and Search
################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

