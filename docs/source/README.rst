Building
========

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

.. note:: The MF library documentation is contained in MF/docs/source directory and in scripts in MF/mf.  

.. _Sphinx home page: http://sphinx.pocoo.org
.. _Github repository: http://github.com/marinkaz/mf
