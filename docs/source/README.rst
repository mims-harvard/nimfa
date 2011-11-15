Building
========

For building the documentation use Sphinx 1.0 or newer. Sphinx is available at `Sphinx home page`_ and
nimfa library documentation sources are available at `Github repository`_. Before building documentation, 
please install nimfa library.

Documentation can be built by issuing::

    make html

Resulting documentation is saved to `html` directory. Without make 
utility, execute::

	cd docs/source

    sphinx-build -b html <source dir [.]>  <build dir [html]>
    
from the nimfa root directory. 

.. note:: The nimfa library documentation is contained in ``docs`` source directory and scripts are in ``nimfa`` directory.  

.. _Sphinx home page: http://sphinx.pocoo.org
.. _Github repository: http://github.com/marinkaz/mf
