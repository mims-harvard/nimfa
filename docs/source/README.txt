Building
========

For building the documentation use Sphinx 1.0 or newer. It is available at
http://sphinx.pocoo.org/. Before building documentation, please install MF library.
Documentation can be built by issuing::

    make html

Resulting documentation is saved to `html` directory. Without make 
utility, follow::

	cd docs/source

    sphinx-build -b html <source dir [.]>  <build dir [html]>



