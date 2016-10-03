Nimfa
-----

[![build: passing](https://img.shields.io/travis/marinkaz/nimfa.svg)](https://travis-ci.org/marinkaz/nimfa)
[![build: passing](https://coveralls.io/repos/marinkaz/nimfa/badge.svg)](https://coveralls.io/github/marinkaz/nimfa?branch=master)

Nimfa is a Python module that implements many algorithms for nonnegative matrix factorization. Nimfa is distributed under the BSD license.

The project was started in 2011 by Marinka Zitnik as a Google Summer of Code project, and since
then many volunteers have contributed. See AUTHORS file for a complete list of contributors.

It is currently maintained by a team of volunteers.

Important links
---------------

- Official source code repo: https://github.com/marinkaz/nimfa
- HTML documentation (stable release): http://nimfa.biolab.si
- Download releases: http://github.com/marinkaz/nimfa/releases
- Issue tracker: http://github.com/marinkaz/nimfa/issues

Dependencies
------------

Nimfa is tested to work under Python 2.7 and Python 3.4.

The required dependencies to build the software are NumPy >= 1.7.0,
SciPy >= 0.12.0.

For running the examples Matplotlib >= 1.1.1 is required.

Install
-------

This package uses setuptools, which is a common way of installing
python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:
    
    sudo python setup.py install

For more detailed installation instructions,
see the web page http://nimfa.biolab.si

Use
---

Run alternating least squares nonnegative matrix factorization with projected gradients and Random Vcol initialization algorithm on medulloblastoma gene expression data::

    >>> import nimfa
    >>> V = nimfa.examples.medulloblastoma.read(normalize=True)
    >>> lsnmf = nimfa.Lsnmf(V, seed='random_vcol', rank=50, max_iter=100)
    >>> lsnmf_fit = lsnmf()
    >>> print('Rss: %5.4f' % lsnmf_fit.fit.rss())
    Rss: 0.2668
    >>> print('Evar: %5.4f' % lsnmf_fit.fit.evar())
    Evar: 0.9997
    >>> print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
    K-L divergence: 38.8744
    >>> print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())
    Sparseness, W: 0.7297, H: 0.8796


Cite
----

    @article{Zitnik2012,
      title     = {Nimfa: A Python Library for Nonnegative Matrix Factorization},
      author    = {Zitnik, Marinka and Zupan, Blaz},
      journal   = {Journal of Machine Learning Research},
      volume    = {13},
      pages     = {849-853},
      year      = {2012}
    }
