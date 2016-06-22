Nimfa
-----

|Travis|_
|Coverage|_

.. |Travis| image:: https://travis-ci.org/marinkaz/nimfa.svg?branch=master
.. _Travis: https://travis-ci.org/marinkaz/nimfa

.. |Coverage| image:: https://coveralls.io/repos/marinkaz/nimfa/badge.svg?branch=master&service=github
.. _Coverage: https://coveralls.io/github/marinkaz/nimfa?branch=master

Nimfa is a Python module that implements many algorithms for nonnegative matrix factorization.

The project was started in 2011 by Marinka Zitnik as a Google Summer of Code project, and since
then many volunteers have contributed. See the AUTHORS.rst file for a complete list of contributors.

It is currently maintained by a team of volunteers.

Important links
---------------

- Official source code repo: https://github.com/marinkaz/nimfa
- HTML documentation (stable release): http://nimfa.biolab.si
- Download releases: http://github.com/marinkaz/nimfa/releases
- Issue tracker: http://github.com/marinkaz/nimfa/issues

Dependencies
------------

Nimfa is tested to work under Python 2.6, Python 2.7, and Python 3.4.

The required dependencies to build the software are NumPy >= 1.7.0,
SciPy >= 0.12.0.

For running the examples Matplotlib >= 1.1.1 is required.

Install
-------

This package uses setuptools, which is a common way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

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

::

    @article{Zitnik2012,
      title     = {Nimfa: A Python Library for Nonnegative Matrix Factorization},
      author    = {Zitnik, Marinka and Zupan, Blaz},
      journal   = {Journal of Machine Learning Research},
      volume    = {13},
      pages     = {849-853},
      year      = {2012}
    }


License
-------

Nimfa - A Python Library for Nonnegative Matrix Factorization
Copyright (C) 2011-2016

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


JMLR Warranty
-------------

THIS SOURCE CODE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF MACHINE LEARNING RESEARCH (JMLR) 
AND JMLR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY 
AND RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR JMLR'S PUBLISHERS AND DISTRIBUTORS, WILL BE 
LIABLE FOR DAMAGES OF ANY KIND RESULTING FROM ITS USE. 

Without limiting the generality of the foregoing, neither the author, nor JMLR, nor JMLR's publishers and distributors, warrant that 
the Source Code will be error-free, will operate without interruption, or will meet the needs of the user.
